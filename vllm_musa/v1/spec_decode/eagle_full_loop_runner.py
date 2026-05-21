# MUSA-0090 step 3: EagleFullLoopRunner — captures vLLM's N-step Eagle3 draft
# loop as ONE cudagraph per batch size (vs vLLM's iterative per-step PIECEWISE
# dispatch). Adapted from SGLang's EAGLEDraftCudaGraphRunner shape to vLLM types.
#
# Design ref: ../../../../generated/musa0090_impl/step1-design-doc.md §3.3
# Q1 cudagraph smoke (proved correctness): ../../../../generated/musa0090_impl/
#     step1_5-q1-cudagraph-smoke-result.md
#
# This file is the skeleton. The hot-path inner loop (_capture_one_batch_size)
# has fully-typed signatures + explicit step-by-step pseudocode for the
# step-4 implementation to fill in. Step 3's deliverable is the structural
# scaffolding so the patch can hook in.

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from .attn_backend_array import (
    StepMetadataIndexing,
    build_per_step_attn_metadata_array,
)
from .spec_info import (
    EagleDraftBuffers,
    EagleFullLoopCaptureContext,
    EagleFullLoopReplayResult,
)

logger = logging.getLogger(__name__)


class EagleFullLoopRunner:
    """Captures the full N-step Eagle3 draft loop as one cudagraph per batch size.

    Public surface:
        __init__(proposer, num_speculative_tokens, cudagraph_capture_sizes,
                 hidden_size, device)
        capture()                           # one-time at server boot
        can_run(batch_size) -> bool         # dispatcher predicate
        replay(target_hidden_states, next_token_ids, common_attn_metadata,
               batch_size) -> EagleFullLoopReplayResult

    Architecture (mirrors SGLang's EAGLEDraftCudaGraphRunner):
      1. At init: hold reference to vLLM's EagleProposer. Don't capture yet.
      2. On first proposer.propose() call (lazy): the patch invokes
         runner.capture() to build one graph per cudagraph_capture_sizes
         entry. Each graph captures the full N-step draft loop using buffers
         from spec_info.py + per-step metadata from attn_backend_array.py.
      3. At runtime: patched propose() checks runner.can_run(bs); if true,
         dispatch to runner.replay() — single graph.replay() + 2 .copy_()
         calls (the input carry-over). No Python loop.

    Critical invariants (from design doc §6):
      - All per-step state mutations as in-graph tensor ops only.
      - Attn-metadata array tensors are VIEWS into buffers (not rebuilt).
      - forward_batch is reused, not rebuilt.
      - Graph capture before any spec request (warm boot expected).
      - MUSA-only: no-op on CUDA (gated at the patch level).
    """

    # ---- construction ----

    def __init__(
        self,
        proposer: Any,  # vllm.v1.spec_decode.eagle.EagleProposer
        num_speculative_tokens: int,
        cudagraph_capture_sizes: list[int],
        hidden_size: int,
        device: torch.device,
        topk: int = 1,  # chain drafting; tree path is MUSA-0094's scope
    ):
        if num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
            )
        if not cudagraph_capture_sizes:
            raise ValueError("cudagraph_capture_sizes must not be empty")

        self.proposer = proposer
        # MUSA-0109 hybrid (2026-05-21): the runner's captured graph covers
        # only the 1-token CHAIN tail. Draft token 0 comes from an eager
        # first forward over the verified sequence (replay() / vLLM's
        # propose() first-forward), which is variable-/multi-token and not
        # capturable. So the captured loop is num_speculative_tokens - 1
        # steps; replay() prepends draft-0 to reach num_speculative_tokens.
        self.num_spec_tokens = num_speculative_tokens
        self.num_steps = num_speculative_tokens - 1
        self.capture_sizes: list[int] = sorted(set(cudagraph_capture_sizes))
        self.max_bs: int = max(self.capture_sizes)
        self.hidden_size = hidden_size
        self.topk = topk
        self.device = device

        # MUSA-0109 A1: detect TREE mode. When the proposer was configured
        # with a `speculative_token_tree`, it exposes `cu_drafts_per_level`
        # / `child_drafts_per_level` (llm_base_proposer.py:280-300) and the
        # tree has depth > 1. In tree mode the runner captures the
        # `propose_tree` loop (`_run_tree_inner`); otherwise the chain loop.
        #
        # MUSA-0109 (2026-05-20): `propose_tree` only runs when the draft
        # attention backend is TREE_ATTN. On the MUSA SOTA stack the draft
        # always resolves to FLASH_ATTN (TREE_ATTN is registered but absent
        # from `_get_backend_priorities`), so `propose_tree` is NEVER called
        # and the draft is chain-style even with a `speculative_token_tree`
        # configured — the tree only shapes verification. Confirmed by the
        # MUSA-0109 POC probe and a live instrumented `propose()` probe
        # (2026-05-20): the tree-branch return never fires. Capturing a tree
        # draft loop here produces an N-node output the chain-oriented engine
        # bookkeeping (`draft_token_ids_cpu` sized `num_spec_tokens`) cannot
        # consume. Tree mode therefore stays opt-in behind
        # `VLLM_MUSA_EAGLE_TREE=1`; it is only meaningful once
        # `MUSATreeAttentionBackend` is actually selected for the draft.
        import os as _os_tree
        cu = getattr(proposer, "cu_drafts_per_level", None)
        _tree_opt_in = _os_tree.environ.get("VLLM_MUSA_EAGLE_TREE", "0") == "1"
        self._tree_mode: bool = _tree_opt_in and bool(cu) and len(cu) > 1
        if self._tree_mode:
            self._cu_drafts_per_level = [int(x) for x in cu]
            self._child_drafts_per_level = [
                int(x) for x in proposer.child_drafts_per_level
            ]
            self._total_drafts = self._cu_drafts_per_level[-1]
            self._tree_depth = len(self._cu_drafts_per_level)
        else:
            self._cu_drafts_per_level = []
            self._child_drafts_per_level = []
            self._total_drafts = num_speculative_tokens
            self._tree_depth = 0

        # Per-batch-size capture state. Populated by capture(); read by replay().
        self.contexts: dict[int, EagleFullLoopCaptureContext] = {}
        self._captured: bool = False
        # MUSA-0109: the context length baked into the captured draft
        # attention (`max_seq_len` is a host int -> graph grid size, cannot
        # change after capture). Set by capture(); `can_run()` rejects
        # requests whose context exceeds it so they fall back to the
        # iterative path instead of replaying a graph that would truncate
        # the draft attention.
        self._capture_seq_len: int = 0

        # MUSA-0109 (2026-05-21) FWDPROBE: VLLM_MUSA_EAGLE_FWDPROBE=1 captures
        # the draft model's per-step (last_hidden, next_hidden) into debug
        # buffers, then replay() runs the graph AND an eager pass over the
        # SAME seeded inputs and logs a direct graph-vs-eager comparison
        # (norm + cosine similarity). cos≈1 on last_hidden ⇒ the captured
        # model forward is correct and bug A is downstream (sampling/d2t);
        # cos≪1 ⇒ the captured model forward itself is corrupted.
        import os as _os_fp
        self._fwdprobe: bool = (
            _os_fp.environ.get("VLLM_MUSA_EAGLE_FWDPROBE", "0") == "1"
        )
        self._fwdprobe_count: int = 0
        self._fwdprobe_step: int = 0
        self._fwdprobe_active: bool = False
        self._fwdprobe_hooks_installed: bool = False
        self._dbg: dict[str, torch.Tensor] = {}
        if self._fwdprobe and self.num_steps > 0:
            # last/next = model() return; embed/attn/mlp = submodule
            # forward-hook outputs (filled by _install_fwdprobe_hooks).
            for _k in ("embed", "attn", "mlp", "last", "next"):
                self._dbg[_k] = torch.zeros(
                    self.num_steps, self.max_bs, hidden_size,
                    dtype=torch.bfloat16, device=device,
                )

        logger.info(
            "MUSA-0090/0109 EagleFullLoopRunner: configured for "
            "num_speculative_tokens=%d, capture_sizes=%s, max_bs=%d, "
            "hidden_size=%d, mode=%s%s",
            self.num_spec_tokens, self.capture_sizes, self.max_bs,
            self.hidden_size,
            "TREE" if self._tree_mode else "chain",
            (
                f" (cu_drafts={self._cu_drafts_per_level}, "
                f"total_drafts={self._total_drafts})"
                if self._tree_mode else ""
            ),
        )

    # ---- dispatcher predicate ----

    def can_run(self, batch_size: int, max_seq_len: int | None = None) -> bool:
        """Whether there's a captured graph compatible with this request.

        Strict equality match on batch size (no padding to next capture
        size); for BS=1 4k/1k workloads (the /goal shape), exact match
        suffices.

        `max_seq_len`: the request's current context length. The captured
        draft attention bakes `max_seq_len` as a host int, so a request
        whose context (plus the `num_steps` draft positions) exceeds the
        captured length must fall back to the iterative path — replaying
        the graph would truncate the draft attention and collapse
        acceptance. When None the seq-len guard is skipped.
        """
        if not self._captured:
            return False
        if batch_size not in self.contexts:
            return False
        if (
            max_seq_len is not None
            and self._capture_seq_len > 0
            and max_seq_len + self.num_spec_tokens > self._capture_seq_len
        ):
            return False
        return True

    # ---- one-time capture (called lazily on first propose()) ----

    def capture(self, base_metadata: Any, in_graph_capture: bool = False) -> None:
        """Capture one cudagraph per entry in self.capture_sizes.

        `base_metadata` is the CommonAttentionMetadata used as the template
        for per-step metadata.

        `in_graph_capture`: True when the caller is ALREADY inside vllm's
        `graph_capture(device)` context (the MUSA-0109 capture-at-boot path
        — `capture_model()` holds `graph_capture()` open across the whole
        boot capture phase). In that case `_capture_n_step_loop` must NOT
        re-enter `graph_capture()` (nesting `ca_comm.capture()` would
        assert); it captures with a plain `torch.cuda.graph(...)` on the
        already-active capture stream. False = standalone (legacy lazy)
        path: the runner opens its own `graph_capture()`.

        After this returns successfully, self._captured is True and
        self.contexts[bs] is populated for every bs in self.capture_sizes.

        Idempotent: subsequent calls are no-ops if already captured.
        """
        if self._captured:
            return
        # MUSA-0109: record the context length baked into the capture so
        # can_run() can reject longer-context requests.
        self._capture_seq_len = int(getattr(base_metadata, "max_seq_len", 0) or 0)
        if not hasattr(self.proposer, "model"):
            raise RuntimeError(
                "EagleProposer does not expose `model` attribute; check vLLM "
                "version compatibility (expected v0.20.1.dev0 shape — the "
                "draft model is exposed as `proposer.model`, NOT "
                "`proposer.draft_model`)."
            )

        # MUSA-0109 FWDPROBE: install submodule hooks BEFORE warmup/capture so
        # the captured graph includes the per-submodule snapshot copies.
        if self._fwdprobe:
            self._install_fwdprobe_hooks()

        # MUSA-0109 spike: use the platform's GLOBAL graph pool (lazily
        # created, shared across target + draft + spec captures). This
        # mirrors sglang's `device_module.graph_pool_handle()` shared-pool
        # pattern, which is the working reference on MUSA. The prior
        # per-runner `torch.cuda.graph_pool_handle()` created a SEPARATE
        # pool, and replay across distinct pools triggered the
        # torch_musa 2.9.0 allocator bug (`MUSA error: unknown error`
        # at capture/replay). Reusing vllm's existing global pool avoids
        # cross-pool interactions during replay.
        #
        # MUSA-0109 (2026-05-20) diagnostic: VLLM_MUSA_EAGLE_PER_RUNNER_POOL=1
        # forces a FRESH per-runner pool instead. The "per-runner pool
        # triggers the allocator bug" finding above predates the
        # capture-stream fix (capture now runs on a dedicated stream). The
        # shared global pool is LIVE — vllm's target graphs are captured
        # into it at boot and replayed before this lazy mid-request draft
        # capture. Capturing into a live shared pool is the prime suspect
        # for the post-capture hang; a fresh per-runner pool isolates the
        # draft capture from that live pool. One-shot retest lever.
        import os as _os
        _force_per_runner_pool = (
            _os.environ.get("VLLM_MUSA_EAGLE_PER_RUNNER_POOL", "0") == "1"
        )
        pool = None
        if _force_per_runner_pool:
            pool = torch.cuda.graph_pool_handle()
            logger.info(
                "MUSA-0109: VLLM_MUSA_EAGLE_PER_RUNNER_POOL=1 — using a "
                "fresh per-runner graph pool (isolated from vllm's live "
                "shared pool); pool=%s", pool,
            )
        else:
            try:
                from vllm.platforms import current_platform
                pool = current_platform.get_global_graph_pool()
            except Exception as exc:
                logger.warning(
                    "MUSA-0109: failed to acquire vllm global graph pool "
                    "(%s); will fall back to a fresh per-runner pool.",
                    exc,
                )
            # Validate the returned handle BEFORE using it. A platform that
            # doesn't implement get_global_graph_pool may return None;
            # passing None to torch.cuda.graph(..., pool=...) silently
            # allocates from the default pool.
            if pool is None:
                pool = torch.cuda.graph_pool_handle()
                logger.warning(
                    "MUSA-0109: get_global_graph_pool() returned None; "
                    "falling back to per-runner pool. pool=%s",
                    pool,
                )
            else:
                logger.info(
                    "MUSA-0109: EagleFullLoopRunner using vllm GLOBAL graph "
                    "pool (shared with target captures); pool=%s",
                    pool,
                )

        for bs in self.capture_sizes:
            try:
                ctx = self._capture_one_batch_size(
                    bs, base_metadata, pool, in_graph_capture=in_graph_capture
                )
                self.contexts[bs] = ctx
                logger.info(
                    "MUSA-0090 captured Eagle3 full-loop graph for bs=%d "
                    "(buffer footprint %.2f MiB)",
                    bs, ctx.memory_footprint_bytes() / 1024 / 1024,
                )
            except Exception as exc:
                logger.exception(
                    "MUSA-0090 graph capture FAILED at bs=%d: %s. "
                    "Falling back to iterative path for this size.",
                    bs, exc,
                )
                # Don't propagate — the patch will fall back to vLLM's
                # iterative path when can_run(bs) returns False for this size.

        self._captured = True

    def _capture_one_batch_size(
        self,
        batch_size: int,
        base_metadata: Any,
        pool: Any,
        in_graph_capture: bool = False,
    ) -> EagleFullLoopCaptureContext:
        """Capture the N-step draft loop for one batch size.

        STEP 4 implementation target. Step 3 leaves this as a structured
        skeleton with explicit pseudocode for what each phase must do.

        Phase 1: allocate buffers
        Phase 2: build per-step metadata array (views into buffers)
        Phase 3: warm-up the draft model on this bs (kernels must be
                 compiled before graph capture per torch.cuda.graph docs)
        Phase 4: open torch.cuda.graph context, drive the N-step loop with
                 in-place tensor ops only
        Phase 5: wrap and return context
        """
        if batch_size <= 0 or batch_size > self.max_bs:
            raise ValueError(
                f"batch_size {batch_size} out of range [1, {self.max_bs}]"
            )

        # MUSA-0109 A1: tree mode captures the propose_tree loop instead.
        if self._tree_mode:
            return self._capture_one_tree(
                batch_size, base_metadata, pool, in_graph_capture
            )

        # Phase 1: allocate buffers for this batch size.
        # block_table_tensor is held as a reference (not copied); we read it
        # via the slot-mapping kernel inside the captured graph.
        block_table_tensor = getattr(base_metadata, "block_table_tensor", None)
        if block_table_tensor is None:
            raise RuntimeError(
                "base_metadata lacks block_table_tensor; cannot capture without it"
            )
        buffers = EagleDraftBuffers.allocate(
            max_bs=self.max_bs,
            num_steps=self.num_steps,
            hidden_size=self.hidden_size,
            topk=self.topk,
            block_table_tensor=block_table_tensor,
            device=self.device,
        )

        # MUSA-0109 (2026-05-20): seed the seq_lens buffers to the capture
        # context length BEFORE Phase 2 builds the per-step metadata. The
        # FlashAttention builder derives `cu_seqlens_k` / scheduler metadata
        # from `seq_lens` at build time and bakes them into the graph; if
        # the buffers are still zero here those derived fields are baked for
        # a 0-length context. `base_metadata.max_seq_len` carries the
        # capture context length (set by the boot-capture patch). replay()
        # overwrites these buffers with the real per-request seq_lens.
        _cap_seq_len = int(getattr(base_metadata, "max_seq_len", 1) or 1)
        buffers.seq_lens_per_step.fill_(_cap_seq_len)
        buffers.seq_lens_in.fill_(_cap_seq_len)

        # Phase 2: pre-build per-step metadata array (views into buffers).
        # Pass `proposer` so the helper can call
        # proposer.build_per_group_and_layer_attn_metadata() to produce
        # per-layer backend-specific metadata that has the use_cascade /
        # common_prefix_len / etc. fields the compiled draft model expects
        # (see attn_backend_array.py docstring for the rationale).
        attn_metadata_array = build_per_step_attn_metadata_array(
            base_metadata=base_metadata,
            buffers=buffers,
            batch_size=batch_size,
            proposer=self.proposer,
        )
        # Verify the indexing math before committing to graph capture.
        # Skip verify_strict for per-layer dict metadata (StepMetadataIndexing
        # reads `.slot_mapping.data_ptr()` which is a CommonAttentionMetadata
        # field, not a per-layer-dict attribute).
        if attn_metadata_array and not isinstance(attn_metadata_array[0], dict):
            StepMetadataIndexing.inspect(attn_metadata_array).verify_strict()

        # Phase 3: warm-up. STEP 4 TODO — run one eager iteration to compile
        # kernels and exercise allocator hot-paths. Per torch.cuda.graph docs,
        # this is required to avoid stream/allocator surprises during capture.
        # Pattern (from Q1 smoke):
        #   s = torch.cuda.Stream()
        #   s.wait_stream(torch.cuda.current_stream())
        #   with torch.cuda.stream(s):
        #       self._run_one_step_eager(buffers, attn_metadata_array, 0,
        #                                 batch_size)
        #   torch.cuda.current_stream().wait_stream(s)
        #   torch.cuda.synchronize()
        self._warmup_eager(buffers, attn_metadata_array, batch_size)

        # Phase 4: capture the N-step loop. STEP 4 TODO — this is the
        # core deliverable. Pseudocode:
        #
        #   graph = torch.cuda.CUDAGraph()
        #   with torch.cuda.graph(graph, pool=pool):
        #       # Step 0: seed from input carry-over (target last hidden).
        #       # Copy bonus_token_ids_in -> input_ids_per_step[0]
        #       # Copy target_hidden_states_in -> hidden_states_per_step[0]
        #       buffers.input_ids_per_step[0, :batch_size].copy_(
        #           buffers.bonus_token_ids_in[:batch_size]
        #       )
        #       buffers.hidden_states_per_step[0, :batch_size].copy_(
        #           buffers.target_hidden_states_in[:batch_size]
        #       )
        #
        #       # The N-step loop body.
        #       for step_idx in range(self.num_steps):
        #           # Set forward_batch state from buffers[step_idx].
        #           # vLLM passes forward_batch through set_forward_context;
        #           # we DON'T use set_forward_context here because we
        #           # want the loop INSIDE the graph (no Python re-entry).
        #           forward_batch.input_ids = buffers.input_ids_per_step[step_idx, :batch_size]
        #           forward_batch.positions = buffers.positions_per_step[step_idx, :batch_size]
        #           forward_batch.slot_mapping = ... (from metadata_array[step_idx])
        #           forward_batch.attn_metadata = attn_metadata_array[step_idx]
        #
        #           # Call draft model forward (captures all its kernels).
        #           logits, hidden = self.proposer.draft_model(forward_batch)
        #           buffers.hidden_states_per_step[step_idx + 1].copy_(hidden)
        #
        #           # Sample topk=1 (chain). Captured in-graph.
        #           probs = torch.softmax(logits, dim=-1)
        #           topk_p, topk_idx = torch.topk(probs, k=self.topk)
        #           buffers.topk_p_per_step[step_idx + 1].copy_(topk_p)
        #           buffers.topk_index_per_step[step_idx + 1].copy_(topk_idx)
        #           buffers.input_ids_per_step[step_idx + 1, :batch_size].copy_(
        #               topk_idx.flatten()
        #           )
        #
        #           # Update positions + slot_mapping for next step via the
        #           # existing fused Triton kernel (already MUSA-adapted
        #           # for the .pyc-cache issue we hit in MUSA-0089).
        #           eagle_step_update_slot_mapping_and_metadata(
        #               positions_1d=buffers.positions_per_step[step_idx + 1],
        #               block_table_tensor=buffers.block_table_tensor,
        #               seq_lens=buffers.seq_lens_per_step[step_idx + 1],
        #               out_slot_mapping=buffers.slot_mapping_per_step[step_idx + 1],
        #               input_batch_size=batch_size,
        #               ...
        #           )
        #
        #           # Write the step's chosen token to the output tensor.
        #           buffers.draft_token_ids_out[:batch_size, step_idx].copy_(
        #               buffers.input_ids_per_step[step_idx + 1, :batch_size]
        #           )
        #
        #   # End of context manager = graph capture complete.
        graph = self._capture_n_step_loop(
            buffers, attn_metadata_array, batch_size, pool,
            in_graph_capture=in_graph_capture,
        )

        # Phase 5: wrap up and return.
        return EagleFullLoopCaptureContext(
            batch_size=batch_size,
            graph=graph,
            pool=pool,
            buffers=buffers,
            attn_metadata_array=attn_metadata_array,
        )

    # ---- tree capture (MUSA-0109 A1 increment 4) ----

    def _capture_one_tree(
        self,
        batch_size: int,
        base_metadata: Any,
        pool: Any,
        in_graph_capture: bool,
    ) -> EagleFullLoopCaptureContext:
        """Capture the narrow-d5 tree draft loop for one batch size.

        Mirrors `_capture_one_batch_size` but for the tree path: allocate
        `EagleTreeDraftBuffers`, build the per-level tree-attention metadata
        (plus the init-forward metadata as entry 0), warm up, capture one
        graph wrapping `_run_tree_inner`.
        """
        from .attn_backend_array import build_per_level_tree_attn_metadata
        from .spec_info import EagleTreeDraftBuffers

        block_table_tensor = getattr(base_metadata, "block_table_tensor", None)
        if block_table_tensor is None:
            raise RuntimeError(
                "base_metadata lacks block_table_tensor; cannot capture tree"
            )

        buffers = EagleTreeDraftBuffers.allocate(
            max_bs=self.max_bs,
            cu_drafts_per_level=self._cu_drafts_per_level,
            child_drafts_per_level=self._child_drafts_per_level,
            hidden_size=self.hidden_size,
            block_table_tensor=block_table_tensor,
            device=self.device,
        )

        # Metadata: entry 0 = the init forward (1 token/req) — vllm's
        # `propose()` builds it via build_per_group_and_layer_attn_metadata;
        # entries 1.. = the per-level accumulated-prefix forwards.
        _per_group, init_per_layer = (
            self.proposer.build_per_group_and_layer_attn_metadata(base_metadata)
        )
        level_metadata = build_per_level_tree_attn_metadata(
            base_metadata, buffers, batch_size, self.proposer
        )
        metadata_per_level = [init_per_layer] + level_metadata

        # Warm up the draft model on the tree shapes before capture.
        self._warmup_tree(buffers, metadata_per_level, batch_size)

        # Capture one graph wrapping the static-unrolled tree loop.
        graph = self._capture_tree_loop(
            buffers, metadata_per_level, batch_size, pool,
            in_graph_capture=in_graph_capture,
        )

        return EagleFullLoopCaptureContext(
            batch_size=batch_size,
            graph=graph,
            pool=pool,
            buffers=buffers,
            attn_metadata_array=metadata_per_level,
        )

    def _warmup_tree(
        self, buffers: Any, metadata_per_level: list, batch_size: int
    ) -> None:
        """Two eager passes of `_run_tree_inner` with full sync between —
        settles JIT/allocator state before capture (same rationale as
        `_warmup_eager`)."""
        from vllm.distributed import get_tp_group

        try:
            tp_barrier = get_tp_group().barrier
        except Exception:
            tp_barrier = lambda: None  # noqa: E731

        s = torch.cuda.Stream()
        for _ in range(2):
            torch.cuda.synchronize()
            tp_barrier()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_tree_inner(buffers, metadata_per_level, batch_size)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        tp_barrier()

    def _capture_tree_loop(
        self,
        buffers: Any,
        metadata_per_level: list,
        batch_size: int,
        pool: Any,
        in_graph_capture: bool = False,
    ) -> Any:
        """Capture `_run_tree_inner` as one graph. Same capture discipline as
        `_capture_n_step_loop` (the `in_graph_capture` distinction)."""
        graph = torch.cuda.CUDAGraph()
        if in_graph_capture:
            with torch.cuda.graph(
                graph, pool=pool, stream=torch.cuda.current_stream()
            ):
                self._run_tree_inner(buffers, metadata_per_level, batch_size)
            return graph

        from vllm.distributed.parallel_state import (
            graph_capture as _vllm_graph_capture,
        )

        with _vllm_graph_capture(self.device) as _gc_ctx:
            with torch.cuda.graph(graph, pool=pool, stream=_gc_ctx.stream):
                self._run_tree_inner(buffers, metadata_per_level, batch_size)
        return graph

    # ---- replay (hot path) ----

    def replay(
        self,
        target_hidden_states: torch.Tensor,  # bf16 [bs, hidden_size] OR [num_tokens, hidden_size]
        next_token_ids: torch.Tensor,        # int32 [bs] (the bonus from verify)
        common_attn_metadata: Any,            # for assertion + step-0 metadata seeding
        batch_size: int,
        target_positions: torch.Tensor | None = None,  # int64 [num_tokens] (the verify pass's positions)
        token_indices_to_sample: torch.Tensor | None = None,  # int [bs] (idx in num_tokens of last token per seq)
        target_token_ids: torch.Tensor | None = None,  # int [num_tokens] verified token ids (first forward)
        num_rejected_tokens_gpu: torch.Tensor | None = None,  # [bs] padded-drafter rejected count
    ) -> EagleFullLoopReplayResult:
        """Replay the captured graph for this batch size.

        Single .replay() call + N .copy_() for input carry-over (5 buffers:
        hidden_states, bonus_token_ids, positions, slot_mapping, seq_lens).
        Versus vLLM's iterative path which pays N × (set_forward_context +
        metadata rebuild + dispatcher dispatch) per spec round.

        Args:
            target_hidden_states: target model's hidden states, shape
                [num_tokens, hidden_size] OR [batch_size, hidden_size].
                The runner selects the per-sequence-last-token slice via
                token_indices_to_sample.
            next_token_ids: bonus token ids from the previous verify pass,
                shape [batch_size]. The "free" token that target accepted.
            common_attn_metadata: source of step-0 slot_mapping and seq_lens
                seeding. The runner reads .slot_mapping[token_indices_to_sample]
                and .seq_lens directly.
            batch_size: actual batch size; must match a captured size.
            target_positions: optional [num_tokens] tensor of the verify pass's
                positions. The runner reads target_positions[token_indices_to_sample]
                to seed step-0 positions. If None, step-0 positions are left
                at the buffer's current value (legacy behavior — wrong, but
                preserved for backward compat during step 5k transition).
            token_indices_to_sample: optional [bs] tensor naming the index
                (within num_tokens) of the last token per sequence. If None,
                computed from common_attn_metadata.query_start_loc[1:] - 1.

        Returns:
            EagleFullLoopReplayResult with .draft_token_ids of shape
            [batch_size, num_speculative_tokens].
        """
        if not self._captured:
            raise RuntimeError("replay() called before capture()")
        if batch_size not in self.contexts:
            raise ValueError(
                f"no captured graph for batch_size={batch_size}; "
                f"capture_sizes={self.capture_sizes}"
            )

        ctx = self.contexts[batch_size]
        p = self.proposer

        # ---- PHASE 1: eager first forward over the verified sequence ----
        # Produces draft token 0 + the hidden/position state that seeds the
        # captured chain. vLLM's Eagle propose() first-forward is variable-
        # length (the K+1 verified tokens / the prompt on the first call) so
        # it cannot be a fixed CUDAGraph — run it eagerly, by calling the
        # proposer's own methods, so it is correct by construction.
        draft0, carry_hidden, carry_positions, cad = self._eager_first_forward(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            batch_size=batch_size,
        )

        # ---- PHASE 2: bridge — first-forward state -> captured-chain step-0
        # metadata. Mirrors propose() lines 528-580: the padded-drafter
        # seq-len adjustment, then ONE eagle_step_update advancing the carry
        # position/seq_len/slot_mapping by one (the chain's first forward is
        # at carry_position + 1, exactly as propose()'s chain iter 0).
        from vllm.v1.spec_decode.utils import (
            eagle_step_update_slot_mapping_and_metadata,
        )
        seq_lens = cad.seq_lens
        if self.num_spec_tokens > 1 and num_rejected_tokens_gpu is not None:
            seq_lens = seq_lens - num_rejected_tokens_gpu
        chain_seq = seq_lens[:batch_size].to(torch.int32).clone()

        cm_block_table = getattr(cad, "block_table_tensor", None)
        if cm_block_table is not None and cm_block_table.numel() > 0:
            src_bs = min(cm_block_table.shape[0], batch_size)
            src_nb = min(
                cm_block_table.shape[1], ctx.buffers.block_table_tensor.shape[1]
            )
            ctx.buffers.block_table_tensor[:src_bs, :src_nb].copy_(
                cm_block_table[:src_bs, :src_nb]
            )

        # eagle_step_update does seq_lens += 1 in place and writes the
        # advanced positions / slot_mapping into the runner buffers.
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=carry_positions[:batch_size],
            block_table_tensor=ctx.buffers.block_table_tensor,
            seq_lens=chain_seq,
            block_size=int(getattr(p, "block_size", 64) or 64),
            max_model_len=int(getattr(p, "max_model_len", 196_608) or 196_608),
            out_clamped_positions=ctx.buffers.positions_in[:batch_size],
            out_slot_mapping=ctx.buffers.slot_mapping_in[:batch_size],
            input_batch_size=batch_size,
        )
        ctx.buffers.seq_lens_in[:batch_size].copy_(chain_seq)

        # ---- PHASE 3: seed the captured chain's step-0 input ----
        ctx.buffers.bonus_token_ids_in[:batch_size].copy_(
            draft0.to(ctx.buffers.bonus_token_ids_in.dtype)
        )
        ctx.buffers.target_hidden_states_in[:batch_size].copy_(carry_hidden)

        # ---- PHASE 4: replay the captured chain (num_spec_tokens - 1 steps).
        # VLLM_MUSA_EAGLE_EAGER_REPLAY=1 runs the chain eagerly (diagnostic —
        # splits a captured-graph bug from a loop-logic bug).
        # VLLM_MUSA_EAGLE_FWDPROBE=1 runs the graph THEN an eager pass over
        # the same seeded inputs and logs a graph-vs-eager comparison of the
        # draft model's per-step outputs (bug-A localisation). ----
        import os as _os_er
        if _os_er.environ.get("VLLM_MUSA_EAGLE_EAGER_REPLAY", "0") == "1":
            self._run_n_step_inner(
                ctx.buffers, ctx.attn_metadata_array, batch_size
            )
        elif getattr(self, "_fwdprobe", False) and self.num_steps > 0:
            # Graph replay -> snapshot the captured draft outputs; then an
            # eager re-run of the IDENTICAL _run_n_step_inner (it re-seeds
            # step-0 from the still-seeded *_in buffers) -> snapshot eager
            # outputs. The eager pass also fills draft_token_ids_out last,
            # so the probe run still serves correct tokens.
            ctx.graph.replay()
            g = {k: v.clone() for k, v in self._dbg.items()}
            g_out = ctx.buffers.draft_token_ids_out[:batch_size].clone()
            self._run_n_step_inner(
                ctx.buffers, ctx.attn_metadata_array, batch_size
            )
            e = {k: v.clone() for k, v in self._dbg.items()}
            e_out = ctx.buffers.draft_token_ids_out[:batch_size].clone()
            self._fwdprobe_log(g, e, g_out, e_out)
        else:
            ctx.graph.replay()

        # MUSA-0109 (2026-05-21) CHAINDBG: dumps the full chain trajectory.
        # Fires after PHASE 4 in EITHER mode (graph or eager) — the per-step
        # buffers are populated by both, and the GPU->CPU sync here is safe
        # (replay() is plain Python, never captured). Comparing the graph
        # dump vs the eager dump pinpoints whether the captured graph
        # diverges in bookkeeping (positions/slots) or the model forward.
        if _os_er.environ.get("VLLM_MUSA_EAGLE_CHAINDBG", "0") == "1":
            _c = getattr(self, "_replay_dbg_count", 0)
            if _c < 12:
                self._replay_dbg_count = _c + 1
                b = ctx.buffers
                logger.info(
                    "MUSA-0109-CHAINDBG#%d draft0=%s carry_pos=%s | "
                    "seeded(bonus=%s pos_in=%s seq_in=%s slot_in=%s) | "
                    "in_ids=%s positions=%s seq_lens=%s slots=%s | out=%s | "
                    "hid_norms=%s",
                    _c,
                    draft0.reshape(-1).tolist(),
                    carry_positions.reshape(-1)[:batch_size].tolist(),
                    b.bonus_token_ids_in[:batch_size].reshape(-1).tolist(),
                    b.positions_in[:batch_size].reshape(-1).tolist(),
                    b.seq_lens_in[:batch_size].reshape(-1).tolist(),
                    b.slot_mapping_in[:batch_size].reshape(-1).tolist(),
                    b.input_ids_per_step[:, :batch_size].reshape(-1).tolist(),
                    b.positions_per_step[:, :batch_size].reshape(-1).tolist(),
                    b.seq_lens_per_step[:, :batch_size].reshape(-1).tolist(),
                    b.slot_mapping_per_step[:, :batch_size].reshape(-1).tolist(),
                    b.draft_token_ids_out[:batch_size].reshape(-1).tolist(),
                    [
                        round(
                            float(
                                b.hidden_states_per_step[s, 0].float().norm().item()
                            ),
                            1,
                        )
                        for s in range(self.num_steps)
                    ],
                )
                # MUSA-0109 (2026-05-21) SLOTDBG: the chain reads context KV
                # via buffers.block_table_tensor; _eager_first_forward writes
                # the verified KV via cad.slot_mapping. If those disagree the
                # chain reads context it cannot find. Dump both + block_size.
                try:
                    _bs_blk = int(
                        getattr(self.proposer, "block_size", 0) or 0
                    )
                    _cad_sm = getattr(cad, "slot_mapping", None)
                    _cad_bt = getattr(cad, "block_table_tensor", None)
                    logger.info(
                        "MUSA-0109-SLOTDBG#%d block_size=%s | "
                        "cad.slot_mapping[-6:]=%s | cad.block_table[0,:8]=%s "
                        "| buffers.block_table[0,:8]=%s",
                        _c,
                        _bs_blk,
                        (_cad_sm.reshape(-1)[-6:].tolist()
                         if _cad_sm is not None else None),
                        (_cad_bt[0, :8].tolist()
                         if _cad_bt is not None and _cad_bt.numel() else None),
                        b.block_table_tensor[0, :8].tolist(),
                    )
                except Exception as _e:
                    logger.info("MUSA-0109-SLOTDBG#%d failed: %s", _c, _e)

        # ---- PHASE 5: assemble [draft0] ++ chain -> [bs, num_spec_tokens].
        # Clone out of the CUDAGraph pool (MUSA-0090 layer-2: a cross-stream
        # copy from pool memory fails MUDNN err 999). ----
        draft0_col = draft0.view(batch_size, 1)
        if self.num_steps > 0:
            chain = ctx.buffers.draft_token_ids_out[:batch_size]
            out = torch.cat([draft0_col.to(chain.dtype), chain], dim=1).clone()
        else:
            out = draft0_col.clone()
        return EagleFullLoopReplayResult(
            draft_token_ids=out,
            batch_size=batch_size,
        )

    def _fwdprobe_log(
        self,
        g: dict[str, torch.Tensor],
        e: dict[str, torch.Tensor],
        g_out: torch.Tensor,
        e_out: torch.Tensor,
    ) -> None:
        """Log a per-step, per-submodule graph-vs-eager comparison (MUSA-0109
        bug-A localisation). For each probe point (embed/attn/mlp/last/next)
        logs cosine similarity + norms. The FIRST probe point whose cosine
        drops is the op that does not replay correctly under capture."""
        c = self._fwdprobe_count
        if c >= 6:
            return
        self._fwdprobe_count = c + 1
        import torch.nn.functional as _F

        def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
            return float(
                _F.cosine_similarity(
                    a.float().reshape(-1), b.float().reshape(-1), dim=0
                )
            )

        for s in range(self.num_steps):
            parts = []
            for k in ("embed", "attn", "mlp", "last", "next"):
                gk = g[k][s, 0]
                ek = e[k][s, 0]
                parts.append(
                    "%s[cos=%.4f gN=%.1f eN=%.1f]"
                    % (k, _cos(gk, ek), float(gk.float().norm()),
                       float(ek.float().norm()))
                )
            logger.info(
                "MUSA-0109-FWDPROBE#%d step%d | %s", c, s, " ".join(parts)
            )
        logger.info(
            "MUSA-0109-FWDPROBE#%d g_out=%s e_out=%s",
            c, g_out.reshape(-1).tolist(), e_out.reshape(-1).tolist(),
        )

    def _install_fwdprobe_hooks(self) -> None:
        """Register forward hooks on the draft model's embed/attn/mlp so the
        captured graph snapshots each submodule's per-step output (MUSA-0109
        bug-A localisation). Hooks are inert unless `self._fwdprobe_active`
        (set only around the chain's `_run_n_step_inner`, NOT the eager
        first forward). The snapshot `.copy_()` is capture-safe."""
        if self._fwdprobe_hooks_installed:
            return
        self._fwdprobe_hooks_installed = True
        model = self.proposer.model

        def _make_hook(label: str):
            buf = self._dbg[label]

            def _hook(_m: Any, _inp: Any, out: Any) -> None:
                if not self._fwdprobe_active:
                    return
                t = out[0] if isinstance(out, tuple) else out
                if not isinstance(t, torch.Tensor) or t.dim() < 2:
                    return
                if t.shape[-1] != buf.shape[-1]:
                    return
                n = min(t.shape[0], buf.shape[1])
                buf[self._fwdprobe_step, :n].copy_(t[:n].detach())

            return _hook

        hooked = []
        for name, mod in model.named_modules():
            if name.endswith("embed_tokens"):
                mod.register_forward_hook(_make_hook("embed"))
                hooked.append((name, "embed"))
            elif name.endswith("layers.0.self_attn"):
                mod.register_forward_hook(_make_hook("attn"))
                hooked.append((name, "attn"))
            elif name.endswith("layers.0.mlp"):
                mod.register_forward_hook(_make_hook("mlp"))
                hooked.append((name, "mlp"))
        logger.info("MUSA-0109 FWDPROBE submodule hooks installed: %s", hooked)

    def _eager_first_forward(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: Any,
        num_rejected_tokens_gpu: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Run vLLM's Eagle `propose()` first-forward EAGERLY (not captured).

        Mirrors `EagleProposer.propose()` lines 433-516 by calling the
        proposer's own methods — the first forward is variable-length (the
        verified sequence / prompt) so it cannot be a fixed CUDAGraph;
        running it eagerly here is correct by construction. Returns
        `(draft_token_0 [bs], carry_hidden [bs, h], carry_positions [bs],
        cad)` — the seed for the captured chain.
        """
        from vllm.forward_context import set_forward_context

        p = self.proposer
        th = target_hidden_states
        if p.method in ("eagle3", "dflash") and hasattr(
            p.model, "combine_hidden_states"
        ):
            th = p.model.combine_hidden_states(th)

        num_tokens, tidx, cad = p.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=th,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        _per_group, per_layer = p.build_per_group_and_layer_attn_metadata(cad)
        cg_mode, num_input_tokens, dp = p._determine_batch_execution_and_padding(
            num_tokens
        )
        model_kwargs, slot_sz = p.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, None
        )
        with set_forward_context(
            per_layer,
            p.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=dp,
            cudagraph_runtime_mode=cg_mode,
            slot_mapping=p._get_slot_mapping(slot_sz, cad.slot_mapping),
        ):
            ret = p.model(**model_kwargs)
        if isinstance(ret, tuple):
            last_hidden, hidden = ret
        else:
            last_hidden, hidden = ret, ret

        draft0 = (
            p._greedy_sample(last_hidden[tidx]).to(torch.int32).view(batch_size)
        )
        carry_hidden = hidden[tidx]
        if target_positions.dim() == 2:
            carry_positions = target_positions[0][tidx]
        else:
            carry_positions = target_positions[tidx]
        return draft0, carry_hidden, carry_positions.to(torch.int64), cad

    # ---- helpers (step 4: real implementations) ----

    def _warmup_eager(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
    ) -> None:
        """Run TWO eager passes through the N-step loop with full sync between,
        BEFORE graph capture.

        MUSA-0109 layer-4 fix (2026-05-17): match SGLang's `_capture_init`
        pattern (sglang/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py:217)
        which is the difference between "MUSA-0090 always crashes" and
        "SGLang works on similar hardware". Specifically:

          1. torch.cuda.synchronize() — drain all pending GPU work
          2. tp_group.barrier() — cross-rank sync; ensures all TP ranks
             see the same allocator state before next warmup
          3. run_once_fn() — actually exercise the workload
          4. on_after_cuda_graph_warmup hook — backend-specific cleanup
          5. REPEAT 2x — settle JIT/Inductor compiles, allocator pool
             pages, and TP comm channels fully BEFORE capture

        Hypothesis: torch_musa's CUDAGraph + allocator interaction bug
        triggers when capture happens with un-settled allocator state.
        SGLang's 2x warmup + TP barrier pattern ensures the state is
        stable before capture opens.
        """
        from vllm.distributed import get_tp_group

        try:
            tp_group = get_tp_group()
            tp_barrier = lambda: tp_group.barrier()
        except Exception as exc:
            logger.debug("TP group unavailable for warmup barrier: %s", exc)
            tp_barrier = lambda: None

        # Side-stream warmup pattern (preserved from Q1 smoke)
        s = torch.cuda.Stream()
        for warmup_iter in range(2):
            torch.cuda.synchronize()
            tp_barrier()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_n_step_inner(buffers, attn_metadata_array, batch_size)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            # Per-attn-backend hook (SGLang's on_after_cuda_graph_warmup)
            hook = getattr(
                getattr(self.proposer, "draft_attn_backend", None),
                "on_after_cuda_graph_warmup",
                None,
            )
            if hook is not None:
                try:
                    hook()
                except Exception as exc:
                    logger.debug("on_after_cuda_graph_warmup raised: %s", exc)
        # Final barrier + sync to ensure all ranks see settled state
        torch.cuda.synchronize()
        tp_barrier()

    def _capture_n_step_loop(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
        pool: Any,
        in_graph_capture: bool = False,
    ) -> Any:
        """Open the torch.cuda.graph context and drive the N-step loop with
        in-place tensor ops only. Returns the captured CUDAGraph object.

        `in_graph_capture=True` means the caller is already inside vllm's
        `graph_capture(device)` (the capture-at-boot path); we then capture
        with a plain `torch.cuda.graph(...)` on the current (capture) stream
        instead of opening a nested `graph_capture()`.

        Critical invariants enforced inside `_run_n_step_inner`:
          - No Python list growth (no .append, no torch.cat).
          - No new tensor allocations (everything writes into `buffers`).
          - No tensor.item() / .tolist() / .cpu() calls (would CUDA-sync).
          - set_forward_context is allowed (pure Python state mutation; the
            actual GPU work it wraps is captured by torch.cuda.graph).
        """
        # torch.cuda.CUDAGraph is routed to torch_musa.musa_graph.MUSAGraph on
        # MUSA via torchada — proven correctness-wise by the Q1 smoke.
        graph = torch.cuda.CUDAGraph()

        # MUSA-0109 (2026-05-20): capture inside vllm's `graph_capture()`
        # context manager — NOT a bare `torch.cuda.graph(stream=...)`.
        #
        # History: an earlier fix gave the runner a dedicated capture stream
        # (`torch.cuda.graph(graph, pool=pool, stream=<own Stream>)`). That
        # cleared the capture-time `MUSA error: unknown error` — but the
        # `--load-format dummy` fast-loop then surfaced the real next failure:
        #   `MUSA error: operation not permitted when stream is capturing`
        # raised from the process-group watchdog thread. The TP=8 draft loop
        # contains custom all-reduces; capturing a collective is only valid
        # while the custom-all-reduce communicator is in capture mode.
        #
        # vllm's `graph_capture(device)` is exactly that wrapper: it enters
        # `GroupCoordinator.graph_capture()` -> `ca_comm.capture()`, which
        # puts the custom all-reduce into graph-capture mode (registers the
        # capture buffers, switches to the IPC-pointer path). It ALSO provides
        # the dedicated capture stream + `wait_stream` fence. This is the same
        # context vllm's own target-model capture uses, and the analogue of
        # SGLang's `GroupCoordinator.graph_capture()`. Capturing the draft
        # loop without it is what triggered the watchdog violation.
        if in_graph_capture:
            # MUSA-0109 capture-at-boot: vllm's `graph_capture()` is ALREADY
            # open (capture_model() holds it across the whole boot capture
            # phase, gpu_model_runner.py:6093). Re-entering it would nest
            # `ca_comm.capture()` and assert. Issuing multiple
            # `torch.cuda.graph(...)` captures inside one `graph_capture()`
            # is vllm's normal pattern (the CUDAGraphWrapper does it per
            # piecewise graph). Capture on the current stream — which, under
            # the active `graph_capture()`, IS the dedicated capture stream.
            with torch.cuda.graph(
                graph, pool=pool, stream=torch.cuda.current_stream()
            ):
                self._run_n_step_inner(buffers, attn_metadata_array, batch_size)
            return graph

        # Standalone (legacy lazy) path: open our own graph_capture().
        from vllm.distributed.parallel_state import (
            graph_capture as _vllm_graph_capture,
        )

        with _vllm_graph_capture(self.device) as _gc_ctx:
            with torch.cuda.graph(graph, pool=pool, stream=_gc_ctx.stream):
                self._run_n_step_inner(buffers, attn_metadata_array, batch_size)
        return graph

    def _run_n_step_inner(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
    ) -> None:
        """The N-step draft loop body. Called both by _warmup_eager (outside
        graph) and _capture_n_step_loop (inside graph). MUST use only
        in-place tensor ops + tensor-view assignments that the graph can
        capture.

        This mirrors vLLM's `EagleProposer.propose()` loop body
        (vllm/v1/spec_decode/llm_base_proposer.py:554-651) but
        statically-allocates everything ahead of time.
        """
        # Lazy imports to avoid pulling vLLM internals at module-load time.
        from vllm.forward_context import set_forward_context
        from vllm.config import CUDAGraphMode
        from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata

        proposer = self.proposer
        if getattr(self, "_fwdprobe", False):
            self._fwdprobe_active = True
        # Step-0 seed: copy the input carry-over into the per-step buffers.
        # bonus_token_ids_in is the bonus token from the previous verify pass
        # (the target's accepted last token); it's the input to the first
        # draft forward.
        buffers.input_ids_per_step[0, :batch_size].copy_(
            buffers.bonus_token_ids_in[:batch_size]
        )
        buffers.hidden_states_per_step[0, :batch_size].copy_(
            buffers.target_hidden_states_in[:batch_size]
        )
        # MUSA-0090 step 5k (2026-05-16): seed step-0 metadata from input
        # buffers populated by replay(). Without these copies, the captured
        # graph reads zeros at step 0 — wrong RoPE positions, wrong KV slot,
        # wrong attention seq_len. The eagle_step kernel below propagates
        # positions[step] -> positions[step+1] by adding 1; so position 0
        # wrong cascades through all downstream steps. This was the root
        # cause of acceptance=0.00 at positions 3-6 in step 5h bench.
        buffers.positions_per_step[0, :batch_size].copy_(
            buffers.positions_in[:batch_size]
        )
        buffers.slot_mapping_per_step[0, :batch_size].copy_(
            buffers.slot_mapping_in[:batch_size]
        )
        buffers.seq_lens_per_step[0, :batch_size].copy_(
            buffers.seq_lens_in[:batch_size]
        )

        # The N-step loop. Inside torch.cuda.graph(), every GPU op below is
        # captured. The Python for-loop unrolls at trace time — N is a static
        # int, not a tensor.
        for step in range(self.num_steps):
            if getattr(self, "_fwdprobe", False):
                self._fwdprobe_step = step
            # Views into this step's slice of the buffers. .narrow-style
            # indexing on a contiguous tensor returns a view; the captured
            # graph captures the kernel launches that consume these views.
            input_ids_view = buffers.input_ids_per_step[step, :batch_size]
            positions_view = buffers.positions_per_step[step, :batch_size]
            hidden_view = buffers.hidden_states_per_step[step, :batch_size]
            slot_mapping_view = buffers.slot_mapping_per_step[step, :batch_size]

            # Build the model kwargs. vLLM's draft model signature:
            #   model(input_ids=..., positions=..., inputs_embeds=None,
            #         hidden_states=...)
            # Returns either (last_hidden, hidden) tuple or just last_hidden.
            model_kwargs = {
                "input_ids": input_ids_view,
                "positions": positions_view,
                "inputs_embeds": None,
            }
            if getattr(proposer, "pass_hidden_states_to_model", True):
                # Eagle3 always passes hidden states from the previous step.
                model_kwargs["hidden_states"] = hidden_view

            # set_forward_context sets vLLM's thread-local context (attn
            # metadata, slot_mapping, num_tokens) for the model call.
            # Critical: cudagraph_runtime_mode=NONE — we're capturing OUR
            # OWN graph; vLLM's PIECEWISE shouldn't re-fire.
            #
            # MUSA-0109 (2026-05-21) FIX: `slot_mapping` MUST be passed — and
            # it must be the per-LAYER DICT, not a raw tensor. vLLM's KV-cache
            # write (`get_attention_context` -> `do_kv_cache_update`,
            # attention.py:703) reads the write slots from
            # `forward_context.slot_mapping[layer_name]` and SKIPS the write
            # entirely when that is None. Without this kwarg the chain's draft
            # model never writes its KV, so chain step s+1 reads stale KV for
            # step s's position -> draft acceptance collapses after position 0.
            # `proposer._get_slot_mapping(n, tensor)` builds the dict (and
            # copies the tensor into the proposer's stable `_slot_mapping_
            # buffer`, which the captured graph then reads — capture-safe).
            sm_dict = proposer._get_slot_mapping(batch_size, slot_mapping_view)
            with set_forward_context(
                attn_metadata_array[step],
                proposer.vllm_config,
                num_tokens=batch_size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=sm_dict,
            ):
                ret_hidden_states = proposer.model(**model_kwargs)
                if isinstance(ret_hidden_states, tuple):
                    last_hidden_states, next_hidden = ret_hidden_states
                else:
                    last_hidden_states = ret_hidden_states
                    next_hidden = ret_hidden_states

            # Sample greedy via the proposer's helper (lm_head + argmax).
            # Result dtype is int64 by default; vllm casts to int32 because
            # "tensor.argmax() returns int64 by default" and Eagle compile
            # requires int32 (per vllm comment line 556).
            draft_token_ids = proposer._greedy_sample(
                last_hidden_states[:batch_size]
            ).to(torch.int32)

            # Write the drafted token to the output tensor.
            buffers.draft_token_ids_out[:batch_size, step].copy_(draft_token_ids)

            # MUSA-0109 FWDPROBE: snapshot the draft model's raw outputs
            # (capture-safe in-place copies into persistent debug buffers).
            if getattr(self, "_fwdprobe", False):
                self._dbg["last"][step, :batch_size].copy_(
                    last_hidden_states[:batch_size]
                )
                self._dbg["next"][step, :batch_size].copy_(
                    next_hidden[:batch_size]
                )

            # Update state for step+1 if not the last step.
            if step + 1 < self.num_steps:
                # Stage next step's input.
                buffers.input_ids_per_step[step + 1, :batch_size].copy_(
                    draft_token_ids
                )
                buffers.hidden_states_per_step[step + 1, :batch_size].copy_(
                    next_hidden[:batch_size]
                )

                # MUSA-0090 step 5k: propagate seq_lens forward before the
                # kernel. The kernel does in-place +1 on its `seq_lens` arg
                # (reads, adds 1, writes back). If we passed
                # seq_lens_per_step[step+1] directly without seeding, the
                # kernel reads 0 and writes 1 — instead of reading
                # actual_seq_lens+step and writing actual_seq_lens+step+1.
                # Copy step's seq_lens to step+1's, then the kernel's
                # in-place +1 produces the correct step+1 value.
                buffers.seq_lens_per_step[step + 1, :batch_size].copy_(
                    buffers.seq_lens_per_step[step, :batch_size]
                )
                # Increment positions + recompute slot_mapping for step+1.
                # The fused kernel writes positions, seq_lens, slot_mapping
                # all in one launch — exactly what vLLM's iterative path
                # does at line 569-578 of llm_base_proposer.py.
                eagle_step_update_slot_mapping_and_metadata(
                    positions_1d=positions_view,
                    block_table_tensor=buffers.block_table_tensor,
                    seq_lens=buffers.seq_lens_per_step[step + 1, :batch_size],
                    block_size=getattr(proposer, "block_size", 16),
                    max_model_len=getattr(proposer, "max_model_len", 196_608),
                    out_clamped_positions=buffers.positions_per_step[
                        step + 1, :batch_size
                    ],
                    out_slot_mapping=buffers.slot_mapping_per_step[
                        step + 1, :batch_size
                    ],
                    input_batch_size=batch_size,
                )

    # ---- tree-aware loop body (MUSA-0109 A1 increment 2) ----

    def _run_tree_inner(
        self,
        buffers: Any,  # EagleTreeDraftBuffers
        metadata_per_level: list,
        batch_size: int,
    ) -> None:
        """Tree draft-loop body — narrow-d5 (MUSA-0109 A1). GPU-only; the
        `for level` loop is static so it unrolls at capture time, letting one
        torch.cuda.graph capture all forwards at their fixed query_lens.

        Faithful port of vLLM `propose()` initial-forward + `propose_tree()`
        (llm_base_proposer.py:460-1160), restructured to write into fixed
        pre-allocated tree buffers (no torch.cat).

        `metadata_per_level`: length `tree_depth`. [0] = init forward
        (1 token/req); [L+1] = the accumulated-prefix forward at level L
        (query_len cu_drafts_per_level[L]).
        """
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        proposer = self.proposer
        cu = buffers.cu_drafts_per_level          # e.g. (2,4,6,7,8)
        child = buffers.child_drafts_per_level    # e.g. (2,1,1,0,1)
        tree_depth = buffers.tree_depth
        total = buffers.total_drafts
        bs = batch_size
        h = buffers.hidden_size
        pass_hidden = getattr(proposer, "pass_hidden_states_to_model", True)
        block_size = int(getattr(proposer, "block_size", 64) or 64)

        # --- precompute per-node RoPE positions + KV slot mappings ---
        # Static given the base (bonus-token) position; all GPU ops, fully
        # capturable. Node p at tree level L (cu[L-1] <= p < cu[L]) has RoPE
        # position base+(L+1) — all nodes at a level share the depth.
        base_pos = buffers.positions_in[:bs]  # int64 [bs]
        for L in range(tree_depth):
            lo = int(cu[L - 1]) if L > 0 else 0
            hi = int(cu[L])
            buffers.tree_positions[:bs, lo:hi].copy_(
                (base_pos + (L + 1)).unsqueeze(1).expand(bs, hi - lo)
            )
        # KV slot per node: flattened offset 1..total gives each node a
        # distinct cache slot (propose_tree's flattened_draft_positions).
        offsets = torch.arange(
            1, total + 1, device=base_pos.device, dtype=torch.int64
        )
        abs_pos = base_pos.unsqueeze(1) + offsets.unsqueeze(0)  # [bs, total]
        block_num = abs_pos // block_size
        block_ids = (
            buffers.block_table_tensor[:bs].to(torch.int64).gather(1, block_num)
        )
        buffers.tree_slot_mapping[:bs, :].copy_(
            block_ids * block_size + abs_pos % block_size
        )

        def _fwd(meta, ids, pos, hid):
            kw = {"input_ids": ids, "positions": pos, "inputs_embeds": None}
            if pass_hidden:
                kw["hidden_states"] = hid
            with set_forward_context(
                meta,
                proposer.vllm_config,
                num_tokens=int(ids.shape[0]),
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                r = proposer.model(**kw)
            return (r[0], r[1]) if isinstance(r, tuple) else (r, r)

        def _sample(last_hidden, n_children):
            logits = proposer.model.compute_logits(last_hidden)
            if n_children <= 1:
                return logits.argmax(dim=-1).to(torch.int32)
            return torch.topk(
                logits, n_children, dim=-1
            ).indices.to(torch.int32)

        # --- initial forward: the bonus token -> root tokens ---
        last_h, next_h = _fwd(
            metadata_per_level[0],
            buffers.bonus_token_ids_in[:bs],
            base_pos,
            buffers.target_hidden_states_in[:bs],
        )
        n_root = int(cu[0])
        root = _sample(last_h[:bs], int(child[0])).view(bs, -1)[:, :n_root]
        buffers.tree_token_ids[:bs, 0:n_root].copy_(root)
        buffers.tree_hidden_states[:bs, 0:n_root].copy_(
            next_h[:bs].unsqueeze(1).expand(bs, n_root, h)
        )

        # --- tree level loop (static — unrolls at capture) ---
        level_num = n_root
        for level in range(tree_depth - 1):
            qlen = int(cu[level])
            last_h, next_h = _fwd(
                metadata_per_level[level + 1],
                buffers.tree_token_ids[:bs, :qlen].reshape(-1),
                buffers.tree_positions[:bs, :qlen].reshape(-1),
                buffers.tree_hidden_states[:bs, :qlen].reshape(bs * qlen, h),
            )
            # The last `level_num` nodes of the prefix are this level's
            # frontier — we sample their children.
            front_last = last_h[: bs * qlen].view(bs, qlen, h)[:, -level_num:]
            front_next = next_h[: bs * qlen].view(bs, qlen, h)[:, -level_num:]
            lo, hi = int(cu[level]), int(cu[level + 1])
            n_new = hi - lo
            nch = int(child[level + 1])
            toks = _sample(
                front_last.reshape(bs * level_num, h), nch
            ).view(bs, -1)[:, :n_new]
            buffers.tree_token_ids[:bs, lo:hi].copy_(toks)
            carry = (
                front_next.repeat_interleave(nch, dim=1)
                if nch > 1
                else front_next
            )
            buffers.tree_hidden_states[:bs, lo:hi].copy_(carry[:, :n_new])
            level_num = n_new

        buffers.draft_token_ids_out[:bs, :].copy_(
            buffers.tree_token_ids[:bs, :]
        )

    # ---- memory accounting ----

    def memory_footprint_bytes(self) -> int:
        """Total memory used by all captured contexts (buffer-only, excludes
        the shared graph memory pool)."""
        return sum(c.memory_footprint_bytes() for c in self.contexts.values())

    # ---- introspection ----

    def __repr__(self) -> str:
        return (
            f"EagleFullLoopRunner("
            f"N={self.num_steps}, "
            f"capture_sizes={self.capture_sizes}, "
            f"captured={self._captured}, "
            f"num_contexts={len(self.contexts)}, "
            f"hidden_size={self.hidden_size}, "
            f"topk={self.topk})"
        )
