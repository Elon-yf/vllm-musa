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
        if topk != 1:
            # Tree drafting is a follow-up ticket (MUSA-0094); chain-only here.
            raise NotImplementedError(
                f"topk={topk} (tree drafting) is out of scope for MUSA-0090; "
                "chain-only path (topk=1). Tree path is MUSA-0094."
            )

        self.proposer = proposer
        self.num_steps = num_speculative_tokens
        self.capture_sizes: list[int] = sorted(set(cudagraph_capture_sizes))
        self.max_bs: int = max(self.capture_sizes)
        self.hidden_size = hidden_size
        self.topk = topk
        self.device = device

        # Per-batch-size capture state. Populated by capture(); read by replay().
        self.contexts: dict[int, EagleFullLoopCaptureContext] = {}
        self._captured: bool = False

        logger.info(
            "MUSA-0090 EagleFullLoopRunner: configured for "
            "num_speculative_tokens=%d, capture_sizes=%s, max_bs=%d, "
            "hidden_size=%d, topk=%d",
            self.num_steps, self.capture_sizes, self.max_bs,
            self.hidden_size, self.topk,
        )

    # ---- dispatcher predicate ----

    def can_run(self, batch_size: int) -> bool:
        """Whether there's a captured graph compatible with this batch size.

        Strict equality match for now (no padding to next capture size).
        Step 4 may add padding logic if batch size variance is high; for
        BS=1 4k/1k workloads (the /goal shape), exact match suffices.
        """
        if not self._captured:
            return False
        return batch_size in self.contexts

    # ---- one-time capture (called lazily on first propose()) ----

    def capture(self, base_metadata: Any) -> None:
        """Capture one cudagraph per entry in self.capture_sizes.

        Called lazily on the first propose() call. `base_metadata` is the
        CommonAttentionMetadata vLLM would otherwise pass to the iterative
        propose loop; used as the template for per-step metadata.

        After this returns successfully, self._captured is True and
        self.contexts[bs] is populated for every bs in self.capture_sizes.

        Idempotent: subsequent calls are no-ops if already captured.
        """
        if self._captured:
            return
        if not hasattr(self.proposer, "draft_model"):
            raise RuntimeError(
                "EagleProposer does not expose draft_model; check vLLM version "
                "compatibility (expected v0.20.1.dev0 shape)."
            )

        # Acquire a shared graph memory pool. Step 4 should share with the
        # target model's pool if vllm-musa exposes a hook; for now, create
        # a fresh pool — slightly more memory but simpler.
        # Note: torch.cuda.graph_pool_handle() works on MUSA via torchada.
        pool = torch.cuda.graph_pool_handle()

        for bs in self.capture_sizes:
            try:
                ctx = self._capture_one_batch_size(bs, base_metadata, pool)
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

        # Phase 2: pre-build per-step metadata array (views into buffers).
        attn_metadata_array = build_per_step_attn_metadata_array(
            base_metadata=base_metadata,
            buffers=buffers,
            batch_size=batch_size,
        )
        # Verify the indexing math before committing to graph capture.
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
        )

        # Phase 5: wrap up and return.
        return EagleFullLoopCaptureContext(
            batch_size=batch_size,
            graph=graph,
            pool=pool,
            buffers=buffers,
            attn_metadata_array=attn_metadata_array,
        )

    # ---- replay (hot path) ----

    def replay(
        self,
        target_hidden_states: torch.Tensor,  # bf16 [bs, hidden_size]
        next_token_ids: torch.Tensor,        # int32 [bs] (the bonus from verify)
        common_attn_metadata: Any,            # for assertion only at replay time
        batch_size: int,
    ) -> EagleFullLoopReplayResult:
        """Replay the captured graph for this batch size.

        Single .replay() call + 2 .copy_() for input carry-over. Versus
        vLLM's iterative path which pays N × (set_forward_context + metadata
        rebuild + dispatcher dispatch) per spec round.

        Args:
            target_hidden_states: target model's last hidden state, shape
                [batch_size, hidden_size]. Used as input to step 0 of the
                draft loop.
            next_token_ids: bonus token ids from the previous verify pass,
                shape [batch_size]. The "free" token that target accepted.
            common_attn_metadata: passed for assertion only (verifies the
                caller's metadata is compatible with the captured shape).
            batch_size: actual batch size; must match a captured size.

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

        # Copy inputs into the runner's pre-allocated input buffers.
        # These copies happen OUTSIDE the graph — they're regular eager ops.
        ctx.buffers.target_hidden_states_in[:batch_size].copy_(target_hidden_states)
        ctx.buffers.bonus_token_ids_in[:batch_size].copy_(next_token_ids)

        # Single graph replay. All N draft forwards + sampling + slot-mapping
        # updates happen inside this one call.
        ctx.graph.replay()

        # Output is a view into the persistent buffer; caller must consume
        # before the next replay overwrites it.
        return EagleFullLoopReplayResult(
            draft_token_ids=ctx.buffers.draft_token_ids_out[:batch_size],
            batch_size=batch_size,
        )

    # ---- helpers (step-4 implementation targets) ----

    def _warmup_eager(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
    ) -> None:
        """Run one eager pass through the N-step loop before graph capture.

        STEP 4 TODO. Required by torch.cuda.graph to ensure all kernels are
        compiled and allocator pages are warm. The eager pass writes to the
        same buffers the captured graph will write to — i.e., it's the
        functional equivalent of the captured loop but executed outside
        any graph context.

        Per Q1 smoke (generated/musa0090_impl/step1_5-q1-cudagraph-smoke-result.md):
        run inside a side stream and join before the capture context opens.
        """
        # STEP 4: implement using the same kernel calls that will be
        # captured. Avoid any Python-level data-dependent control flow inside
        # this method that wouldn't replay correctly later.
        raise NotImplementedError("_warmup_eager: STEP 4 implementation pending")

    def _capture_n_step_loop(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
        pool: Any,
    ) -> Any:
        """Open the torch.cuda.graph context and drive the N-step loop with
        in-place tensor ops only. Returns the captured CUDAGraph object.

        STEP 4 TODO. The pseudocode in _capture_one_batch_size's docstring
        is the spec. Critical invariants enforced here:
          - No Python list growth (no .append, no torch.cat).
          - No new tensor allocations (everything writes into `buffers`).
          - No set_forward_context / dispatcher calls (loop is inside graph).
          - No tensor.item() / .tolist() / .cpu() calls (would CUDA-sync).
        """
        # STEP 4: torch.cuda.graph() + for loop. See pseudocode in
        # _capture_one_batch_size() docstring.
        raise NotImplementedError(
            "_capture_n_step_loop: STEP 4 implementation pending. "
            "Pseudocode in _capture_one_batch_size() docstring."
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
