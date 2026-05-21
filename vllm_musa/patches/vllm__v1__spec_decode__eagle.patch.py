# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0090 / MUSA-0109: Eagle3 full-loop CUDAGraph runner — dispatch + capture.

This patch wires `EagleFullLoopRunner` (vllm_musa/v1/spec_decode/) — which
captures vLLM's N-step Eagle3 draft loop as ONE cudagraph — into vLLM's
spec-decode path.

Capture-at-boot (MUSA-0109, 2026-05-20)
---------------------------------------
The runner is CAPTURED AT BOOT, not lazily on the first `propose()`.
Diagnosis (see ticket MUSA-0109) showed lazy mid-request capture is
fundamentally broken: capturing the collective-containing draft loop must
happen inside vLLM's `graph_capture()` context, and `graph_capture()`
cannot run mid-request — it collides with the engine's live serving
collectives (gloo `Connection reset by peer`).

`GPUModelRunner.capture_model()` holds `graph_capture(device)` open across
the whole boot capture phase, calling `drafter.dummy_run(...,
is_graph_capturing=True)` inside it. This patch hooks that:

  - `SpecDecodeBaseProposer.dummy_run` is wrapped. On the first
    `is_graph_capturing=True` call (boot, inside `graph_capture()`), it
    builds the `EagleFullLoopRunner`, synthesizes a bs=1
    `CommonAttentionMetadata`, and calls `runner.capture(cam,
    in_graph_capture=True)` — capture happens in the quiescent boot phase
    with the custom-AR communicator already in capture mode.
  - `EagleProposer.propose` is wrapped as a REPLAY-ONLY dispatcher: if a
    boot-captured runner exists, replay it; otherwise fall through to
    vLLM's iterative path. No lazy capture.

Gating: `VLLM_MUSA_EAGLE_RUNNER=1` opt-in (default off — the runner is not
yet end-to-end verified). With it off, both wrappers are pass-throughs.

Patch style: `PATCHES = []` — no file mutation; the monkey-patch install
is the import side effect. Idempotent via the `_musa_eagle_patched`
class marker.
"""

PATCHES: list = []  # No file mutation; the monkey-patch install is the side effect.

import logging
import os

_log = logging.getLogger(__name__)

_RUNNER_ENABLED = os.environ.get("VLLM_MUSA_EAGLE_RUNNER", "0") == "1"

# MUSA-0109 (2026-05-20): the boot capture bakes the draft attention's
# `max_seq_len` (a host int -> grid size) into the CUDAGraph; per-request
# `seq_lens`/`block_table`/`slot_mapping` are runner buffers and DO update
# at replay, but `max_seq_len` cannot. The earlier synthesis used
# `max_seq_len=1`, so the captured draft attention only ever iterated ~1
# KV block -> the draft saw a 1-token context at replay -> near-0%
# acceptance (measured 30.8 vs 86.9 tok/s). The capture must bake a
# `max_seq_len` >= the longest context the runner will replay. It is set
# generously here (covers the 4k-in/1k-out bench + margin); a request
# whose context exceeds it falls back to the iterative path via
# `can_run()`. Larger = correct for longer contexts but more wasted KV
# iteration per draft step, so it is a tunable, not max_model_len.
_CAPTURE_MAX_SEQ_LEN = int(
    os.environ.get("VLLM_MUSA_EAGLE_CAPTURE_SEQ_LEN", "8192")
)

# CRITICAL ORDER: import vllm_musa.v1.spec_decode.utils BEFORE eagle /
# llm_base_proposer. vllm-musa's utils module installs a MUSA-Triton-adapted
# `eagle_prepare_next_token_padded_kernel`. Importing the proposer modules
# first would bind the UPSTREAM unpatched kernel, which fails at MUSA Triton
# compile time with 'mismatched type for valid_count'.
import vllm_musa.v1.spec_decode.utils  # noqa: F401 — prime the kernel monkey-patch

try:
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
except ImportError as exc:  # pragma: no cover
    _log.debug(
        "MUSA-0090 eagle patch: cannot import proposer classes "
        "(probably an unsupported vllm version). Skipping: %s", exc,
    )
    EagleProposer = None  # type: ignore
    SpecDecodeBaseProposer = None  # type: ignore


def _is_musa() -> bool:
    try:
        from vllm.platforms import current_platform

        return bool(current_platform.is_musa())
    except Exception:  # pragma: no cover
        return False


def _probe_hidden_size(proposer) -> int:
    """Probe the draft model hidden size across vllm config layouts."""
    hidden_size = 0
    mc = getattr(proposer, "vllm_config", None)
    if mc is not None:
        model_config = getattr(mc, "model_config", None)
        if model_config is not None:
            try:
                hidden_size = int(model_config.get_hidden_size())
            except AttributeError:
                hidden_size = int(
                    getattr(getattr(model_config, "hf_config", None),
                            "hidden_size", 0)
                )
    if hidden_size <= 0:
        try:
            hidden_size = int(proposer.model.config.hidden_size)
        except Exception:
            hidden_size = 3072  # M2.5 draft default; surfaces in capture if wrong
        _log.warning("MUSA-0109: hidden_size probe fell back to %d", hidden_size)
    return hidden_size


def _build_runner(proposer, cudagraph_capture_sizes):
    """Construct (do not capture) an EagleFullLoopRunner for `proposer`."""
    import torch

    from vllm_musa.v1.spec_decode.eagle_full_loop_runner import EagleFullLoopRunner

    num_speculative_tokens = int(getattr(proposer, "num_speculative_tokens", 0))
    if num_speculative_tokens < 1:
        raise RuntimeError(
            f"num_speculative_tokens={num_speculative_tokens} is invalid; "
            "spec-decode appears not properly configured."
        )
    hidden_size = _probe_hidden_size(proposer)
    device = getattr(proposer, "device", None) or torch.device("cuda")
    return EagleFullLoopRunner(
        proposer=proposer,
        num_speculative_tokens=num_speculative_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        hidden_size=hidden_size,
        device=device,
    )


def _synthesize_base_metadata(proposer, batch_size: int = 1):
    """Build a structurally valid bs=1 `CommonAttentionMetadata` for capture.

    `EagleFullLoopRunner` overrides `block_table_tensor`, `slot_mapping`, and
    `seq_lens` with its own buffers (see attn_backend_array.py
    `build_per_step_attn_metadata_array`), so only structural validity +
    sane scalars matter here. `block_table_tensor` is sized so the runner's
    replay buffer (allocated from this shape) is not truncated for real
    long-context requests.
    """
    import torch

    from vllm.v1.attention.backend import CommonAttentionMetadata

    device = getattr(proposer, "device", None) or torch.device("cuda")

    # block_size: probe proposer / vllm cache config; MUSA page size is 64.
    block_size = int(getattr(proposer, "block_size", 0) or 0)
    if block_size <= 0:
        try:
            block_size = int(proposer.vllm_config.cache_config.block_size)
        except Exception:
            block_size = 64

    # max_model_len: probe proposer / vllm model config.
    max_model_len = int(getattr(proposer, "max_model_len", 0) or 0)
    if max_model_len <= 0:
        try:
            max_model_len = int(proposer.vllm_config.model_config.max_model_len)
        except Exception:
            max_model_len = 8192
    max_blocks = (max_model_len + block_size - 1) // block_size

    # MUSA-0109: capture with a realistic context length. `max_seq_len` is
    # baked into the CUDAGraph as a host int (it sizes the draft attention
    # grid); `seq_lens` here is what the per-step metadata builder derives
    # `cu_seqlens_k` / scheduler metadata from. Both must reflect a real
    # decode context, not 1, or the captured draft attention truncates to a
    # ~1-token context at replay. Clamp to max_model_len.
    capture_seq_len = min(_CAPTURE_MAX_SEQ_LEN, max_model_len)
    qsl = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    qsl_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
    seq_lens = torch.full(
        (batch_size,), capture_seq_len, dtype=torch.int32, device=device
    )
    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device
    )
    slot_mapping = torch.zeros(batch_size, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=capture_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )


def _draft_attn_kv_cache_blocks(proposer) -> int:
    """Largest `num_blocks` across the draft model's attention kv_cache
    tensors (0 if none bound yet).

    MUSA-0109 bug A: a captured CUDAGraph bakes the kv_cache DEVICE POINTER
    into the draft attention kernel. At boot, vLLM's capture phase runs the
    draft against a small dummy kv_cache (~8 blocks); the real per-engine
    cache (tens of thousands of blocks) is a DIFFERENT tensor. Capturing
    against the dummy makes the replayed draft attention read stale memory
    -> garbage attention output (cos~=0 vs eager). Boot capture must be
    deferred until this returns a real (large) block count.
    """
    import torch

    model = getattr(proposer, "model", None)
    if model is None:
        return 0
    best = 0
    for _name, mod in model.named_modules():
        kvc = getattr(mod, "kv_cache", None)
        if kvc is None:
            continue
        if isinstance(kvc, (list, tuple)):
            tensors = [t for t in kvc if isinstance(t, torch.Tensor)]
        elif isinstance(kvc, torch.Tensor):
            tensors = [kvc]
        else:
            tensors = []
        for t in tensors:
            if t.dim() >= 3:
                best = max(best, int(t.shape[1]))
            elif t.numel() > 0:
                best = max(best, int(t.numel()))
    return best


def _boot_capture_runner(proposer) -> None:
    """Build + capture the EagleFullLoopRunner at boot.

    MUST be called while inside vLLM's `graph_capture(device)` context (the
    boot capture phase). Sets `proposer._musa_full_loop_runner` on success,
    or to None on failure (dispatcher then uses the iterative path).
    """
    # BS=1 is the M2.5+Eagle3 SOTA workload; capture only that for now.
    runner = _build_runner(proposer, cudagraph_capture_sizes=[1])
    base_metadata = _synthesize_base_metadata(proposer, batch_size=1)
    runner.capture(base_metadata, in_graph_capture=True)
    proposer._musa_full_loop_runner = runner
    _log.info(
        "MUSA-0109: EagleFullLoopRunner captured AT BOOT "
        "(N=%d, capture_sizes=[1], buffer_footprint=%.2f MiB)",
        runner.num_steps, runner.memory_footprint_bytes() / 1024 / 1024,
    )


def _make_boot_capture_dummy_run(_original_dummy_run):
    """Wrap `SpecDecodeBaseProposer.dummy_run` to capture the runner at boot.

    On the first `is_graph_capturing=True` call (boot, inside
    `graph_capture()`), trigger `_boot_capture_runner` after the original
    dummy run. One-shot via `_musa_boot_capture_attempted`.
    """

    def _musa_boot_capture_dummy_run(self, *args, **kwargs):
        result = _original_dummy_run(self, *args, **kwargs)

        if not _RUNNER_ENABLED or not _is_musa():
            return result
        if not kwargs.get("is_graph_capturing", False):
            return result
        if getattr(self, "_musa_full_loop_runner", None) is not None:
            return result  # already captured

        # MUSA-0109 bug A: the captured draft attention bakes the kv_cache
        # device pointer. Defer capture until the REAL kv_cache is bound —
        # capturing against the boot dummy (~8 blocks) makes the replayed
        # draft attention read stale memory. Probe each is_graph_capturing
        # dummy_run; capture on the first one where the cache looks real.
        blocks = _draft_attn_kv_cache_blocks(self)
        n = getattr(self, "_musa_boot_capture_probe_n", 0) + 1
        self._musa_boot_capture_probe_n = n
        _log.info(
            "MUSA-0109 boot-capture probe #%d: draft attn kv_cache "
            "num_blocks=%d (need >=64 for the real cache)", n, blocks,
        )
        if blocks < 64:
            return result  # still the boot dummy cache — wait
        if getattr(self, "_musa_boot_capture_attempted", False):
            return result  # one-shot — do not retry
        self._musa_boot_capture_attempted = True

        try:
            _boot_capture_runner(self)
        except Exception as exc:
            _log.warning(
                "MUSA-0109: boot capture of EagleFullLoopRunner failed (%s); "
                "spec-decode will use vLLM's iterative path.", exc,
            )
            self._musa_full_loop_runner = None
        return result

    return _musa_boot_capture_dummy_run


def _make_dispatched_propose(_original_propose):
    """Wrap `EagleProposer.propose` as a replay-only dispatcher.

    If a boot-captured `EagleFullLoopRunner` exists, replay it; otherwise
    fall through to vLLM's iterative path. No lazy capture (capture happens
    only at boot — see `_make_boot_capture_dummy_run`).
    """

    def _musa_dispatched_propose(self, *args, **kwargs):
        if not _RUNNER_ENABLED or not _is_musa():
            return _original_propose(self, *args, **kwargs)

        runner = getattr(self, "_musa_full_loop_runner", None)
        if runner is None:
            return _original_propose(self, *args, **kwargs)

        # Extract args. EagleProposer.propose signature (vllm v0.20.1.dev0):
        #   def propose(self, target_token_ids, target_positions,
        #               target_hidden_states, next_token_ids,
        #               token_indices_to_sample, common_attn_metadata,
        #               sampling_metadata, ...)
        try:
            target_positions = (
                kwargs.get("target_positions")
                if "target_positions" in kwargs
                else args[1]
            )
            target_hidden_states = (
                kwargs.get("target_hidden_states")
                if "target_hidden_states" in kwargs
                else args[2]
            )
            next_token_ids = (
                kwargs.get("next_token_ids")
                if "next_token_ids" in kwargs
                else args[3]
            )
            token_indices_to_sample = (
                kwargs.get("token_indices_to_sample")
                if "token_indices_to_sample" in kwargs
                else (args[4] if len(args) > 4 else None)
            )
            common_attn_metadata = (
                kwargs.get("common_attn_metadata")
                if "common_attn_metadata" in kwargs
                else args[5]
            )
            # MUSA-0109 hybrid: the eager first forward needs the verified
            # token ids (args[0]) and, for the padded-drafter seq-len
            # adjustment, num_rejected_tokens_gpu (args[8], optional).
            target_token_ids = (
                kwargs.get("target_token_ids")
                if "target_token_ids" in kwargs
                else args[0]
            )
            num_rejected_tokens_gpu = (
                kwargs.get("num_rejected_tokens_gpu")
                if "num_rejected_tokens_gpu" in kwargs
                else (args[8] if len(args) > 8 else None)
            )
        except (IndexError, KeyError):
            _log.debug(
                "MUSA-0090: propose() signature mismatch; fallback to original"
            )
            return _original_propose(self, *args, **kwargs)

        batch_size = int(next_token_ids.shape[0])
        # MUSA-0109: `common_attn_metadata.max_seq_len` is a host int (the
        # batch's longest context) — no GPU sync. A request longer than the
        # captured context must use the iterative path.
        _req_max_seq_len = int(getattr(common_attn_metadata, "max_seq_len", 0) or 0)
        if not runner.can_run(batch_size, max_seq_len=_req_max_seq_len):
            return _original_propose(self, *args, **kwargs)

        try:
            result = runner.replay(
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                common_attn_metadata=common_attn_metadata,
                batch_size=batch_size,
                target_positions=target_positions,
                token_indices_to_sample=token_indices_to_sample,
                target_token_ids=target_token_ids,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
            return result.draft_token_ids
        except Exception as exc:
            _log.warning(
                "MUSA-0090: runner.replay failed (%s); falling back to "
                "vLLM iterative path for this call", exc,
            )
            return _original_propose(self, *args, **kwargs)

    return _musa_dispatched_propose


# Idempotent install.
if (
    EagleProposer is not None
    and SpecDecodeBaseProposer is not None
    and not getattr(EagleProposer, "_musa_eagle_patched", False)
):
    SpecDecodeBaseProposer.dummy_run = _make_boot_capture_dummy_run(
        SpecDecodeBaseProposer.dummy_run
    )
    EagleProposer.propose = _make_dispatched_propose(EagleProposer.propose)
    EagleProposer._musa_eagle_patched = True
    _log.info(
        "MUSA-0090/0109: eagle runner patched — boot-capture hook on "
        "SpecDecodeBaseProposer.dummy_run + replay dispatcher on "
        "EagleProposer.propose (VLLM_MUSA_EAGLE_RUNNER=%s)",
        "1" if _RUNNER_ENABLED else "0",
    )
