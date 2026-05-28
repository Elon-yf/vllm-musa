# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 — gpu_model_runner.py.

Wires the caller-side path that makes the draft model's CUDAGraphWrapper(FULL)
actually capture entries during boot (instead of failing at request time with
``validate_cudagraph_capturing_enabled``).

Three changes:
  1. Add ``supports_sd_full_graph`` flag (True for Eagle + padded drafter).
  2. ``_dummy_run`` captures ``spec_decode_cm`` from ``_build_attention_metadata``
     and passes it to ``drafter.dummy_run`` as ``common_attn_metadata``.
  3. ``_dummy_run`` extends the ``use_cudagraphs`` predicate so FULL captures
     are enabled when ``supports_sd_full_graph``.

Skipped from PR #34880 (lower-priority, not strictly required to make the
draft FULL capture path engage on MUSA's BS=1 SOTA workload):
  - ``ExecuteModelState.batch_desc`` stash + ``propose_draft_token_ids``
    target_model_batch_desc threading. Our llm_base_proposer.patch.py
    derives ``uniform_decode`` locally from common_attn_metadata.max_query_len
    instead of taking it as an external arg.
  - ``input_batch.num_reqs_padded`` setter at execute_model.
  - ``_capture_cudagraphs`` clear_graphs cleanup.
  - ``_build_attn_group_metadata`` supports_sd_full_graph short-circuit
    (only matters for unpadded drafter batch, which we don't use).

History: this filename previously held the MUSA-0109 ``VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM``
default-stream workaround that targeted the now-deleted EagleFullLoopRunner
(MUSA-0203 cleanup). That patch is no longer reachable.
"""

# ---- Hunk 1: __init__ — initialize supports_sd_full_graph = False ----
_OLD_INIT_FLAG = """        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding."""

_NEW_INIT_FLAG = """        self.use_aux_hidden_state_outputs = False
        # MUSA-0203 / PR #34880: tracks whether the draft model supports
        # FULL-mode CUDA-graph capture (Eagle + padded drafter batch).
        self.supports_sd_full_graph = False
        # Set up speculative decoding."""

# ---- Hunk 2: __init__ — set the flag for Eagle proposers ----
_OLD_EAGLE_INIT = """            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )"""

_NEW_EAGLE_INIT = """            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
                # MUSA-0203 / PR #34880: enable FULL-mode draft capture when
                # padded drafter batch is enabled.
                self.supports_sd_full_graph = (
                    not self.speculative_config.disable_padded_drafter_batch
                )"""

# ---- Hunk 3: _dummy_run — capture spec_decode_cm from _build_attention_metadata ----
_OLD_BUILD_ATTN = """                attn_metadata, _ = self._build_attention_metadata("""

_NEW_BUILD_ATTN = """                attn_metadata, spec_decode_cm = self._build_attention_metadata("""

# ---- Hunk 4: _dummy_run — extend use_cudagraphs predicate ----
_OLD_USE_CG = """                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager"""

_NEW_USE_CG = """                # MUSA-0203 / PR #34880: Eagle now supports FULL cudagraphs via
                # CUDAGraphWrapper around the draft model (gated on
                # supports_sd_full_graph in __init__).
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and (
                            cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                            or self.supports_sd_full_graph
                        )
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager"""

# ---- Hunk 5: _dummy_run — pass common_attn_metadata to drafter.dummy_run ----
_OLD_DRAFTER_CALL = """                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )"""

_NEW_DRAFTER_CALL = """                self.drafter.dummy_run(
                    num_tokens,
                    common_attn_metadata=spec_decode_cm,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )"""

PATCHES = [
    (_OLD_INIT_FLAG, _NEW_INIT_FLAG),
    (_OLD_EAGLE_INIT, _NEW_EAGLE_INIT),
    (_OLD_BUILD_ATTN, _NEW_BUILD_ATTN),
    (_OLD_USE_CG, _NEW_USE_CG),
    (_OLD_DRAFTER_CALL, _NEW_DRAFTER_CALL),
]
