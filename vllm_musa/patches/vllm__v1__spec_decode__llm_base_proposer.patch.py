# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 — llm_base_proposer.py.

In upstream PR #34880 these hunks target ``vllm/v1/spec_decode/eagle.py``.
On v0.20.0-dev the EagleProposer implementation lives in
``vllm/v1/spec_decode/llm_base_proposer.py`` (eagle.py is a 22-line
stub). The hunks below re-target the same semantic changes to the
v0.20.0-dev file structure.

Changes:
  1. Import ``CUDAGraphWrapper`` + ``BatchDescriptor``.
  2. ``CudagraphDispatcher(self.vllm_config)`` -> ``(..., for_draft_model=True)``.
  3. ``initialize_cudagraph_keys``: pass through target's
     ``cudagraph_mode`` directly (was restricted to PIECEWISE).
  4. ``load_model``: wrap ``self.model`` with
     ``CUDAGraphWrapper(runtime_mode=FULL)`` when target has FULL
     captures (no-op when target is PIECEWISE-only).
  5. ``_determine_batch_execution_and_padding``: return 4-tuple
     (adds ``batch_desc``); thread ``uniform_decode`` +
     ``uniform_decode_query_len`` through.
  6. ``propose``: pass ``batch_descriptor=batch_desc`` to both
     ``set_forward_context`` calls (first forward + per-step loop).

Skipped from PR #34880 in this version (lower-priority for MUSA
PIECEWISE workload — can be added later):
  - ``target_model_batch_desc`` propose() parameter (gpu_model_runner
    caller change required).
  - ``dummy_run`` two-pass capture variant.
  - ``prepare_next_token_ids_padded`` / ``prepare_inputs_padded``
    padding fixes.

When upstream PR #34880 merges (or v0.20.0-dev rebases onto it), this
patch's anchors will stop matching and the file should be deleted.
"""

# ---- Hunk 1: import CUDAGraphWrapper + BatchDescriptor ----
_OLD_IMPORTS = """from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context"""

_NEW_IMPORTS = """from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context"""

# ---- Hunk 2: dispatcher init — for_draft_model=True ----
_OLD_DISPATCHER_INIT = """        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)"""

_NEW_DISPATCHER_INIT = """        # MUSA-0203 / PR #34880: dedicated dispatcher for the draft model so
        # FULL-mode capture keys are registered with uniform_decode_query_len=1
        # (each Eagle decode step is 1 token per request).
        self.cudagraph_dispatcher = CudagraphDispatcher(
            self.vllm_config, for_draft_model=True
        )"""

# ---- Hunk 3: initialize_cudagraph_keys — pass target mode through ----
_OLD_INIT_KEYS = """        \"\"\"Initialize cudagraph dispatcher keys for eagle.

        Eagle only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        \"\"\"
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)"""

_NEW_INIT_KEYS = """        \"\"\"Initialize cudagraph dispatcher keys for eagle.

        MUSA-0203 / PR #34880: pass the target's cudagraph_mode through
        directly. With ``for_draft_model=True`` the dispatcher now
        registers both the legacy mixed-mode (PIECEWISE) keys AND the
        new FULL-mode keys at uniform_decode_query_len=1, so the draft
        model can dispatch FULL captures when the target uses FULL or
        FULL_AND_PIECEWISE.
        \"\"\"
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_dispatcher.initialize_cudagraph_keys(cudagraph_mode)"""

# ---- Hunk 4: load_model — wrap with CUDAGraphWrapper ----
_OLD_LOAD_MODEL = """self.model = self._get_model()

        # Find draft layers"""

_NEW_LOAD_MODEL = """self.model = self._get_model()
        # MUSA-0203 / PR #34880: wrap the draft model with FULL-mode
        # CUDAGraphWrapper when target uses FULL captures. CUDAGraphWrapper
        # is a no-op when the runtime mode at call time doesn't match FULL,
        # so this is safe when target is PIECEWISE-only.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.vllm_config.parallel_config.use_ubatching
            and not self.speculative_config.disable_padded_drafter_batch
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )

        # Find draft layers"""

# ---- Hunk 5: _determine_batch_execution_and_padding signature + dispatch + return ----
_OLD_DETERMINE = """    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),"""

_NEW_DETERMINE = """    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        use_cudagraphs: bool = True,
        uniform_decode_query_len: int | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            uniform_decode,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
            uniform_decode_query_len=uniform_decode_query_len,"""

# Need to also patch the return statement of _determine_batch_execution_and_padding
# to 4-tuple. Find the unique anchor.
_OLD_DETERMINE_RETURN = """        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp"""

_NEW_DETERMINE_RETURN = """        return cudagraph_mode, batch_desc, num_tokens_padded, num_tokens_across_dp"""

# ---- Hunk 6: propose() first-forward _determine_batch unpack ----
# In v0.20.0-dev the propose() first forward is:
#   cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
#       self._determine_batch_execution_and_padding(num_tokens)
#   )
# We need 4-tuple unpack now.
_OLD_PROPOSE_FIRST_DET = """        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )"""

_NEW_PROPOSE_FIRST_DET = """        cudagraph_runtime_mode, batch_desc, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )"""

# ---- Hunk 7: propose() first-forward set_forward_context — thread batch_desc ----
_OLD_FFC_FIRST = """            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),"""

_NEW_FFC_FIRST = """            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_desc,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),"""

# ---- Hunk 8: propose() loop _determine_batch unpack ----
# The second occurrence is inside the loop, with `batch_size` not `num_tokens`.
_OLD_PROPOSE_LOOP_DET = """        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )"""

_NEW_PROPOSE_LOOP_DET = """        # MUSA-0203 / PR #34880: subsequent decode passes dispatch with
        # (batch_size, uniform=True, qdl=1) so the draft's FULL-mode capture
        # keys are exercised.
        cudagraph_runtime_mode, batch_desc, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(
                batch_size, uniform_decode=True, uniform_decode_query_len=1
            )
        )"""

# ---- Hunk 9: propose() loop set_forward_context — thread batch_desc ----
_OLD_FFC_LOOP = """                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),"""

_NEW_FFC_LOOP = """                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                slot_mapping=self._get_slot_mapping(input_batch_size),"""

# ---- Hunk 10: dummy_run() unpack — match 4-tuple shape ----
_OLD_DUMMY_UNPACK = """                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )"""

_NEW_DUMMY_UNPACK = """                cudagraph_runtime_mode, _, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )"""

PATCHES = [
    (_OLD_IMPORTS, _NEW_IMPORTS),
    (_OLD_DISPATCHER_INIT, _NEW_DISPATCHER_INIT),
    (_OLD_INIT_KEYS, _NEW_INIT_KEYS),
    (_OLD_LOAD_MODEL, _NEW_LOAD_MODEL),
    (_OLD_DETERMINE, _NEW_DETERMINE),
    (_OLD_DETERMINE_RETURN, _NEW_DETERMINE_RETURN),
    (_OLD_PROPOSE_FIRST_DET, _NEW_PROPOSE_FIRST_DET),
    (_OLD_FFC_FIRST, _NEW_FFC_FIRST),
    (_OLD_PROPOSE_LOOP_DET, _NEW_PROPOSE_LOOP_DET),
    (_OLD_FFC_LOOP, _NEW_FFC_LOOP),
    (_OLD_DUMMY_UNPACK, _NEW_DUMMY_UNPACK),
]
