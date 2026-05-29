# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0400/0402: dflash.py compatibility for the MUSA gpu_model_runner spec-decode path.

Two hunks:

1. (MUSA-0203, #34880 backport) After ``vllm__v1__spec_decode__llm_base_proposer``
   patches ``_determine_batch_execution_and_padding`` to return a 4-tuple,
   ``DFlashProposer.dummy_run`` must unpack the new shape too.

2. (MUSA-0402) The MUSA ``vllm__v1__worker__gpu_model_runner`` patch (Hunk 5)
   passes ``common_attn_metadata=spec_decode_cm`` to ``drafter.dummy_run`` for
   *every* proposer, but the upstream ``DFlashProposer.dummy_run`` override does
   not declare it (parallel drafting uses a single pass and does not consume it)
   -> boot ``TypeError: dummy_run() got an unexpected keyword argument``. Add the
   kwarg to the signature; the dflash body ignores it. ``CommonAttentionMetadata``
   is already imported in dflash.py.
"""

# Hunk 2 (MUSA-0402): accept (and ignore) common_attn_metadata in the dflash
# dummy_run override so the unconditional MUSA gpu_model_runner call type-checks.
_OLD_SIG = """    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:"""

_NEW_SIG = """    def dummy_run(
        self,
        num_tokens: int,
        common_attn_metadata: CommonAttentionMetadata | None = None,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:"""

# Hunk 1 (MUSA-0203 / #34880): 4-tuple unpack of _determine_batch_execution_and_padding.
_OLD = """        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )"""

_NEW = """        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, _, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )"""

PATCHES = [(_OLD_SIG, _NEW_SIG), (_OLD, _NEW)]
