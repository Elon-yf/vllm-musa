# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 — dflash.py tuple shape.

After ``vllm__v1__spec_decode__llm_base_proposer`` patches
``_determine_batch_execution_and_padding`` to return 4-tuple,
``DFlashProposer.dummy_run`` must unpack the new shape too.
"""

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

PATCHES = [(_OLD, _NEW)]
