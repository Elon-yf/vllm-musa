# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.fused_moe.deep_gemm_moe.
"""

PATCHES = [
    # Patch DeepGemmExperts.__init__, deep_gemm get_mk_alignment_for_contiguous_layout return [256,256] which not support ds.
    (
        "assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()",
        '''logger.debug_once(
            f'Warning: delete assert quant_config.block_shape={quant_config.block_shape} == '
            f'get_mk_alignment_for_contiguous_layout()={get_mk_alignment_for_contiguous_layout()}, '
            f'which may cause performance degradation or errors.'
        )''',
    ),
]
