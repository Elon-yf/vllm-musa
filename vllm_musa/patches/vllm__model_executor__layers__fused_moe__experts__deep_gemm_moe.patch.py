# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.
"""

_OLD = """        if activation == MoEActivation.SILU:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
            )
"""

_NEW = """        if activation == MoEActivation.SILU:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            from vllm.platforms import current_platform

            if current_platform.is_musa():
                from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
                    silu_mul_per_token_group_quant_fp8_musa,
                )

                return silu_mul_per_token_group_quant_fp8_musa(
                    input=input,
                    output=output,
                    use_ue8m0=use_ue8m0,
                )
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
            )
"""

PATCHES = [
    (_OLD, _NEW),
]
