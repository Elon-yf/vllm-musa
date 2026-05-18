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
    # MUSA mate's ragged_moe_gemm_8bit asserts scale_a.IsContiguous().
    # Both grouped FP8 GEMM call sites must materialise contiguous scales.
    # Discovered MUSA-0046 2026-05-14 while bringing up MiniMax-M2.7 FP8 on
    # yeahdongcn50: profile_run aborted with
    #   tvm.error.InternalError: Check failed: (scale_a.IsContiguous()) is
    #   false: scale_a must be contiguous
    # The dense FP8 path has the equivalent fix in patch #2 of
    # vllm__model_executor__layers__quantization__utils__fp8_utils.patch.py.
    # Gate+up GEMM (first call site):
    (
        """        mm1_out = _resize_cache(workspace2, (M_sum, N))
        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids
        )""",
        """        mm1_out = _resize_cache(workspace2, (M_sum, N))
        # MUSA mate ragged_moe_gemm_8bit asserts scale tensors are contiguous.
        a1q_scale = a1q_scale.contiguous()
        w1_scale = self.w1_scale.contiguous()
        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, w1_scale), mm1_out, expert_ids
        )""",
    ),
    # Down GEMM (second call site):
    (
        """        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )""",
        """        mm2_out = _resize_cache(workspace2, (M_sum, K))
        # MUSA mate ragged_moe_gemm_8bit asserts scale tensors are contiguous.
        a2q_scale = a2q_scale.contiguous()
        w2_scale = self.w2_scale.contiguous()
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, w2_scale), mm2_out, expert_ids
        )""",
    ),
]
