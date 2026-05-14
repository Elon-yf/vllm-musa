# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.quantization.utils.fp8_utils.
"""

PATCHES = [
    # Patch per_token_group_quant_fp8 where per_token_group_fp8_quant need x is_contiguous
    (
        "if current_platform.is_cuda() and x.is_contiguous():",
        "if (current_platform.is_cuda() or current_platform.is_musa()) and x.is_contiguous():",
    ),
    (
        """        assert self.deepgemm_input_quant_op is not None
        q_input, input_scale = self.deepgemm_input_quant_op(input_2d)
        output = torch.empty(
""",
        """        assert self.deepgemm_input_quant_op is not None
        q_input, input_scale = self.deepgemm_input_quant_op(input_2d)
        # Ensure scale tensors are contiguous for DeepGemm
        input_scale = input_scale.contiguous()
        weight_scale = weight_scale.contiguous()
        output = torch.empty(
""",
    ),
    # MUSA namespace redirect: per_token_group_fp8_quant kernel is built by
    # vllm-musa under torch.ops._C_musa_ops, not torch.ops._C. The previous
    # patch (entry #1 above) enables MUSA to take the CUDA-style fast path,
    # so we must also redirect the actual op call to the _C_musa_ops
    # namespace. Without this the call raises:
    #   AttributeError: '_OpNamespace' '_C' object has no attribute
    #   'per_token_group_fp8_quant'
    # This affects MiniMax-M2.7 FP8 startup. Discovered MUSA-0046 2026-05-14.
    # Only the unpacked variant is redirected; _packed has no MUSA equivalent
    # and is on a different code path not exercised by MiniMax-M2.7.
    (
        "torch.ops._C.per_token_group_fp8_quant(",
        "torch.ops._C_musa_ops.per_token_group_fp8_quant(",
    ),
]
