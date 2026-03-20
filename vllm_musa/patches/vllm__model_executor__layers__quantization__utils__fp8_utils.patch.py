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
]
