# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.platforms import current_platform

from vllm_musa import _custom_ops as musa_ops


def _run_musa(
    self,
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    assert input_scale is None
    q_input, input_scale = torch.ops.vllm.triton_per_token_group_quant_fp8(
        input_2d,
        self.act_quant_group_shape.col,
    )
    return musa_ops.musa_w8a8_scaled_mm(
        q_input,
        weight,
        out_dtype=input_2d.dtype,
        scale_a=input_scale,
        scale_b=weight_scale,
        bias=None,
    )


def _dispatch_w8a8_blockscale_op(
    self,
    use_cutlass: bool,
    use_aiter_and_is_supported: bool,
) -> tuple[
    Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ],
        torch.Tensor,
    ],
    QuantFP8 | None,
]:
    if use_cutlass:
        return self._run_cutlass, (
            QuantFP8(
                False,
                self.act_quant_group_shape,
                column_major_scales=True,
                use_ue8m0=False,
            )
        )
    if use_aiter_and_is_supported:
        return self._run_aiter, None
    # ==================== MUSA ADAPTATION ====================
    if current_platform.is_musa():
        return self._run_musa, None
    # ========================== END ==========================
    return self._run_triton, (
        QuantFP8(
            False,
            self.act_quant_group_shape,
            column_major_scales=False,
            use_ue8m0=False,
        )
    )


def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor, ws: torch.Tensor, quant_block_shape: tuple[int], use_e8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return wq, ws


import vllm.model_executor.layers.quantization.utils.fp8_utils

vllm.model_executor.layers.quantization.utils.fp8_utils.deepgemm_post_process_fp8_weight_block = (
    deepgemm_post_process_fp8_weight_block
)
vllm.model_executor.layers.quantization.utils.fp8_utils.W8A8BlockFp8LinearOp._run_musa = (
    _run_musa
)
vllm.model_executor.layers.quantization.utils.fp8_utils.W8A8BlockFp8LinearOp._dispatch_w8a8_blockscale_op = (
    _dispatch_w8a8_blockscale_op
)
