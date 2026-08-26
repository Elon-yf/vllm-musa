# SPDX-License-Identifier: Apache-2.0
"""MATE-backed FlashInfer FP8 ScaledMM provider for MUSA."""

from __future__ import annotations

import torchada  # noqa: F401
import torch
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.utils.flashinfer import (
    bmm_fp8,
    has_musa_flashinfer_bmm_fp8,
)


def _as_mate_scalar_scale(scale: torch.Tensor, name: str) -> torch.Tensor:
    if scale.numel() != 1:
        raise RuntimeError(
            f"MATE FlashInfer per-tensor BMM requires one {name} value, "
            f"got shape {tuple(scale.shape)}"
        )
    return scale.reshape(())


def _musa_flashinfer_bmm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return bmm_fp8(
        a,
        b,
        _as_mate_scalar_scale(a_scale, "activation scale"),
        _as_mate_scalar_scale(b_scale, "weight scale"),
        out_dtype,
        out=None,
        backend="auto",
    )


def _musa_flashinfer_bmm_fp8_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del a_scale, b_scale
    return torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]),
        device=a.device,
        dtype=out_dtype,
    )


direct_register_custom_op(
    op_name="musa_flashinfer_bmm_fp8",
    op_func=_musa_flashinfer_bmm_fp8,
    fake_impl=_musa_flashinfer_bmm_fp8_fake,
)


class MUSAFlashInferFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Use MATE's FlashInfer-compatible BMM for per-tensor FP8 Linear."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_musa():
            return False, "requires MUSA."
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            if capability is not None:
                compute_capability = capability.major * 10 + capability.minor
        if compute_capability != 31:
            return False, "requires an MP31 MUSA device."
        if not has_musa_flashinfer_bmm_fp8():
            return False, "requires MATE flashinfer.bmm_fp8."
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not config.activation_quant_key.scale.group_shape.is_per_tensor():
            return False, "requires per-tensor activation scales."
        if not config.weight_quant_key.scale.group_shape.is_per_tensor():
            return False, "requires per-tensor weight scales."
        if config.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "requires BF16 or FP16 output."
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output = torch.ops.vllm.musa_flashinfer_bmm_fp8(
            A.unsqueeze(0),
            B.unsqueeze(0),
            As,
            Bs,
            out_dtype,
        ).view(A.shape[0], B.shape[1])
        if bias is not None:
            output = output + bias
        return output.view(*output_shape)


__all__ = ["MUSAFlashInferFP8ScaledMMLinearKernel"]
