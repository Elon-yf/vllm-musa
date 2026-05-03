import torch

from vllm.model_executor.kernels.linear import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform

from vllm_musa import _custom_ops as musa_ops


def _scale_2d(scale: torch.Tensor | None) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.dim() == 0:
        return scale.reshape(1, 1)
    if scale.dim() == 1:
        return scale.reshape(-1, 1)
    return scale.contiguous()


class MUSAFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_musa():
            return False, "requires MUSA."
        return True, None

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s, getattr(layer, "weight_scale_inv", None)),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

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
        qweight = B.t().contiguous()
        if qweight.shape[0] % 128 == 0 and qweight.shape[1] % 128 == 0:
            qweight_scales = Bs.contiguous()
        else:
            qweight_scales = _scale_2d(Bs)
        output = musa_ops.musa_fused_gemv(
            A.contiguous(),
            qweight,
            _scale_2d(As),
            qweight_scales,
        )
        if bias is not None:
            output = output + bias
        return output.to(out_dtype).view(*output_shape)


class MUSAFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_musa():
            return False, "requires MUSA."
        return True, None

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return super().can_implement(c)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return musa_ops.musa_fused_gemv(
            A.contiguous(),
            B.contiguous(),
            As.contiguous(),
            Bs.contiguous(),
        )
