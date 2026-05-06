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


def _is_square_transposed_weight_view(
    weight: torch.Tensor, original_out_features: int, original_in_features: int
) -> bool:
    return (
        original_out_features == original_in_features
        and weight.dim() == 2
        and weight.shape == (original_out_features, original_in_features)
        and weight.stride(0) == 1
        and weight.stride(1) != 1
    )


def _logical_width_block_counts(logical_widths: list[int]) -> list[int]:
    return [(width + 127) // 128 for width in logical_widths]


def _expand_fp8_weight_scales(
    qweight_scales: torch.Tensor | None,
    qweight: torch.Tensor,
    logical_block_counts: list[int] | None = None,
) -> torch.Tensor | None:
    if qweight_scales is None:
        return None
    output_blocks = (qweight.shape[0] + 127) // 128
    input_blocks = (qweight.shape[1] + 127) // 128
    scale_rows, scale_columns = qweight_scales.shape
    if scale_rows == 1 or scale_rows == output_blocks:
        expanded_rows = qweight_scales
    elif logical_block_counts is not None and scale_rows == len(logical_block_counts):
        if sum(logical_block_counts) != output_blocks:
            raise ValueError(
                "MUSA FP8 GEMV logical shard block counts must match output blocks; "
                f"got {logical_block_counts} for {output_blocks} output blocks"
            )
        expanded_rows = qweight_scales.repeat_interleave(
            torch.tensor(logical_block_counts, device=qweight_scales.device), dim=0
        )
    elif 1 < scale_rows < output_blocks and output_blocks % scale_rows == 0:
        expanded_rows = qweight_scales.repeat_interleave(
            output_blocks // scale_rows, dim=0
        )
    else:
        raise ValueError(
            "MUSA FP8 GEMV requires either a single global weight scale, "
            "one weight scale row per 128 output channels, or logical shard scales "
            "matching layer.logical_widths; "
            f"got {scale_rows} scale rows for {output_blocks} output blocks"
        )
    if scale_columns not in {1, input_blocks}:
        raise ValueError(
            "MUSA FP8 GEMV requires either a single global input-block scale or "
            "one weight scale column per 128 input channels; "
            f"got {scale_columns} scale columns for {input_blocks} input blocks"
        )
    return expanded_rows.expand(output_blocks, input_blocks).contiguous()


class MUSAFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logical_widths = getattr(layer, "logical_widths", None)
        if logical_widths is None:
            self.logical_block_counts = None
        else:
            self.logical_block_counts = _logical_width_block_counts(list(logical_widths))
        super().process_weights_after_loading(layer)

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
        if hasattr(layer, w_s):
            weight_scale = getattr(layer, w_s)
        elif w_s == "weight_scale" and hasattr(layer, "weight_scale_inv"):
            weight_scale = getattr(layer, "weight_scale_inv")
        else:
            raise AttributeError(
                f"{type(layer).__name__} has no FP8 weight scale attribute "
                f"{w_s!r} or 'weight_scale_inv'"
            )
        return (
            getattr(layer, w),
            weight_scale,
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
        original_out_features, original_in_features = self.config.weight_shape
        if B.shape == (original_out_features, original_in_features):
            if _is_square_transposed_weight_view(
                B, original_out_features, original_in_features
            ):
                qweight = B.t().contiguous()
            else:
                qweight = B.contiguous()
        elif B.shape == (original_in_features, original_out_features):
            qweight = B.t().contiguous()
        else:
            raise ValueError(
                "MUSA FP8 GEMV weight shape must match either the original "
                "(out_features, in_features) weight layout or upstream's stored "
                "(in_features, out_features) FP8 layout; "
                f"got {tuple(B.shape)} for weight_shape={self.config.weight_shape}"
            )
        actual_output_shape = [*output_shape[:-1], original_out_features]
        qweight_scales = _expand_fp8_weight_scales(
            _scale_2d(Bs), qweight, getattr(self, "logical_block_counts", None)
        )
        output = musa_ops.musa_fused_gemv(
            A.contiguous(),
            qweight,
            _scale_2d(As),
            qweight_scales,
        )
        if bias is not None:
            output = output + bias
        return output.to(out_dtype).view(*actual_output_shape)


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
