# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

try:
    from vllm.model_executor.layers.layernorm import fused_add_rms_norm
except ImportError:
    fused_add_rms_norm = None

from vllm_musa import _custom_ops as musa_ops
from vllm_musa.jit_kernel.csrc import norm as musa_jit_norm
from vllm_musa.utils.environ import envs


def _can_use_musa_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    hidden_size = x.shape[-1]
    return (
        envs.VLLM_MUSA_FUSED_ADD_RMSNORM.get()
        and x.device.type == "musa"
        and residual.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and residual.dim() == 2
        and weight.dim() == 1
        and x.shape == residual.shape
        and hidden_size == weight.numel()
        and hidden_size % 8 == 0
        and hidden_size <= 16384
        and x.dtype in (torch.float16, torch.bfloat16)
        and residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and hasattr(torch.ops, "_C_musa_ops")
        and hasattr(torch.ops._C_musa_ops, "musa_fused_add_rms_norm")
    )


def _can_use_musa_jit_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    hidden_size = x.shape[-1]
    return (
        x.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and weight.dim() == 1
        and hidden_size > 0
        and hidden_size == weight.numel()
        and hidden_size <= 32768
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def _can_use_musa_jit_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether the validated Qwen3.6-27B specialization is applicable."""
    hidden_size = x.shape[-1]
    return (
        x.device.type == "musa"
        and residual.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and residual.dim() == 2
        and weight.dim() == 1
        and x.shape == residual.shape
        # The standalone kernel wins at all row counts, but an opaque custom
        # op boundary is not profitable for per-token decode. Keep the path on
        # prefill-sized batches where clean E2E A/B shows a repeatable gain.
        and x.shape[0] >= 64
        and hidden_size == weight.numel()
        # H2048 regresses Qwen3.6-MoE E2E despite winning standalone. Keep
        # this path scoped to the measured dense-model width.
        and hidden_size == 5120
        # The Qwen3.6 E2E gate was run in BF16. Keep FP16 available through
        # the low-level kernel, but do not auto-dispatch it without an E2E A/B.
        and x.dtype == torch.bfloat16
        and residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def _musa_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if _can_use_musa_fused_add_rms_norm(x, residual, weight):
        musa_ops.musa_fused_add_rms_norm(x, residual, weight, eps)
        return x, residual
    if fused_add_rms_norm is not None:
        return fused_add_rms_norm(x, residual, weight, eps)
    return None


@RMSNorm.register_oot
class MusaRMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
            or self.variance_size_override is not None
        ):
            return self.forward_native(x, residual)

        weight = self.weight.data
        eps = self.variance_epsilon

        if residual is not None:
            out = _musa_fused_add_rmsnorm(x, residual, weight, eps)
            if out is not None:
                return out
            return self.forward_native(x, residual)

        if _can_use_musa_jit_rmsnorm(x, weight):
            return musa_jit_norm.rmsnorm(x, weight, eps)

        return nn.functional.rms_norm(x, (self.hidden_size,), weight, eps)


@GemmaRMSNorm.register_oot
class MusaGemmaRMSNorm(GemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
            or getattr(self, "variance_size_override", None) is not None
        ):
            return self.forward_native(x, residual)

        weight = self.weight.data
        if residual is not None:
            if _can_use_musa_jit_fused_add_rmsnorm(x, residual, weight):
                return musa_jit_norm.fused_add_rmsnorm(
                    x,
                    residual,
                    weight,
                    self.variance_epsilon,
                    gemma=True,
                )
            return self.forward_native(x, residual)

        if _can_use_musa_jit_rmsnorm(x, weight):
            return musa_jit_norm.gemma_rmsnorm(x, weight, self.variance_epsilon)

        return self.forward_native(x, residual)
