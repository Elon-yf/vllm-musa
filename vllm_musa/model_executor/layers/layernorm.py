# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_current_vllm_config
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


def _qwen3_5_dense_fused_add_rmsnorm_min_rows(
    hidden_size: int,
    vllm_config: VllmConfig | None = None,
) -> int | None:
    """Return a row threshold guaranteeing at least 64 prefill tokens.

    Qwen3.6 checkpoints use vLLM's Qwen3.5 text architecture.  Limit the
    in-place specialization to the exact model configuration used for the E2E
    gate, and keep every possible decode-only batch on the functional path.
    """
    if vllm_config is None:
        vllm_config = get_current_vllm_config()

    model_config = vllm_config.model_config
    text_config = model_config.hf_text_config
    if (
        getattr(text_config, "model_type", None) != "qwen3_5_text"
        or hidden_size != 5120
        or getattr(model_config, "quantization", None) is not None
        or vllm_config.parallel_config.tensor_parallel_size != 8
    ):
        return None

    max_decode_query_len = 1
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None:
        max_decode_query_len += int(speculative_config.num_speculative_tokens or 0)
    max_decode_rows = vllm_config.scheduler_config.max_num_seqs * max_decode_query_len
    return max_decode_rows + 64


def _can_use_musa_jit_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    min_rows: int | None,
) -> bool:
    """Whether the validated Qwen3.6-27B specialization is applicable."""
    hidden_size = x.shape[-1]
    return (
        min_rows is not None
        and x.device.type == "musa"
        and residual.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and residual.dim() == 2
        and weight.dim() == 1
        and x.shape == residual.shape
        # The threshold is 64 rows above the configured decode-only ceiling,
        # so this cannot accidentally select a high-concurrency decode batch.
        and x.shape[0] >= min_rows
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
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps)
        self._musa_fused_add_rmsnorm_min_rows = (
            _qwen3_5_dense_fused_add_rmsnorm_min_rows(hidden_size)
        )

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
            if _can_use_musa_jit_fused_add_rmsnorm(
                x,
                residual,
                weight,
                self._musa_fused_add_rmsnorm_min_rows,
            ):
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
