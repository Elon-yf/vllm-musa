# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PaddleOCR-VL-only MUSA rotary adapters.

The visual path reuses vllm-musa's existing strided rotary kernel. The
language path delegates to the pinned vLLM MRoPE implementation. Nothing here
registers a global CustomOp or changes platform defaults.
"""

from __future__ import annotations

import torch


def paddle_musa_fast_path(vllm_config: object) -> bool:
    """Narrow PaddleOCR-VL-1.6 MUSA scope; return false for other models."""
    if getattr(torch.version, "musa", None) is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) != "paddleocr_vl":
        return False
    vision = getattr(hf_config, "vision_config", None)
    text = getattr(hf_config, "text_config", hf_config)
    rope_parameters = getattr(text, "rope_parameters", None) or getattr(
        hf_config, "rope_parameters", None
    )
    return (
        getattr(vision, "hidden_size", None) == 1152
        and (
            getattr(vision, "num_hidden_layers", None) == 27
            or getattr(vision, "depth", None) == 27
        )
        and getattr(vision, "num_attention_heads", None) == 16
        and getattr(vision, "patch_size", None) == 14
        and getattr(vision, "image_size", None) == 384
        and getattr(text, "hidden_size", None) == 1024
        and getattr(text, "num_attention_heads", None) == 16
        and getattr(text, "num_key_value_heads", None) == 2
        and list((rope_parameters or {}).get("mrope_section", ())) == [16, 24, 24]
    )


def paddle_apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Use the existing MUSA rotary kernel with Paddle's FP32 contract."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Paddle rotary expects rank 3 or 4, got {x.ndim}")
    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError("cos and sin must have shape [seqlen, rotary_dim // 2]")
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    x_work = x if x.dtype == torch.float32 else x.float()
    cache = torch.cat((cos.float(), sin.float()), dim=-1).to(x_work.dtype)
    cache = cache.contiguous()
    positions = torch.arange(
        x_work.shape[1], device=x_work.device, dtype=torch.long
    ).expand(x_work.shape[0], -1)
    output = x_work.clone()
    from vllm_musa.jit_kernel.csrc.rope import rotary_embedding

    rotary_embedding(positions, output, None, x_work.shape[-1], cache, True)
    if squeeze:
        output = output.squeeze(0)
    return output.to(dtype=x.dtype) if x.dtype != torch.float32 else output


class PaddleMusaApplyRotary(torch.nn.Module):
    """Model-local ApplyRotary adapter; no global CustomOp registration."""

    def __init__(self, inner: torch.nn.Module, fast: bool) -> None:
        super().__init__()
        self.inner = inner
        self.fast = fast

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fast:
            return self.inner(x, cos, sin)
        x, cos, sin, origin_shape, origin_dtype = self.inner._pre_process(
            x, cos, sin
        )
        output = paddle_apply_rotary(x, cos, sin)
        return self.inner._post_process(output, origin_shape, origin_dtype)


class PaddleMusaRotaryWrapper(torch.nn.Module):
    """Keep the original language MRoPE module and its caches."""

    def __init__(self, inner: torch.nn.Module, fast: bool) -> None:
        super().__init__()
        self.inner = inner
        self.fast = fast

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        if not self.fast or positions.ndim != 2:
            return self.inner(positions, query, key)
        return self.inner.forward_cuda(positions, query, key)


def install_paddle_musa_rotary(
    visual: torch.nn.Module,
    language: torch.nn.Module,
    fast: bool,
) -> None:
    """Replace only Paddle's visual Apply and language MRoPE instances."""
    if not fast:
        return
    for module in list(visual.modules()):
        if module.__class__.__name__ == "SiglipAttention":
            module.apply_rotary_emb = PaddleMusaApplyRotary(
                module.apply_rotary_emb, fast=True
            )
    for module in list(language.modules()):
        self_attn = getattr(module, "self_attn", None)
        rotary_emb = getattr(self_attn, "rotary_emb", None)
        if rotary_emb is not None and hasattr(rotary_emb, "mrope_section"):
            module.self_attn.rotary_emb = PaddleMusaRotaryWrapper(
                rotary_emb, fast=True
            )
