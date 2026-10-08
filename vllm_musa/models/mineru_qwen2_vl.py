# SPDX-License-Identifier: Apache-2.0
"""MinerU-local Qwen2-VL rotary dispatch.

This file is intentionally model-local.  It only selects existing MUSA and
upstream rotary implementations after the exact MinerU geometry is matched;
the generic rotary custom ops remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


_TEXT_SIGNATURE = (896, 24, 14, 2, 64)
_VISION_SIGNATURE = (1280, 32, 16, 80)
_MROPE_SECTION = (8, 12, 12)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _mrope_section(config: Any) -> tuple[int, ...] | None:
    for owner in (config, _get(config, "text_config")):
        for name in ("mrope_section",):
            value = _get(owner, name)
            if value is not None:
                return tuple(int(v) for v in value)
        for name in ("rope_parameters", "rope_scaling"):
            value = _get(owner, name)
            if isinstance(value, dict) and value.get("mrope_section") is not None:
                return tuple(int(v) for v in value["mrope_section"])
    return None


def is_mineru_qwen2_vl_config(config: Any) -> bool:
    text = _get(config, "text_config", config)
    vision = _get(config, "vision_config")
    if vision is None or _get(config, "model_type") != "qwen2_vl":
        return False
    text_sig = (
        _get(text, "hidden_size"),
        _get(text, "num_hidden_layers"),
        _get(text, "num_attention_heads"),
        _get(text, "num_key_value_heads"),
        _get(text, "head_dim")
        or _get(text, "hidden_size") // _get(text, "num_attention_heads"),
    )
    vision_hidden = _get(vision, "embed_dim") or _get(vision, "hidden_size")
    vision_heads = _get(vision, "num_heads") or _get(vision, "num_attention_heads")
    vision_sig = (
        vision_hidden,
        _get(vision, "depth") or _get(vision, "num_hidden_layers"),
        vision_heads,
        vision_hidden // vision_heads,
    )
    return text_sig == _TEXT_SIGNATURE and vision_sig == _VISION_SIGNATURE and (
        _mrope_section(config) == _MROPE_SECTION
    )


def register_mineru_attention_backends(
    config: Any, registrar: Callable[[], None] | None = None
) -> bool:
    """Keep the existing FA registration local to the MinerU construction."""
    if not is_mineru_qwen2_vl_config(config):
        return False
    if registrar is None:
        from vllm_musa.platform import register_attention_backends

        registrar = register_attention_backends
    registrar()
    return True


def _musa_visual_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Adapt Qwen2-VL's cos/sin API to the existing MUSA rotary helper."""
    from vllm_musa.jit_kernel import rotary_embedding

    origin_shape = x.shape
    if x.ndim == 3:
        x = x.unsqueeze(0)
    batch, seq_len, _, head_size = x.shape
    out = x.contiguous().clone()
    positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
    positions = positions.expand(batch, seq_len).contiguous()
    cos_sin_cache = torch.cat((cos, sin), dim=-1).to(x.dtype).contiguous()
    rotary_embedding(
        positions,
        out,
        None,
        head_size,
        cos_sin_cache,
        True,
    )
    return out.squeeze(0) if len(origin_shape) == 3 else out


class _MineruVisualRotary(nn.Module):
    def __init__(self, reference: nn.Module):
        super().__init__()
        self.reference = reference

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if (
            x.device.type == "musa"
            and x.dtype == torch.bfloat16
            and x.shape[-1] == 80
            and cos.shape[-1] == 40
            and sin.shape == cos.shape
            and getattr(self.reference, "is_neox_style", True)
            and not getattr(self.reference, "enable_fp32_compute", False)
        ):
            return _musa_visual_rotary(x, cos, sin)
        return self.reference.forward_native(x, cos, sin)


class _MineruMRotary(nn.Module):
    def __init__(self, reference: nn.Module):
        super().__init__()
        self.reference = reference

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ):
        if (
            query.device.type == "musa"
            and positions.ndim == 2
            and key is not None
            and tuple(getattr(self.reference, "mrope_section", ()))
            == _MROPE_SECTION
            and getattr(self.reference, "head_size", None) == 64
            and getattr(self.reference, "rotary_dim", None) == 64
            and query.shape[-1] == 896
            and key.shape[-1] == 128
        ):
            from vllm.model_executor.layers.rotary_embedding.mrope import (
                triton_mrope,
            )

            cache = self.reference._match_cos_sin_cache_dtype(query)
            cos_sin = cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            q_shape, k_shape = query.shape, key.shape
            q, k = triton_mrope(
                query,
                key,
                cos,
                sin,
                self.reference.mrope_section,
                self.reference.head_size,
                self.reference.rotary_dim,
                self.reference.mrope_interleaved,
                self.reference.is_neox_style,
            )
            return q.reshape(q_shape), k.reshape(k_shape)
        return self.reference.forward_native(positions, query, key, offsets)


def _replace_children(module: nn.Module, predicate, factory) -> int:
    changed = 0
    for name, child in list(module.named_children()):
        if predicate(child):
            setattr(module, name, factory(child))
            changed += 1
        else:
            changed += _replace_children(child, predicate, factory)
    return changed


def patch_mineru_rotary(model: nn.Module) -> tuple[int, int]:
    visual = _replace_children(
        model,
        lambda m: m.__class__.__name__ == "ApplyRotaryEmb",
        _MineruVisualRotary,
    )
    language = _replace_children(
        model,
        lambda m: m.__class__.__name__ == "MRotaryEmbedding"
        and tuple(getattr(m, "mrope_section", ())) == _MROPE_SECTION
        and getattr(m, "head_size", None) == 64,
        _MineruMRotary,
    )
    return visual, language
