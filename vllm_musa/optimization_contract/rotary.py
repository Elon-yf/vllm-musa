"""Shared adapters for model-selected MUSA rotary paths."""

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_musa.jit_kernel import rotary_embedding
from vllm_musa.utils.environ import envs

from .types import ModelFamily, ModelRole, MusaOptimizationContract, OptimizationFeature

class MusaVisionApplyRotaryEmb(ApplyRotaryEmb):
    """Adapt a vision x/cos/sin call to the existing MUSA rotary kernel."""

    def __init__(
        self,
        *,
        is_neox_style: bool,
        enable_fp32_compute: bool,
    ) -> None:
        super().__init__(
            enforce_enable=True,
            is_neox_style=is_neox_style,
            enable_fp32_compute=enable_fp32_compute,
        )

    def forward_oot(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "musa" or envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(x, cos, sin)
        x, cos, sin, shape, dtype = self._pre_process(x, cos, sin)
        output = x.clone()
        batch, seq_len, _, head_size = output.shape
        cache = torch.cat((cos, sin), dim=-1).to(output.dtype).contiguous()
        positions = torch.arange(
            seq_len, device=output.device, dtype=torch.long
        ).expand(batch, -1)
        query = output
        rotary_embedding(
            positions, query, None, head_size, cache, self.is_neox_style
        )
        return self._post_process(output, shape, dtype)


class MusaMRotaryEmbedding(nn.Module):
    """Use upstream MRoPE except for the validated MUSA 2-D-position case."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        args = (
            (positions, query, key)
            if offsets is None
            else (positions, query, key, offsets)
        )
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.inner.forward_native(*args)
        if query.device.type == "musa" and positions.ndim == 2 and key is not None:
            return self.inner.forward_cuda(*args)
        return self.inner(*args)


def install_vision_rotary(
    attentions: Iterable[nn.Module | None],
    *,
    expected_blocks: int | None = None,
) -> bool:
    """Install one adapter across a complete, already-selected vision stack."""
    active = tuple(attention for attention in attentions if attention is not None)
    if (
        not active
        or (expected_blocks is not None and len(active) != expected_blocks)
        or any(
            not isinstance(
                getattr(attention, "apply_rotary_emb", None), ApplyRotaryEmb
            )
            for attention in active
        )
    ):
        return False
    for attention in active:
        rotary = attention.apply_rotary_emb
        attention.apply_rotary_emb = MusaVisionApplyRotaryEmb(
            is_neox_style=rotary.is_neox_style,
            enable_fp32_compute=rotary.enable_fp32_compute,
        )
    return True


def install_mrope_rotary(
    layers: Iterable[nn.Module],
    *,
    set_neox_style: bool = False,
) -> None:
    """Wrap eligible language RoPE modules with the shared MUSA adapter."""
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        rotary = getattr(attention, "rotary_emb", None)
        if rotary is None:
            continue
        if set_neox_style:
            rotary.is_neox_style = True
        attention.rotary_emb = MusaMRotaryEmbedding(rotary)


def _paddleocr_vl_geometry(config: object) -> tuple[object, ...]:
    vision = getattr(config, "vision_config", None)
    text = getattr(config, "text_config", config)
    section = None
    for rope in (
        getattr(text, "rope_parameters", None),
        getattr(text, "rope_scaling", None),
        getattr(config, "rope_parameters", None),
        getattr(config, "rope_scaling", None),
    ):
        if isinstance(rope, dict) and rope.get("mrope_section") is not None:
            value = rope["mrope_section"]
            section = tuple(value) if isinstance(value, (list, tuple)) else None
            break
    return (
        getattr(vision, "hidden_size", None),
        getattr(vision, "num_hidden_layers", None),
        getattr(vision, "depth", None),
        getattr(vision, "num_attention_heads", None),
        getattr(vision, "patch_size", None),
        getattr(vision, "image_size", None),
        getattr(text, "hidden_size", None),
        getattr(text, "num_attention_heads", None),
        getattr(text, "num_key_value_heads", None),
        section,
    )


def resolve_rotary_contract(
    contract: MusaOptimizationContract, model_config: object
) -> MusaOptimizationContract:
    """Select exact model rotary geometry without model rules in the resolver."""
    config = getattr(model_config, "hf_config", None)
    if getattr(config, "model_type", None) == "paddleocr_vl":
        model = replace(contract.model, family=ModelFamily.PADDLEOCR_VL, role=ModelRole.TEXT)
        geometry = _paddleocr_vl_geometry(config)
        enabled = (
            geometry[0] == 1152
            and 27 in geometry[1:3]
            and geometry[3:] == (16, 14, 384, 1024, 16, 2, (16, 24, 24))
            and getattr(torch.version, "musa", None) is not None
        )
        features = frozenset({OptimizationFeature.PADDLEOCR_VL_ROTARY} if enabled else ())
        return replace(
            contract, model=model, profile="paddleocr_vl",
            supported_features=features, preferred_features=features,
        )
    return contract


def install_model_rotary(owner: nn.Module, vllm_config: object) -> bool:
    """Select and install every supported model rotary route in one place."""
    from vllm.platforms import current_platform

    if not current_platform.is_musa():
        return False
    from . import OptimizationFeature, resolve_optimization_contract

    features = resolve_optimization_contract(vllm_config).preferred_features
    paddle = getattr(OptimizationFeature, "PADDLEOCR_VL_ROTARY", None)
    if paddle is not None and paddle in features:
        visual = getattr(getattr(owner, "visual", None), "vision_model", None)
        visual = getattr(getattr(visual, "encoder", None), "layers", ())
        vision = ((getattr(layer, "self_attn", None) for layer in visual), {"expected_blocks": 27})
        language = (getattr(getattr(getattr(owner, "language_model", None), "model", None), "layers", ()), {"set_neox_style": True})
    else:
        return False
    if not install_vision_rotary(vision[0], **vision[1]):
        return False
    if language is not None:
        install_mrope_rotary(language[0], **language[1])
    return True
