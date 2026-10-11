"""Shared adapters for model-selected MUSA rotary paths."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_musa.jit_kernel import rotary_embedding


class MusaVisionRotaryPositions(nn.Module):
    """Keep positions alive when a visual rotary call is graph captured."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_positions", None, persistent=False)
        self._key: tuple[int, int, torch.device] | None = None
        self._graph_pinned = False

    def get(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        try:
            capturing = device.type == "musa" and bool(
                torch.cuda.is_current_stream_capturing()
            )
        except (AttributeError, RuntimeError):
            capturing = device.type == "musa"
        key = (batch, seq_len, device)
        if self._positions is not None and self._key == key:
            self._graph_pinned |= capturing
            return self._positions
        positions = torch.arange(seq_len, device=device).repeat(batch)
        if not capturing and not self._graph_pinned:
            self._positions, self._key = positions, key
        return positions


class MusaVisionApplyRotaryEmb(ApplyRotaryEmb):
    """Adapt a vision x/cos/sin call to the existing MUSA rotary kernel."""

    def __init__(
        self,
        *,
        is_neox_style: bool,
        enable_fp32_compute: bool,
        inplace: bool = False,
        flatten: bool = False,
        positions_cache: MusaVisionRotaryPositions | None = None,
        required_bf16_neox_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            enforce_enable=True,
            is_neox_style=is_neox_style,
            enable_fp32_compute=enable_fp32_compute,
        )
        self.inplace = inplace
        self.flatten = flatten
        self.positions_cache = positions_cache
        self.required_bf16_neox_shape = required_bf16_neox_shape

    def forward_oot(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "musa":
            return self.forward_native(x, cos, sin)
        if self.required_bf16_neox_shape is not None:
            head_size, cos_width = self.required_bf16_neox_shape
            if (
                x.dtype != torch.bfloat16
                or x.shape[-1] != head_size
                or cos.ndim != 2
                or cos.shape[-1] != cos_width
                or sin.shape != cos.shape
                or not self.is_neox_style
                or self.enable_fp32_compute
            ):
                return self.forward_native(x, cos, sin)
        x, cos, sin, shape, dtype = self._pre_process(x, cos, sin)
        output = x.contiguous() if self.inplace else x.clone()
        batch, seq_len, heads, head_size = output.shape
        cache = torch.cat((cos, sin), dim=-1).to(output.dtype).contiguous()
        if self.flatten:
            positions = (
                self.positions_cache.get(batch, seq_len, output.device)
                if self.positions_cache is not None
                else torch.arange(seq_len, device=output.device).repeat(batch)
            )
            query = output.reshape(batch * seq_len, heads * head_size)
        else:
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

    def __init__(
        self, inner: nn.Module, qk_hidden_sizes: tuple[int, int] | None = None
    ) -> None:
        super().__init__()
        self.inner = inner
        self.qk_hidden_sizes = qk_hidden_sizes

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
        if (
            query.device.type == "musa"
            and positions.ndim == 2
            and key is not None
            and (
                self.qk_hidden_sizes is None
                or (query.shape[-1], key.shape[-1]) == self.qk_hidden_sizes
            )
        ):
            return self.inner.forward_cuda(*args)
        return self.inner(*args)


def install_vision_rotary(
    attentions: Iterable[nn.Module | None],
    *,
    inplace: bool = False,
    flatten: bool = False,
    expected_blocks: int | None = None,
    required_bf16_neox_shape: tuple[int, int] | None = None,
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
    positions = MusaVisionRotaryPositions() if flatten else None
    for attention in active:
        rotary = attention.apply_rotary_emb
        attention.apply_rotary_emb = MusaVisionApplyRotaryEmb(
            is_neox_style=rotary.is_neox_style,
            enable_fp32_compute=rotary.enable_fp32_compute,
            inplace=inplace,
            flatten=flatten,
            positions_cache=positions,
            required_bf16_neox_shape=required_bf16_neox_shape,
        )
    return True


def install_mrope_rotary(
    layers: Iterable[nn.Module],
    *,
    mrope_section: tuple[int, ...] | None = None,
    qk_hidden_sizes: tuple[int, int] | None = None,
    head_size: int | None = None,
    rotary_dim: int | None = None,
    set_neox_style: bool = False,
) -> None:
    """Wrap eligible language RoPE modules with the shared MUSA adapter."""
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        rotary = getattr(attention, "rotary_emb", None)
        if (
            rotary is None
            or (mrope_section is not None and tuple(getattr(rotary, "mrope_section", ()) or ()) != mrope_section)
            or (head_size is not None and getattr(rotary, "head_size", None) != head_size)
            or (rotary_dim is not None and getattr(rotary, "rotary_dim", None) != rotary_dim)
        ):
            continue
        if set_neox_style:
            rotary.is_neox_style = True
        attention.rotary_emb = MusaMRotaryEmbedding(
            rotary, qk_hidden_sizes=qk_hidden_sizes
        )


def install_model_rotary(owner: nn.Module, vllm_config: object) -> bool:
    """Select and install every supported model rotary route in one place."""
    from vllm.platforms import current_platform

    if not current_platform.is_musa():
        return False
    from . import OptimizationFeature, resolve_optimization_contract

    features = resolve_optimization_contract(vllm_config).preferred_features
    paddle = getattr(OptimizationFeature, "PADDLEOCR_VL_ROTARY", None)
    mineru = getattr(OptimizationFeature, "MINERU_QWEN2_VL_ROTARY", None)
    qwen35 = getattr(OptimizationFeature, "QWEN35_VISION_ROTARY_BF16", None)
    if paddle is not None and paddle in features:
        visual = getattr(getattr(owner, "visual", None), "vision_model", None)
        visual = getattr(getattr(visual, "encoder", None), "layers", ())
        vision = ((getattr(layer, "self_attn", None) for layer in visual), {"expected_blocks": 27})
        language = (getattr(getattr(getattr(owner, "language_model", None), "model", None), "layers", ()), {"set_neox_style": True})
    elif mineru is not None and mineru in features:
        visual = getattr(owner, "visual", None)
        vision = ((getattr(block, "attn", None) for block in getattr(visual, "blocks", ())), {"expected_blocks": 32, "required_bf16_neox_shape": (80, 40)})
        language = (getattr(getattr(getattr(owner, "language_model", None), "model", None), "layers", ()), {"mrope_section": (8, 12, 12), "qk_hidden_sizes": (896, 128), "head_size": 64, "rotary_dim": 64})
    elif qwen35 is not None and qwen35 in features:
        visual = getattr(owner, "visual", None)
        vision = ((block.attn for block in getattr(visual, "blocks", ())), {"expected_blocks": 12, "inplace": True, "flatten": True, "required_bf16_neox_shape": (64, 32)})
        language = None
    else:
        return False
    if not install_vision_rotary(vision[0], **vision[1]):
        return False
    if language is not None:
        install_mrope_rotary(language[0], **language[1])
    return True
