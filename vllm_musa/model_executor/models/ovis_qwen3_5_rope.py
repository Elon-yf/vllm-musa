import torch
from torch import nn


class _OvisRotaryPositionCache(nn.Module):
    """One shared position buffer, retaining storage referenced by a graph."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_positions", None, persistent=False)
        self._key = None
        self._graph_pinned = False

    @staticmethod
    def _is_capturing(device: torch.device) -> bool:
        if device.type != "musa":
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except (AttributeError, RuntimeError):
            # Unknown capture state must not evict graph-referenced storage.
            return True

    def get(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (batch, seq_len, device)
        capturing = self._is_capturing(device)
        if (
            self._positions is not None
            and self._key == key
            and self._positions.device == device
        ):
            if capturing:
                self._graph_pinned = True
            return self._positions
        positions = torch.arange(seq_len, device=device, dtype=torch.long).repeat(batch)
        # A captured graph keeps the old address. Other shapes use temporary
        # positions rather than replacing that buffer; capture misses belong to
        # the graph's allocation pool and never mutate this shared cache.
        if not capturing and not self._graph_pinned:
            self._positions = positions
            self._key = key
        return positions


class OvisVisionApplyRotaryEmb(nn.Module):
    """Model-local adapter to the existing MUSA rotary custom op."""

    def __init__(
        self,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
        positions_cache: _OvisRotaryPositionCache | None = None,
    ) -> None:
        super().__init__()
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        self.positions_cache = (
            positions_cache
            if positions_cache is not None
            else _OvisRotaryPositionCache()
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "musa":
            raise RuntimeError("OvisVisionApplyRotaryEmb is MUSA-only")
        origin_dtype = x.dtype
        if self.enable_fp32_compute:
            x = x.float()
            cos = cos.float()
            sin = sin.float()
        batch, seq_len, num_heads, head_size = x.shape
        query = x.reshape(batch * seq_len, num_heads * head_size)
        positions = self.positions_cache.get(batch, seq_len, x.device)
        cos_sin_cache = torch.cat((cos, sin), dim=-1).contiguous()
        from vllm_musa.jit_kernel.csrc.rope import rotary_embedding

        rotary_embedding(
            positions,
            query,
            None,
            head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query.view(batch, seq_len, num_heads, head_size).to(origin_dtype)


def enable_ovis_vision_rope(visual: nn.Module) -> int:
    replaced = 0
    positions_cache = _OvisRotaryPositionCache()
    for block in visual.blocks:
        attention = getattr(block, "attn", None)
        rotary = getattr(attention, "apply_rotary_emb", None)
        if rotary is None:
            continue
        attention.apply_rotary_emb = OvisVisionApplyRotaryEmb(
            is_neox_style=rotary.is_neox_style,
            enable_fp32_compute=rotary.enable_fp32_compute,
            positions_cache=positions_cache,
        )
        replaced += 1
    if replaced != len(visual.blocks):
        raise RuntimeError(
            f"Ovis vision rotary adaptation replaced {replaced}/"
            f"{len(visual.blocks)} blocks"
        )
    return replaced
