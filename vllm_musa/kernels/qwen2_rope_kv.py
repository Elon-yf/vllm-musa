# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-shape Qwen2 RoPE and NHD KV-cache update for MUSA.

This module is intentionally not imported from :mod:`vllm_musa.kernels`.
The flash-attention backend imports it lazily only when the upstream
RoPE/KV-cache fusion pass has selected the fused path.

The kernel covers the single validated CosyVoice/Qwen2 shape only:
BF16 Q14/KV2 with a 64-element NeoX rotary head and a 64-token NHD cache
block.  Unsupported calls fail closed so the backend can retain the existing
RoPE plus reshape-and-cache fallback.
"""

from __future__ import annotations

import torch
from torch import Tensor
from vllm.triton_utils import tl, triton

_NUM_Q_HEADS = 14
_NUM_KV_HEADS = 2
_HEAD_DIM = 64
_HALF_HEAD_DIM = _HEAD_DIM // 2
_Q_SIZE = _NUM_Q_HEADS * _HEAD_DIM
_KV_SIZE = _NUM_KV_HEADS * _HEAD_DIM
_PACKED_WIDTH = _Q_SIZE + 2 * _KV_SIZE
_CACHE_BLOCK_SIZE = 64
_MAX_FUSED_TOKENS = 1
_BLOCK_HEADS = 4


@triton.jit
def _qwen2_rope_kv_cache_kernel(
    packed_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    packed_stride_t,
    cos_sin_stride_t,
    key_cache_stride_b,
    key_cache_stride_s,
    value_cache_stride_b,
    value_cache_stride_s,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    """Rotate one token/head tile and scatter its K/V cache entries."""
    token = tl.program_id(0).to(tl.int64)
    heads = (tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS))[:, None]
    half_dims = tl.arange(0, HALF_HEAD_DIM)[None, :]
    full_dims = tl.arange(0, HEAD_DIM)[None, :]
    total_heads = NUM_Q_HEADS + NUM_KV_HEADS
    valid_head = heads < total_heads
    is_q = heads < NUM_Q_HEADS
    is_k = (heads >= NUM_Q_HEADS) & valid_head
    kv_heads = tl.where(is_k, heads - NUM_Q_HEADS, 0)

    position = tl.load(positions_ptr + token).to(tl.int64)
    cache_base = position * cos_sin_stride_t
    cosine = tl.load(cos_sin_cache_ptr + cache_base + half_dims).to(tl.float32)
    sine = tl.load(cos_sin_cache_ptr + cache_base + half_dims + HALF_HEAD_DIM).to(
        tl.float32
    )

    row_base = token * packed_stride_t + tl.where(
        is_q,
        heads * HEAD_DIM,
        Q_SIZE + kv_heads * HEAD_DIM,
    )
    q_x = tl.load(
        packed_ptr + row_base + half_dims,
        mask=is_q & valid_head,
        other=0.0,
    ).to(tl.float32)
    q_y = tl.load(
        packed_ptr + row_base + HALF_HEAD_DIM + half_dims,
        mask=is_q & valid_head,
        other=0.0,
    ).to(tl.float32)
    q_rot_x = (q_x * cosine - q_y * sine).to(tl.bfloat16)
    q_rot_y = (q_y * cosine + q_x * sine).to(tl.bfloat16)
    tl.store(
        packed_ptr + row_base + half_dims,
        q_rot_x,
        mask=is_q & valid_head,
    )
    tl.store(
        packed_ptr + row_base + HALF_HEAD_DIM + half_dims,
        q_rot_y,
        mask=is_q & valid_head,
    )

    k_x = tl.load(packed_ptr + row_base + half_dims, mask=is_k, other=0.0).to(
        tl.float32
    )
    k_y = tl.load(
        packed_ptr + row_base + HALF_HEAD_DIM + half_dims,
        mask=is_k,
        other=0.0,
    ).to(tl.float32)
    k_rot_x = (k_x * cosine - k_y * sine).to(tl.bfloat16)
    k_rot_y = (k_y * cosine + k_x * sine).to(tl.bfloat16)
    tl.store(packed_ptr + row_base + half_dims, k_rot_x, mask=is_k)
    tl.store(
        packed_ptr + row_base + HALF_HEAD_DIM + half_dims,
        k_rot_y,
        mask=is_k,
    )

    # Negative slots are padding and must leave both cache tensors untouched.
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    has_slot = slot >= 0
    safe_slot = tl.where(has_slot, slot, 0)
    block = safe_slot // CACHE_BLOCK_SIZE
    block_offset = safe_slot - block * CACHE_BLOCK_SIZE
    cache_mask = is_k & has_slot
    key_cache_base = (
        block * key_cache_stride_b
        + block_offset * key_cache_stride_s
        + kv_heads * HEAD_DIM
    )
    tl.store(
        key_cache_ptr + key_cache_base + half_dims,
        k_rot_x,
        mask=cache_mask,
    )
    tl.store(
        key_cache_ptr + key_cache_base + half_dims + HALF_HEAD_DIM,
        k_rot_y,
        mask=cache_mask,
    )
    value = tl.load(
        packed_ptr
        + token * packed_stride_t
        + Q_SIZE
        + KV_SIZE
        + kv_heads * HEAD_DIM
        + full_dims,
        mask=cache_mask,
        other=0.0,
    )
    value_cache_base = (
        block * value_cache_stride_b
        + block_offset * value_cache_stride_s
        + kv_heads * HEAD_DIM
    )
    tl.store(
        value_cache_ptr + value_cache_base + full_dims,
        value,
        mask=cache_mask,
    )


def _is_musa_tensor(tensor: Tensor) -> bool:
    return tensor.device.type == "musa"


def _qwen2_rope_kv_cache_support(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    is_neox: bool,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
) -> tuple[bool, str]:
    tensors = (
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
    )
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"
    if len({tensor.device for tensor in tensors}) != 1:
        return False, "all tensors must be on the same MUSA device"
    if not is_neox:
        return False, "only NeoX rotary layout is supported"

    data_tensors = (query, key, value, cos_sin_cache, key_cache, value_cache)
    if any(tensor.dtype != torch.bfloat16 for tensor in data_tensors):
        return False, "query/key/value/cos-sin/cache tensors must be BF16"
    if positions.dtype != torch.int64 or slot_mapping.dtype != torch.int64:
        return False, "positions and slot_mapping must be int64"

    num_tokens = query.shape[0] if query.dim() == 3 else -1
    if query.shape != (num_tokens, _NUM_Q_HEADS, _HEAD_DIM):
        return False, "query must have shape [tokens, 14, 64]"
    expected_kv_shape = (num_tokens, _NUM_KV_HEADS, _HEAD_DIM)
    if key.shape != expected_kv_shape or value.shape != expected_kv_shape:
        return False, "key and value must have shape [tokens, 2, 64]"
    if not 0 <= num_tokens <= _MAX_FUSED_TOKENS:
        return False, f"token count must be in [0, {_MAX_FUSED_TOKENS}]"
    if positions.shape != (num_tokens,):
        return False, "positions must be 1D and match the token count"
    if slot_mapping.shape != (num_tokens,):
        return False, "slot_mapping must be 1D and match the token count"

    expected_cache_tail = (_CACHE_BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM)
    if key_cache.dim() != 4 or key_cache.shape[1:] != expected_cache_tail:
        return False, "key_cache must have NHD shape [blocks, 64, 2, 64]"
    if (
        value_cache.shape != key_cache.shape
        or value_cache.stride() != key_cache.stride()
    ):
        return False, "value_cache shape and strides must match key_cache"
    if key_cache.shape[0] == 0:
        return False, "cache must contain at least one block"

    for name, tensor, row_width in (
        ("query", query, _NUM_Q_HEADS * _HEAD_DIM),
        ("key", key, _NUM_KV_HEADS * _HEAD_DIM),
        ("value", value, _NUM_KV_HEADS * _HEAD_DIM),
    ):
        if tensor.stride(2) != 1 or tensor.stride(1) != _HEAD_DIM:
            return False, f"{name} head and head-dim axes must be contiguous"
        if tensor.stride(0) < row_width:
            return False, f"{name} token rows must not overlap"

    # The fused pass presents Q/K/V as views of the contiguous packed QKV
    # result.  Keeping this alias gate is what lets the kernel use the
    # compile-time packed offsets and avoids a second, stride-heavy variant.
    element_size = query.element_size()
    if (
        query.data_ptr() + _Q_SIZE * element_size != key.data_ptr()
        or query.data_ptr() + (_Q_SIZE + _KV_SIZE) * element_size != value.data_ptr()
    ):
        return False, "Q/K/V must be packed aliases"

    cache_row_width = _NUM_KV_HEADS * _HEAD_DIM
    cache_block_width = _CACHE_BLOCK_SIZE * cache_row_width
    if (
        key_cache.stride(3) != 1
        or key_cache.stride(2) != _HEAD_DIM
        or key_cache.stride(1) != cache_row_width
        or key_cache.stride(0) < cache_block_width
    ):
        return False, "cache tensors must use non-overlapping NHD strides"
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] != _HEAD_DIM:
        return False, "cos_sin_cache must have shape [positions, 64]"
    if cos_sin_cache.stride(1) != 1 or cos_sin_cache.stride(0) < _HEAD_DIM:
        return False, "cos_sin_cache rows must be contiguous and non-overlapping"
    if not positions.is_contiguous() or not slot_mapping.is_contiguous():
        return False, "positions and slot_mapping must be contiguous"
    return True, ""


def supports_qwen2_rope_kv_cache(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    is_neox: bool,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
) -> bool:
    """Return whether the exact Qwen2 fused kernel can handle a call."""
    supported, _ = _qwen2_rope_kv_cache_support(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        is_neox,
        key_cache,
        value_cache,
        slot_mapping,
    )
    return supported


def try_qwen2_rope_kv_cache(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    is_neox: bool,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
) -> bool:
    """Apply fused RoPE/cache update, returning ``False`` for fallback calls.

    ``query`` and ``key`` are mutated in place.  For every non-negative slot,
    the corresponding rotated key and unmodified value are written into the
    NHD paged caches.  Negative slots do not modify either cache.
    """
    supported, _ = _qwen2_rope_kv_cache_support(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        is_neox,
        key_cache,
        value_cache,
        slot_mapping,
    )
    if not supported:
        return False

    _launch_qwen2_rope_kv_cache(
        query,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
    )
    return True


def _launch_qwen2_rope_kv_cache(
    query: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
) -> None:
    """Launch after the caller has established the exact support envelope."""
    num_tokens = query.shape[0]
    if num_tokens == 0:
        return
    _qwen2_rope_kv_cache_kernel[
        (num_tokens, triton.cdiv(_NUM_Q_HEADS + _NUM_KV_HEADS, _BLOCK_HEADS))
    ](
        query,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        _PACKED_WIDTH,
        cos_sin_cache.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        NUM_Q_HEADS=_NUM_Q_HEADS,
        NUM_KV_HEADS=_NUM_KV_HEADS,
        HEAD_DIM=_HEAD_DIM,
        HALF_HEAD_DIM=_HALF_HEAD_DIM,
        Q_SIZE=_Q_SIZE,
        KV_SIZE=_KV_SIZE,
        CACHE_BLOCK_SIZE=_CACHE_BLOCK_SIZE,
        BLOCK_HEADS=_BLOCK_HEADS,
        num_warps=4,
        num_stages=2,
    )


__all__ = ["supports_qwen2_rope_kv_cache", "try_qwen2_rope_kv_cache"]
