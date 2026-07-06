# SPDX-License-Identifier: Apache-2.0
"""Fused qk-norm + neox RoPE (+ optional KV-cache write) for Qwen3 dense GQA decode.

One TileLang kernel replaces the separate q_norm / k_norm / rotary_embedding
launches vLLM-MUSA issues per Qwen3 dense decode step. Two entry points:

- ``qk_norm_rope_inplace`` (3->1): q_norm + k_norm + neox rope, writing q and k
  in place; the attention backend keeps ``reshape_and_cache``.
- ``qk_norm_rope_kv_insert`` (4->1): additionally writes the normed+roped k and
  the untouched v straight into the paged KV cache.

Both wrappers issue exactly one captured GPU op (the fused kernel), so they are
CUDAGraph-capture-safe. ``register_and_patch`` registers the opaque custom op and
rebinds ``Qwen3Attention.__init__``/``forward`` onto the 3->1 path.

Grid ``(num_tokens, num_q_heads + num_kv_heads)``, head_dim threads = one element
per thread. ``head_slot < num_q_heads`` -> a q head (norm+rope -> q_out); else a k
head (norm+rope -> key_cache[slot], or k in place when write_kv=False). The q/k
branch is on blockIdx.y (block-uniform, no warp divergence); all syncs sit outside
it. RMSNorm reduces over head_dim in fp32; neox rope reads bf16 cos/sin cast to
fp32 to match the vLLM rope path.
"""

from functools import lru_cache
from typing import Optional

import torch

import tilelang
import tilelang.language as T
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.deepseek_v4_jit.kernel_common import (
    _patch_tilelang_musa_wrapper,
    _tilelang_musa_pass_configs,
)

logger = init_logger(__name__)

_patch_tilelang_musa_wrapper()

_OP_NAME = "musa_fused_qk_norm_rope"
# block_size only sizes the read-never dummy paged cache on the 3->1 path; the
# kernel never indexes it when write_kv=False.
_DECODE_BLOCK_SIZE = 16


def _warp_reduce_sum(value):
    mask = T.tvm_warp_activemask()
    value += T.tvm_warp_shuffle_down(mask, value, 16, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 8, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 4, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 2, 32, 32)
    value += T.tvm_warp_shuffle_down(mask, value, 1, 32, 32)
    return T.tvm_warp_shuffle(mask, value, 0, 32, 32)


@lru_cache(maxsize=None)
def qk_norm_rope_kv_kernel(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    write_kv: bool = True,
):
    """Compile the fused kernel for one (heads, head_dim, block_size) shape.

    head_dim must equal the rope dim (Qwen3 8B/32B use full rotary, rot_dim==128)
    and be a multiple of 32 so the block reduction is warp-clean.

    write_kv=True  (4->1): normed+roped k -> key_cache[slot], v -> value_cache[slot].
    write_kv=False (3->1): normed+roped k -> k in place; caches/v/slot ignored. Used
      for the low-risk decode path where the attention backend still owns
      reshape_and_cache (no backend surgery).
    """
    assert head_dim % 32 == 0, "head_dim must be a multiple of the warp size"
    assert head_dim <= 1024
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.dynamic("num_blocks")
    num_positions = T.dynamic("num_positions")
    total_heads = num_q_heads + num_kv_heads
    half = head_dim // 2
    threads = head_dim
    warps_per_cta = threads // 32

    @tilelang.jit(target="musa", pass_configs=_tilelang_musa_pass_configs(tilelang))
    def _build():
        @T.prim_func
        def _kernel(
            q: T.Tensor((num_tokens, num_q_heads, head_dim), T.bfloat16),
            k: T.Tensor((num_tokens, num_kv_heads, head_dim), T.bfloat16),
            v: T.Tensor((num_tokens, num_kv_heads, head_dim), T.bfloat16),
            q_out: T.Tensor((num_tokens, num_q_heads, head_dim), T.bfloat16),
            q_norm_w: T.Tensor((head_dim,), T.bfloat16),
            k_norm_w: T.Tensor((head_dim,), T.bfloat16),
            cos_sin_cache: T.Tensor((num_positions, head_dim), T.bfloat16),
            positions: T.Tensor((num_tokens,), T.int64),
            slot_mapping: T.Tensor((num_tokens,), T.int64),
            key_cache: T.Tensor(
                (num_blocks, block_size, num_kv_heads, head_dim), T.bfloat16
            ),
            value_cache: T.Tensor(
                (num_blocks, block_size, num_kv_heads, head_dim), T.bfloat16
            ),
            eps: T.float32,
        ):
            with T.Kernel(num_tokens, total_heads, threads=threads) as (
                token_id,
                head_slot,
            ):
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32

                x = T.alloc_local((1,), T.float32)
                w = T.alloc_local((1,), T.float32)
                partial = T.alloc_local((1,), T.float32)
                warp_sums = T.alloc_shared((warps_per_cta,), T.float32)
                normed = T.alloc_shared((head_dim,), T.float32)

                is_q = head_slot < num_q_heads
                # --- load head element + per-head norm weight (q or k) ---
                if is_q:
                    x[0] = T.cast(q[token_id, head_slot, tx], T.float32)
                    w[0] = T.cast(q_norm_w[tx], T.float32)
                else:
                    x[0] = T.cast(k[token_id, head_slot - num_q_heads, tx], T.float32)
                    w[0] = T.cast(k_norm_w[tx], T.float32)

                # --- RMSNorm reduction over head_dim (fp32 accumulate) ---
                partial[0] = x[0] * x[0]
                partial[0] = _warp_reduce_sum(partial[0])
                if lane == 0:
                    warp_sums[warp] = partial[0]
                T.sync_threads()
                partial[0] = T.if_then_else(tx < warps_per_cta, warp_sums[tx], 0.0)
                if warp == 0:
                    partial[0] = _warp_reduce_sum(partial[0])
                    if lane == 0:
                        warp_sums[0] = T.rsqrt(partial[0] / float(head_dim) + eps)
                T.sync_threads()
                normed[tx] = x[0] * warp_sums[0] * w[0]
                T.sync_threads()

                # --- neox RoPE over the full head (bf16 cos/sin, fp32 math) ---
                pos = T.Cast("int32", positions[token_id])
                o = T.alloc_local((1,), T.float32)
                c = T.alloc_local((1,), T.float32)
                s = T.alloc_local((1,), T.float32)
                if tx < half:
                    c[0] = T.cast(cos_sin_cache[pos, tx], T.float32)
                    s[0] = T.cast(cos_sin_cache[pos, half + tx], T.float32)
                    o[0] = normed[tx] * c[0] - normed[tx + half] * s[0]
                else:
                    c[0] = T.cast(cos_sin_cache[pos, tx - half], T.float32)
                    s[0] = T.cast(cos_sin_cache[pos, tx], T.float32)
                    o[0] = normed[tx] * c[0] + normed[tx - half] * s[0]

                # --- store: q_out, or roped-k (in place, or into the paged cache) ---
                if is_q:
                    q_out[token_id, head_slot, tx] = T.cast(o[0], T.bfloat16)
                else:
                    hk = head_slot - num_q_heads
                    if write_kv:
                        loc = T.Cast("int32", slot_mapping[token_id])
                        if loc >= 0:
                            blk = loc // block_size
                            off = loc % block_size
                            key_cache[blk, off, hk, tx] = T.cast(o[0], T.bfloat16)
                            value_cache[blk, off, hk, tx] = v[token_id, hk, tx]
                    else:
                        k[token_id, hk, tx] = T.cast(o[0], T.bfloat16)

        return _kernel

    return _build()


def qk_norm_rope_kv_insert(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    eps: float,
    q_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """4->1 fusion: q_norm + k_norm + neox rope + KV-cache write. One captured op.

    q  : [num_tokens, num_q_heads, head_dim] bf16
    k,v: [num_tokens, num_kv_heads, head_dim] bf16
    cos_sin_cache: [num_positions, head_dim] bf16  (first half cos, second half sin)
    key/value_cache: [num_blocks, block_size, num_kv_heads, head_dim] bf16 (NHD)
    Returns q_out (normed+roped q); k/v are written into the caches in place.
    """
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k.shape[1]
    block_size = key_cache.shape[1]
    if q_out is None:
        q_out = q  # in-place: rope reads from shared, so overwriting q is safe
    kernel = qk_norm_rope_kv_kernel(
        num_q_heads, num_kv_heads, head_dim, block_size, write_kv=True
    )
    kernel(
        q,
        k,
        v,
        q_out,
        q_norm_w,
        k_norm_w,
        cos_sin_cache,
        positions,
        slot_mapping,
        key_cache,
        value_cache,
        float(eps),
    )
    return q_out


_DUMMY_CACHE: dict = {}


def _dummies(num_kv_heads, head_dim, block_size, device):
    """Module-cached read-never caches for the 3->1 path (no per-call alloc ->
    capture-safe)."""
    key = (num_kv_heads, head_dim, block_size, device)
    d = _DUMMY_CACHE.get(key)
    if d is None:
        kc = torch.zeros(
            1, block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        d = (kc, torch.zeros_like(kc))
        _DUMMY_CACHE[key] = d
    return d


def qk_norm_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    block_size: int = _DECODE_BLOCK_SIZE,
) -> None:
    """3->1 fusion: q_norm + k_norm + neox rope, writing q and k IN PLACE.

    Replaces the q_norm / k_norm / rotary_embedding launches; the attention
    backend still owns reshape_and_cache.  Exactly one captured GPU op.
    """
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k.shape[1]
    kc, vc = _dummies(num_kv_heads, head_dim, block_size, q.device)
    kernel = qk_norm_rope_kv_kernel(
        num_q_heads, num_kv_heads, head_dim, block_size, write_kv=False
    )
    kernel(
        q,
        k,
        k,
        q,
        q_norm_w,
        k_norm_w,
        cos_sin_cache,
        positions,
        positions,
        kc,
        vc,
        float(eps),
    )


def _fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    qk_norm_rope_inplace(
        q, k, q_norm_w, k_norm_w, cos_sin_cache, positions, float(eps), int(block_size)
    )


def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    return


def _patched_forward(
    self, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # contiguous so the fused kernel sees the standard [T, heads, hd] layout
    # (the qkv split slices share qkv's row stride, which the kernel can't index).
    q = q.contiguous()
    k = k.contiguous()
    q3 = q.view(-1, self.num_heads, self.head_dim)
    k3 = k.view(-1, self.num_kv_heads, self.head_dim)
    torch.ops.vllm.musa_fused_qk_norm_rope(
        q3,
        k3,
        self.q_norm.weight,
        self.k_norm.weight,
        self._fused_cos_sin,
        positions,
        self.q_norm.variance_epsilon,
        _DECODE_BLOCK_SIZE,
    )
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def register_and_patch(qwen3_attention_cls) -> bool:
    """Register the opaque fused op and rebind Qwen3Attention onto the 3->1 path.

    Registers ``musa_fused_qk_norm_rope`` (opaque custom op so Dynamo sees a single
    call) and monkeypatches ``__init__`` (pre-cast the rotary cos/sin cache to bf16
    once) and ``forward`` (fused q_norm+k_norm+rope in place of the three launches;
    the attention backend keeps ``reshape_and_cache``). Idempotent per process via a
    class marker. Returns True once the class is patched.
    """
    if getattr(qwen3_attention_cls, "_musa_fused_qknorm", False):
        return True

    if not hasattr(torch.ops.vllm, _OP_NAME):
        direct_register_custom_op(
            op_name=_OP_NAME,
            op_func=_fused_qk_norm_rope,
            mutates_args=["q", "k"],
            fake_impl=_fused_qk_norm_rope_fake,
        )

    _orig_init = qwen3_attention_cls.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # pre-cast cos/sin to bf16 once; cache on the (shared) rotary module so all
        # layers using the same rope reuse one bf16 copy.
        rope = self.rotary_emb
        cached = getattr(rope, "_musa_fused_cos_sin", None)
        if cached is None:
            cached = rope.cos_sin_cache.to(dtype=torch.bfloat16)
            rope._musa_fused_cos_sin = cached
        self._fused_cos_sin = cached

    qwen3_attention_cls.__init__ = _init
    qwen3_attention_cls.forward = _patched_forward
    qwen3_attention_cls._musa_fused_qknorm = True
    logger.info(
        "Installed fused qk-norm+RoPE kernel on %s", qwen3_attention_cls.__name__
    )
    return True
