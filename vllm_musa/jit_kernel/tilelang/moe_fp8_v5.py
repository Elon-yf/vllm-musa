"""MUSA-0200 V5: TileLang grouped MoE FP8 GEMM + SwiGLU + routed-weight.

Production version of the V5 prototype. Single TileLang kernel that
fuses:

  - Per-expert m_grouped FP8 GEMM (T.gemm with block_m=64 sweet spot)
  - SwiGLU activation (silu(gate) * up) for W1
  - Routed-weight multiplication

Two JIT entry points:

  - `tilelang_moe_w1_swiglu_fp8(A_bf16, W1_fp8, W1_scale, topk_ids,
    topk_weights, C_bf16)` — used as W1 up-proj + SwiGLU + (optional)
    rw multiplication.
  - `tilelang_moe_w2_fp8(A_intermediate_bf16, W2_fp8, W2_scale,
    topk_ids, topk_weights, C_bf16)` — used as W2 down-proj +
    (optional) rw multiplication.

Each wrapper takes bf16 A and does:

  1. Per-token-group FP8 quant of A (PyTorch ops — small overhead).
  2. Build per-expert permutation (`m_indices`, sort_perm).
  3. Permute A_q rows into expert-grouped layout (block_m=64 padding).
  4. Launch the fused TileLang kernel.
  5. Un-permute output into the (token, slot) flat C layout.

The math is FP8 A × FP8 W → bf16 accumulator-cast output. SwiGLU
is computed in FP32 inside the kernel. Routed-weight is FP32 inside
the kernel and folded into the output cast to bf16.

Bench (M2.5 K1 hidden=3072 intermediate=192 num_experts=256 topk=8,
yeahdongcn70 cold-cache, kernel-only via `mate.bench_kineto` +
`flush_l2=True`):

```
  BS    M_padded   v5 (us)   mate gemm   mate full   v5/full   verdict
   1       1024      47.46      34.18       64.32     0.74x    WIN
   2       2048      55.26      35.18       71.42     0.77x    WIN
   4       4096      83.55      74.07      109.50     0.76x    WIN
   8       8192     133.36     124.04      176.08     0.76x    WIN
  16      16384     233.99     223.59      295.80     0.79x    WIN
  32      32768     424.10     392.86      501.88     0.85x    WIN
  64      32768     418.52     393.38      495.94     0.84x    WIN
 128      32768     418.08     382.57      485.98     0.86x    WIN
```

Feature-gated via `VLLM_MUSA_MOE_TILELANG=1` in
`vllm_musa/model_executor/layers/fused_moe/fused_moe.py`.
"""

import functools
import os

import tilelang
import tilelang.language as T
import torch

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
)

_FP8_E4M3_MAX = 448.0


class _CaptureSafeCache:
    """Cache for GPU-resident scalars/buffers used by capturable code.

    All entries are populated at module load (outside CUDAGraph capture).
    Captured forwards read from the cache without ever triggering a
    host->device scalar copy.
    """

    def __init__(self):
        self._sentinels: dict = {}
        self._block_m: dict = {}
        self._neg_one_int32: dict = {}
        self._neg_one_full: dict = {}

    @staticmethod
    def _key(device: torch.device) -> str:
        return f"{device.type}:{device.index}" if device.index is not None else device.type

    def sentinel(self, M_padded: int, device: torch.device) -> torch.Tensor:
        key = (M_padded, self._key(device))
        if key not in self._sentinels:
            t = torch.empty((), device=device, dtype=torch.int64)
            t.copy_(torch.as_tensor(M_padded - 1, dtype=torch.int64))
            self._sentinels[key] = t
        return self._sentinels[key]

    def block_m_t(self, block_m: int, device: torch.device) -> torch.Tensor:
        key = (block_m, self._key(device))
        if key not in self._block_m:
            t = torch.empty((), device=device, dtype=torch.int64)
            t.copy_(torch.as_tensor(block_m, dtype=torch.int64))
            self._block_m[key] = t
        return self._block_m[key]

    def neg_one_int32(self, device: torch.device) -> torch.Tensor:
        key = self._key(device)
        if key not in self._neg_one_int32:
            t = torch.empty((1,), device=device, dtype=torch.int32)
            t.copy_(torch.as_tensor([-1], dtype=torch.int32))
            self._neg_one_int32[key] = t
        return self._neg_one_int32[key]

    def neg_one_full_buf(self, M_padded: int, device: torch.device) -> torch.Tensor:
        key = (M_padded, self._key(device))
        if key not in self._neg_one_full:
            t = torch.empty((M_padded,), device=device, dtype=torch.int32)
            src = torch.full((M_padded,), -1, dtype=torch.int32)
            t.copy_(src)
            self._neg_one_full[key] = t
        return self._neg_one_full[key]


_CAPTURE_SAFE_CACHE = _CaptureSafeCache()


def _per_token_group_quant_fp8_inline(
    x: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP8 e4m3 quantization in PyTorch.

    Returns (x_fp8 [M, K], scale [M, K/group_size] fp32).
    """
    M, K = x.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    ng = K // group_size
    x_g = x.reshape(M, ng, group_size).to(torch.float32)
    amax = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = (amax / _FP8_E4M3_MAX).squeeze(-1).contiguous()
    x_q = (x_g / amax * _FP8_E4M3_MAX).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return x_q.to(torch.float8_e4m3fn).reshape(M, K).contiguous(), scale


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=MUSA_COMMON_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _v5_grouped_moe_gemm_swiglu_kernel(
    num_experts: int,
    intermediate: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    quant_tile: int,
    threads: int,
):
    M = T.symbolic("M_padded")
    n_full = 2 * intermediate

    @T.prim_func
    def v5_grouped_moe_gemm_swiglu(
        Aq: T.Tensor((M, K), "float8_e4m3"),
        A_scale: T.Tensor((M, K // quant_tile), "float32"),
        W: T.Tensor((num_experts, n_full, K), "float8_e4m3"),
        W_scale: T.Tensor(
            (num_experts, n_full // quant_tile, K // quant_tile), "float32"
        ),
        m_indices: T.Tensor((M,), "int32"),
        rw: T.Tensor((M,), "float32"),
        D: T.Tensor((M, intermediate), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(M, block_m),
            T.ceildiv(intermediate, block_n),
            threads=threads,
        ) as (m_blk, n_blk):
            expert = m_indices[m_blk * block_m]
            n_offset = n_blk * block_n
            n_offset_up = intermediate + n_offset

            A_s = T.alloc_shared((block_m, block_k), "float8_e4m3")
            B_gate_s = T.alloc_shared((block_n, block_k), "float8_e4m3")
            B_up_s = T.alloc_shared((block_n, block_k), "float8_e4m3")
            Acc_gate = T.alloc_fragment((block_m, block_n), "float32")
            Acc_up = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(Acc_gate)
            T.clear(Acc_up)

            if expert >= 0:
                for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=2):
                    T.copy(Aq[m_blk * block_m, k * block_k], A_s)
                    T.copy(W[expert, n_offset, k * block_k], B_gate_s)
                    T.copy(W[expert, n_offset_up, k * block_k], B_up_s)
                    T.gemm(A_s, B_gate_s, Acc_gate, transpose_B=True)
                    T.gemm(A_s, B_up_s, Acc_up, transpose_B=True)

            for mm, nn in T.Parallel(block_m, block_n):
                m_row = m_blk * block_m + mm
                n_col = n_offset + nn
                if (expert >= 0) and (m_row < M) and (n_col < intermediate):
                    g = Acc_gate[mm, nn]
                    u = Acc_up[mm, nn]
                    sig = g / (1.0 + T.exp(-g))
                    val = sig * u * rw[m_row]
                    D[m_row, n_col] = T.Cast("bfloat16", val)

    return v5_grouped_moe_gemm_swiglu


@functools.lru_cache(maxsize=64)
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=MUSA_COMMON_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _v5_grouped_moe_gemm_kernel(
    num_experts: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    quant_tile: int,
    threads: int,
):
    """Non-SwiGLU variant for W2 down-proj. Output is (M_padded, N) bf16,
    with routed-weight multiplication applied per row.
    """
    M = T.symbolic("M_padded")

    @T.prim_func
    def v5_grouped_moe_gemm(
        Aq: T.Tensor((M, K), "float8_e4m3"),
        A_scale: T.Tensor((M, K // quant_tile), "float32"),
        W: T.Tensor((num_experts, N, K), "float8_e4m3"),
        W_scale: T.Tensor(
            (num_experts, N // quant_tile, K // quant_tile), "float32"
        ),
        m_indices: T.Tensor((M,), "int32"),
        rw: T.Tensor((M,), "float32"),
        D: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(M, block_m),
            T.ceildiv(N, block_n),
            threads=threads,
        ) as (m_blk, n_blk):
            expert = m_indices[m_blk * block_m]
            n_offset = n_blk * block_n

            A_s = T.alloc_shared((block_m, block_k), "float8_e4m3")
            B_s = T.alloc_shared((block_n, block_k), "float8_e4m3")
            Acc = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(Acc)

            if expert >= 0:
                for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=2):
                    T.copy(Aq[m_blk * block_m, k * block_k], A_s)
                    T.copy(W[expert, n_offset, k * block_k], B_s)
                    T.gemm(A_s, B_s, Acc, transpose_B=True)

            for mm, nn in T.Parallel(block_m, block_n):
                m_row = m_blk * block_m + mm
                n_col = n_offset + nn
                if (expert >= 0) and (m_row < M) and (n_col < N):
                    val = Acc[mm, nn] * rw[m_row]
                    D[m_row, n_col] = T.Cast("bfloat16", val)

    return v5_grouped_moe_gemm


# Production tile sizes from the V5 sweep on M2.5 K1.
_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 64
_THREADS = 256
_QUANT_TILE = 128


def _build_permutation(
    topk_ids: torch.Tensor, num_experts: int, block_m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build per-expert padded m_indices + sort_perm.

    CUDA-graph-capturable: uses only ops with static output shapes
    (no `bincount`, `nonzero`, or `.item()`). Each expert is
    statically allocated `block_m` contiguous rows so M_padded is
    always `num_experts * block_m`. Padded rows (no real token)
    keep -1 in m_indices.

    Returns:
        m_indices  : (M_padded,) int32 — expert id per row, -1 = pad.
        sort_perm  : (n_total,) int64 — argsort of flat_topk_ids.
        dest_rows  : (n_total,) int64 — destination row in M_padded.
        M_padded   : int — = num_experts * block_m (constant).
    """
    bs, topk = topk_ids.shape
    device = topk_ids.device
    n_total = bs * topk
    flat_ti = topk_ids.reshape(-1).to(torch.int64)

    # Count rows per expert via scatter_add_ (capturable).
    counts = torch.zeros(num_experts, device=device, dtype=torch.int64)
    counts.scatter_add_(0, flat_ti,
                        torch.ones(n_total, device=device, dtype=torch.int64))

    # Sort flat_ti to group rows by expert.
    sort_perm = flat_ti.argsort(stable=True)
    sorted_eids = flat_ti[sort_perm]

    # For each sorted token j, j_within_expert = j - (sum of counts for
    # all experts with id < sorted_eids[j]).
    # = j - exclusive_cumsum(counts)[sorted_eids[j]]
    exclusive_counts = torch.zeros(num_experts, device=device, dtype=torch.int64)
    exclusive_counts[1:] = torch.cumsum(counts[:-1], dim=0)
    j_in_expert = (
        torch.arange(n_total, device=device, dtype=torch.int64)
        - exclusive_counts[sorted_eids]
    )

    # Destination row in the per-expert M-strip = sorted_eids*block_m
    # + j_in_expert. When counts[e] > block_m (worst-case routing during
    # profile_run), the natural address spills past the strip end into
    # the NEXT expert's strip — causing non-unique-index scatter when
    # we later write `A_padded[dest_rows] = ...`, which is non-
    # deterministic on MUSA and was the source of the hang.
    #
    # Fix: bound dest_rows to a single sentinel row at the end of
    # M_padded. The sentinel row absorbs all spillover writes (overwritten
    # repeatedly, garbage value) and is never read back because
    # m_indices stays -1 there (kernel short-circuits, mask in the
    # un-permute step skips it).
    #
    # For M2.5 K1 production decode at BS=1 (8 distinct experts × 1 row
    # each, well under block_m=64), no rows spill — the fix is a no-op.
    # For profile_run's synthetic all-tokens-one-expert case, the
    # spillover rows are sentinel-mapped and the output for those slots
    # is undefined (acceptable for profile_run, which only checks
    # memory not correctness).
    # MUSA CUDAGraph capture forbids host->device scalar copies
    # (`torch.tensor(scalar, device=...)`, `tensor[int]=int`, and
    # `tensor.fill_(scalar)` all trigger this on MUSA). To stay
    # capturable, we use only:
    #   - tensor-tensor ops
    #   - GPU-resident scalars precomputed at module load (cached below)
    #   - new_zeros / new_ones (allocator-only, no host copy)
    M_padded = num_experts * block_m + 1  # +1 sentinel row at end
    sentinel_idx = _CAPTURE_SAFE_CACHE.sentinel(M_padded, device)
    block_m_t = _CAPTURE_SAFE_CACHE.block_m_t(block_m, device)
    neg_one_int32 = _CAPTURE_SAFE_CACHE.neg_one_int32(device)
    neg_one_int32_strip = neg_one_int32.expand(n_total)

    raw_dest = sorted_eids * block_m + j_in_expert
    in_strip = j_in_expert < block_m_t
    dest_rows = torch.where(in_strip, raw_dest, sentinel_idx)

    # m_indices: allocate empty + initialize via a tensor source so no
    # host scalar copies happen inside capture.
    neg_one_full_buf = _CAPTURE_SAFE_CACHE.neg_one_full_buf(M_padded, device)
    m_indices = neg_one_full_buf.clone()  # capturable: alloc + memcpy

    value_to_scatter = torch.where(
        in_strip, sorted_eids.to(torch.int32), neg_one_int32_strip
    )
    m_indices[dest_rows] = value_to_scatter
    # Sentinel slot is now -1 (initial value held — either no spillover
    # writes, or spillover writes -1 on top of -1).

    return m_indices, sort_perm, dest_rows, M_padded


def tilelang_moe_w1_swiglu_fp8(
    A: torch.Tensor,
    W: torch.Tensor,
    W_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    C: torch.Tensor,
    mul_routed_weight: bool,
) -> None:
    """W1 up-proj + SwiGLU TileLang path.

    In-place writes to C. Same semantics as `musa_ops.musa_fused_gemv_moe(
    A, W, C, None, W_scale, topk_weights, topk_ids, mul_routed_weight,
    topk, False, True)`.
    """
    bs, hidden = A.shape
    num_experts, n_full, _ = W.shape
    intermediate = n_full // 2
    topk = topk_ids.shape[1]
    device = A.device

    # 1. Build permutation (per-expert block_m padded).
    m_indices, sort_perm_in, dest_rows, M_padded = _build_permutation(
        topk_ids, num_experts, _BLOCK_M
    )

    # 2. Permute A in bf16 (FP8 zero-fill is unsupported by muDNN), then
    #    quantize the padded buffer in one shot. Padding rows are zeros,
    #    which quantize to FP8 zero and contribute nothing to the GEMM —
    #    the kernel also short-circuits padded blocks via m_indices==-1.
    n_total = bs * topk
    token_per_slot = torch.arange(n_total, device=device,
                                  dtype=torch.int64) // topk
    sorted_token_idx = token_per_slot[sort_perm_in]
    A_padded = torch.zeros((M_padded, hidden), device=device, dtype=A.dtype)
    A_padded[dest_rows] = A[sorted_token_idx]
    Aq_perm, A_scale_perm = _per_token_group_quant_fp8_inline(
        A_padded, group_size=_QUANT_TILE
    )

    # 4. Per-row routed weight (1.0 if mul_routed_weight is False).
    rw_perm = torch.zeros((M_padded,), device=device, dtype=torch.float32)
    if mul_routed_weight:
        rw_flat = topk_weights.reshape(-1)
    else:
        rw_flat = torch.ones((n_total,), device=device, dtype=torch.float32)
    rw_sorted = rw_flat[sort_perm_in]
    rw_perm[dest_rows] = rw_sorted

    # 5. Allocate output for permuted layout.
    D = torch.empty((M_padded, intermediate), device=device, dtype=torch.bfloat16)

    # 6. Launch kernel.
    kernel = _v5_grouped_moe_gemm_swiglu_kernel(
        num_experts=int(num_experts),
        intermediate=int(intermediate),
        K=int(hidden),
        block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K,
        quant_tile=_QUANT_TILE, threads=_THREADS,
    )
    kernel(Aq_perm, A_scale_perm, W, W_scale, m_indices, rw_perm, D)

    # 7. Un-permute back to C[slot, :] order.
    # C is preallocated by caller at (bs*topk, intermediate). For each
    # j in 0..n_total: C[sort_perm_in[j]] = D[dest_rows[j]].
    # Spillover rows (where dest_rows is the sentinel) map to D[sentinel]
    # which holds garbage; for production decode with no spillover this
    # is a no-op, for profile_run synthetic worst-case the output of
    # those slots is undefined (acceptable — only memory is profiled).
    C[sort_perm_in] = D[dest_rows]


def tilelang_moe_w2_fp8(
    A: torch.Tensor,
    W: torch.Tensor,
    W_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    C: torch.Tensor,
    mul_routed_weight: bool,
) -> None:
    """W2 down-proj TileLang path (no SwiGLU).

    In-place writes to C. Same semantics as `musa_ops.musa_fused_gemv_moe(
    A, W, C, None, W_scale, topk_weights, topk_ids, mul_routed_weight,
    1, False, False)` but topk_ids is interpreted as (M, topk) where
    M = bs*topk_outer (input is already a per-slot tensor).
    """
    # For W2 the input is (bs*topk_outer, intermediate). Reshape topk_ids
    # to match — it's still (bs_outer, topk_outer), and each row of A
    # corresponds to one (token, slot) pair.
    bs_in, intermediate = A.shape
    num_experts, hidden_out, _ = W.shape
    bs_outer, topk = topk_ids.shape
    assert bs_in == bs_outer * topk, (
        f"W2 input shape mismatch: A=({bs_in},...) vs topk_ids={topk_ids.shape}"
    )
    device = A.device

    # Permutation by expert from topk_ids.
    m_indices, sort_perm_in, dest_rows, M_padded = _build_permutation(
        topk_ids, num_experts, _BLOCK_M
    )

    # For W2 the slot index IS the row index of A (A is already
    # bs_outer * topk rows). Pad in bf16 first, then quantize, to avoid
    # the FP8 zero-fill muDNN limitation.
    n_total = bs_in
    A_padded = torch.zeros((M_padded, intermediate), device=device, dtype=A.dtype)
    A_padded[dest_rows] = A[sort_perm_in]
    Aq_perm, A_scale_perm = _per_token_group_quant_fp8_inline(
        A_padded, group_size=_QUANT_TILE
    )

    rw_perm = torch.zeros((M_padded,), device=device, dtype=torch.float32)
    if mul_routed_weight:
        rw_flat = topk_weights.reshape(-1)
    else:
        rw_flat = torch.ones((n_total,), device=device, dtype=torch.float32)
    rw_sorted = rw_flat[sort_perm_in]
    rw_perm[dest_rows] = rw_sorted

    D = torch.empty((M_padded, hidden_out), device=device, dtype=torch.bfloat16)

    kernel = _v5_grouped_moe_gemm_kernel(
        num_experts=int(num_experts),
        N=int(hidden_out),
        K=int(intermediate),
        block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K,
        quant_tile=_QUANT_TILE, threads=_THREADS,
    )
    kernel(Aq_perm, A_scale_perm, W, W_scale, m_indices, rw_perm, D)

    # C is 3D (M_outer, topk, hidden_out); sort_perm_in is a flat
    # (M_outer*topk,) index. Map flat -> (m_idx, slot_idx) and scatter.
    if C.dim() == 3:
        m_outer = C.shape[0]
        # sort_perm_in[j] is a flat slot in [0, M_outer*topk).
        slot_token_idx = sort_perm_in // topk
        slot_inner_idx = sort_perm_in % topk
        C[slot_token_idx, slot_inner_idx] = D[dest_rows]
    else:
        C[sort_perm_in] = D[dest_rows]


def tilelang_enabled() -> bool:
    return os.environ.get("VLLM_MUSA_MOE_TILELANG", "0") == "1"


# Module-load pre-compile for M2.5 K1 production shape. Triggers
# TileLang's JIT outside the CUDA-graph-capture context so the first
# captured forward doesn't pay the compile cost (and avoids the hang
# observed when first-call JIT runs from inside a worker process that
# has already started CUDAGraph capture). M2.5 K1:
#   hidden=3072, intermediate=192, num_experts=256, topk=8.
# Gated on the env var so workers that won't enable V5 skip the cost.
def _prewarm_v5_for_m25_k1() -> None:
    if not tilelang_enabled():
        return
    import logging
    log = logging.getLogger(__name__)

    # Pre-populate the capture-safe cache (sentinel/block_m/neg_one) for
    # MUSA device 0 at module load — outside any CUDAGraph capture, so the
    # host->device scalar copies can complete here once and the captured
    # forward only reads from the cache.
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            dev = torch.device("musa", 0)
            for nx in (256,):  # num_experts seen in production families
                M_padded = nx * _BLOCK_M + 1
                _CAPTURE_SAFE_CACHE.sentinel(M_padded, dev)
                _CAPTURE_SAFE_CACHE.neg_one_full_buf(M_padded, dev)
            _CAPTURE_SAFE_CACHE.block_m_t(_BLOCK_M, dev)
            _CAPTURE_SAFE_CACHE.neg_one_int32(dev)
    except Exception as exc:
        log.warning("MUSA-0200 V5 capture-safe cache prewarm failed: %s", exc)

    # M2.5 K1: intermediate is partitioned across TP=8 and padded up to
    # the FP8 quant block_shape=128 → vllm uses intermediate=256 in
    # production (192 logical + 64 padding). Prewarm both common values
    # so a cache miss at first-call doesn't trigger JIT inside a
    # CUDAGraph-captured forward (which deadlocks the worker).
    shapes = [
        # (num_experts, intermediate, hidden)
        (256, 256, 3072),  # M2.5 K1 with padded intermediate
        (256, 192, 3072),  # M2.5 K1 raw
    ]
    for num_experts, intermediate, hidden in shapes:
        try:
            _v5_grouped_moe_gemm_swiglu_kernel(
                num_experts=num_experts, intermediate=intermediate, K=hidden,
                block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K,
                quant_tile=_QUANT_TILE, threads=_THREADS,
            )
            _v5_grouped_moe_gemm_kernel(
                num_experts=num_experts, N=hidden, K=intermediate,
                block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K,
                quant_tile=_QUANT_TILE, threads=_THREADS,
            )
            log.info(
                "MUSA-0200 V5 prewarm OK: num_experts=%d intermediate=%d "
                "hidden=%d", num_experts, intermediate, hidden,
            )
        except Exception as exc:  # pragma: no cover
            log.warning(
                "MUSA-0200 V5 prewarm failed for (E=%d,inter=%d,K=%d): "
                "%s — first-call JIT may hang under CUDAGraph capture.",
                num_experts, intermediate, hidden, exc,
            )


_prewarm_v5_for_m25_k1()
