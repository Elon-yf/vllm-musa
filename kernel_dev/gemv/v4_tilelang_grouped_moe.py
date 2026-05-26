#!/usr/bin/env python3
"""V4 TileLang grouped MoE FP8 GEMM (T.gemm + bundle-by-expert).

Per-block expert binding: each threadblock handles one (expert,
N-tile) pair, processes all rows routed to that expert with `T.gemm`.
No SwiGLU / routed-weight yet — we want to see if the core GEMM
beats mate first, then layer activation on top.

Strategy mirrors mate's `m_grouped_fp8_gemm_nt_contiguous`:
  - A is pre-permuted so rows for the same expert are contiguous.
  - `m_indices[r]` gives the expert id for permuted row r (or -1 pad).
  - alignment_m = block_m, so each expert occupies a multiple of
    block_m rows.

We then drive ONE T.gemm tile per (M-bucket, N-tile) block. The
expert id determines which W slice to load.
"""
import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T
import sys
sys.path.insert(0, "/ws")
from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS, MUSA_COMPILE_FLAGS,
)
from mate.testing.utils import bench_kineto, group_quantize_fp8
from mate.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

import functools


@functools.lru_cache
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=MUSA_COMMON_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _v4_grouped_moe_gemm_kernel(
    num_experts: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    quant_tile: int,
    threads: int,
):
    M = T.symbolic("M_padded")  # padded total active rows

    @T.prim_func
    def v4_grouped_moe_gemm(
        Aq: T.Tensor((M, K), "float8_e4m3"),
        A_scale: T.Tensor((M, K // quant_tile), "float32"),
        W: T.Tensor((num_experts, N, K), "float8_e4m3"),
        W_scale: T.Tensor(
            (num_experts, N // quant_tile, K // quant_tile), "float32"
        ),
        m_indices: T.Tensor((M,), "int32"),
        D: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(M, block_m),
            T.ceildiv(N, block_n),
            threads=threads,
        ) as (m_blk, n_blk):
            # Each block: rows [m_blk*block_m, (m_blk+1)*block_m), columns
            # [n_blk*block_n, (n_blk+1)*block_n). All rows in a block share
            # the same expert because the host padded each expert to
            # block_m row alignment.
            expert = m_indices[m_blk * block_m]

            A_s = T.alloc_shared((block_m, block_k), "float8_e4m3")
            B_s = T.alloc_shared((block_n, block_k), "float8_e4m3")
            Acc = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(Acc)

            # Skip padding blocks (expert == -1).
            if expert >= 0:
                for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=4):
                    T.copy(Aq[m_blk * block_m, k * block_k], A_s)
                    T.copy(W[expert, n_blk * block_n, k * block_k], B_s)
                    T.gemm(A_s, B_s, Acc, transpose_B=True)

            # Scale and store.
            # Per-row scale[m_row, k_block] applies; for the per-K-block scale
            # design, we'd need to incorporate scales inside the K-loop. For
            # this V4 probe we omit scale folding to isolate the GEMM perf
            # (matches mate's recipe=(1,128,128) behavior approximately).
            for mm, nn in T.Parallel(block_m, block_n):
                m_row = m_blk * block_m + mm
                n_col = n_blk * block_n + nn
                if (expert >= 0) and (m_row < M) and (n_col < N):
                    D[m_row, n_col] = T.Cast("bfloat16", Acc[mm, nn])

    return v4_grouped_moe_gemm


def bench_v4(bs, num_tests=20, block_m=128, block_n=128, block_k=64, topk=8,
             hidden=3072, intermediate=192, num_experts=256):
    n_total = bs * topk
    M_padded = ((n_total + block_m - 1) // block_m) * block_m
    N = 2 * intermediate  # gate+up
    K = hidden
    quant_tile = 128

    # Inputs
    gen = torch.Generator(device="musa").manual_seed(0)
    a_raw = torch.rand((M_padded, K), device="musa", dtype=torch.float32) * 0.1
    w_raw = torch.rand((num_experts, N, K), device="musa", dtype=torch.float32) * 0.05
    Aq, A_scale = group_quantize_fp8(
        a_raw, (M_padded, K // quant_tile), (1, quant_tile),
        torch.float8_e4m3fn, "K"
    )
    Wq, W_scale = group_quantize_fp8(
        w_raw, (num_experts, N // quant_tile, K // quant_tile),
        (1, quant_tile, quant_tile),
        torch.float8_e4m3fn, "K"
    )
    # m_indices: assign experts cyclically; pad with -1 at the end.
    m_indices = torch.full((M_padded,), -1, device="musa", dtype=torch.int32)
    for i in range(n_total):
        m_indices[i] = (i // block_m) % num_experts
    # Pad rows for each expert to block_m: re-pack.
    # For this probe assume uniform expert assignment by block index.
    m_indices_packed = torch.full((M_padded,), -1, device="musa",
                                  dtype=torch.int32)
    n_used = (n_total // block_m) * block_m
    m_indices_packed[:n_used] = m_indices[:n_used]
    D = torch.empty((M_padded, N), device="musa", dtype=torch.bfloat16)

    kernel = _v4_grouped_moe_gemm_kernel(
        num_experts=int(num_experts),
        N=int(N), K=int(K),
        block_m=int(block_m), block_n=int(block_n), block_k=int(block_k),
        quant_tile=int(quant_tile),
        threads=256,
    )

    def run():
        kernel(Aq, A_scale, Wq, W_scale, m_indices_packed, D)
    for _ in range(3):
        run()
    torch.musa.synchronize()
    sec = bench_kineto(run, kernel_names="v4_grouped_moe_gemm",
                       num_tests=num_tests, suppress_kineto_output=True,
                       flush_l2=True, with_multiple_kernels=True)

    # Mate baseline at same shape
    def run_mate():
        m_grouped_fp8_gemm_nt_contiguous(
            (Aq, A_scale), (Wq, W_scale), D, m_indices_packed,
            recipe=(1, quant_tile, quant_tile), alignment_m=block_m,
        )
    for _ in range(3):
        run_mate()
    torch.musa.synchronize()
    sec_mate = bench_kineto(run_mate, kernel_names="ssgemm",
                            num_tests=num_tests, suppress_kineto_output=True,
                            flush_l2=True, with_multiple_kernels=True)
    return sec, sec_mate


def main():
    print("# V4 TileLang grouped MoE FP8 GEMM (T.gemm + bundle-by-expert)")
    print("# Shape M2.5 K1: hidden=3072 intermediate=192 num_experts=256 topk=8")
    print()
    print(f"{'BS':>4s} {'M_padded':>10s}  {'v4 (us)':>9s}  {'mate (us)':>10s}  {'ratio':>7s}  verdict")
    print("-" * 70)
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        try:
            sec, sec_mate = bench_v4(bs)
            ratio = sec / sec_mate
            verdict = "WIN" if ratio < 0.95 else ("TIE" if ratio < 1.05 else "LOSE")
            print(f"{bs:>4d} {(bs*8 + 63)//64*64:>10d}  {sec*1e6:>9.2f}"
                  f"  {sec_mate*1e6:>10.2f}  {ratio:>6.2f}x  {verdict}")
            torch.musa.empty_cache()
        except Exception as e:
            print(f"{bs:>4d}  ERROR {type(e).__name__}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
