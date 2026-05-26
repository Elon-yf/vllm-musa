#!/usr/bin/env python3
"""Simplest possible TileLang FP8 GEMM test.

Goal: prove TileLang generates SQMMA on MUSA. If T.gemm of FP8xFP8 -> bf16
compiles and runs faster than scalar MAC, then V3 SQMMA via TileLang is the
viable path.
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

import functools


@functools.lru_cache
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=MUSA_COMMON_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _fp8_gemm_kernel(M, N, K, block_M, block_N, block_K, threads):

    @T.prim_func
    def fp8_gemm_kernel(
        A: T.Tensor((M, K), "float8_e4m3"),
        B: T.Tensor((N, K), "float8_e4m3"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M),
            T.ceildiv(N, block_N),
            threads=threads,
        ) as (bm, bn):
            A_s = T.alloc_shared((block_M, block_K), "float8_e4m3")
            B_s = T.alloc_shared((block_N, block_K), "float8_e4m3")
            C_r = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_r)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[bm * block_M, k * block_K], A_s)
                T.copy(B[bn * block_N, k * block_K], B_s)
                T.gemm(A_s, B_s, C_r, transpose_B=True)

            for mm, nn in T.Parallel(block_M, block_N):
                C[bm * block_M + mm, bn * block_N + nn] = T.Cast("bfloat16", C_r[mm, nn])

    return fp8_gemm_kernel


def main():
    M, N, K = 128, 384, 3072

    kernel = _fp8_gemm_kernel(
        M=M, N=N, K=K,
        block_M=32, block_N=64, block_K=64,
        threads=128,
    )

    # Random FP8 inputs
    A = torch.randn(M, K, device="musa", dtype=torch.float32).clamp(-3, 3)
    B = torch.randn(N, K, device="musa", dtype=torch.float32).clamp(-3, 3)
    A8 = A.to(torch.float8_e4m3fn)
    B8 = B.to(torch.float8_e4m3fn)
    C = torch.zeros(M, N, device="musa", dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        kernel(A8, B8, C)
    torch.musa.synchronize()

    # Correctness check (rough)
    C_ref = (A.float() @ B.float().T).to(torch.bfloat16)
    print(f"max abs err: {(C.float() - C_ref.float()).abs().max().item():.4f}")
    print(f"max abs ref: {C_ref.float().abs().max().item():.4f}")

    # Bench
    from mate.testing.utils import bench_kineto
    sec = bench_kineto(lambda: kernel(A8, B8, C),
                       kernel_names="fp8_gemm_kernel",
                       num_tests=20, suppress_kineto_output=True,
                       flush_l2=True, with_multiple_kernels=True)
    print(f"TileLang FP8 GEMM (M={M},N={N},K={K}): {sec*1e6:.2f} us")

    # Roofline check
    flops = 2 * M * N * K
    tflops = flops / sec / 1e12
    print(f"Achieved: {tflops:.1f} TFLOP/s (S5000 FP8 peak: 1000 TFLOP/s)")


if __name__ == "__main__":
    main()
