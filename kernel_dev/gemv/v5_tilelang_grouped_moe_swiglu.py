#!/usr/bin/env python3
"""V5 TileLang grouped MoE FP8 GEMM + SwiGLU + routed-weight (fused).

Adds SwiGLU activation and routed-weight multiplication inside the
V4 grouped MoE kernel. Output is (M_padded, intermediate) bf16.

The fused single-kernel design is the op-level win lever: mate's
m_grouped_fp8_gemm_nt_contiguous only does the GEMM and needs a
separate silu+mul+scale kernel. V5 does it all in one launch.

Per block:
  - Block handles block_m rows × block_n columns of the OUTPUT (intermediate).
  - Computes acc_gate (W[expert, :intermediate]) and acc_up (W[expert,
    intermediate:]) via TWO T.gemm calls into separate fragments.
  - In-register: silu(gate) * up * routed_weight -> bf16 output.
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
def _v5_grouped_moe_gemm_swiglu_kernel(
    num_experts: int,
    intermediate: int,  # output dim per row (= N/2 of W)
    K: int,             # hidden
    block_m: int,
    block_n: int,       # output dim block (<= intermediate)
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
                for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=4):
                    T.copy(Aq[m_blk * block_m, k * block_k], A_s)
                    T.copy(W[expert, n_offset, k * block_k], B_gate_s)
                    T.copy(W[expert, n_offset_up, k * block_k], B_up_s)
                    T.gemm(A_s, B_gate_s, Acc_gate, transpose_B=True)
                    T.gemm(A_s, B_up_s, Acc_up, transpose_B=True)

            # SwiGLU + routed_weight fused write-back.
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


def bench_v5(bs, num_tests=20, block_m=64, block_n=64, block_k=64, topk=8,
             hidden=3072, intermediate=192, num_experts=256):
    n_total = bs * topk
    M_padded = ((n_total + block_m - 1) // block_m) * block_m
    n_full = 2 * intermediate
    K = hidden
    quant_tile = 128

    # Inputs
    a_raw = torch.rand((M_padded, K), device="musa", dtype=torch.float32) * 0.1
    w_raw = torch.rand((num_experts, n_full, K), device="musa", dtype=torch.float32) * 0.05
    Aq, A_scale = group_quantize_fp8(
        a_raw, (M_padded, K // quant_tile), (1, quant_tile),
        torch.float8_e4m3fn, "K"
    )
    Wq, W_scale = group_quantize_fp8(
        w_raw, (num_experts, n_full // quant_tile, K // quant_tile),
        (1, quant_tile, quant_tile),
        torch.float8_e4m3fn, "K"
    )
    # V5 uses its own block_m. Each active expert pads to block_m rows.
    n_active_experts = min(n_total, num_experts)
    M_padded_v5 = max(n_active_experts * block_m, block_m)
    m_indices = torch.full((M_padded_v5,), -1, device="musa", dtype=torch.int32)
    rw = torch.zeros((M_padded_v5,), device="musa", dtype=torch.float32)
    rows_per_expert_full = n_total // max(n_active_experts, 1)
    remainder = n_total - rows_per_expert_full * n_active_experts
    a_raw = torch.rand((M_padded_v5, K), device="musa", dtype=torch.float32) * 0.1
    Aq, A_scale = group_quantize_fp8(
        a_raw, (M_padded_v5, K // quant_tile), (1, quant_tile),
        torch.float8_e4m3fn, "K"
    )
    for e in range(n_active_experts):
        rows_here = rows_per_expert_full + (1 if e < remainder else 0)
        for r in range(rows_here):
            row = e * block_m + r
            m_indices[row] = e
            rw[row] = 1.0
    M_padded = M_padded_v5
    D = torch.empty((M_padded, intermediate), device="musa", dtype=torch.bfloat16)
    D_mate_size = M_padded

    kernel = _v5_grouped_moe_gemm_swiglu_kernel(
        num_experts=int(num_experts),
        intermediate=int(intermediate), K=int(K),
        block_m=int(block_m), block_n=int(block_n), block_k=int(block_k),
        quant_tile=int(quant_tile),
        threads=256,
    )

    def run():
        kernel(Aq, A_scale, Wq, W_scale, m_indices, rw, D)
    for _ in range(3):
        run()
    torch.musa.synchronize()
    sec = bench_kineto(run, kernel_names="v5_grouped_moe_gemm_swiglu",
                       num_tests=num_tests, suppress_kineto_output=True,
                       flush_l2=True, with_multiple_kernels=True)

    # Mate baseline: requires alignment_m in {128, 256}. Use 128.
    mate_align = 128
    M_padded_mate = max(n_active_experts * mate_align, mate_align)
    m_indices_mate = torch.full((M_padded_mate,), -1, device="musa", dtype=torch.int32)
    rw_mate = torch.zeros((M_padded_mate,), device="musa", dtype=torch.float32)
    for e in range(n_active_experts):
        rows_here = rows_per_expert_full + (1 if e < remainder else 0)
        for r in range(rows_here):
            row = e * mate_align + r
            m_indices_mate[row] = e
            rw_mate[row] = 1.0
    a_raw_mate = torch.rand((M_padded_mate, K), device="musa", dtype=torch.float32) * 0.1
    Aq_mate, A_scale_mate = group_quantize_fp8(
        a_raw_mate, (M_padded_mate, K // quant_tile), (1, quant_tile),
        torch.float8_e4m3fn, "K"
    )
    D_mate = torch.empty((M_padded_mate, n_full), device="musa", dtype=torch.bfloat16)

    def run_mate():
        m_grouped_fp8_gemm_nt_contiguous(
            (Aq_mate, A_scale_mate), (Wq, W_scale), D_mate, m_indices_mate,
            recipe=(1, quant_tile, quant_tile), alignment_m=mate_align,
        )
    for _ in range(3):
        run_mate()
    torch.musa.synchronize()
    sec_mate_gemm = bench_kineto(
        run_mate, kernel_names="ssgemm",
        num_tests=num_tests, suppress_kineto_output=True,
        flush_l2=True, with_multiple_kernels=True,
    )

    # Mate full op approximation: GEMM + separate SwiGLU + scale (PyTorch native)
    def run_mate_full():
        m_grouped_fp8_gemm_nt_contiguous(
            (Aq_mate, A_scale_mate), (Wq, W_scale), D_mate, m_indices_mate,
            recipe=(1, quant_tile, quant_tile), alignment_m=mate_align,
        )
        gate, up = D_mate.chunk(2, dim=-1)
        D2 = torch.nn.functional.silu(gate) * up
        D2.mul_(rw_mate.unsqueeze(-1).to(D2.dtype))
    for _ in range(3):
        run_mate_full()
    torch.musa.synchronize()
    # Use bench_gpu_time to measure full call time (covers all kernels)
    from mate.testing.utils import bench_gpu_time
    times = bench_gpu_time(
        run_mate_full,
        repeat_iters=num_tests, l2_flush=True, l2_flush_size_mb=64,
    )
    import statistics
    sec_mate_full = statistics.median(times) * 1e-3  # ms -> seconds

    return sec, sec_mate_gemm, sec_mate_full


def main():
    print("# V5 TileLang grouped MoE FP8 GEMM + SwiGLU + routed-weight (fused)")
    print("# Shape M2.5 K1: hidden=3072 intermediate=192 num_experts=256 topk=8")
    print()
    print(f"{'BS':>4s} {'M_padded':>10s}  {'v5 (us)':>9s}  {'mate gemm':>10s}  "
          f"{'mate full':>10s}  {'v5/full':>8s}  verdict")
    print("-" * 80)
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        try:
            sec, sec_mate_gemm, sec_mate_full = bench_v5(bs)
            ratio = sec / sec_mate_full
            verdict = "WIN" if ratio < 0.95 else ("TIE" if ratio < 1.05 else "LOSE")
            n_total = bs * 8
            block_m = 128
            n_active = min(n_total, 256)
            M_padded_real = n_active * block_m
            print(f"{bs:>4d} {M_padded_real:>10d}  {sec*1e6:>9.2f}"
                  f"  {sec_mate_gemm*1e6:>10.2f}  {sec_mate_full*1e6:>10.2f}"
                  f"  {ratio:>7.2f}x  {verdict}")
            torch.musa.empty_cache()
        except Exception as e:
            print(f"{bs:>4d}  ERROR {type(e).__name__}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
