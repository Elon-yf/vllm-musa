#!/usr/bin/env python3
"""MUSA-0153 cold-cache bench for csrc/musa/gemv.mu — the dominant kernel.

Targets `musa_fused_gemv_moe` at the production M2.5 shape: hidden=3072,
intermediate=192, num_experts=256, topk=8 (per-rank, TP=8 no-EP).

Two regimes from the profile (73.4 % combined kernel time):
  K1: w1 fused swiglu — A bf16 [M, 3072], B fp8 [256, 384, 3072]
       (output is M*topk*192 after swiglu halving)
  K2: w2                — A bf16 [M*topk, 192], B fp8 [256, 3072, 192]
       (output is M*3072)

Roofline anchor: sphere-kb claim s5000.practical_gddr6_bandwidth_updated.
Both regimes are bandwidth-bound at M=1 (weight load dominates).

Usage:

  python3 benchmarks/op_perf/bench_fused_gemv_moe.py
  VLLM_MUSA_GEMV_MOE_BLOCK=32x4 python3 benchmarks/op_perf/bench_fused_gemv_moe.py
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401

for _so in (
    "/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
    "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so",
):
    if os.path.exists(_so):
        torch.ops.load_library(_so)

from mate.testing.utils import bench_kineto  # noqa: E402

_DEVICE = "musa"
_ROOFLINE_GBPS = float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200"))


@dataclass(frozen=True)
class M25Shape:
    """M2.5 per-rank shape (TP=8 no-EP)."""
    hidden: int = 3072
    intermediate: int = 192
    num_experts: int = 256
    topk: int = 8


SHAPES = M25Shape()


def _quantize_to_fp8_e4m3(t: torch.Tensor, group_size: int):
    """Per-group symmetric quantize bf16 weight to fp8_e4m3 + fp32 scale."""
    assert t.dim() in (2, 3)
    # group along last dim
    K = t.size(-1)
    assert K % group_size == 0, f"{K} % {group_size} != 0"
    num_groups = K // group_size
    front_shape = t.shape[:-1]
    g = t.reshape(*front_shape, num_groups, group_size).float()
    amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    # fp8_e4m3 max = 448
    scale = (amax / 448.0).squeeze(-1)            # [..., num_groups]
    q = (g / amax.clamp(min=1e-12) * 448.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q = q.reshape(*front_shape, K)
    return q.contiguous(), scale.contiguous()


def _make_k1_inputs(bseqlen: int, shp: M25Shape):
    """K1: w1 fused swiglu (hidden → 2*intermediate per expert)."""
    g = torch.Generator(device=_DEVICE).manual_seed(0)
    hidden_states = (
        torch.randn((bseqlen, shp.hidden), dtype=torch.bfloat16, device=_DEVICE, generator=g)
        * 0.1
    ).contiguous()
    w1_raw = (
        torch.randn(
            (shp.num_experts, 2 * shp.intermediate, shp.hidden),
            dtype=torch.bfloat16, device=_DEVICE, generator=g,
        ) * 0.05
    ).contiguous()
    w1, w1_scale = _quantize_to_fp8_e4m3(w1_raw, group_size=128)
    # output buffer: [M*topk, 2*intermediate] BEFORE swiglu halving, but
    # csrc kernel actually writes [M*topk, intermediate] (after halving)
    out = torch.empty((bseqlen * shp.topk, shp.intermediate),
                      dtype=torch.bfloat16, device=_DEVICE)
    # topk routing — pick 8 random experts per token
    topk_ids = torch.stack(
        [torch.randperm(shp.num_experts, device=_DEVICE)[: shp.topk] for _ in range(bseqlen)]
    ).to(torch.int32).contiguous()
    topk_weights = torch.softmax(
        torch.randn((bseqlen, shp.topk), dtype=torch.float32, device=_DEVICE, generator=g),
        dim=-1,
    ).contiguous()
    return hidden_states, w1, out, w1_scale, topk_weights, topk_ids


def _make_k2_inputs(bseqlen: int, shp: M25Shape):
    """K2: w2 (intermediate → hidden per expert), no swiglu."""
    g = torch.Generator(device=_DEVICE).manual_seed(0)
    interm = (
        torch.randn((bseqlen * shp.topk, shp.intermediate),
                    dtype=torch.bfloat16, device=_DEVICE, generator=g)
        * 0.1
    ).contiguous()
    w2_raw = (
        torch.randn(
            (shp.num_experts, shp.hidden, shp.intermediate),
            dtype=torch.bfloat16, device=_DEVICE, generator=g,
        ) * 0.05
    ).contiguous()
    # K=intermediate=192. Group must divide K. 192 = 64 * 3 → group=64.
    w2, w2_scale = _quantize_to_fp8_e4m3(w2_raw, group_size=64)
    out = torch.empty((bseqlen, shp.hidden), dtype=torch.bfloat16, device=_DEVICE)
    topk_ids = torch.stack(
        [torch.randperm(shp.num_experts, device=_DEVICE)[: shp.topk] for _ in range(bseqlen)]
    ).to(torch.int32).contiguous()
    topk_weights = torch.softmax(
        torch.randn((bseqlen, shp.topk), dtype=torch.float32, device=_DEVICE, generator=g),
        dim=-1,
    ).contiguous()
    return interm, w2, out, w2_scale, topk_weights, topk_ids


def _bytes_k1(bseqlen: int, shp: M25Shape) -> int:
    """Bytes touched by the K1 (w1 swiglu) call.

    Read: A [M, hidden] bf16 + active weights B [topk*M, 2*intermediate, hidden] fp8
          + active B_scale [topk*M, 2*intermediate, hidden/128] fp32 +
          topk_ids/topk_weights (small).
    Write: out [M*topk, intermediate] bf16.

    Each token activates topk=8 experts; the load is per-(token, expert).
    """
    a_bytes = bseqlen * shp.hidden * 2
    active = bseqlen * shp.topk
    b_bytes = active * 2 * shp.intermediate * shp.hidden * 1            # fp8
    b_scale_bytes = active * 2 * shp.intermediate * (shp.hidden // 128) * 4  # fp32
    out_bytes = bseqlen * shp.topk * shp.intermediate * 2
    return a_bytes + b_bytes + b_scale_bytes + out_bytes


def _bytes_k2(bseqlen: int, shp: M25Shape) -> int:
    """Bytes touched by the K2 (w2) call."""
    a_bytes = bseqlen * shp.topk * shp.intermediate * 2
    active = bseqlen * shp.topk
    b_bytes = active * shp.hidden * shp.intermediate * 1
    b_scale_bytes = active * shp.hidden * (shp.intermediate // 64) * 4
    out_bytes = bseqlen * shp.hidden * 2
    return a_bytes + b_bytes + b_scale_bytes + out_bytes


def _flops_k1(bseqlen: int, shp: M25Shape) -> int:
    # active_experts * (2 * M * 2_intermediate * hidden) MACs (×2 for FMA)
    return 2 * (bseqlen * shp.topk) * (2 * shp.intermediate) * shp.hidden


def _flops_k2(bseqlen: int, shp: M25Shape) -> int:
    return 2 * (bseqlen * shp.topk) * shp.hidden * shp.intermediate


def _bench_k1(bseqlen: int, shp: M25Shape, num_tests=30) -> dict:
    A, w1, C, w1_scale, topk_w, topk_ids = _make_k1_inputs(bseqlen, shp)
    op = torch.ops._C_musa_ops.musa_fused_gemv_moe

    def runner():
        op(A, w1, C, None, w1_scale, topk_w, topk_ids,
           False,            # mul_routed_weight (False for w1)
           int(shp.topk),
           False,            # use_int4_w4a16
           True)             # use_swigelu

    # warmup
    for _ in range(3):
        runner()
    torch.musa.synchronize()
    seconds = bench_kineto(
        runner,
        kernel_names="musa_gemv_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError("bench_kineto returned 0 for musa_gemv_kernel")
    bytes_moved = _bytes_k1(bseqlen, shp)
    flops = _flops_k1(bseqlen, shp)
    return {
        "regime": "K1_w1_swiglu",
        "bseqlen": bseqlen,
        "latency_us": seconds * 1e6,
        "bytes": bytes_moved,
        "GB_s": bytes_moved / seconds / 1e9,
        "TFLOPS": flops / seconds / 1e12,
        "pct_peak_bw": (bytes_moved / seconds / 1e9) / _ROOFLINE_GBPS * 100,
        "pct_peak_fp8": (flops / seconds / 1e12) / 1000 * 100,  # S5000 fp8 peak
    }


def _bench_k2(bseqlen: int, shp: M25Shape, num_tests=30) -> dict:
    A, w2, C, w2_scale, topk_w, topk_ids = _make_k2_inputs(bseqlen, shp)
    op = torch.ops._C_musa_ops.musa_fused_gemv_moe

    def runner():
        op(A, w2, C, None, w2_scale, topk_w, topk_ids,
           True,            # mul_routed_weight (True for w2)
           1,               # topk=1 because A already includes the topk dim
           False,
           False)

    for _ in range(3):
        runner()
    torch.musa.synchronize()
    seconds = bench_kineto(
        runner,
        kernel_names="musa_gemv_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError("bench_kineto returned 0 for musa_gemv_kernel")
    bytes_moved = _bytes_k2(bseqlen, shp)
    flops = _flops_k2(bseqlen, shp)
    return {
        "regime": "K2_w2",
        "bseqlen": bseqlen,
        "latency_us": seconds * 1e6,
        "bytes": bytes_moved,
        "GB_s": bytes_moved / seconds / 1e9,
        "TFLOPS": flops / seconds / 1e12,
        "pct_peak_bw": (bytes_moved / seconds / 1e9) / _ROOFLINE_GBPS * 100,
        "pct_peak_fp8": (flops / seconds / 1e12) / 1000 * 100,
    }


def _print_row(r):
    print(f"{r['regime']:<14s} BS={r['bseqlen']:>3d}  "
          f"lat={r['latency_us']:>8.2f}us  "
          f"BW={r['GB_s']:>8.2f} GB/s ({r['pct_peak_bw']:>5.1f}% of 1200) "
          f"compute={r['TFLOPS']:>6.2f} TFLOPS ({r['pct_peak_fp8']:>4.1f}% of 1000)")


def main():
    parser = argparse.ArgumentParser(__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bseqlens", type=int, nargs="+",
                        default=[1, 6, 16, 64, 256])
    parser.add_argument("--regime", choices=["k1", "k2", "both"], default="both")
    parser.add_argument("--num-tests", type=int, default=30)
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        print("ERROR: MUSA not available", file=sys.stderr); sys.exit(2)
    print(f"# MUSA-0153 musa_fused_gemv_moe cold-cache bench")
    print(f"# device: {torch.musa.get_device_name(0)}")
    print(f"# M2.5 per-rank shape: hidden=3072, intermediate=192, num_experts=256, topk=8")
    print(f"# roofline: 1200 GB/s practical R+W (sphere-kb); FP8 peak 1000 TFLOPS")
    print(f"# block_n x block_k: " + os.environ.get("VLLM_MUSA_GEMV_MOE_BLOCK", "adaptive"))
    print()
    for bs in args.bseqlens:
        if args.regime in ("k1", "both"):
            r = _bench_k1(bs, SHAPES, args.num_tests)
            _print_row(r)
        if args.regime in ("k2", "both"):
            r = _bench_k2(bs, SHAPES, args.num_tests)
            _print_row(r)
        torch.musa.empty_cache()


if __name__ == "__main__":
    main()
