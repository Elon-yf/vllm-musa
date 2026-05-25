#!/usr/bin/env python3
"""MUSA-0150 v2 cold-cache bench for csrc/musa/fused_add_rmsnorm.mu.

Replaces benchmarks/op_perf/bench_fused_add_rmsnorm.py — that script used
`time.perf_counter_ns` + no L2 flush + 20 warmup + 100 iters, which is the
warm-cache/host-timer pattern .claude/rules/musa-kernel-bench.md forbids. The
85.9 % "SOTA" claimed by the V14 probe was measured under that regime.

This bench uses mate.testing.utils.bench_kineto with flush_l2=True, the same
harness MR 238 uses on the MUSA-native JIT kernels. Each iteration sees a
cold L2 (8 GB int32 .zero_() between fn() calls) so the reported GB/s is
the kernel's true effective bandwidth, not the warm-L2 artifact.

Production shapes for M2.5 decode (per-rank, TP=8, no-EP):
  - hidden=3072 (per-rank shard of full hidden=24576)
  - rows ∈ {1, 6, 16, 64, 4096} (decode-single, narrow-d5 verify, BS sweep, prefill)

Also includes the V14 probe's headline shape M=4096 N=8192 for direct
comparison to the documented "1470 GB/s warm-cache" claim.

Usage (inside authorized MUSA container, /ws editable-installed):

  python3 benchmarks/op_perf/bench_fused_add_rmsnorm_v2.py
  python3 benchmarks/op_perf/bench_fused_add_rmsnorm_v2.py \\
      --shapes M2.5_decode probe_v14

  # Force a single block_x for debugging:
  VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X=1024 \\
      python3 benchmarks/op_perf/bench_fused_add_rmsnorm_v2.py

Roofline = S5000 GDDR6 peak = 1600 GB/s; pass_ratio = 0.95 → target 1520 GB/s
for memory-bound large-M shapes. Small-M (M≤16) is launch-overhead-bound,
not bandwidth-bound — see PERF NOTE in csrc/musa/fused_add_rmsnorm.mu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch

# Prime MUSA patches before any torch.cuda symbols are captured.
import torchada  # noqa: F401
import torch_musa  # noqa: F401

# Load torch.ops directly from the .so files to avoid the vllm-plugin
# circular-import error that fires when `import vllm` triggers
# `vllm.plugins.load_plugins_by_group()` against a partially-initialized
# vllm_musa module.
for _so in (
    "/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
    "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so",
):
    if os.path.exists(_so):
        torch.ops.load_library(_so)

from mate.testing.utils import bench_kineto  # noqa: E402

_DEVICE = "musa"
# S5000 GDDR6 bandwidth anchor. The kernel is read+write heavy (4MN read +
# 2MN write per call), so the right ceiling is sphere-kb's authoritative
# claim s5000.practical_gddr6_bandwidth_updated: 1200 GB/s for read+write
# traffic (not the 1600 GB/s theoretical peak from mtforge/docs/HARDWARE.md
# — that's pure-read marketing). pass_ratio 0.95 → target 1140 GB/s.
_ROOFLINE_GBPS = float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200"))


@dataclass(frozen=True)
class Shape:
    name: str
    rows: int
    hidden: int


# M2.5 per-rank decode (TP=8 no-EP): hidden=3072 (per-rank shard of 24576)
# Common rows in the workload:
#   1   = greedy decode single token
#   6   = narrow-d5 chain verify (1 + 5 spec tokens)
#   16  = small concurrency
#   64  = larger concurrency
#   4096 = prefill at 4k input
SHAPE_SETS: dict[str, tuple[Shape, ...]] = {
    "M2.5_decode_per_rank": (
        Shape("decode_single", 1, 3072),
        Shape("verify_d5_chain", 6, 3072),
        Shape("bs16_decode", 16, 3072),
        Shape("bs64_decode", 64, 3072),
        Shape("bs256_decode", 256, 3072),
        Shape("prefill_4k", 4096, 3072),
    ),
    "M2.5_decode_full_hidden": (
        Shape("decode_single_h24576", 1, 24576),
        Shape("verify_d5_chain_h24576", 6, 24576),
        Shape("bs16_h24576", 16, 24576),
        Shape("bs256_h24576", 256, 24576),
    ),
    "probe_v14_headline": (
        # Shapes the V14 probe published in sphere-kb/probes/.../RESULTS.md
        Shape("v14_M256_N4096", 256, 4096),
        Shape("v14_M128_N4096", 128, 4096),
        Shape("v14_M512_N12288", 512, 12288),
        Shape("v14_M4096_N12288", 4096, 12288),
        Shape("v14_M4096_N8192", 4096, 8192),
    ),
    "mr238_layernorm_gated_reference": (
        # MR 238's published `rms_norm_gated M4096 N8192` row = 1017 GB/s TileLang
        # (not directly comparable — different op — but anchors the shape).
        Shape("mr238_M4096_N8192", 4096, 8192),
    ),
}


def _bytes_moved(rows: int, hidden: int, dtype: torch.dtype) -> int:
    """Bytes read + written by one fused_add_rmsnorm call.

    Read: input[M, N], residual[M, N], weight[N]
    Write: residual[M, N] (sum), input[M, N] (normed)
    Total = 4 * M * N + N (all dtype-sized).
    """
    elem = torch.empty((), dtype=dtype).element_size()
    return (4 * rows * hidden + hidden) * elem


def _flops(rows: int, hidden: int) -> int:
    """FLOPs per fused_add_rmsnorm call.

    Per element (M*N total):
      add: 1 FLOP
      sq:  1 FLOP
      mul (output): 2 FLOPs (inv_rms * weight * fused)
    Per row:
      rsqrt: ~2 FLOPs
    Total = 4 * M * N + 2 * M  ≈ 4 M N.
    """
    return 4 * rows * hidden + 2 * rows


def _make_inputs(rows: int, hidden: int, dtype: torch.dtype):
    g = torch.Generator(device=_DEVICE).manual_seed(0)
    inp = torch.randn((rows, hidden), dtype=dtype, device=_DEVICE, generator=g) * 0.1
    res = torch.randn((rows, hidden), dtype=dtype, device=_DEVICE, generator=g) * 0.1
    w = torch.randn((hidden,), dtype=dtype, device=_DEVICE, generator=g) * 0.5 + 1.0
    return inp.contiguous(), res.contiguous(), w.contiguous()


def _resolve_op(prefer: str):
    """Return the torch op pointer for the requested fused_add_rmsnorm variant.

    `prefer` is one of:
      "v14"      — vllm-musa native MUSA-0150 V14 kernel
                   (torch.ops._C_musa_ops.musa_fused_add_rms_norm).
      "upstream" — upstream vllm kernel
                   (torch.ops._C.fused_add_rms_norm in vllm/_C.so).
    """
    if prefer == "v14":
        op = getattr(torch.ops, "_C_musa_ops", None)
        if op is not None and hasattr(op, "musa_fused_add_rms_norm"):
            return op.musa_fused_add_rms_norm, "_C_musa_ops::musa_fused_add_rms_norm"
        raise RuntimeError(
            "V14 op torch.ops._C_musa_ops.musa_fused_add_rms_norm is not loaded. "
            "Confirm /ws/vllm_musa/_C.cpython-*.so is loaded via torch.ops.load_library."
        )
    if prefer == "upstream":
        op = getattr(torch.ops, "_C", None)
        if op is not None and hasattr(op, "fused_add_rms_norm"):
            return op.fused_add_rms_norm, "_C::fused_add_rms_norm"
        raise RuntimeError(
            "Upstream op torch.ops._C.fused_add_rms_norm is not loaded. "
            "Confirm /ws/vllm/_C.cpython-*.so is loaded via torch.ops.load_library."
        )
    raise ValueError(f"prefer must be 'v14' or 'upstream', got {prefer!r}")


def _bench_shape(shape: Shape, dtype: torch.dtype, kernel_substr: str,
                 prefer: str, num_tests: int = 30) -> dict:
    inp, res, w = _make_inputs(shape.rows, shape.hidden, dtype)
    op_fn, op_name = _resolve_op(prefer)
    eps = 1e-5

    def _runner():
        # in-place: residual ← input + residual, input ← rmsnorm(residual)
        op_fn(inp, res, w, eps)

    # Warmup outside the measurement (mate.bench_kineto already does an extra
    # warmup pass inside the profiler schedule).
    for _ in range(3):
        _runner()
    torch.musa.synchronize()

    seconds = bench_kineto(
        _runner,
        kernel_names=kernel_substr,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,                 # MANDATORY (see musa-kernel-bench.md)
    )
    if seconds <= 0:
        raise RuntimeError(
            f"bench_kineto returned 0 for {kernel_substr!r}; check substring match"
        )
    bytes_moved = _bytes_moved(shape.rows, shape.hidden, dtype)
    flops = _flops(shape.rows, shape.hidden)
    gbps = bytes_moved / seconds / 1e9
    tflops = flops / seconds / 1e12
    return {
        "shape": shape.name,
        "rows": shape.rows,
        "hidden": shape.hidden,
        "dtype": str(dtype).removeprefix("torch."),
        "bytes": bytes_moved,
        "flops": flops,
        "latency_us": seconds * 1e6,
        "GB_s": gbps,
        "TFLOPS": tflops,
        "pct_peak_bw": gbps / _ROOFLINE_GBPS * 100,
        "kernel": op_name,
    }


def _print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'shape':<28s} {'rows':>6s} {'hidden':>7s} {'dtype':>6s} "
        f"{'latency_us':>12s} {'GB/s':>10s} {'%peak':>7s} {'TFLOPS':>9s}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['shape']:<28s} {r['rows']:>6d} {r['hidden']:>7d} {r['dtype']:>6s} "
            f"{r['latency_us']:>12.3f} {r['GB_s']:>10.2f} {r['pct_peak_bw']:>6.1f}% "
            f"{r['TFLOPS']:>9.4f}"
        )


def main():
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["M2.5_decode_per_rank", "probe_v14_headline"],
        choices=list(SHAPE_SETS.keys()),
        help="Which shape set(s) to bench.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Tensor dtype (fused_add_rmsnorm runs on bf16/fp16).",
    )
    parser.add_argument(
        "--variant",
        choices=["v14", "upstream", "both"],
        default="both",
        help="Which fused_add_rmsnorm impl to bench. 'both' runs V14 and "
        "upstream side-by-side for the same shapes.",
    )
    parser.add_argument(
        "--kernel-substr-v14",
        default="vllm_musa::fused_add_rmsnorm_kernel",
        help="Substring matched in profiler table for the V14 kernel.",
    )
    parser.add_argument(
        "--kernel-substr-upstream",
        default="vllm::fused_add_rms_norm_kernel",
        help="Substring matched in profiler table for the upstream kernel.",
    )
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human table.",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        print("ERROR: MUSA device not available", file=sys.stderr)
        sys.exit(2)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    forced = os.environ.get("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X", "auto")

    print(f"# MUSA-0150 cold-cache bench (mate.bench_kineto + flush_l2=True)")
    print(f"# device: {torch.musa.get_device_name(0)}")
    print(f"# dtype:  {args.dtype}")
    print(f"# block_x dispatch: {'forced=' + forced if forced != 'auto' else 'adaptive (V14)'}")
    print(f"# roofline: {_ROOFLINE_GBPS:.0f} GB/s (S5000 GDDR6 peak)")
    print(f"# pass_ratio 0.95 → target {_ROOFLINE_GBPS*0.95:.0f} GB/s")
    print(f"# num_tests: {args.num_tests}")

    variants = ["v14", "upstream"] if args.variant == "both" else [args.variant]
    substrs = {
        "v14": args.kernel_substr_v14,
        "upstream": args.kernel_substr_upstream,
    }

    all_rows: list[dict] = []
    for shape_set in args.shapes:
        print(f"\n## {shape_set}")
        rows_by_variant: dict[str, list[dict]] = {}
        for variant in variants:
            rows = []
            for shape in SHAPE_SETS[shape_set]:
                try:
                    r = _bench_shape(shape, dtype, substrs[variant], variant,
                                     args.num_tests)
                    r["variant"] = variant
                    rows.append(r)
                except Exception as exc:
                    rows.append({
                        "shape": shape.name, "rows": shape.rows,
                        "hidden": shape.hidden, "variant": variant,
                        "dtype": str(dtype).removeprefix("torch."),
                        "error": str(exc)[:200],
                    })
            rows_by_variant[variant] = rows
            if args.json:
                continue
            print(f"\n### variant = {variant}")
            _print_table([r for r in rows if "error" not in r])
            for r in rows:
                if "error" in r:
                    print(f"  ERROR {r['shape']:<28s}: {r['error']}")

        # Side-by-side speedup table when both variants ran
        if len(variants) == 2 and not args.json:
            print(f"\n### V14 vs upstream speedup")
            print(f"{'shape':<28s} {'rows':>6s} {'hidden':>7s} "
                  f"{'v14_us':>10s} {'up_us':>10s} {'speedup':>9s} "
                  f"{'v14 GB/s':>10s} {'up GB/s':>10s}")
            print("-" * 95)
            v14_by_shape = {r['shape']: r for r in rows_by_variant.get('v14', [])
                            if 'error' not in r}
            up_by_shape = {r['shape']: r for r in rows_by_variant.get('upstream', [])
                           if 'error' not in r}
            for sname in [s.name for s in SHAPE_SETS[shape_set]]:
                v = v14_by_shape.get(sname)
                u = up_by_shape.get(sname)
                if v and u:
                    sp = u['latency_us'] / v['latency_us']
                    print(f"{sname:<28s} {v['rows']:>6d} {v['hidden']:>7d} "
                          f"{v['latency_us']:>10.3f} {u['latency_us']:>10.3f} "
                          f"{sp:>8.2f}x {v['GB_s']:>10.2f} {u['GB_s']:>10.2f}")

        all_rows.extend([r for variant in variants for r in rows_by_variant[variant]])
        torch.musa.empty_cache()

    if args.json:
        print(json.dumps({"all": all_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
