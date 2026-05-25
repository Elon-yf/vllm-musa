#!/usr/bin/env python3
"""MUSA-0152 cold-cache bench for csrc/musa/quantization/per_token_group_quant.cu.

Replaces the previous bench (`time.perf_counter_ns` + no flush_l2 + warm
iters). Uses mate.testing.utils.bench_kineto + flush_l2=True per the
.claude/rules/musa-kernel-bench.md discipline.

Targets `per_token_group_quant_128_register_kernel<bf16, fp8>` (the M2.5
FP8 quant fast path; group_size=128, threads_per_group=16,
elems_per_thread=8). Pure bandwidth-bound (read bf16 input, write fp8
output + fp32 scale per group).

M2.5 per-rank shapes (TP=8 no-EP):
  intermediate=192 (sharded), hidden=3072 per rank.
  Active token counts for decode/prefill mix.

Roofline: 1200 GB/s practical R+W ceiling (sphere-kb
s5000.practical_gddr6_bandwidth_updated).
"""
from __future__ import annotations

import argparse
import json
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
_GROUP_SIZE = 128


@dataclass(frozen=True)
class Shape:
    name: str
    num_tokens: int
    hidden: int


SHAPE_SETS: dict[str, tuple[Shape, ...]] = {
    "M2.5_decode_per_rank": (
        Shape("decode_single", 1, 3072),
        Shape("verify_d5_chain", 6, 3072),
        Shape("bs16_decode", 16, 3072),
        Shape("bs64_decode", 64, 3072),
        Shape("bs256_decode", 256, 3072),
        Shape("prefill_4k", 4096, 3072),
    ),
    "wider": (
        # MoE / DeepSeek-like wider hidden
        Shape("T4096_H6144", 4096, 6144),
        Shape("T4096_H8192", 4096, 8192),
        Shape("T4096_H12288", 4096, 12288),
    ),
}


def _bytes_moved(shape: Shape, dtype: torch.dtype) -> int:
    """Read bf16 input + write fp8 output + write fp32 scale per group."""
    elem_in = torch.empty((), dtype=dtype).element_size()
    elem_fp8 = 1
    elem_scale = 4
    num_groups = (shape.hidden // _GROUP_SIZE)
    return (shape.num_tokens * shape.hidden * (elem_in + elem_fp8)
            + shape.num_tokens * num_groups * elem_scale)


def _make_inputs(shape: Shape, dtype: torch.dtype):
    g = torch.Generator(device=_DEVICE).manual_seed(0)
    inp = (torch.randn((shape.num_tokens, shape.hidden), dtype=dtype,
                       device=_DEVICE, generator=g) * 0.1).contiguous()
    # FP8 Fill op unsupported on MUSA — use empty() instead of zeros().
    output_q = torch.empty((shape.num_tokens, shape.hidden),
                           dtype=torch.float8_e4m3fn, device=_DEVICE)
    num_groups = shape.hidden // _GROUP_SIZE
    output_s = torch.empty((shape.num_tokens, num_groups),
                           dtype=torch.float32, device=_DEVICE)
    return inp, output_q, output_s


def _resolve_op():
    """Resolve the op pointer. torch.ops uses __getattr__ to materialize
    OpOverloadPacket on demand, so hasattr() works once load_library() ran."""
    op = torch.ops._C_musa_ops
    for name in ("per_token_group_fp8_quant", "per_token_group_quant_fp8",
                 "per_token_group_quant_8bit", "musa_per_token_group_fp8_quant"):
        try:
            return getattr(op, name), f"_C_musa_ops::{name}"
        except (AttributeError, RuntimeError):
            continue
    raise RuntimeError("per_token_group_quant op not found in _C_musa_ops.")


def _bench_shape(shape: Shape, dtype: torch.dtype, num_tests: int = 30) -> dict:
    inp, output_q, output_s = _make_inputs(shape, dtype)
    op_fn, op_name = _resolve_op()
    eps = 1e-10
    fp8_min = -448.0
    fp8_max = 448.0

    # Schema from torch_bindings: (input, output_q, output_s, group_size,
    # eps, fp8_min, fp8_max, scale_ue8m0, dummy_is_scale_transposed,
    # dummy_is_tma_aligned)
    def runner():
        op_fn(inp, output_q, output_s, _GROUP_SIZE, eps, fp8_min, fp8_max,
              False, False, False)

    try:
        runner()
        torch.musa.synchronize()
    except Exception as exc:
        raise RuntimeError(f"runner call failed: {exc}")

    for _ in range(3):
        runner()
    torch.musa.synchronize()

    seconds = bench_kineto(
        runner,
        kernel_names="per_token_group_quant_128_register_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError("bench_kineto returned 0; check kernel name match")
    bytes_moved = _bytes_moved(shape, dtype)
    gbps = bytes_moved / seconds / 1e9
    return {
        "shape": shape.name,
        "num_tokens": shape.num_tokens,
        "hidden": shape.hidden,
        "dtype": str(dtype).removeprefix("torch."),
        "bytes": bytes_moved,
        "latency_us": seconds * 1e6,
        "GB_s": gbps,
        "pct_peak_bw": gbps / _ROOFLINE_GBPS * 100,
        "kernel": op_name,
    }


def _print_table(rows):
    hdr = (f"{'shape':<22s} {'T':>6s} {'H':>6s} {'dtype':>8s} "
           f"{'latency_us':>12s} {'GB/s':>10s} {'%peak':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['shape']:<22s} {r['num_tokens']:>6d} {r['hidden']:>6d} "
              f"{r['dtype']:>8s} {r['latency_us']:>12.3f} "
              f"{r['GB_s']:>10.2f} {r['pct_peak_bw']:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+",
                        default=["M2.5_decode_per_rank", "wider"],
                        choices=list(SHAPE_SETS.keys()))
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        print("ERROR: MUSA not available", file=sys.stderr); sys.exit(2)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print("# MUSA-0152 cold-cache bench (mate.bench_kineto + flush_l2=True)")
    print(f"# device: {torch.musa.get_device_name(0)}")
    print(f"# dtype:  {args.dtype}")
    print(f"# roofline: {_ROOFLINE_GBPS:.0f} GB/s (S5000 GDDR6 R+W practical)")

    all_rows = []
    for shape_set in args.shapes:
        print(f"\n## {shape_set}")
        rows = []
        for shape in SHAPE_SETS[shape_set]:
            try:
                rows.append(_bench_shape(shape, dtype, args.num_tests))
            except Exception as exc:
                rows.append({"shape": shape.name, "error": str(exc)[:200]})
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            _print_table([r for r in rows if "error" not in r])
            for r in rows:
                if "error" in r:
                    print(f"  ERROR {r['shape']:<28s}: {r['error']}")
        all_rows.extend(rows)
        torch.musa.empty_cache()

    if args.json:
        print(json.dumps({"all": all_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
