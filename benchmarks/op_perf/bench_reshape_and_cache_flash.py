#!/usr/bin/env python3
"""MUSA-0151 cold-cache bench for csrc/musa/cache_kernels.mu.

Replaces the previous bench (time.perf_counter_ns + no L2 flush + warm
iters). Uses mate.testing.utils.bench_kineto with flush_l2=True per
.claude/rules/musa-kernel-bench.md.

Op: reshape_and_cache_flash_nhd_kernel<T, BLOCK_X=512, TOKENS_PER_BLOCK>.
Pure bandwidth-bound gather (key, value) -> scatter (kv-cache), 4 * T *
num_heads_kv * head_size * sizeof(dtype) bytes per call.

Production M2.5 shapes (per-rank, TP=8 no-EP):
  num_heads_kv=1 (full kv_heads=8 sharded TP=8), head_size=128.
  vecs_per_token = num_heads_kv * head_size / 8 = 16  -> TPB=8 default.
  num_tokens ∈ {1, 6, 16, 64, 256, 4096}.

Roofline anchor: sphere-kb claim s5000.practical_gddr6_bandwidth_updated
= 1200 GB/s for read+write traffic (this kernel is 2T*H*D read +
2T*H*D write = 4T*H*D total, all GDDR6 traffic).

Usage:

  python3 benchmarks/op_perf/bench_reshape_and_cache_flash.py
  VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK=4 python3 benchmarks/...

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


@dataclass(frozen=True)
class Shape:
    name: str
    num_tokens: int
    num_heads: int
    head_size: int
    block_size: int = 16
    num_blocks: int = 16384  # enough for max_model_len = 196608


SHAPE_SETS: dict[str, tuple[Shape, ...]] = {
    "M2.5_decode_per_rank": (
        Shape("decode_single", 1, 1, 128),
        Shape("verify_d5_chain", 6, 1, 128),
        Shape("bs16_decode", 16, 1, 128),
        Shape("bs64_decode", 64, 1, 128),
        Shape("bs256_decode", 256, 1, 128),
        Shape("prefill_4k", 4096, 1, 128),
    ),
    # Wider configs (full kv_heads on rank, non-TP) for diversity
    "wide_heads": (
        Shape("T256_H4_D128", 256, 4, 128),
        Shape("T256_H8_D128", 256, 8, 128),
        Shape("T4096_H4_D128", 4096, 4, 128),
        Shape("T4096_H8_D128", 4096, 8, 128),
    ),
}


def _bytes_moved(shape: Shape, dtype: torch.dtype) -> int:
    """Read key + value, write key_cache + value_cache. 4 * T * H * D * dt."""
    elem = torch.empty((), dtype=dtype).element_size()
    return 4 * shape.num_tokens * shape.num_heads * shape.head_size * elem


def _make_inputs(shape: Shape, dtype: torch.dtype):
    g = torch.Generator(device=_DEVICE).manual_seed(0)
    key = torch.randn((shape.num_tokens, shape.num_heads, shape.head_size),
                      dtype=dtype, device=_DEVICE, generator=g).contiguous()
    value = torch.randn_like(key)
    kc = torch.zeros((shape.num_blocks, shape.block_size, shape.num_heads,
                      shape.head_size), dtype=dtype, device=_DEVICE)
    vc = torch.zeros_like(kc)
    # Slot mapping: each token writes to a unique slot
    max_slot = shape.num_blocks * shape.block_size
    slots = torch.arange(shape.num_tokens, dtype=torch.int64, device=_DEVICE)
    slots = (slots * 17) % max_slot  # pseudo-random but deterministic spread
    return key, value, kc, vc, slots


def _resolve_op():
    op = torch.ops._C_musa_ops
    for name in ("musa_reshape_and_cache_flash_nhd", "reshape_and_cache_flash"):
        if hasattr(op, name):
            return getattr(op, name), f"_C_musa_ops::{name}"
    raise RuntimeError(
        "Could not find reshape_and_cache_flash op. Available: "
        + ", ".join(n for n in dir(op) if "cache" in n.lower())
    )


def _bench_shape(shape: Shape, dtype: torch.dtype, num_tests: int = 30) -> dict:
    key, value, kc, vc, slots = _make_inputs(shape, dtype)
    op_fn, op_name = _resolve_op()

    def _runner():
        op_fn(key, value, kc, vc, slots)

    for _ in range(3):
        _runner()
    torch.musa.synchronize()

    seconds = bench_kineto(
        _runner,
        kernel_names="reshape_and_cache_flash_nhd_kernel",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError(
            "bench_kineto returned 0; check kernel name substring match"
        )
    bytes_moved = _bytes_moved(shape, dtype)
    gbps = bytes_moved / seconds / 1e9
    return {
        "shape": shape.name,
        "num_tokens": shape.num_tokens,
        "num_heads": shape.num_heads,
        "head_size": shape.head_size,
        "dtype": str(dtype).removeprefix("torch."),
        "bytes": bytes_moved,
        "latency_us": seconds * 1e6,
        "GB_s": gbps,
        "pct_peak_bw": gbps / _ROOFLINE_GBPS * 100,
        "kernel": op_name,
    }


def _print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'shape':<22s} {'T':>6s} {'H':>3s} {'D':>4s} {'dtype':>8s} "
        f"{'latency_us':>12s} {'GB/s':>10s} {'%peak':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['shape']:<22s} {r['num_tokens']:>6d} {r['num_heads']:>3d} "
            f"{r['head_size']:>4d} {r['dtype']:>8s} "
            f"{r['latency_us']:>12.3f} {r['GB_s']:>10.2f} "
            f"{r['pct_peak_bw']:>6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+",
                        default=["M2.5_decode_per_rank", "wide_heads"],
                        choices=list(SHAPE_SETS.keys()))
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        print("ERROR: MUSA not available", file=sys.stderr)
        sys.exit(2)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    forced_tpb = os.environ.get("VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK", "auto")

    print("# MUSA-0151 cold-cache bench (mate.bench_kineto + flush_l2=True)")
    print(f"# device: {torch.musa.get_device_name(0)}")
    print(f"# dtype:  {args.dtype}")
    print(f"# tokens_per_block: {'forced=' + forced_tpb if forced_tpb != 'auto' else 'adaptive'}")
    print(f"# roofline: {_ROOFLINE_GBPS:.0f} GB/s (S5000 GDDR6 R+W practical)")

    all_rows: list[dict] = []
    for shape_set in args.shapes:
        print(f"\n## {shape_set}")
        rows = [_bench_shape(shape, dtype, args.num_tests)
                for shape in SHAPE_SETS[shape_set]]
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            _print_table(rows)
        all_rows.extend(rows)
        torch.musa.empty_cache()

    if args.json:
        print(json.dumps({"all": all_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
