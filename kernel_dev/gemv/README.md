# MUSA-0153 standalone kernel-dev harness

Sub-second iteration on `csrc/musa/gemv.mu` without rebuilding all of vllm-musa.

## How it works

`build_and_bench.py` compiles ONLY two files via direct mcc + linker:

  - `/ws/csrc/musa/gemv.mu` — the kernel under optimization
  - `kernel_dev/gemv/wrapper.cpp` — torch.library binding (gemv_dev::musa_fused_gemv_moe)

Both into `/tmp/kernel_dev_gemv/build/gemv_dev.so`. The script loads it via
`torch.ops.load_library` and benches at the M2.5 MoE swiglu shape
(hidden=3072, intermediate=192, num_experts=256, topk=8) cold-cache via
`mate.testing.utils.bench_kineto + flush_l2=True`.

A full vllm-musa rebuild takes ~5 min. This harness rebuilds gemv.mu in
~30 sec. Use it for the V2/V3 iterations on this kernel.

## Usage

Inside the authorized MUSA container (yeahdongcn70, venv
`/root/.virtualenvs/sglang-0.5.6` which has mate>=0.2.0):

```bash
# Push the bench files + edited gemv.mu to /tmp/kernel_dev_gemv/
# (from the host workspace via sshpass+cat)

source /root/.virtualenvs/sglang-0.5.6/bin/activate
MUSA_VISIBLE_DEVICES=0 python /tmp/kernel_dev_gemv/build_and_bench.py \
    --bs 1 6 16 64 128 --num-tests 20

# To force a specific block config:
VLLM_MUSA_GEMV_MOE_BLOCK=32x4 ...

# To clean rebuild (also after wrapper.cpp change):
... --rebuild
```

## Edit-rebuild-bench loop

1. Edit `/ws/csrc/musa/gemv.mu` (the production kernel) on the container.
2. `python build_and_bench.py` — incremental rebuild via mtime check.
3. Read bench output (cold-cache GB/s + % of 1200 GB/s practical peak).
4. Iterate.

When SOTA is reached, the .mu is already in vllm-musa csrc/. A full
`pip install -e . --no-build-isolation` rebuild then picks it up
production-side.

## Current bench shape (M2.5)

- K1 (w1 swiglu): A bf16 [BS, 3072], B fp8 [256, 384, 3072] groupwise.
  output [BS*8, 192] bf16 (after swiglu halving).

Not yet wired in this harness: K2 (w2 down — needs hidden=192 with non-128 scale tile).

## Roofline anchors

- S5000 GDDR6 R+W practical: **1200 GB/s** (sphere-kb
  s5000.practical_gddr6_bandwidth_updated)
- S5000 FP8 SQMMA peak: 1000 TFLOPS — current kernel is bandwidth-bound,
  reaches < 1 % of FP8 peak. Compute is not the lever; bandwidth is.

