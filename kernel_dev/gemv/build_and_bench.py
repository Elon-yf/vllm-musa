#!/usr/bin/env python3
"""MUSA-0153 standalone kernel-dev harness — JIT compile gemv.mu only."""
from __future__ import annotations
import argparse, os, sys, subprocess
import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401
from mate.testing.utils import bench_kineto

WS = "/ws"
BUILD = "/tmp/kernel_dev_gemv/build"
SO_PATH = f"{BUILD}/gemv_dev.so"

EXT_INCLUDE = [
    f"{WS}/csrc_musa", f"{WS}/csrc",
    f"{WS}/third_party/vllm/csrc_musa", f"{WS}/third_party/vllm/csrc",
    "/usr/local/musa/include",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible/aten/src",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible/include",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible/include/torch/csrc/api/include",
    "/usr/local/lib/python3.10/dist-packages/torch_musa/share/torch_musa_codegen",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/include/python3.10",
]
MCC_FLAGS = ["-O3", "-std=c++17", "-fPIC", "-x", "musa", "-mtgpu", "-Od3",
             "-ffast-math", "-fmusa-flush-denormals-to-zero", "-fno-strict-aliasing",
             "-DUSE_MUSA", "-DENABLE_FP8", "--offload-arch=mp_31", "-march=native",
             "-DTORCH_API_INCLUDE_EXTENSION_H", "-DTORCH_EXTENSION_NAME=gemv_dev"]

def build():
    os.makedirs(BUILD, exist_ok=True)
    inc = sum([["-I", p] for p in EXT_INCLUDE], [])
    gemv_src = f"{WS}/csrc/musa/gemv.mu"
    wrap_src = "/tmp/kernel_dev_gemv/wrapper.cpp"
    gemv_o = f"{BUILD}/gemv.o"
    wrap_o = f"{BUILD}/wrapper.o"

    def newer(src, obj):
        return (not os.path.exists(obj)) or os.path.getmtime(src) > os.path.getmtime(obj)

    if newer(gemv_src, gemv_o):
        print(f"[mcc] {gemv_src}", file=sys.stderr)
        subprocess.check_call(["/usr/local/musa/bin/mcc"] + inc + MCC_FLAGS + ["-c", gemv_src, "-o", gemv_o])
    if newer(wrap_src, wrap_o):
        print(f"[mcc] {wrap_src}", file=sys.stderr)
        subprocess.check_call(["/usr/local/musa/bin/mcc"] + inc + MCC_FLAGS + ["-c", wrap_src, "-o", wrap_o])
    if (not os.path.exists(SO_PATH)) or any(
        os.path.getmtime(o) > os.path.getmtime(SO_PATH) for o in [gemv_o, wrap_o]
    ):
        print(f"[link] {SO_PATH}", file=sys.stderr)
        # Link against torch + torch_musa (musa_python carries torch_musa exports).
        subprocess.check_call([
            "/usr/local/musa/bin/mcc", "-shared", "-fPIC", gemv_o, wrap_o,
            "-o", SO_PATH,
            "-L/usr/local/musa/lib", "-lmusart",
            "-L/usr/local/lib/python3.10/dist-packages/torch/lib",
            "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10",
            "-L/usr/local/lib/python3.10/dist-packages/torch_musa/lib",
            "-lmusa_python",
        ])
    torch.ops.load_library(SO_PATH)


class M25:
    hidden = 3072
    intermediate = 192
    num_experts = 256
    topk = 8


def _q(t, gs):
    K = t.size(-1); ng = K // gs; front = t.shape[:-1]
    g = t.reshape(*front, ng, gs).float()
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = (amax / 448.0).squeeze(-1)
    q = (g / amax * 448.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(*front, K).contiguous(), s.contiguous()


def make_k1(bs):
    gn = torch.Generator(device="musa").manual_seed(0)
    A = (torch.randn((bs, M25.hidden), dtype=torch.bfloat16, device="musa", generator=gn) * 0.1).contiguous()
    w_raw = (torch.randn((M25.num_experts, 2*M25.intermediate, M25.hidden),
                         dtype=torch.bfloat16, device="musa", generator=gn) * 0.05).contiguous()
    w1, ws = _q(w_raw, 128)
    C = torch.empty((bs*M25.topk, M25.intermediate), dtype=torch.bfloat16, device="musa")
    ti = torch.stack([torch.randperm(M25.num_experts, device="musa")[:M25.topk] for _ in range(bs)]).to(torch.int32).contiguous()
    tw = torch.softmax(torch.randn((bs, M25.topk), dtype=torch.float32, device="musa", generator=gn), -1).contiguous()
    return A, w1, C, ws, tw, ti


def bytes_k1(bs):
    a = bs * M25.topk
    return bs*M25.hidden*2 + a*2*M25.intermediate*M25.hidden + a*2*M25.intermediate*(M25.hidden//128)*4 + bs*M25.topk*M25.intermediate*2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, nargs="+", default=[1, 6, 16, 64])
    p.add_argument("--num-tests", type=int, default=20)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    if args.rebuild and os.path.exists(BUILD):
        import shutil; shutil.rmtree(BUILD)
    build()
    op = torch.ops.gemv_dev.musa_fused_gemv_moe
    print("# MUSA-0153 standalone kernel-dev bench")
    print(f"# block_config: {os.environ.get('VLLM_MUSA_GEMV_MOE_BLOCK', 'adaptive')}")
    print(f"\n{'BS':>4s} {'lat_us':>9s} {'GB/s':>9s} {'%peak':>7s}")
    for bs in args.bs:
        A, w1, C, ws, tw, ti = make_k1(bs)
        def run(): op(A, w1, C, None, ws, tw, ti, False, int(M25.topk), False, True)
        for _ in range(3): run()
        torch.musa.synchronize()
        sec = bench_kineto(run, kernel_names="musa_gemv_kernel",
                           num_tests=args.num_tests, suppress_kineto_output=True, flush_l2=True)
        gbps = bytes_k1(bs) / sec / 1e9
        print(f"{bs:>4d} {sec*1e6:>9.2f} {gbps:>9.2f} {gbps/1200*100:>6.1f}%")
        torch.musa.empty_cache()


if __name__ == "__main__":
    main()
