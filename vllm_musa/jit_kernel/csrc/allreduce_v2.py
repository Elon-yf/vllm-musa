"""JIT wrapper for the v2 (device-descriptor / push) custom all-reduce.

Ported from SGLang MR 577 (`perf(musa): custom all-reduce v2 for 0.5.16`).
The v1 module in ``allreduce.py`` keeps its own JIT build name; this module
compiles ``distributed/custom_all_reduce_v2.mu`` into a *separate* shared
object so both algorithms can be loaded in the same process and selected at
runtime for A/B validation.

The v2 kernel exposes three launch entries: the TVM-FFI
``vllm_musa_custom_ar_launch_device_descriptor`` (push shots) and two plain
``extern "C"`` entry points for the pull shots,
``vllm_musa_custom_ar_launch_unregistered_raw_nocheck`` and
``vllm_musa_custom_ar_launch_registered_raw_nocheck``.  Upstream MR 577 binds
the pull pair through a 60-line pybind11 module
(``distributed/custom_all_reduce_pybind.cpp``); vllm-musa has no pybind11 JIT
infrastructure, so the same two symbols are bound here with ``ctypes`` against
the shared object that ``load_musa_jit`` already produces.  The kernel and its
arguments are identical; only the binding mechanism differs.
"""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path

import torch

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit

# Shot identifiers, mirroring the v1 constants used by the v2 object.
SHOT_PUSH = 0
SHOT_ONE_STAGE = 1
SHOT_TWO_STAGE = 2
SHOT_PUSH_WIDE = 3
SHOT_TWO_STAGE_512 = 4

# The v2 Signal ABI is fixed by the kernel: every JIT variant must agree on it.
#
#   struct alignas(128) PullSemaphore {   // 128 bytes
#       FlagType flag, counter;
#       FlagType self_counter[8];
#       FlagType peer_counter[2][8];
#   };
#   struct alignas(128) Signal {          // 2 * 512 * 128 == 131072 bytes
#       union { PullSemaphore pull[512]; FlagType push_epoch[512]; };
#       union { PullSemaphore publish[512]; PullSemaphore push_ready[512]; };
#   };
#
# `custom_all_reduce_v2.mu` asserts the same size at compile time, so this
# constant and the kernel cannot drift silently (S2 parity would fail).
_V2_SIGNAL_BLOCKS = 512
_V2_SEMAPHORE_BYTES = 128


def allreduce_v2_meta_size() -> int:
    """Bytes of one v2 Signal region (also the push/pull controller stride)."""
    return 2 * _V2_SIGNAL_BLOCKS * _V2_SEMAPHORE_BYTES


def push_buffer_bytes(nbytes: int, world_size: int) -> int:
    """Staging bytes for the push path: two epochs times one slot per rank."""
    return int(nbytes) * 2 * int(world_size)


def _default_threads_blocks(world_size: int) -> tuple[int, int]:
    # Same defaults as the v1 module so the two arms stay comparable.
    return (1024, 60) if int(world_size) == 8 else (512, 36)


def _compile_config(world_size: int, shot: int) -> tuple[int, int, int, int, int, int, int]:
    default_threads, default_blocks = _default_threads_blocks(world_size)
    is_push = shot in (SHOT_PUSH, SHOT_PUSH_WIDE)
    threads = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_THREADS", default_threads))
    blocks = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_BLOCKS", default_blocks))
    vector_load = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "0"))
    atomic_barrier = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_ATOMIC_BARRIER", "1"))
    max_blocks = int(
        os.environ.get("VLLM_MUSA_CUSTOM_AR_MAX_BLOCKS", str(max(120, blocks)))
    )
    max_blocks = max(max_blocks, blocks)
    # Upstream defaults: push shots use a static grid (dynamic_blocks=0) with
    # the polling wait loop, pull shots use dynamic blocks.  Only the push
    # case is compiled here because v2 pull still runs through the v1 module.
    dynamic_blocks = int(
        os.environ.get(
            "VLLM_MUSA_CUSTOM_AR_DYNAMIC_BLOCKS", "0" if is_push else "1"
        )
    )
    # `push_16b_asm` defaults to on upstream.  This port keeps it off until the
    # MUSA inline-asm store path is validated bit-exactly against the C++ store
    # path on real hardware (VLLM_MUSA_CUSTOM_AR_PUSH_16B_ASM=1 opts in).
    push_16b_asm = int(os.environ.get("VLLM_MUSA_CUSTOM_AR_PUSH_16B_ASM", "0"))
    if not is_push or push_16b_asm == 0:
        push_16b_asm = 0
    return threads, blocks, vector_load, atomic_barrier, max_blocks, dynamic_blocks, push_16b_asm


def _module_name(world_size: int, shot: int) -> str:
    threads, blocks, vector_load, atomic_barrier, max_blocks, dynamic_blocks, push_16b_asm = (
        _compile_config(world_size, shot)
    )
    return (
        f"vllm_musa_custom_all_reduce_v2_s{shot}_ws{int(world_size)}"
        f"_t{threads}_b{blocks}_v{vector_load}_ab{atomic_barrier}"
        f"_mb{max_blocks}_db{dynamic_blocks}_pa{push_16b_asm}"
    )


@functools.lru_cache(maxsize=32)
def _custom_ar_v2_module(world_size: int, shot: int):
    threads, blocks, vector_load, atomic_barrier, max_blocks, dynamic_blocks, push_16b_asm = (
        _compile_config(world_size, shot)
    )
    return load_musa_jit(
        _module_name(world_size, shot),
        ("distributed/custom_all_reduce_v2.mu",),
        extra_musa_cflags=(
            "-Wno-error=address-of-temporary",
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-D__MUSA_ARCH_LIST__=310",
            f"-DSGL_CUSTOM_AR_THREADS={threads}",
            f"-DSGL_CUSTOM_AR_BLOCKS={blocks}",
            f"-DSGL_CUSTOM_AR_VECTOR_LOAD={vector_load}",
            f"-DSGL_CUSTOM_AR_ATOMIC_BARRIER={atomic_barrier}",
            f"-DSGL_CUSTOM_AR_MAX_BLOCKS={max_blocks}",
            f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={dynamic_blocks}",
            f"-DSGL_CUSTOM_AR_PUSH_POLLING={1 if int(shot) in (SHOT_PUSH, SHOT_PUSH_WIDE) else 0}",
            f"-DSGL_CUSTOM_AR_PUSH_16B_ASM={push_16b_asm}",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
        ),
    )


def ensure_compiled(world_size: int, shot: int = SHOT_PUSH) -> None:
    _custom_ar_v2_module(int(world_size), int(shot))


def launch_device_descriptor(
    rank_data_device: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    """Launch a v2 kernel from a device-resident RankData pointer table.

    ``rank_data_device`` must be an int64 device tensor holding ``kMaxRanks``
    pointers; the kernel reads it as ``const RankData*``.  ``out`` is both
    input and output (the push path publishes from the caller's buffer).
    """
    _custom_ar_v2_module(int(world_size), int(shot)).vllm_musa_custom_ar_launch_device_descriptor(
        rank_data_device,
        signal_ptrs_cpu,
        out,
        int(self_signal_ptr),
        int(rank),
        int(world_size),
        int(shot),
    )


# --- pull launch entry points -------------------------------------------------
#
# The two ``extern "C"`` symbols below take raw pointers plus the current MUSA
# stream.  These wrappers mirror upstream's pybind11 bindings one-to-one: the
# tensor arguments are resolved to ``data_ptr()`` and ``out.numel()`` /
# ``dtype_code(out)`` are forwarded exactly as the pybind module did.

_DTYPE_CODES = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}

_raw_libs: dict[str, ctypes.CDLL] = {}


def _raw_symbols(world_size: int, shot: int):
    """Load the compiled v2 module and return its two raw launch callables."""
    name = _module_name(int(world_size), int(shot))
    lib = _raw_libs.get(name)
    if lib is None:
        _custom_ar_v2_module(int(world_size), int(shot))  # ensures the .so exists
        cache_dir = Path(
            os.environ.get("VLLM_MUSA_JIT_CACHE_DIR")
            or (Path.home() / ".cache" / "vllm_musa_jit")
        )
        so_path = cache_dir / name / f"{name}.so"
        if not so_path.is_file():
            raise RuntimeError(
                f"MUSA custom AR v2 module was not built at {so_path}; "
                "the pull launch entry points are unavailable"
            )
        lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
        unregistered = lib.vllm_musa_custom_ar_launch_unregistered_raw_nocheck
        unregistered.restype = None
        unregistered.argtypes = [
            ctypes.c_void_p,  # rank_ptrs
            ctypes.c_void_p,  # signal_ptrs
            ctypes.c_void_p,  # inp
            ctypes.c_void_p,  # out
            ctypes.c_int64,  # self_signal_ptr
            ctypes.c_int64,  # self_buffer_ptr
            ctypes.c_int64,  # push_buffer_bytes
            ctypes.c_int64,  # numel
            ctypes.c_int,  # dtype_code
            ctypes.c_int64,  # stream_value
            ctypes.c_int64,  # rank
            ctypes.c_int64,  # world_size
            ctypes.c_int64,  # shot
        ]
        registered = lib.vllm_musa_custom_ar_launch_registered_raw_nocheck
        registered.restype = None
        registered.argtypes = [
            ctypes.c_void_p,  # rank_ptrs
            ctypes.c_void_p,  # signal_ptrs
            ctypes.c_void_p,  # out
            ctypes.c_int64,  # self_signal_ptr
            ctypes.c_int64,  # numel
            ctypes.c_int,  # dtype_code
            ctypes.c_int64,  # stream_value
            ctypes.c_int64,  # rank
            ctypes.c_int64,  # world_size
            ctypes.c_int64,  # shot
        ]
        _raw_libs[name] = lib
    return (
        lib.vllm_musa_custom_ar_launch_unregistered_raw_nocheck,
        lib.vllm_musa_custom_ar_launch_registered_raw_nocheck,
    )


def _current_stream_value() -> int:
    """Raw handle of the current MUSA stream (what ``getCurrentMUSAStream`` gave)."""
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        current_stream = getattr(module, "current_stream", None)
        if current_stream is None:
            continue
        stream = current_stream()
        for attr in ("musa_stream", "cuda_stream"):
            value = getattr(stream, attr, None)
            if value is None:
                continue
            # 0 is the legacy/default stream handle and is a valid value here;
            # `getCurrentMUSAStream` returns exactly that outside a capture.
            return int(value)
    raise RuntimeError(
        "MUSA custom AR v2 cannot resolve the current stream handle; "
        "the pull launch path needs it"
    )


def _dtype_code(tensor: torch.Tensor) -> int:
    code = _DTYPE_CODES.get(tensor.dtype)
    if code is None:
        raise TypeError(
            "MUSA custom AR v2 only supports fp16/bf16/fp32, got "
            f"{tensor.dtype}"
        )
    return code


def launch_unregistered_func(world_size: int, shot: int):
    """Callable equivalent of upstream's ``launch_unregistered`` pybind binding."""
    symbols = _raw_symbols(int(world_size), int(shot))

    def launch(
        rank_data: torch.Tensor,
        signal_ptrs_cpu: torch.Tensor,
        input: torch.Tensor,
        out: torch.Tensor,
        self_signal_ptr: int,
        self_buffer_ptr: int,
        push_buffer_bytes: int,
        rank: int,
        world_size: int,
        shot: int,
    ) -> None:
        symbols[0](
            ctypes.c_void_p(rank_data.data_ptr()),
            ctypes.c_void_p(signal_ptrs_cpu.data_ptr()),
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(int(self_signal_ptr)),
            ctypes.c_int64(int(self_buffer_ptr)),
            ctypes.c_int64(int(push_buffer_bytes)),
            ctypes.c_int64(int(out.numel())),
            ctypes.c_int(_dtype_code(out)),
            ctypes.c_int64(_current_stream_value()),
            ctypes.c_int64(int(rank)),
            ctypes.c_int64(int(world_size)),
            ctypes.c_int64(int(shot)),
        )

    return launch


def launch_registered_func(world_size: int, shot: int):
    """Callable equivalent of upstream's ``launch_registered`` pybind binding."""
    symbols = _raw_symbols(int(world_size), int(shot))

    def launch(
        rank_data: torch.Tensor,
        signal_ptrs_cpu: torch.Tensor,
        out: torch.Tensor,
        self_signal_ptr: int,
        rank: int,
        world_size: int,
        shot: int,
    ) -> None:
        symbols[1](
            ctypes.c_void_p(rank_data.data_ptr()),
            ctypes.c_void_p(signal_ptrs_cpu.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(int(self_signal_ptr)),
            ctypes.c_int64(int(out.numel())),
            ctypes.c_int(_dtype_code(out)),
            ctypes.c_int64(_current_stream_value()),
            ctypes.c_int64(int(rank)),
            ctypes.c_int64(int(world_size)),
            ctypes.c_int64(int(shot)),
        )

    return launch

# Names used by the ported v2 object layer.  The object calls
# ``launch_*_pybind_func(world_size, shot)`` because upstream resolved them from
# its pybind11 module; here they resolve to the ctypes bindings above.  Aliasing
# keeps the object layer byte-identical to the MR instead of editing its launch
# call sites.
launch_unregistered_pybind_func = launch_unregistered_func
launch_registered_pybind_func = launch_registered_func
