# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import platform
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import torch
from packaging import version


def _ensure_numpy_compatible():
    """Ensure numpy<2 is installed (MUSA/PyTorch compatibility requirement).

    This may need to be called multiple times during setup because vllm
    installation can pull in numpy>=2 as a dependency.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2", "-q"])


def _ensure_torchada_installed():
    """Ensure torchada is installed (needed for torch.cuda patching)."""
    try:
        import torchada  # noqa: F401
    except ImportError:
        print("Installing torchada...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torchada", "--upgrade", "-q"]
        )
        import torchada  # noqa: F401


# Run dependency checks at setup start
_ensure_numpy_compatible()
_ensure_torchada_installed()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
sys.path.insert(0, str(root))

from build_utils.ccache import configure_compiler_cache

third_party = Path("third_party")
arch = platform.machine().lower()


def _read_pins():
    """read the single upstream-pin source of truth (third_party/PINS).

    KEY=VALUE only (no TOML — tomllib is 3.11+, the build box is py3.10). The same
    file is read by Makefile.sync, so the generation base cannot desync from the
    build base across version bumps.
    """
    pins = {}
    pins_path = root / "third_party" / "PINS"
    for line in pins_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pins[key.strip()] = value.split("#", 1)[0].strip()
    return pins


_PINS = _read_pins()

configure_compiler_cache(root)

# Detect editable install (pip install -e .) or develop mode
_is_editable_install = (
    "develop" in sys.argv
    or "editable_wheel" in sys.argv
    or any("--editable" in arg or "-e" in arg for arg in sys.argv)
)

if _is_editable_install:
    Path("vllm").mkdir(exist_ok=True)


def develop_dynamic_library(package_name, source_dir="./"):
    try:
        dist = distribution(package_name)
        install_path = dist.locate_file(package_name)

        target_dir = Path(install_path)
        source_path = Path(source_dir) / "vllm"

        for file_path in source_path.glob("*.so"):
            shutil.copy2(file_path, target_dir)

    except PackageNotFoundError:
        print(f"vLLM is not installed '{package_name}'")


def _warn_if_vllm_shadowed(source_dir):
    """After an editable vLLM install, warn loudly if ``import vllm`` still
    resolves OUTSIDE the clone -- i.e. another vLLM (e.g. a system install
    seen through a ``--system-site-packages`` venv) shadows the appended
    editable finder, so edits to ``third_party/vllm`` will NOT take effect."""
    try:
        origin = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util as u; s = u.find_spec('vllm');"
                " print(s.origin if s and s.origin else '')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
    except Exception:
        return
    expected = str(Path(source_dir).resolve())
    if origin and expected not in origin:
        print(
            "\n*** WARNING: editable vLLM is SHADOWED ***\n"
            f"    `import vllm` resolves to: {origin}\n"
            f"    expected under:           {expected}\n"
            "    Another vLLM is winning (e.g. a system/global install seen via\n"
            "    a --system-site-packages venv). Edits to third_party/vllm will\n"
            "    NOT take effect. Use a clean venv without a pre-existing vLLM.\n"
        )


def get_mcc_version():
    try:
        proc = subprocess.run(
            ["musa_version_query"], capture_output=True, text=True, timeout=5
        )

        if proc.returncode != 0:
            print(f"Warning: musa_version_query failed (exit code: {proc.returncode}")
            return None

        mcc_version = re.search(
            r'mcc:\s*\{[^}]*"version":\s*"([^"]+)"', proc.stdout, re.DOTALL
        )

        if mcc_version:
            return mcc_version.group(1)
        else:
            print(
                f"Warning: failed to get MUSA version, which may cause installation failure"
            )
            return None
    except Exception as e:
        print(
            f"Warning: failed to get MUSA version, which may cause installation failure: {e}"
        )
        return None


class _RepoInfo:
    """Configuration for a third-party git repository."""

    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_VLLM_REPO = _RepoInfo(
    name="vllm",
    git_repository="https://github.com/vllm-project/vllm.git",
    # pin read from third_party/PINS (single source of truth).
    git_tag=_PINS["VLLM_TAG"],
    git_shallow=False,
)

_FLASHINFER_REPO = _RepoInfo(
    name="flashinfer",
    git_repository="https://github.com/flashinfer-ai/flashinfer.git",
    # Keep the prepared MUSA-compatible FlashInfer baseline (intentionally
    # decoupled from upstream vLLM's choice — see the comment in third_party/PINS).
    # pin read from third_party/PINS.
    git_tag=_PINS["FLASHINFER_COMMIT"],
    git_shallow=False,
)

INCLUDE_DIRS = [
    root / "csrc",
    root / _VLLM_REPO.source_dir / "csrc",
    root / _FLASHINFER_REPO.source_dir / "include",
    root / _FLASHINFER_REPO.source_dir / "csrc",
]

# =============================================================================
# C/C++ Source Files for Extension Modules
# =============================================================================

VLLM_CSRC_SOURCES = [
    str(_VLLM_REPO.source_dir / "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cache_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cache_kernels_fused.cu"),
    # (2026-05-28): paged_attention_v1/v2 are CUDA-only and unused
    # on MUSA (vllm uses FlashAttention via mate's flash_attn_varlen_func).
    # Skipping them avoids compile time and mcc frontend failures. Their impl
    # registrations are stripped from third_party/vllm/csrc/torch_bindings.cpp
    # below so torch.ops._C import does not reference unbuilt symbols.
    # str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v1.cu"),
    # str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v2.cu"),
    str(_VLLM_REPO.source_dir / "csrc/attention/merge_attn_states.cu"),
    str(_VLLM_REPO.source_dir / "csrc/sampler.cu"),
    str(_VLLM_REPO.source_dir / "csrc/topk.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cuda_view.cu"),
    str(
        _VLLM_REPO.source_dir
        / "csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
    ),
    str(_VLLM_REPO.source_dir / "csrc/quantization/activation_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cuda_utils_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/custom_all_reduce.cu"),
    str(_VLLM_REPO.source_dir / "csrc/torch_bindings.cpp"),
    str(_VLLM_REPO.source_dir / "csrc/minimax_reduce_rms_kernel.cu"),
]

VLLM_STABLE_CSRC_SOURCES = [
    # v0.22 imports this extension for stable-ABI operator schemas. The CUDA
    # stable kernels require torch/headeronly/core/Dispatch.h, which is not in
    # the current torch_musa stack, so MUSA builds this as a schema-only shim.
    str(_VLLM_REPO.source_dir / "csrc/libtorch_stable/torch_bindings.cpp"),
]

VLLM_MUSA_CSRC_SOURCES = [
    "csrc/musa/torch_bindings.cpp",
    "csrc/musa/gemv.mu",
    "csrc/musa/fused_add_rmsnorm.mu",
    "csrc/musa/cache_kernels.mu",
    "csrc/musa/attention/deepseek_v4_cache_store.mu",
    "csrc/musa/attention/deepseek_v4_fused_qkv_rmsnorm.mu",
    "csrc/musa/attention/deepseek_v4_cache_utils.mu",
    "csrc/musa/attention/deepseek_v4_indexer_topk.mu",
    "csrc/musa/attention/deepseek_v4_sparse_flashmla.mu",
    "csrc/musa/attention/deepseek_v4_inv_rope_fp8_quant.mu",
    "csrc/musa/mhc/deepseek_v4_mhc_pre.mu",
    "csrc/musa/moe/deepseek_v4_topk_softplus_sqrt.mu",
    # XXX (MUSA): This local kernel stays non-stable for now because the
    # upstream v0.22 path moved under csrc/libtorch_stable and depends on
    # stable ABI headers not yet covered by the current MUSA path.
    "csrc/musa/quantization/per_token_group_quant.cu",
    "csrc/musa/sampler.mu",
    str(_FLASHINFER_REPO.source_dir / "csrc/norm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc/renorm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc/sampling.cu"),
]

VLLM_MOE_CSRC_SOURCES = [
    str(_VLLM_REPO.source_dir / "csrc/moe/moe_align_sum_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/topk_softmax_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/topk_softplus_sqrt_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/torch_bindings.cpp"),
]

# =============================================================================
# Source Code Patching Configuration
# =============================================================================

# Files to completely replace with local versions from csrc/
# These are full file overrides where the local copy replaces the upstream version.
CSRC_FILE_OVERRIDES = [
    "csrc/custom_all_reduce.cu",
    "csrc/custom_all_reduce.cuh",
    "csrc/mamba/mamba_ssm/selective_scan_fwd.cu",
    "csrc/quantization/activation_kernels.cu",
]

# Inline text replacements to apply to upstream source files.
# Format: {file_path: [{old_text: new_text}, ...]}
# Special case: empty old_text ("") means prepend new_text to file.
CSRC_TEXT_PATCHES = {
    str(_VLLM_REPO.source_dir / "csrc/topk.cu"): [
        {"#ifndef USE_ROCM": "#ifndef USE_MUSA"}
    ],
    str(_VLLM_REPO.source_dir / "csrc/moe/torch_bindings.cpp"): [
        {"#ifndef USE_ROCM": "#ifndef USE_MUSA"}
    ],
    str(_VLLM_REPO.source_dir / "csrc/torch_bindings.cpp"): [
        {"": '#include "torch_musa/csrc/aten/musa/MUSAContext.h"'},
        {"#ifndef USE_ROCM": "#ifndef USE_MUSA"},
        # paged_attention_v1/v2 are CUDA-only and unused on MUSA
        # (vllm uses FlashAttention via mate). Their .cu sources are dropped
        # from VLLM_CSRC_SOURCES above. Strip the `ops.impl(..., &paged_attention_v*)`
        # references so the linker doesn't need to resolve the unbuilt symbols.
        # The `ops.def(...)` schema declarations are left intact (string-only,
        # no symbol reference) so the operator name remains registered;
        # invoking it on MUSA would error at dispatch time, which is fine
        # because MUSA code paths never call paged_attention.
        {
            '  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);': "  // paged_attention_v1 impl stripped (kernel not built on MUSA)",
        },
        {
            '  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);': "  // paged_attention_v2 impl stripped (kernel not built on MUSA)",
        },
        {
            '  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", torch::kCUDA,\n           &fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert);': "  // MUSA: fused DeepSeek-V4 qnorm/rope/cache impl stripped;\n  // vllm_musa redirects this path to native/JIT MUSA implementations.",
        },
    ],
    str(_VLLM_REPO.source_dir / "csrc/libtorch_stable/torch_bindings.cpp"): [
        {
            '#include "ops.h"': '#if !defined(USE_MUSA)\n#include "ops.h"\n#endif'
        },
        {
            "STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {": "#if !defined(USE_MUSA)\nSTABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {"
        },
        {
            "STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {": "#endif\n#if !defined(USE_MUSA)\nSTABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {"
        },
        {"REGISTER_EXTENSION(_C_stable_libtorch)": "#endif\nREGISTER_EXTENSION(_C_stable_libtorch)"},
    ],
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh"): [
        {
            '#include "../../../../attention/attention_dtypes.h"': '#include "../../../../../csrc_musa/attention_musa/attention_dtypes.h"'
        },
        {
            '#include "../../../../../csrc/attention_musa/attention_dtypes.h"': '#include "../../../../../csrc_musa/attention_musa/attention_dtypes.h"'
        },
    ],
    str(_VLLM_REPO.source_dir / "csrc/attention/merge_attn_states.cu"): [
        {
            '#include "attention_dtypes.h"': '#include "attention_musa/attention_dtypes.h"'
        },
        {
            '#include "attention_utils.cuh"': '#include "attention_musa/attention_utils.cuh"'
        },
        {
            '#include "../quantization/w8a8/fp8/common.cuh"': '#include "quantization/w8a8/fp8/common.cuh"'
        },
    ],
    str(
        _VLLM_REPO.source_dir / "csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh"
    ): [
        {
            '#include "../../cuda_vec_utils.cuh"': '#include "../../../csrc_musa/cuda_vec_utils.cuh"'
        }
    ],
    str(
        _VLLM_REPO.source_dir / "csrc/libtorch_stable/quantization/vectorization.cuh"
    ): [
        {
            "#include <torch/headeronly/util/Float8_e4m3fnuz.h>": "#include <c10/util/Float8_e4m3fnuz.h>"
        },
        {
            "#include <torch/headeronly/util/Float8_e4m3fn.h>": "#include <c10/util/Float8_e4m3fn.h>"
        },
    ],
    str(_VLLM_REPO.source_dir / "csrc/cuda_vec_utils.cuh"): [
        {
            "#include <torch/headeronly/util/BFloat16.h>": "#include <c10/util/BFloat16.h>"
        },
        {"#include <torch/headeronly/util/Half.h>": "#include <c10/util/Half.h>"},
    ],
    str(_VLLM_REPO.source_dir / "csrc/cuda_compat.h"): [
        {
            "cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)": "musaFuncSetAttribute(FUNC, musaFuncAttributeMaxDynamicSharedMemorySize, VAL)"
        }
    ],
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/fp8/common.cuh"): [
        {'#include "../../utils.cuh"': '#include "quantization/utils.cuh"'},
    ],
    str(
        _VLLM_REPO.source_dir / "csrc/quantization/fused_kernels/quant_conversions.cuh"
    ): [
        {
            '#include "../w8a8/fp8/common.cuh"': '#include "quantization/w8a8/fp8/common.cuh"'
        },
    ],
    str(
        _VLLM_REPO.source_dir
        / "csrc/libtorch_stable/quantization/fused_kernels/quant_conversions.cuh"
    ): [
        {
            '#include "../../../quantization/w8a8/fp8/common.cuh"': '#include "quantization/w8a8/fp8/common.cuh"'
        },
    ],
    str(
        _VLLM_REPO.source_dir
        / "csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
    ): [
        {
            '#include "../w8a8/fp8/common.cuh"': '#include "quantization/w8a8/fp8/common.cuh"'
        },
    ],
    str(_VLLM_REPO.source_dir / "csrc/moe/moe_align_sum_kernels.cu"): [
        {'#include "../dispatch_utils.h"': '#include "dispatch_utils.h"'},
        {'#include "../cuda_compat.h"': '#include "cuda_compat.h"'},
    ],
    str(_VLLM_REPO.source_dir / "csrc/attention/attention_kernels.cuh"): [
        {
            '#include "attention_dtypes.h"': '#include "attention_musa/attention_dtypes.h"'
        },
        {
            '#include "attention_utils.cuh"': '#include "attention_musa/attention_utils.cuh"'
        },
        {
            '#include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"': '#include "../quantization_musa/w8a8/fp8/nvidia/quant_utils.cuh"'
        },
    ],
    str(_VLLM_REPO.source_dir / "csrc/type_convert.cuh"): [
        {"defined(USE_ROCM)": "defined(USE_MUSA)"}
    ],
    str(_VLLM_REPO.source_dir / "vllm/_custom_ops.py"): [
        {
            'if hasattr(torch.ops._C, "gptq_marlin_24_gemm"):': 'if not hasattr(torch.ops._C, "gptq_marlin_24_gemm"):'
        }
    ],
    str(_VLLM_REPO.source_dir / "csrc/activation_kernels.cu"): [
        {"CUDA_VERSION >= 12090 &&": "VLLM_CUDA_HAS_VERSION_CHECK &&"},
        {
            '#include "cuda_vec_utils.cuh"\n#include "dispatch_utils.h"': '#include "dispatch_utils.h"\n#include "cuda_vec_utils.cuh"\n\n'
            "#if defined(__CUDACC__) && defined(CUDA_VERSION)\n"
            "#define VLLM_CUDA_HAS_VERSION_CHECK (CUDA_VERSION >= 12090)\n"
            "#else\n"
            "#define VLLM_CUDA_HAS_VERSION_CHECK 0\n"
            "#endif"
        },
    ],
}

# =============================================================================
# Compiler and Linker Configuration
# =============================================================================

CXX_FLAGS = ["force_mcc"]
LINK_LIBRARIES = ["c10", "torch", "torch_python", "musart"]
EXTRA_LINK_ARGS = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{arch}-linux-gnu",
    "-lmublasLt",
]

# Detect MTGPU target architecture
DEFAULT_MTGPU_TARGET = "mp_31"
MTGPU_TARGET = os.environ.get("MTGPU_TARGET")

if MTGPU_TARGET:
    print(f"Using MTGPU_TARGET from environment: {MTGPU_TARGET}")
else:
    MTGPU_TARGET = DEFAULT_MTGPU_TARGET

if "MTGPU_TARGET" not in os.environ and torch.musa.is_available():
    try:
        device_props = torch.musa.get_device_properties(0)
        MTGPU_TARGET = f"mp_{device_props.major}{device_props.minor}"
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
elif "MTGPU_TARGET" not in os.environ:
    print(f"Warning: torch.musa not available. Using default target: {MTGPU_TARGET}")

SUPPORTED_MTGPU_TARGETS = ["mp_22", "mp_31"]
if MTGPU_TARGET not in SUPPORTED_MTGPU_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture '{MTGPU_TARGET}'. "
        f"Expected one of: {SUPPORTED_MTGPU_TARGETS}"
    )
    sys.exit(1)

MCC_FLAGS = [
    "-DNDEBUG",
    "-O3",
    "-fPIC",
    "-std=c++17",
    "-x",
    "musa",
    "-mtgpu",
    "-Od3",
    "-ffast-math",
    "-fmusa-flush-denormals-to-zero",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-DUSE_MUSA",
]

mcc_version = get_mcc_version()
if mcc_version:
    try:
        # mcc_version can be compared normally for types 5.1.0, 5.1.0-rc1, v5.1.0, etc
        if version.parse(mcc_version) > version.parse("5.0.0"):
            # (2026-05-28): vllm-musa used to disable SLP vectorization
            # on mcc 5.1.0+ via `-mllvm -vectorize-slp=false`, with the comment
            # "After mtcc implements vectorization length restrictions, this
            # option can be removed." A/B on M2.5+Eagle3 BS=1 cookbook shows
            # this disable was costing perf; mate's JIT compile does NOT set it.
            # Re-enabling SLP vectorization on mcc 5.1.0+ to test toolchain win.
            # If something regresses, set VLLM_MUSA_DISABLE_SLP=1 to opt back.
            if os.environ.get("VLLM_MUSA_DISABLE_SLP", "0") == "1":
                MCC_FLAGS += ["-mllvm", "-vectorize-slp=false"]
            # / PR #50 review: mate's mcc-specific load-clustering hints
            # (mate/mate/jit/gemm_ops.py CUDA_FLAGS). Gated to mcc > 5.0.0 -- they
            # are validated on 5.1.0; an older/unsupported mcc/LLVM may not
            # recognize these -mllvm options and would otherwise hard-fail the
            # build even when the version gate would allow it. Opt out with
            # VLLM_MUSA_DISABLE_LOAD_CLUSTER=1.
            if os.environ.get("VLLM_MUSA_DISABLE_LOAD_CLUSTER", "0") != "1":
                MCC_FLAGS += [
                    "-mllvm",
                    "-mtgpu-load-cluster-mutation=1",
                    "-mllvm",
                    "--num-dwords-of-load-in-mutation=64",
                ]
    except Exception as e:
        print(
            f"Warning: failed to get MUSA version, which may cause installation failure: {e}"
        )

if MTGPU_TARGET == "mp_31":
    MCC_FLAGS.append("-DENABLE_FP8")

COMPILE_ARGS = {
    "mcc": MCC_FLAGS,
    "cxx": CXX_FLAGS,
}

# =============================================================================
# Extension Modules
# =============================================================================

EXT_MODULES = [
    CUDAExtension(
        name="vllm._C",
        sources=VLLM_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm._C_stable_libtorch",
        sources=VLLM_STABLE_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm_musa._C",
        sources=VLLM_MUSA_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm._moe_C",
        sources=VLLM_MOE_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
]


class _CustomBuildExt(BuildExtension):
    """Custom build extension that clones third-party repositories before building."""

    @staticmethod
    def _clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
        """Clone a git repository and checkout a specific tag/commit."""
        repo_path.parent.mkdir(exist_ok=True)
        if not repo_path.exists():
            clone_cmd = ["git", "clone"]
            if git_shallow:
                clone_cmd += ["--depth", "1"]
            clone_cmd += [repo_url, str(repo_path)]
            subprocess.check_call(clone_cmd)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
        else:
            subprocess.check_call(["git", "fetch", "--all"], cwd=repo_path)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
        # the vllm/ir/tolerances.py float4 edit is now a build-applied
        # cat-1 patch (series/0055-MUSA-vllm.ir.tolerances.patch) classified by the
        # manifest, not an ad-hoc sed here. _apply_musa_patch_series applies it to
        # the cloned vLLM after checkout.

    @staticmethod
    def _install_vllm(repo_path):
        """Install the cloned + patched vLLM, using the existing torch.

        When vllm-musa itself is installed editable (``pip install -e .``),
        vLLM is installed EDITABLE from ``third_party/vllm`` so a developer can
        edit the patched vLLM source in place -- edits take effect at runtime;
        commit them in ``third_party/vllm`` and run
        ``python tools/musa_sync.py regen`` to capture them as ``series/``
        patches. A regular (non-editable) install bakes vLLM in.

        Sole-vLLM requirement: the editable install only takes effect if no
        other vLLM is importable. The PEP 660 editable finder is appended to
        ``sys.meta_path``, so a vLLM already on ``sys.path`` (e.g. a system
        install seen through a ``--system-site-packages`` venv) shadows it.
        Use a clean venv without a pre-existing vLLM for editable development;
        ``_warn_if_vllm_shadowed`` flags it after install if it happens.
        """
        source_dir = Path(repo_path)

        env = os.environ.copy()
        env["VLLM_TARGET_DEVICE"] = "empty"

        # Dev (editable) installs of vllm-musa install vLLM editable too, so
        # third_party/vllm edits are live; regular installs bake it in.
        vllm_install_cmd = [sys.executable, "-m", "pip", "install"]
        if _is_editable_install:
            vllm_install_cmd.append("-e")
        vllm_install_cmd += [str(source_dir), "--no-build-isolation", "-v"]

        steps = [
            {
                "name": "Install vllm use existing torch",
                "cmd": f"cd {source_dir} && python use_existing_torch.py",
                "shell": True,
                "env": None,
            },
            {
                "name": "Install vllm build requirements",
                "cmd": [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(source_dir / "requirements" / "build" / "cuda.txt"),
                ],
                "shell": False,
                "env": None,
            },
            {
                "name": (
                    "Install vllm EDITABLE (dev)"
                    if _is_editable_install
                    else "Install vllm without target device"
                ),
                "cmd": vllm_install_cmd,
                "shell": False,
                "env": env,
            },
        ]

        for step in steps:
            print(f"{step['name']}")

            if step["shell"]:
                print(f"Command: {step['cmd']}")
                subprocess.check_call(step["cmd"], shell=True, env=step["env"])
            else:
                print(f"Command: {' '.join(step['cmd'])}")
                subprocess.check_call(step["cmd"], env=step["env"])

    @staticmethod
    def _apply_file_overrides(repo_path):
        """Copy local patched files to replace upstream versions."""
        for file_path in CSRC_FILE_OVERRIDES:
            src_path = Path(root) / file_path
            dst_path = Path(repo_path) / file_path

            print(f"Applying file override: {file_path}")
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            except (OSError, IOError) as e:
                print(f"Error applying file override {file_path}: {e}")

    @staticmethod
    def _apply_text_patches():
        """Apply inline text replacements to upstream source files."""
        for file_path, replacement_rules in CSRC_TEXT_PATCHES.items():
            if not Path(file_path).exists():
                print(f"Skipping missing text patch target: {file_path}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            for rule in replacement_rules:
                for old_str, new_str in rule.items():
                    if old_str == "":
                        # Empty old_str means prepend content (if not already present)
                        if new_str not in content:
                            content = new_str + "\n" + content
                    elif old_str == "CUDA_VERSION >= 12090":
                        # Special case: only replace in lines without the macro definition
                        lines = content.split("\n")
                        new_lines = []
                        for line in lines:
                            if "VLLM_CUDA_HAS_VERSION_CHECK" in line:
                                new_lines.append(line)
                            else:
                                new_lines.append(line.replace(old_str, new_str))
                        content = "\n".join(new_lines)
                    else:
                        content = content.replace(old_str, new_str)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Applied text patches: {file_path}")
            else:
                print(f"Skipping (already patched): {file_path}")

    @staticmethod
    def _apply_musa_patch_series(repo_path):
        """apply the vLLM-MUSA unified-diff series to the cloned vLLM
        at build time, so the installed vLLM is
        pre-patched. This is the only vLLM source-patch mechanism; there is no
        runtime source-patching and no fallback. No-op until
        ``vllm_musa/patches/series`` is populated (the ``Makefile.sync``
        bootstrap).

        ``build_apply.py`` is stdlib-only and loaded by file path so this does
        not import the ``vllm_musa`` package before it is installed. ``strict``
        makes a drifted patch fail the build loudly (the pinned vLLM moved —
        regenerate the series via ``make -f Makefile.sync``).
        """
        if os.environ.get("VLLM_MUSA_NO_BUILD_PATCH", "0") == "1":
            return
        repo = Path(repo_path)
        series = Path(root) / "vllm_musa" / "patches" / "series"
        if not repo.is_dir() or not series.is_dir():
            return
        import importlib.util as _ilu

        ba_path = Path(root) / "vllm_musa" / "patches" / "build_apply.py"
        spec = _ilu.spec_from_file_location("_musa_build_apply", ba_path)
        ba = _ilu.module_from_spec(spec)
        spec.loader.exec_module(ba)
        for name, status in ba.apply_patch_series(repo, series, strict=True):
            print(f"MUSA build patch: {status:16} {name}")

    def run(self):
        if os.environ.get("SKIP_THIRD_PARTY", "0") == "1":
            print("Skipping third-party repositories cloning (SKIP_THIRD_PARTY=1)")
        else:
            print("Cloning third-party repositories...")
            self._clone_and_checkout(
                _VLLM_REPO.source_dir,
                _VLLM_REPO.git_repository,
                _VLLM_REPO.git_tag,
                _VLLM_REPO.git_shallow,
            )
            self._clone_and_checkout(
                _FLASHINFER_REPO.source_dir,
                _FLASHINFER_REPO.git_repository,
                _FLASHINFER_REPO.git_tag,
                _FLASHINFER_REPO.git_shallow,
            )
            print("Third-party repositories ready.")

        # pre-patch the cloned vLLM (Python patch series) BEFORE
        # installing it, so the installed vLLM is patched at build time rather
        # than rewritten at runtime. No-op until the series dir is populated.
        self._apply_musa_patch_series(_VLLM_REPO.source_dir)

        self._install_vllm(_VLLM_REPO.source_dir)

        # Re-ensure numpy<2 after vllm installation (vllm may pull in numpy>=2)
        _ensure_numpy_compatible()

        # csrc divergence is now part of the build-time series — cat-2
        # text-edits + cat-3 whole-file diffs, applied above by
        # _apply_musa_patch_series alongside the cat-1 python patches. The legacy
        # CSRC_FILE_OVERRIDES / CSRC_TEXT_PATCHES str-replace mechanism is retired
        # (the series is the single, reviewable, drift-loud path; a moved csrc
        # anchor now fails `git apply` loudly instead of silently no-opping).

        super().run()


setup(
    ext_modules=EXT_MODULES,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    include_package_data=False,
    # Force these dependencies even with --no-build-isolation
    # (pyproject.toml dependencies aren't processed with --no-build-isolation)
    install_requires=[
        "torchada>=0.1.52",
        "mthreads-ml-py>=2.2.11",
        "numpy<2",
        "openai>=2.24.0",
    ],
)

if _is_editable_install:
    develop_dynamic_library("vllm")
    _warn_if_vllm_shadowed(_VLLM_REPO.source_dir)
