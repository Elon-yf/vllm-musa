from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PATCH = (
    ROOT
    / "vllm_musa/patches/series/0131-MUSA-route-upstream-FlashInfer-callers-to-MATE.patch"
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_mate_flashinfer_private_requirements_are_coherent() -> None:
    requirements = _read("requirements/musa_private.txt")
    expected = {
        "mate==0.2.6",
        "flashinfer-python==0.2.6+musa",
        "tilelang_musa==0.1.12+musa.2",
        "apache-tvm-ffi==0.1.11.post1+musa.1",
    }
    assert expected.issubset(set(requirements.splitlines()))


def test_docker_import_check_covers_mate_flashinfer_runtime() -> None:
    dockerfile = _read("docker/musa.Dockerfile")
    for distribution, module in (
        ("mate", "mate"),
        ("flashinfer-python", "flashinfer"),
        ("tilelang_musa", "tilelang"),
        ("apache-tvm-ffi", "tvm_ffi"),
    ):
        assert f'("{distribution}", "{module}"' in dockerfile
        assert f'requirement_prefix("{distribution}")' in dockerfile
        assert f'"{distribution}"' in dockerfile.split("exact_version_dists =", 1)[1]


def test_only_upstream_flashinfer_callers_are_patched() -> None:
    patch = UPSTREAM_PATCH.read_text(encoding="utf-8")
    assert "has_flashinfer_bmm_fp8" in patch
    assert "trtllm_batch_decode_with_kv_cache_mla" in patch
    for unsupported in (
        "bmm_bf16",
        "gemm_fp8_nt_groupwise",
        "group_deepgemm_fp8_nt_groupwise",
        "batch_deepgemm_fp8_nt_groupwise",
        "mla_rope_quantize_fp8",
        "get_batch_decode_metadata_mla",
    ):
        assert unsupported not in patch
    assert "_MUSA_FLASHINFER_DISTRIBUTION = \"flashinfer-python\"" in patch
    assert "_MUSA_FLASHINFER_VERSION = \"0.2.6+musa\"" in patch
    assert "except OSError:" in patch
    assert "ModuleNotFoundError, OSError" not in patch
    assert (
        'if current_platform.device_type != "musa":\n'
        "+        return has_flashinfer()"
    ) in patch


def test_mate_route_does_not_extend_upstream_env_surface() -> None:
    patch = UPSTREAM_PATCH.read_text(encoding="utf-8")
    assert "vllm/envs.py" not in patch
    assert "VLLM_MUSA_" not in patch


def test_upstream_flashinfer_fp8_provider_is_reused_on_musa() -> None:
    patch = UPSTREAM_PATCH.read_text(encoding="utf-8")
    selector = _read(
        "vllm_musa/patches/series/0052-MUSA-vllm.model_executor.kernels.linear.patch"
    )
    assert 'op_name="bmm_fp8"' in patch
    assert "A_scale.reshape(())" in patch
    assert "compute_capability != 31" in patch
    assert "torch.ops.vllm.bmm_fp8" in patch
    assert "flashinfer_scaled_fp8_mm_out" in patch
    assert "_call_flashinfer_bmm_fp8(" in patch
    assert 'assert out.device.type in ("cuda", "musa")' in patch
    assert "FlashInferFP8ScaledMMLinearKernel" in selector
    assert "MUSAFlashInferFP8ScaledMMLinearKernel" not in selector
    assert not (
        ROOT / "vllm_musa/model_executor/kernels/linear/scaled_mm/flashinfer.py"
    ).exists()


def test_upstream_sparse_mla_backend_is_reused_on_musa() -> None:
    patch = UPSTREAM_PATCH.read_text(encoding="utf-8")
    platform = _read("vllm_musa/platform.py")
    assert "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py" in patch
    assert "has_flashinfer_sparse_mla" in patch
    assert "BLOCK_N=index_block_n" in patch
    assert 'index_block_n = 64 if current_platform.device_type == "musa"' in patch
    assert 'lse_base_on_e: bool = current_platform.device_type == "musa"' in patch
    assert 'kv_cache_dtype in ("fp8", "fp8_e4m3")' in patch
    assert "qk_nope_head_dim 128 or 192" not in patch
    assert 'current_platform.device_type != "musa"' in patch
    assert "AttentionBackendEnum.FLASHINFER_MLA_SPARSE" in platform
    assert "MUSAFlashInferMLASparseBackend" not in platform
    assert not (
        ROOT / "vllm_musa/v1/attention/backends/mla/flashinfer_sparse.py"
    ).exists()


def test_complete_cuda_flashinfer_gate_stays_disabled_on_musa() -> None:
    patch = UPSTREAM_PATCH.read_text(encoding="utf-8")
    assert 'if current_platform.device_type == "musa":\n+        return False' in patch
    assert "has_flashinfer_cubin" in patch
    assert "shutil.which(\"nvcc\")" not in patch


def test_native_flashinfer_source_pin_remains_separate() -> None:
    pins = _read("third_party/PINS")
    setup = _read("setup.py")
    assert "FLASHINFER_COMMIT=" in pins
    assert 'name="flashinfer"' in setup
    assert "_FLASHINFER_REPO" in setup
    assert "_FLASHINFER_REPO.source_dir / \"include\"" in setup
    # The MATE wrapper wheel is Python-only; native sampler/norm/renorm and
    # JIT code still consume the separately pinned FlashInfer checkout.
    assert "_find_vendored_flashinfer_root" in _read(
        "vllm_musa/jit_kernel/utils.py"
    )
