from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_mate_flashinfer_private_requirements_are_coherent() -> None:
    requirements = _read("requirements/musa_private.txt")
    expected = {
        "mate==0.2.6",
        "mate-mubin==0.2.6",
        "flash_attn_3==0.2.6+musa",
        "flash_mla==0.2.6+musa",
        "deep-gemm==0.2.6+musa",
        "flashinfer-python==0.2.6+musa",
        "sageattention==0.2.6+musa",
        "tilelang_musa==0.1.12+musa.2",
        "apache-tvm-ffi==0.1.11.post1+musa.1",
    }
    assert expected.issubset(set(requirements.splitlines()))


def test_docker_import_check_covers_mate_flashinfer_cohort() -> None:
    dockerfile = _read("docker/musa.Dockerfile")
    for distribution, module in (
        ("mate", "mate"),
        ("mate-mubin", "mate_mubin"),
        ("flash_attn_3", "flash_attn_3"),
        ("flash_mla", "flash_mla"),
        ("deep-gemm", "deep_gemm"),
        ("flashinfer-python", "flashinfer"),
        ("sageattention", "sageattention"),
        ("tilelang_musa", "tilelang"),
        ("apache-tvm-ffi", "tvm_ffi"),
    ):
        assert f'("{distribution}", "{module}"' in dockerfile
        assert f'requirement_prefix("{distribution}")' in dockerfile
        assert f'"{distribution}"' in dockerfile.split("exact_version_dists =", 1)[1]


def test_musa_flashinfer_adapter_exposes_required_symbols() -> None:
    source = _read("vllm_musa/utils/flashinfer.py")
    for symbol in (
        "bmm_bf16",
        "bmm_fp8",
        "gemm_fp8_nt_groupwise",
        "group_deepgemm_fp8_nt_groupwise",
        "batch_deepgemm_fp8_nt_groupwise",
        "mla_rope_quantize_fp8",
        "get_batch_decode_metadata_mla",
        "trtllm_batch_decode_with_kv_cache_mla",
    ):
        assert f'def {symbol}' in source
    assert "flashinfer-python" in source
    assert "nvcc" not in source


def test_flashinfer_sparse_backend_is_registered_on_musa() -> None:
    platform = _read("vllm_musa/platform.py")
    backend = _read("vllm_musa/v1/attention/backends/mla/flashinfer_sparse.py")
    assert "AttentionBackendEnum.FLASHINFER_MLA_SPARSE" in platform
    assert "MUSAFlashInferMLASparseBackend" in platform
    assert "@register_backend(AttentionBackendEnum.FLASHINFER_MLA_SPARSE)" in backend
    assert "trtllm_batch_decode_with_kv_cache_mla" in backend
    assert (
        'supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fp8", "fp8_e4m3"]'
        in backend
    )
    assert "return [64]" in backend
    assert "requires head_size=576" in backend
    assert "requires index_topk divisible by 64" in backend
    assert "has_musa_flashinfer_sparse_decode" in backend
    assert "capability.minor == 1" in backend
    assert "device_capability.minor != 1" in backend


def test_flashinfer_bmm_uses_a_musa_provider_not_a_global_availability_patch() -> None:
    provider = _read(
        "vllm_musa/model_executor/kernels/linear/scaled_mm/flashinfer.py"
    )
    patch = _read(
        "vllm_musa/patches/series/0052-MUSA-vllm.model_executor.kernels.linear.patch"
    )
    assert "MUSAFlashInferFP8ScaledMMLinearKernel" in provider
    assert "has_musa_flashinfer_bmm_fp8" in provider
    assert "musa_flashinfer_bmm_fp8" in provider
    assert "compute_capability != 31" in provider
    assert "MUSAFlashInferFP8ScaledMMLinearKernel" in patch
    assert not (
        ROOT
        / "vllm_musa/patches/series/0131-MUSA-vllm.utils.flashinfer-mate-wrapper.patch"
    ).exists()


def test_mate_wrapper_keeps_native_flashinfer_headers_separate() -> None:
    source = _read("vllm_musa/jit_kernel/utils.py")
    assert "_find_vendored_flashinfer_root" in source
    assert '"third_party/flashinfer"' in source
    assert 'flashinfer_root / "data"' in source
