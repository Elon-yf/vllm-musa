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
        ("deep-gemm", "deep_gemm"),
        ("flashinfer-python", "flashinfer"),
    ):
        assert f'("{distribution}", "{module}"' in dockerfile
        assert f'"{distribution}"' in dockerfile.split("exact_version_dists =", 1)[1]


def test_musa_flashinfer_adapter_exposes_required_symbols() -> None:
    source = _read("vllm_musa/utils/flashinfer.py")
    for symbol in (
        "bmm_bf16",
        "bmm_fp8",
        "gemm_fp8_nt_groupwise",
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


def test_flashinfer_upstream_patch_is_a_musa_availability_seam() -> None:
    patch = _read(
        "vllm_musa/patches/series/0131-MUSA-vllm.utils.flashinfer-mate-wrapper.patch"
    )
    assert "vllm/utils/flashinfer.py" in patch
    assert "current_platform.is_musa()" in patch
    assert "MATE-backed Python wrapper" in patch
