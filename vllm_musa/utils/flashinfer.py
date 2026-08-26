# SPDX-License-Identifier: Apache-2.0
"""MUSA-facing accessors for the MATE-backed FlashInfer wrapper."""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import torchada  # noqa: F401
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_WRAPPER_DISTRIBUTION = "flashinfer-python"
_WRAPPER_VERSION = "0.2.6+musa"


def flashinfer_wrapper_version() -> str | None:
    """Return the installed MATE-backed FlashInfer wrapper version."""
    try:
        return version(_WRAPPER_DISTRIBUTION)
    except PackageNotFoundError:
        return None


def _load_symbol(module_name: str, symbol_name: str) -> Callable[..., Any] | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    if spec is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    symbol = getattr(module, symbol_name, None)
    return symbol if callable(symbol) else None


def has_musa_flashinfer_wrapper() -> bool:
    """Return whether the expected MATE-backed FlashInfer package is installed."""
    installed = flashinfer_wrapper_version()
    if installed != _WRAPPER_VERSION:
        return False
    return importlib.util.find_spec("flashinfer") is not None


def _has_wrapper_symbol(module_name: str, symbol_name: str) -> bool:
    return has_musa_flashinfer_wrapper() and _load_symbol(
        module_name, symbol_name
    ) is not None


def has_musa_flashinfer_bmm_fp8() -> bool:
    """Return whether the one upstream production GEMM caller is available."""
    return _has_wrapper_symbol("flashinfer", "bmm_fp8")


def has_musa_flashinfer_gemm() -> bool:
    """Return whether all five public MATE FlashInfer GEMM APIs exist."""
    return has_musa_flashinfer_wrapper() and all(
        _load_symbol("flashinfer.gemm", name) is not None
        for name in (
            "bmm_bf16",
            "bmm_fp8",
            "gemm_fp8_nt_groupwise",
            "group_deepgemm_fp8_nt_groupwise",
            "batch_deepgemm_fp8_nt_groupwise",
        )
    )


def has_musa_flashinfer_sparse_decode() -> bool:
    """Return whether the upstream Sparse MLA decode caller is available."""
    return _has_wrapper_symbol(
        "flashinfer.decode", "trtllm_batch_decode_with_kv_cache_mla"
    )


def has_musa_flashinfer_sparse_mla() -> bool:
    """Return whether all three public MATE Sparse MLA APIs exist."""
    return has_musa_flashinfer_wrapper() and all(
        _load_symbol(module_name, symbol_name) is not None
        for module_name, symbol_name in (
            ("flashinfer.rope", "mla_rope_quantize_fp8"),
            ("flashinfer.decode", "get_batch_decode_metadata_mla"),
            ("flashinfer.decode", "trtllm_batch_decode_with_kv_cache_mla"),
        )
    )


def _require_symbol(module_name: str, symbol_name: str) -> Callable[..., Any]:
    if not has_musa_flashinfer_wrapper():
        raise RuntimeError(
            "MATE-backed FlashInfer wrapper is unavailable; expected "
            f"{_WRAPPER_DISTRIBUTION}=={_WRAPPER_VERSION}"
        )
    symbol = _load_symbol(module_name, symbol_name)
    if symbol is None:
        raise RuntimeError(
            f"MATE-backed FlashInfer symbol {module_name}.{symbol_name} is unavailable"
        )
    return symbol


def bmm_bf16(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Call MATE's FlashInfer-compatible BF16/FP16 BMM."""
    return _require_symbol("flashinfer.gemm", "bmm_bf16")(*args, **kwargs)


def bmm_fp8(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Call MATE's FlashInfer-compatible FP8 BMM."""
    return _require_symbol("flashinfer.gemm", "bmm_fp8")(*args, **kwargs)


def gemm_fp8_nt_groupwise(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Call MATE's FlashInfer-compatible groupwise FP8 GEMM."""
    return _require_symbol("flashinfer.gemm", "gemm_fp8_nt_groupwise")(
        *args, **kwargs
    )


def group_deepgemm_fp8_nt_groupwise(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Call MATE's FlashInfer-compatible contiguous grouped FP8 GEMM."""
    return _require_symbol(
        "flashinfer.gemm", "group_deepgemm_fp8_nt_groupwise"
    )(*args, **kwargs)


def batch_deepgemm_fp8_nt_groupwise(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Call MATE's FlashInfer-compatible masked grouped FP8 GEMM."""
    return _require_symbol(
        "flashinfer.gemm", "batch_deepgemm_fp8_nt_groupwise"
    )(*args, **kwargs)


def mla_rope_quantize_fp8(*args: Any, **kwargs: Any) -> Any:
    """Call MATE's fused MLA RoPE and FP8 quantization operation."""
    return _require_symbol("flashinfer.rope", "mla_rope_quantize_fp8")(
        *args, **kwargs
    )


def get_batch_decode_metadata_mla(*args: Any, **kwargs: Any) -> Any:
    """Create reusable MATE Sparse MLA decode metadata."""
    return _require_symbol("flashinfer.decode", "get_batch_decode_metadata_mla")(
        *args, **kwargs
    )


def trtllm_batch_decode_with_kv_cache_mla(*args: Any, **kwargs: Any) -> Any:
    """Call MATE's FlashInfer-compatible Sparse MLA decode operation."""
    return _require_symbol(
        "flashinfer.decode", "trtllm_batch_decode_with_kv_cache_mla"
    )(*args, **kwargs)


__all__ = [
    "batch_deepgemm_fp8_nt_groupwise",
    "bmm_bf16",
    "bmm_fp8",
    "flashinfer_wrapper_version",
    "gemm_fp8_nt_groupwise",
    "get_batch_decode_metadata_mla",
    "group_deepgemm_fp8_nt_groupwise",
    "has_musa_flashinfer_bmm_fp8",
    "has_musa_flashinfer_gemm",
    "has_musa_flashinfer_sparse_decode",
    "has_musa_flashinfer_sparse_mla",
    "has_musa_flashinfer_wrapper",
    "mla_rope_quantize_fp8",
    "trtllm_batch_decode_with_kv_cache_mla",
]
