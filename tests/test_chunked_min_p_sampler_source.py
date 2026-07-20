# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the default-on chunked min-p dispatch."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_min_p_dispatch_has_no_environment_gate() -> None:
    source = _read("vllm_musa/_custom_ops.py")
    environ = _read("vllm_musa/utils/environ.py")

    assert "envs." not in source
    assert "CHUNKED_MIN_P_SAMPLER" not in environ


def test_min_p_dispatch_keeps_the_validated_contract_guard() -> None:
    source = _read("vllm_musa/_custom_ops.py")

    assert "def _can_use_chunked_min_p_sampler(" in source
    assert "deterministic" in source
    assert "and probs.dtype == torch.float32" in source
    assert "and probs.shape[1] in _QWEN_MIN_P_VOCABS" in source
    assert "and 0 < probs.shape[0] <= _QWEN_MIN_P_MAX_BATCH" in source
    assert "and probs.is_contiguous()" in source
    assert "and _is_validated_musa_device(probs.device)" in source
    assert "device_id=device_id" in source
    assert "and _is_supported_musa_generator(generator, probs.device)" in source
    assert "musa_chunked_min_p_sampling_from_probs.default" in source
    assert "min_p_sampling_from_probs.default" in source


def test_min_p_dispatch_preserves_unsupported_input_fallbacks() -> None:
    source = _read("vllm_musa/_custom_ops.py")

    assert "indices.dtype == torch.int32" in source
    assert "indices.is_contiguous()" in source
    assert "maybe_min_p_arr.dtype == torch.float32" in source
    assert "maybe_min_p_arr.is_contiguous()" in source
    assert "input_probs = probs" in source
    assert "input_min_p_arr = maybe_min_p_arr" in source
