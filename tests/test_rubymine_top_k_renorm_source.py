# SPDX-License-Identifier: Apache-2.0
"""Source contracts for default guarded top-k dispatch."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_top_k_dispatch_has_no_environment_gate() -> None:
    source = (ROOT / "vllm_musa/_custom_ops.py").read_text(encoding="utf-8")
    environ = (ROOT / "vllm_musa/utils/environ.py").read_text(encoding="utf-8")

    assert "RUBYMINE_TOP_K_RENORM" not in source
    assert "RUBYMINE_TOP_K_RENORM" not in environ
    assert "probs.shape[1] in (151936, 248320)" in source


def test_top_k_kernel_sanitizes_malformed_probabilities() -> None:
    source = (ROOT / "csrc/musa/top_k_renorm.mu").read_text(encoding="utf-8")

    assert "kOneBits = 0x3f800000u" in source
    assert "sanitize_probability" in source
    assert "atomicExch(&invalid_probability, 1)" in source
    assert "selected_total > 0.0f" in source
