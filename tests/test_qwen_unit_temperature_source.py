# SPDX-License-Identifier: Apache-2.0
"""Source contract for the Qwen unit-temperature divide skip."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0091-perf-skip-qwen-unit-temperature-divide.patch"
)
DECODE_PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0090-perf-reuse-uniform-decode-logits-indices.patch"
)
SAMPLER = ROOT / "vllm_musa" / "v1" / "sample" / "topk_topp_sampler.py"


def test_qwen_unit_temperature_metadata_gate_is_fail_closed() -> None:
    source = PATCH.read_text()
    cumulative_source = DECODE_PATCH.read_text() + source

    assert "uniform_temperature: float | None = None" in source
    assert "self.vocab_size in (151936, 248320)" in source
    assert "self.all_random" in source
    assert "num_reqs > 0" in source
    assert "temperature_cpu == np.float32(1.0)" in source
    assert "VLLM_MUSA_QWEN_SKIP_UNIT_TEMPERATURE" in source
    assert '("0", "false", "no", "off")' in source
    assert "uniform_temperature=uniform_temperature" in source
    assert "current_platform.is_musa()" in source
    assert "not self.is_pooling_model" in source
    assert "self.speculative_config is None" in source
    for architecture in (
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        assert architecture in cumulative_source


def test_qwen_unit_temperature_sampler_keeps_original_fallback() -> None:
    source = SAMPLER.read_text()

    assert "def _can_skip_legacy_qwen_unit_temperature" in source
    assert 'getattr(sampler, "_musa_qwen_skip_unit_temperature", False)' in source
    assert 'getattr(sampling_metadata, "all_random", False)' in source
    assert "_is_qwen_sampler_vocab(logits)" in source
    assert 'sampling_metadata, "uniform_temperature", None' in source
    assert "if not _can_skip_legacy_qwen_unit_temperature" in source
    assert "logits = self.apply_temperature(" in source
