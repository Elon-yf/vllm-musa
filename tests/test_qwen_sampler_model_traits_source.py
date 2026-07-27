"""Source contract for attribute-gated Qwen sampler fast paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0090-perf-musa-unify-Qwen-runtime-fast-paths.patch"
)


def test_sampler_traits_are_derived_from_model_architecture() -> None:
    source = PATCH.read_text()

    assert "VLLM_MUSA_QWEN" not in source
    assert (
        "+        self.sampler._musa_qwen_family = is_musa_qwen_text_generation"
        in source
    )
    assert "+            qwen_sampling_architectures = {" in source
    assert "+            self.sampler._musa_qwen_family = any(" in source
    for architecture in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        assert architecture in source


def test_both_runner_generations_propagate_qwen_traits() -> None:
    source = PATCH.read_text()

    assert (
        "diff --git a/vllm/v1/worker/gpu_model_runner.py "
        "b/vllm/v1/worker/gpu_model_runner.py"
    ) in source
    assert (
        "diff --git a/vllm/v1/worker/gpu/model_runner.py "
        "b/vllm/v1/worker/gpu/model_runner.py"
    ) in source
