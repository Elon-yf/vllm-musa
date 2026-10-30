"""Source contract for model-local DeepSeek-OCR backend priority."""

from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa"
    / "patches"
    / "series"
    / "0194-MUSA-DeepSeek-OCR-model-backend-priority.patch"
)


def test_route_is_model_config_scoped() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "vllm/model_executor/models/config.py" in source
    assert "class DeepseekOCRForCausalLMConfig" in source
    assert '"DeepseekOCRForCausalLM": DeepseekOCRForCausalLMConfig' in source
    assert "vllm_musa/platform.py" not in source


def test_model_priority_preserves_explicit_override_and_fallback() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "mm_encoder_attn_backend is not None" in source
    assert "candidates = [AttentionBackendEnum.TRITON_ATTN]" in source
    assert 'vision_config.width' in source
    assert 'clip-l-14-224' in source
    assert "supports_head_size" in source
    assert "supports_dtype(torch.float32)" in source
    assert "supports_compute_capability" in source
    assert "multimodal_config.mm_encoder_attn_backend = fallback" in source
