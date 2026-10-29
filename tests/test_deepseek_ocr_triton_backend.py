"""Source contract for the DeepSeek-OCR MUSA vision backend default."""

from pathlib import Path


PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa"
    / "patches"
    / "series"
    / "0194-MUSA-route-DeepSeek-OCR-ViT-attention-to-Triton.patch"
)


def test_deepseek_ocr_patch_sets_model_scoped_triton_default() -> None:
    source = PATCH.read_text(encoding="utf-8")

    assert "vllm/model_executor/models/deepseek_ocr.py" in source
    assert "_is_musa_runtime()" in source
    assert "multimodal_config.mm_encoder_attn_backend is None" in source
    assert "AttentionBackendEnum.TRITON_ATTN" in source
    assert "vllm_musa/platform.py" not in source
