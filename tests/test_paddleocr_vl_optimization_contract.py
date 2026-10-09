"""PaddleOCR-VL rotary eligibility within the shared optimization contract."""

from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    ModelFamily,
    OptimizationFeature,
    resolve_optimization_contract,
)


def _config() -> SimpleNamespace:
    hf_config = SimpleNamespace(
        model_type="paddleocr_vl",
        vision_config=SimpleNamespace(
            hidden_size=1152,
            num_hidden_layers=27,
            num_attention_heads=16,
            patch_size=14,
            image_size=384,
        ),
        text_config=SimpleNamespace(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=2,
            rope_parameters={"mrope_section": [16, 24, 24]},
        ),
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype="bfloat16")
    )


def _prefers(config: SimpleNamespace) -> bool:
    return resolve_optimization_contract(config).prefers(
        OptimizationFeature.PADDLEOCR_VL_ROTARY
    )


def test_exact_paddle_config_gets_only_paddle_rotary() -> None:
    contract = resolve_optimization_contract(_config())
    assert contract.model.family is ModelFamily.PADDLEOCR_VL
    assert contract.profile == "paddleocr_vl"
    assert contract.preferred_features == {OptimizationFeature.PADDLEOCR_VL_ROTARY}
    assert contract.supported_features == contract.preferred_features


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("vision", "hidden_size", 1024),
        ("vision", "num_hidden_layers", 26),
        ("vision", "num_attention_heads", 12),
        ("vision", "patch_size", 16),
        ("vision", "image_size", 448),
        ("text", "hidden_size", 2048),
        ("text", "num_attention_heads", 8),
        ("text", "num_key_value_heads", 4),
        ("text", "rope_parameters", {"mrope_section": [16, 24, 23]}),
    ],
)
def test_one_geometry_mismatch_disables_rotary(
    owner: str, field: str, value: object
) -> None:
    config = _config()
    hf_config = config.model_config.hf_config
    target = hf_config.vision_config if owner == "vision" else hf_config.text_config
    setattr(target, field, value)
    assert not _prefers(config)
    assert resolve_optimization_contract(config).model.family is ModelFamily.PADDLEOCR_VL


def test_depth_alias_retains_old_or_semantics() -> None:
    config = _config()
    vision = config.model_config.hf_config.vision_config
    vision.num_hidden_layers = 26
    vision.depth = 27
    assert _prefers(config)


def test_outer_rope_fallback_and_hf_text_config_precedence() -> None:
    config = _config()
    hf = config.model_config.hf_config
    hf.text_config.rope_parameters = None
    hf.rope_parameters = {"mrope_section": [16, 24, 24]}
    config.model_config.hf_text_config = SimpleNamespace(hidden_size=999)
    assert _prefers(config)


def test_other_model_same_shapes_does_not_get_paddle_feature() -> None:
    config = _config()
    config.model_config.hf_config.model_type = "qwen2_vl"
    assert not _prefers(config)
    assert resolve_optimization_contract(config).model.family is not ModelFamily.PADDLEOCR_VL


def test_paddle_outer_identity_wins_over_qwen_like_text_metadata() -> None:
    config = _config()
    hf = config.model_config.hf_config
    hf.architectures = ["Qwen3_5ForConditionalGeneration"]
    hf.text_config.model_type = "qwen3_5_text"
    assert _prefers(config)
    assert resolve_optimization_contract(config).model.family is ModelFamily.PADDLEOCR_VL
