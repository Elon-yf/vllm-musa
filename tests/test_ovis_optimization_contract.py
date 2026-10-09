from types import SimpleNamespace

import pytest
import torch

from vllm_musa.optimization_contract import (
    ModelFamily,
    MusaOptimizationContract,
    OptimizationFeature,
    resolve_optimization_contract,
)


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=SimpleNamespace(
            model_type="qwen3_5_text",
            hidden_size=1024,
            intermediate_size=3584,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            vocab_size=248320,
        ),
        vision_config=SimpleNamespace(
            hidden_size=768,
            depth=12,
            num_heads=12,
            out_hidden_size=1024,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
    )


def _contract(
    hf_config: SimpleNamespace, dtype: object = torch.bfloat16
) -> MusaOptimizationContract:
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config.text_config,
        architectures=hf_config.architectures,
        dtype=dtype,
    )
    return resolve_optimization_contract(model_config=model_config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, "bfloat16", "torch.bfloat16"])
def test_exact_ovis_adds_vision_feature_without_losing_qwen_features(
    dtype: object,
) -> None:
    contract = _contract(_hf_config(), dtype)
    assert contract.model.family is ModelFamily.QWEN35_36
    assert contract.prefers(OptimizationFeature.OVIS_QWEN35_VISION_ROTARY)
    assert contract.prefers(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, None, "float32", "BFLOAT16"]
)
def test_ovis_rejects_other_dtypes(dtype: object) -> None:
    assert not _contract(_hf_config(), dtype).prefers(
        OptimizationFeature.OVIS_QWEN35_VISION_ROTARY
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("architectures",), ["Qwen3_5MoeForConditionalGeneration"]),
        (("model_type",), "qwen3_5_moe"),
        (("text_config", "model_type"), "qwen3_5_moe_text"),
        (("text_config", "hidden_size"), 2048),
        (("text_config", "intermediate_size"), 4096),
        (("text_config", "num_hidden_layers"), 25),
        (("text_config", "num_attention_heads"), 16),
        (("text_config", "num_key_value_heads"), 4),
        (("text_config", "head_dim"), 128),
        (("text_config", "vocab_size"), 151936),
        (("vision_config", "hidden_size"), 1024),
        (("vision_config", "depth"), 24),
        (("vision_config", "num_heads"), 8),
        (("vision_config", "out_hidden_size"), 768),
        (("vision_config", "patch_size"), 14),
        (("vision_config", "spatial_merge_size"), 1),
        (("vision_config", "temporal_patch_size"), 1),
    ],
)
def test_ovis_rejects_each_geometry_miss(
    path: tuple[str, ...], value: object
) -> None:
    config = _hf_config()
    owner = config if len(path) == 1 else getattr(config, path[0])
    setattr(owner, path[-1], value)
    assert not _contract(config).prefers(
        OptimizationFeature.OVIS_QWEN35_VISION_ROTARY
    )


def test_ovis_uses_raw_hf_config_and_ordinary_qwen_keeps_text_features() -> None:
    config = _hf_config()
    model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=SimpleNamespace(
            model_type="qwen3_5_text", hidden_size=2048, vocab_size=248320
        ),
        architectures=config.architectures,
        dtype=torch.bfloat16,
    )
    contract = resolve_optimization_contract(model_config=model_config)
    assert contract.prefers(OptimizationFeature.OVIS_QWEN35_VISION_ROTARY)
    assert contract.prefers(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)

    config.vision_config = None
    ordinary = _contract(config)
    assert ordinary.model.family is ModelFamily.QWEN35_36
    assert ordinary.prefers(OptimizationFeature.QWEN35_INTERLEAVED_MROPE_QK)
    assert not ordinary.prefers(OptimizationFeature.OVIS_QWEN35_VISION_ROTARY)


def test_ovis_does_not_accept_vision_field_aliases() -> None:
    config = _hf_config()
    config.vision_config.hidden_size = 1024
    config.vision_config.embed_dim = 768
    assert not _contract(config).prefers(
        OptimizationFeature.OVIS_QWEN35_VISION_ROTARY
    )

    config = _hf_config()
    config.vision_config.num_heads = None
    config.vision_config.num_attention_heads = 12
    assert not _contract(config).prefers(
        OptimizationFeature.OVIS_QWEN35_VISION_ROTARY
    )
