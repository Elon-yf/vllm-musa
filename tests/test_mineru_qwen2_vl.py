# SPDX-License-Identifier: Apache-2.0
"""The MinerU route is a strict Qwen2-VL optimization contract."""

from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    ModelFamily,
    OptimizationFeature,
    resolve_optimization_contract,
)
from vllm_musa.optimization_contract.qwen import matches_mineru_qwen2_vl_config


def _raw_config():
    return SimpleNamespace(
        model_type="qwen2_vl",
        architectures=["Qwen2VLForConditionalGeneration"],
        text_config=SimpleNamespace(
            model_type="qwen2_vl",
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            rope_scaling={"mrope_section": [8, 12, 12]},
        ),
        vision_config=SimpleNamespace(embed_dim=1280, depth=32, num_heads=16),
    )


def _vllm_config(raw=None, *, normalized_text=None, tp=1, pp=1):
    raw = raw or _raw_config()
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=raw,
            hf_text_config=normalized_text or raw.text_config,
            architectures=raw.architectures,
            dtype="bfloat16",
            enforce_eager=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp,
        ),
    )


def test_raw_hf_config_wins_over_normalized_text():
    raw = _raw_config()
    normalized = SimpleNamespace(model_type="qwen2", hidden_size=1024)
    contract = resolve_optimization_contract(
        _vllm_config(raw, normalized_text=normalized)
    )
    assert contract.model.family is ModelFamily.QWEN2
    assert contract.preferred_features == {
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    }
    raw.text_config.hidden_size = 1024
    normalized.hidden_size = 896
    assert not resolve_optimization_contract(
        _vllm_config(raw, normalized_text=normalized)
    ).prefers(OptimizationFeature.MINERU_QWEN2_VL_ROTARY)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("text_config", "hidden_size", 1024),
        ("text_config", "num_hidden_layers", 25),
        ("text_config", "num_attention_heads", 16),
        ("text_config", "num_key_value_heads", 4),
        ("text_config", "head_dim", "64"),
        ("vision_config", "embed_dim", 1024),
        ("vision_config", "depth", 24),
        ("vision_config", "num_heads", 20),
    ],
)
def test_geometry_mismatch_misses(owner, field, value):
    raw = _raw_config()
    setattr(getattr(raw, owner), field, value)
    assert not matches_mineru_qwen2_vl_config(raw)
    assert not resolve_optimization_contract(_vllm_config(raw)).prefers(
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    )


def test_outer_mrope_section_takes_precedence():
    raw = _raw_config()
    raw.mrope_section = [8, 12, 11]
    assert not matches_mineru_qwen2_vl_config(raw)
    raw.mrope_section = [8, 12, 12]
    raw.text_config.rope_scaling = {"mrope_section": [8, 12, 11]}
    assert matches_mineru_qwen2_vl_config(raw)


@pytest.mark.parametrize("section", [None, [], [12, 8, 12], [8, 12, 12, 0]])
def test_wrong_mrope_section_misses(section):
    raw = _raw_config()
    raw.text_config.rope_scaling = {"mrope_section": section}
    assert not matches_mineru_qwen2_vl_config(raw)


def test_only_exact_qwen2_vl_subclass_gets_feature():
    raw = _raw_config()
    raw.model_type = "qwen2_5_vl"
    assert not resolve_optimization_contract(_vllm_config(raw)).prefers(
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    )
    raw.model_type = "qwen2_vl"
    raw.architectures = ["OtherForConditionalGeneration"]
    assert not resolve_optimization_contract(_vllm_config(raw)).prefers(
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    )


def test_existing_qwen_feature_survives_mineru_subclass():
    raw = _raw_config()
    normalized = SimpleNamespace(
        model_type="qwen2", hidden_size=896, intermediate_size=4864,
        num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
    )
    config = _vllm_config(raw, normalized_text=normalized)
    config.model_config.enforce_eager = False
    features = resolve_optimization_contract(config).preferred_features
    assert features == {
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY,
        OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT,
    }
    raw.vision_config.depth = 24
    assert resolve_optimization_contract(config).preferred_features == {
        OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT,
    }


@pytest.mark.parametrize(("tp", "pp"), [(2, 1), (1, 2)])
def test_parallel_configuration_keeps_visual_eligibility(tp, pp):
    assert resolve_optimization_contract(_vllm_config(tp=tp, pp=pp)).prefers(
        OptimizationFeature.MINERU_QWEN2_VL_ROTARY
    )
