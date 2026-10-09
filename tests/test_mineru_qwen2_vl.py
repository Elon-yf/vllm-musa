# SPDX-License-Identifier: Apache-2.0
"""Behavioral cases for the build-applied MinerU Qwen2-VL gate."""

from pathlib import Path
from types import SimpleNamespace

import pytest

PATCH = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa/patches/series/0180-MUSA-add-minimal-MinerU-Qwen2-VL-rotary-dispatch.patch"
)


@pytest.fixture(scope="module")
def gate():
    # Load only the real gate from the build patch; no model-constructor mocks.
    lines = PATCH.read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("+def _is_mineru_qwen2_vl_config(")
    )
    body = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        body.append(line[1:])
    namespace = {"Qwen2VLConfig": object}
    exec("\n".join(body), namespace)
    return namespace["_is_mineru_qwen2_vl_config"]


def _config():
    return SimpleNamespace(
        model_type="qwen2_vl",
        text_config=SimpleNamespace(
            hidden_size=896,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            rope_parameters={"mrope_section": [8, 12, 12]},
        ),
        vision_config=SimpleNamespace(embed_dim=1280, depth=32, num_heads=16),
    )


def test_exact_raw_hf_gate_ignores_conflicting_normalized_text(gate):
    raw = _config()
    model_config = SimpleNamespace(hf_config=raw, hf_text_config=SimpleNamespace(hidden_size=1024))
    assert gate(model_config.hf_config)
    raw.text_config.hidden_size = 1024
    model_config.hf_text_config.hidden_size = 896
    assert not gate(model_config.hf_config)


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
def test_geometry_mismatch_misses(gate, owner, field, value):
    config = _config()
    setattr(getattr(config, owner), field, value)
    assert not gate(config)


def test_flat_text_and_vision_aliases(gate):
    config = _config()
    text = vars(config.text_config).copy()
    text.pop("head_dim")
    del config.text_config
    vars(config).update(text)
    config.vision_config = SimpleNamespace(
        hidden_size=1280, num_hidden_layers=32, num_attention_heads=16
    )
    assert gate(config)


@pytest.mark.parametrize("field", ["mrope_section", "rope_parameters", "rope_scaling"])
def test_supported_rope_sections(gate, field):
    config = _config()
    del config.text_config.rope_parameters
    setattr(
        config,
        field,
        [8, 12, 12] if field == "mrope_section" else {"mrope_section": [8, 12, 12]},
    )
    assert gate(config)


def test_outer_section_precedes_text(gate):
    config = _config()
    config.mrope_section = [8, 12, 11]
    assert not gate(config)
    config.mrope_section = [8, 12, 12]
    config.text_config.rope_parameters = {"mrope_section": [8, 12, 11]}
    assert gate(config)


@pytest.mark.parametrize("section", [None, [], [12, 8, 12], [8, 12, 12, 0]])
def test_wrong_section_misses(gate, section):
    config = _config()
    config.text_config.rope_parameters = {"mrope_section": section}
    assert not gate(config)


def test_other_model_or_missing_vision_misses(gate):
    config = _config()
    config.model_type = "qwen2_5_vl"
    assert not gate(config)
    config.model_type = "qwen2_vl"
    del config.vision_config
    assert not gate(config)
