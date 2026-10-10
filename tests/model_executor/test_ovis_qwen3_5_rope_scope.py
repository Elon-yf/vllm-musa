"""Behavioral scope checks for Ovis vision RoPE selection."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")
import torch  # noqa: E402

from vllm_musa.model_executor.layers.rotary_embedding.base import (  # noqa: E402
    MusaVisionApplyRotaryEmb,
    MusaVisionRotaryPositions,
)


def _config() -> SimpleNamespace:
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


def _visual() -> SimpleNamespace:
    def block() -> SimpleNamespace:
        rotary = SimpleNamespace(is_neox_style=True, enable_fp32_compute=False)
        return SimpleNamespace(attn=SimpleNamespace(apply_rotary_emb=rotary))

    return SimpleNamespace(blocks=[block() for _ in range(12)])


@pytest.fixture
def ovis_hook(monkeypatch: pytest.MonkeyPatch):
    module = pytest.importorskip("vllm.model_executor.models.qwen3_5")
    if not hasattr(module, "_enable_musa_ovis_rope"):
        pytest.skip("Ovis patch is not applied")
    import vllm.platforms
    from vllm.config import VllmConfig, set_current_vllm_config

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(is_musa=lambda: True)
    )
    with set_current_vllm_config(VllmConfig()):
        yield module._enable_musa_ovis_rope


def test_exact_ovis_geometry_installs_only_visual_rope(ovis_hook) -> None:
    visual = _visual()
    ovis_hook(_config(), SimpleNamespace(dtype=torch.bfloat16), visual)
    rotary = [block.attn.apply_rotary_emb for block in visual.blocks]
    assert all(isinstance(item, MusaVisionApplyRotaryEmb) for item in rotary)
    assert all(item.positions_cache is rotary[0].positions_cache for item in rotary)


@pytest.mark.parametrize(
    ("owner", "name", "value"),
    [
        (None, "model_type", "qwen3_5_moe"),
        ("text_config", "hidden_size", 2048),
        ("vision_config", "depth", 24),
        ("vision_config", "num_heads", 8),
    ],
)
def test_other_geometry_keeps_original_layer(ovis_hook, owner, name, value) -> None:
    config = _config()
    setattr(getattr(config, owner) if owner else config, name, value)
    visual = _visual()
    old = visual.blocks[0].attn.apply_rotary_emb
    ovis_hook(config, SimpleNamespace(dtype=torch.bfloat16), visual)
    assert visual.blocks[0].attn.apply_rotary_emb is old


def test_other_dtype_keeps_original_layer(ovis_hook) -> None:
    visual = _visual()
    old = visual.blocks[0].attn.apply_rotary_emb
    ovis_hook(_config(), SimpleNamespace(dtype=torch.float16), visual)
    assert visual.blocks[0].attn.apply_rotary_emb is old


def test_graph_position_buffer_survives_other_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = MusaVisionRotaryPositions()
    device = torch.device("cpu")
    original = cache.get(2, 3, device)
    monkeypatch.setattr(cache, "_is_capturing", lambda _: True)
    assert cache.get(2, 3, device) is original
    cache.get(2, 4, device)
    monkeypatch.setattr(cache, "_is_capturing", lambda _: False)
    assert cache.get(2, 3, device) is original
