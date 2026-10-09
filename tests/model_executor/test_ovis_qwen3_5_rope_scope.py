from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models import qwen3_5
from vllm_musa.model_executor.models.ovis_qwen3_5_rope import (
    OvisVisionApplyRotaryEmb,
    _OvisRotaryPositionCache,
    enable_ovis_vision_rope,
)
from vllm.model_executor.models.qwen3_5 import _is_ovis_qwen35_config


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, "bfloat16", "torch.bfloat16"])
def test_ovis_guard_accepts_exact_config(dtype: torch.dtype | str) -> None:
    assert _is_ovis_qwen35_config(_config(), SimpleNamespace(dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, None, "float32"])
def test_ovis_guard_rejects_other_dtypes(dtype: torch.dtype | str | None) -> None:
    assert not _is_ovis_qwen35_config(_config(), SimpleNamespace(dtype=dtype))


def test_ovis_guard_rejects_other_architecture() -> None:
    config = _config()
    config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
    assert not _is_ovis_qwen35_config(config, SimpleNamespace(dtype=torch.bfloat16))


def test_ovis_guard_rejects_other_hidden_size() -> None:
    config = _config()
    config.text_config.hidden_size = 2048
    assert not _is_ovis_qwen35_config(config, SimpleNamespace(dtype=torch.bfloat16))


def _visual(block_count: int = 12) -> SimpleNamespace:
    class ExistingRotary(torch.nn.Module):
        is_neox_style = True
        enable_fp32_compute = False

    return SimpleNamespace(
        blocks=[
            SimpleNamespace(attn=SimpleNamespace(apply_rotary_emb=ExistingRotary()))
            for _ in range(block_count)
        ]
    )


def test_ovis_replacement_is_block_local_and_shares_one_cache() -> None:
    visual, unrelated = _visual(), _visual()
    assert enable_ovis_vision_rope(visual) == 12
    assert all(
        isinstance(block.attn.apply_rotary_emb, OvisVisionApplyRotaryEmb)
        for block in visual.blocks
    )
    caches = [block.attn.apply_rotary_emb.positions_cache for block in visual.blocks]
    assert all(cache is caches[0] for cache in caches)
    assert all(
        not isinstance(block.attn.apply_rotary_emb, OvisVisionApplyRotaryEmb)
        for block in unrelated.blocks
    )
    enable_ovis_vision_rope(unrelated)
    assert unrelated.blocks[0].attn.apply_rotary_emb.positions_cache is not caches[0]


def test_gate_reports_musa_miss_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    log = Mock()
    monkeypatch.setattr(qwen3_5, "logger", log)
    monkeypatch.setattr(
        qwen3_5, "current_platform", SimpleNamespace(is_musa=lambda: True)
    )
    visual = _visual()
    qwen3_5._enable_ovis_vision_rope(
        _config(), SimpleNamespace(dtype=torch.float32), visual
    )
    assert "disabled" in log.info_once.call_args.args[0]
    assert not isinstance(
        visual.blocks[0].attn.apply_rotary_emb, OvisVisionApplyRotaryEmb
    )
    qwen3_5._enable_ovis_vision_rope(
        _config(), SimpleNamespace(dtype=torch.bfloat16), visual
    )
    assert log.info_once.call_args.args[1:] == (12, 12)
    assert all(
        isinstance(b.attn.apply_rotary_emb, OvisVisionApplyRotaryEmb)
        for b in visual.blocks
    )


def test_non_musa_and_unrelated_model_remain_untouched_and_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = Mock()
    monkeypatch.setattr(qwen3_5, "logger", log)
    monkeypatch.setattr(
        qwen3_5, "current_platform", SimpleNamespace(is_musa=lambda: False)
    )
    visual = _visual()
    qwen3_5._enable_ovis_vision_rope(
        _config(), SimpleNamespace(dtype=torch.bfloat16), visual
    )
    assert not isinstance(
        visual.blocks[0].attn.apply_rotary_emb, OvisVisionApplyRotaryEmb
    )
    monkeypatch.setattr(
        qwen3_5, "current_platform", SimpleNamespace(is_musa=lambda: True)
    )
    config = _config()
    config.model_type = "qwen2_vl"
    qwen3_5._enable_ovis_vision_rope(
        config, SimpleNamespace(dtype=torch.bfloat16), visual
    )
    log.info_once.assert_not_called()


def test_position_cache_reuses_shared_buffer_and_replaces_one_shape() -> None:
    visual = _visual()
    enable_ovis_vision_rope(visual)
    first = visual.blocks[0].attn.apply_rotary_emb.positions_cache
    last = visual.blocks[-1].attn.apply_rotary_emb.positions_cache
    device = torch.device("cpu")
    positions = first.get(2, 3, device)
    assert positions.tolist() == [0, 1, 2, 0, 1, 2]
    assert last.get(2, 3, device) is positions
    changed = last.get(2, 4, device)
    assert changed is not positions
    assert first.get(2, 4, device) is changed
    assert first.get(4, 4, device).shape == (16,)
    assert len(list(first.buffers())) == 1
    assert first.state_dict() == {}
    other_device = first.get(4, 4, torch.device("meta"))
    assert other_device.device.type == "meta"
    assert first.get(4, 4, torch.device("meta")) is other_device


def test_warm_capture_pins_storage_and_other_shapes_do_not_evict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _OvisRotaryPositionCache()
    device = torch.device("cpu")
    warm = cache.get(2, 3, device)
    monkeypatch.setattr(cache, "_is_capturing", lambda _: True)
    assert cache.get(2, 3, device) is warm
    capture_miss = cache.get(2, 4, device)
    assert capture_miss.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    monkeypatch.setattr(cache, "_is_capturing", lambda _: False)
    assert cache.get(2, 4, device) is not capture_miss
    assert cache.get(2, 3, device) is warm
    assert len(list(cache.buffers())) == 1


def test_capture_miss_does_not_publish_graph_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _OvisRotaryPositionCache()
    monkeypatch.setattr(cache, "_is_capturing", lambda _: True)
    cache.get(2, 3, torch.device("cpu"))
    assert list(cache.buffers()) == []
    monkeypatch.setattr(cache, "_is_capturing", lambda _: False)
    eager = cache.get(2, 3, torch.device("cpu"))
    assert cache.get(2, 3, torch.device("cpu")) is eager


def test_musa_capture_query_uses_redirected_api_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(type="musa")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _OvisRotaryPositionCache._is_capturing(device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not _OvisRotaryPositionCache._is_capturing(device)

    def unavailable() -> bool:
        raise RuntimeError("capture state unavailable")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unavailable)
    assert _OvisRotaryPositionCache._is_capturing(device)
    assert not _OvisRotaryPositionCache._is_capturing(torch.device("cpu"))
