# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the packaged MinerU model adapter."""

import builtins
import importlib.util
from pathlib import Path
from textwrap import dedent
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "vllm_musa/models/mineru_qwen2_vl.py"
PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0173-MUSA-add-minimal-MinerU-Qwen2-VL-rotary-dispatch.patch"
)


@pytest.fixture(scope="module")
def mineru() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mineru_adapter_under_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> SimpleNamespace:
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


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("text_config", "hidden_size", 1024),
        ("text_config", "num_hidden_layers", 25),
        ("text_config", "num_attention_heads", 16),
        ("text_config", "num_key_value_heads", 4),
        ("text_config", "head_dim", 128),
        ("vision_config", "embed_dim", 1024),
        ("vision_config", "depth", 24),
        ("vision_config", "num_heads", 20),
    ],
)
def test_geometry_gate_rejects_each_mismatch(
    mineru: ModuleType, owner: str, field: str, value: int
) -> None:
    config = _config()
    assert mineru.is_mineru_qwen2_vl_config(config)
    setattr(getattr(config, owner), field, value)
    assert not mineru.is_mineru_qwen2_vl_config(config)


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen2", None])
def test_geometry_gate_rejects_other_model_types(
    mineru: ModuleType, model_type: str | None
) -> None:
    config = _config()
    config.model_type = model_type
    assert not mineru.is_mineru_qwen2_vl_config(config)


def test_geometry_gate_requires_vision_config(mineru: ModuleType) -> None:
    config = _config()
    del config.vision_config
    assert not mineru.is_mineru_qwen2_vl_config(config)


def test_geometry_gate_accepts_flat_text_and_vision_aliases(mineru: ModuleType) -> None:
    config = _config()
    text = vars(config.text_config).copy()
    text.pop("head_dim")
    del config.text_config
    vars(config).update(text)
    config.vision_config = SimpleNamespace(
        hidden_size=1280, num_hidden_layers=32, num_attention_heads=16
    )
    assert mineru.is_mineru_qwen2_vl_config(config)


@pytest.mark.parametrize("owner_name", ["model", "text"])
@pytest.mark.parametrize("field", ["mrope_section", "rope_parameters", "rope_scaling"])
@pytest.mark.parametrize("section", [[8, 12, 12], (8, 12, 12)])
def test_geometry_gate_accepts_supported_rope_layouts(
    mineru: ModuleType, owner_name: str, field: str, section: list[int] | tuple[int, ...]
) -> None:
    config = _config()
    del config.text_config.rope_parameters
    owner = config if owner_name == "model" else config.text_config
    value = section if field == "mrope_section" else {"mrope_section": section}
    setattr(owner, field, value)
    assert mineru.is_mineru_qwen2_vl_config(config)


@pytest.mark.parametrize("section", [None, [], [12, 8, 12], [8, 12, 11], [8, 12, 12, 0]])
def test_geometry_gate_rejects_other_rope_sections(
    mineru: ModuleType, section: list[int] | None
) -> None:
    config = _config()
    config.text_config.rope_parameters = {"mrope_section": section}
    assert not mineru.is_mineru_qwen2_vl_config(config)


def test_attention_registration_is_model_local_and_takes_no_arguments(
    mineru: ModuleType,
) -> None:
    calls: list[None] = []
    registrar = lambda: calls.append(None)  # noqa: E731
    config = _config()
    config.model_type = "qwen2_5_vl"
    assert not mineru.register_mineru_attention_backends(config, registrar)
    assert calls == []
    assert mineru.register_mineru_attention_backends(_config(), registrar)
    assert calls == [None]


@pytest.mark.parametrize("shape", [(3, 2, 80), (2, 3, 2, 80)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_visual_rotary_on_cpu_preserves_native_call(
    mineru: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    x = torch.zeros(shape, dtype=dtype)
    cos = torch.ones((3, 40), dtype=dtype)
    sin = torch.zeros_like(cos)
    expected = torch.full_like(x, 3)
    reference = nn.Module()
    reference.forward_native = Mock(return_value=expected)
    musa_rotary = Mock(side_effect=AssertionError("CPU must use native rotary"))
    monkeypatch.setattr(mineru, "_musa_visual_rotary", musa_rotary)

    result = mineru._MineruVisualRotary(reference)(x, cos, sin)

    assert result is expected
    reference.forward_native.assert_called_once_with(x, cos, sin)
    musa_rotary.assert_not_called()


@pytest.mark.parametrize("multimodal_positions", [False, True])
@pytest.mark.parametrize("with_key", [False, True])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_language_rotary_on_cpu_preserves_native_call(
    mineru: ModuleType, multimodal_positions: bool, with_key: bool, with_offsets: bool
) -> None:
    positions = torch.arange(3)
    if multimodal_positions:
        positions = positions.expand(3, -1)
    query = torch.zeros((3, 896), dtype=torch.bfloat16)
    key = torch.zeros((3, 128), dtype=query.dtype) if with_key else None
    offsets = torch.ones(3, dtype=torch.long) if with_offsets else None
    expected = (query + 1, key + 2 if key is not None else None)
    reference = nn.Module()
    reference.forward_native = Mock(return_value=expected)

    result = mineru._MineruMRotary(reference)(positions, query, key, offsets)

    assert result is expected
    reference.forward_native.assert_called_once_with(positions, query, key, offsets)


class ApplyRotaryEmb(nn.Module):
    pass


class MRotaryEmbedding(nn.Module):
    def __init__(
        self, section: tuple[int, ...] = (8, 12, 12), head_size: int = 64
    ) -> None:
        super().__init__()
        self.mrope_section = section
        self.head_size = head_size


def test_rotary_replacement_is_recursive_and_limited_to_matching_children(
    mineru: ModuleType,
) -> None:
    visual = ApplyRotaryEmb()
    language = MRotaryEmbedding()
    wrong_section = MRotaryEmbedding(section=(12, 8, 12))
    wrong_head = MRotaryEmbedding(head_size=128)
    unrelated = nn.Identity()
    model = nn.ModuleList(
        [visual, nn.ModuleList([language, wrong_section, wrong_head]), unrelated]
    )
    other_model = nn.ModuleList([ApplyRotaryEmb(), MRotaryEmbedding()])

    assert mineru.patch_mineru_rotary(model) == (1, 1)

    assert isinstance(model[0], mineru._MineruVisualRotary)
    assert model[0].reference is visual
    assert isinstance(model[1][0], mineru._MineruMRotary)
    assert model[1][0].reference is language
    assert model[1][1] is wrong_section
    assert model[1][2] is wrong_head
    assert model[2] is unrelated
    assert type(other_model[0]) is ApplyRotaryEmb
    assert type(other_model[1]) is MRotaryEmbedding


def test_series_patch_only_modifies_existing_qwen2_vl() -> None:
    source = PATCH.read_text()
    target = "vllm/model_executor/models/qwen2_vl.py"
    headers = [line for line in source.splitlines() if line.startswith("diff --git ")]
    assert headers == [f"diff --git a/{target} b/{target}"]
    assert f"--- a/{target}\n+++ b/{target}\n" in source
    assert "new file mode" not in source
    assert "/dev/null" not in source
    assert "vllm_musa.models.mineru_qwen2_vl" in source


def _patch_additions() -> list[str]:
    hunks: list[str] = []
    additions: list[str] = []
    for line in PATCH.read_text().splitlines():
        if line.startswith("@@"):
            if additions:
                hunks.append(dedent("\n".join(additions)))
                additions = []
        elif line.startswith("+") and not line.startswith("+++"):
            additions.append(line[1:])
    if additions:
        hunks.append(dedent("\n".join(additions)))
    return hunks


@pytest.mark.parametrize("musa_version", [None, "test-musa"])
@pytest.mark.parametrize("matching_model", [False, True])
def test_constructor_hooks_import_and_apply_only_for_musa_mineru(
    mineru: ModuleType, musa_version: str | None, matching_model: bool
) -> None:
    config = _config()
    if not matching_model:
        config.text_config.num_hidden_layers = 25
    registrar, replace_rotary = Mock(), Mock()
    helper = SimpleNamespace(
        is_mineru_qwen2_vl_config=mineru.is_mineru_qwen2_vl_config,
        register_mineru_attention_backends=registrar,
        patch_mineru_rotary=replace_rotary,
    )
    import_helper = Mock(return_value=helper)
    model = SimpleNamespace(visual=object(), language_model=object())
    namespace = {
        "__builtins__": {**vars(builtins), "__import__": import_helper},
        "torch": SimpleNamespace(version=SimpleNamespace(musa=musa_version)),
        "config": config,
        "self": model,
    }
    hunks = _patch_additions()
    assert len(hunks) == 3
    for additions in hunks:
        exec(compile(additions, str(PATCH), "exec"), namespace)

    if musa_version is None:
        import_helper.assert_not_called()
    else:
        assert import_helper.call_count == 1
        assert import_helper.call_args.args[0] == "vllm_musa.models.mineru_qwen2_vl"
    if musa_version is not None and matching_model:
        registrar.assert_called_once_with(config)
        assert replace_rotary.call_args_list == [
            call(model.visual),
            call(model.language_model),
        ]
    else:
        registrar.assert_not_called()
        replace_rotary.assert_not_called()
