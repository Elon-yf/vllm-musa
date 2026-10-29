# SPDX-License-Identifier: Apache-2.0
"""Exercise vision backend selection without loading the MUSA runtime."""

from __future__ import annotations

import ast
import sys
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

PLATFORM = Path(__file__).parents[1] / "vllm_musa" / "platform.py"


def _config(architecture: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            multimodal_config=SimpleNamespace(mm_encoder_attn_backend=None),
        )
    )


@pytest.fixture
def selector(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    registry = {}
    lookups = []
    register = Mock()

    class Backend(Enum):
        FLASH_ATTN = auto()
        TRITON_ATTN = auto()
        TORCH_SDPA = auto()

        def get_class(self):
            assert register.called
            lookups.append(self)
            if registry[self] is None:
                raise ImportError("Backend unavailable")
            return registry[self]

    for backend in Backend:
        supported = {"float16", "bfloat16"}
        if backend is Backend.TRITON_ATTN:
            supported.add("float32")
        registry[backend] = SimpleNamespace(
            supports_head_size=Mock(return_value=True),
            supports_dtype=Mock(side_effect=lambda dtype, ds=supported: dtype in ds),
            supports_compute_capability=Mock(return_value=True),
        )

    state = SimpleNamespace(config=_config("DeepseekOCRForCausalLM"))
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config_or_none = lambda: state.config
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)

    tree = ast.parse(PLATFORM.read_text(encoding="utf-8"))
    platform = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MUSAPlatformBase"
    )
    platform.bases = []
    platform.body = [
        node
        for node in platform.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_supported_vit_attn_backends", "get_vit_attn_backend"}
    ]
    module = ast.Module(body=[platform], type_ignores=[])
    namespace = {
        "torch": SimpleNamespace(dtype=str, float32="float32"),
        "logger": Mock(),
        "AttentionBackendEnum": Backend,
        "register_attention_backends": register,
    }
    exec(compile(module, str(PLATFORM), "exec"), namespace)
    cls = namespace["MUSAPlatformBase"]
    cls.get_device_capability = classmethod(lambda cls: (3, 1))
    return SimpleNamespace(
        platform=cls,
        state=state,
        backend=Backend,
        registry=registry,
        lookups=lookups,
    )


def test_deepseek_ocr_prefers_triton_with_fp32_vision(selector) -> None:
    before = repr(selector.state.config)
    result = selector.platform.get_vit_attn_backend(64, "bfloat16")

    assert result is selector.backend.TRITON_ATTN
    assert selector.lookups == [selector.backend.TRITON_ATTN]
    selector.registry[result].supports_dtype.assert_called_once_with("float32")
    assert repr(selector.state.config) == before


@pytest.mark.parametrize(
    "architecture", ["Qwen3VLForConditionalGeneration", "DeepseekOCR2ForCausalLM"]
)
def test_other_models_keep_original_priority_and_dtype(selector, architecture) -> None:
    selector.state.config = _config(architecture)
    result = selector.platform.get_vit_attn_backend(64, "bfloat16")

    assert result is selector.backend.FLASH_ATTN
    assert selector.lookups == [selector.backend.FLASH_ATTN]
    selector.registry[result].supports_dtype.assert_called_once_with("bfloat16")


@pytest.mark.parametrize("config", [None, SimpleNamespace(model_config=None)])
def test_missing_model_context_keeps_original_selection(selector, config) -> None:
    selector.state.config = config
    assert (
        selector.platform.get_vit_attn_backend(64, "bfloat16")
        is selector.backend.FLASH_ATTN
    )


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "TRITON_ATTN", "TORCH_SDPA"])
def test_explicit_override_remains_authoritative(selector, backend) -> None:
    requested = selector.backend[backend]
    assert (
        selector.platform.get_vit_attn_backend(64, "bfloat16", requested) is requested
    )
    assert selector.lookups == []


@pytest.mark.parametrize("failure", ["import", "head_size", "dtype", "capability"])
def test_incompatible_triton_uses_compatible_fallback(selector, failure) -> None:
    triton = selector.backend.TRITON_ATTN
    if failure == "import":
        selector.registry[triton] = None
    else:
        attr = (
            "supports_compute_capability"
            if failure == "capability"
            else f"supports_{failure}"
        )
        check = getattr(selector.registry[triton], attr)
        check.side_effect = None
        check.return_value = False

    result = selector.platform.get_vit_attn_backend(64, "bfloat16")

    assert result is selector.backend.TORCH_SDPA
    assert selector.lookups == [triton, selector.backend.FLASH_ATTN]
    selector.registry[
        selector.backend.FLASH_ATTN
    ].supports_dtype.assert_called_once_with("float32")


def test_model_context_does_not_leak_between_selections(selector) -> None:
    expected = [
        ("DeepseekOCRForCausalLM", selector.backend.TRITON_ATTN),
        ("Qwen3VLForConditionalGeneration", selector.backend.FLASH_ATTN),
        ("DeepseekOCRForCausalLM", selector.backend.TRITON_ATTN),
    ]
    for architecture, backend in expected:
        selector.state.config = _config(architecture)
        assert selector.platform.get_vit_attn_backend(64, "bfloat16") is backend

    assert selector.platform.get_supported_vit_attn_backends() == [
        selector.backend.FLASH_ATTN,
        selector.backend.TRITON_ATTN,
        selector.backend.TORCH_SDPA,
    ]


def test_next_candidate_is_selected_when_it_supports_fp32(selector) -> None:
    selector.registry[selector.backend.TRITON_ATTN] = None
    next_backend = selector.registry[selector.backend.FLASH_ATTN]
    next_backend.supports_dtype.side_effect = None
    next_backend.supports_dtype.return_value = True

    assert (
        selector.platform.get_vit_attn_backend(64, "bfloat16")
        is selector.backend.FLASH_ATTN
    )
    next_backend.supports_dtype.assert_called_once_with("float32")
