# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the optional CosyVoice3 MUSA sampler wrapper."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "vllm_musa/omni/__init__.py"
GPU_AR = ROOT / "vllm_musa/omni/gpu_ar.py"
MODEL = ROOT / "vllm_musa/omni/models/cosyvoice3.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_vllm_omni_entrypoint_is_lazy_and_optional() -> None:
    project = _source(ROOT / "pyproject.toml")

    assert '[project.entry-points."vllm_omni.general_plugins"]' in project
    assert 'musa_custom_ops = "vllm_musa.omni:register_omni_optimizations"' in project


def test_omni_plugin_registers_both_model_registries_lazily() -> None:
    source = _source(PLUGIN)

    assert "vllm_musa.omni.models.cosyvoice3:CosyVoice3Model" in source
    assert "ModelRegistry.register_model" in source
    assert "OmniModelRegistry.register_model" in source
    assert "_MODEL_ARCH_BY_HASH" in source
    assert "_invalidate_cosyvoice3_model_cache()" in source
    assert "models.cosyvoice3 import" not in source


def test_omni_plugin_invalidates_only_stale_cosyvoice3_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_module = types.ModuleType("vllm")
    model_executor_module = types.ModuleType("vllm.model_executor")
    model_loader_module = types.ModuleType("vllm.model_executor.model_loader")
    loader_utils_module = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils_module._MODEL_ARCH_BY_HASH = {
        1: (object(), "CosyVoice3Model"),
        2: (object(), "OtherModel"),
    }
    model_loader_module.utils = loader_utils_module
    model_executor_module.model_loader = model_loader_module
    vllm_module.model_executor = model_executor_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.model_loader", model_loader_module
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils_module,
    )

    spec = importlib.util.spec_from_file_location("test_vllm_musa_omni_plugin", PLUGIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._invalidate_cosyvoice3_model_cache() == 1
    assert list(loader_utils_module._MODEL_ARCH_BY_HASH) == [2]


def _load_gpu_ar_module(monkeypatch: pytest.MonkeyPatch):
    fake_runner_module = types.ModuleType("vllm_omni.worker.gpu_ar_model_runner")

    class FakeGPUARModelRunner:
        def _sampling_metadata_for_model_sampler(self, sampling_metadata):
            return sampling_metadata

    fake_runner_module.GPUARModelRunner = FakeGPUARModelRunner
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.worker.gpu_ar_model_runner",
        fake_runner_module,
    )

    spec = importlib.util.spec_from_file_location("test_vllm_musa_omni_gpu_ar", GPU_AR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, FakeGPUARModelRunner


def test_gpu_ar_patch_ignores_repetition_penalty_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, runner_cls = _load_gpu_ar_module(monkeypatch)
    assert module.install_sampling_metadata_patch()

    runner = runner_cls()
    runner.model_config = SimpleNamespace(model_arch="CosyVoice3Model")
    runner.input_batch = SimpleNamespace(
        frequency_penalties_reqs=set(),
        presence_penalties_reqs=set(),
        repetition_penalties_reqs={"request-0"},
        temperature_cpu=[1.0],
        top_p_cpu=[0.8],
        top_k_cpu=[25],
    )
    model_metadata = SimpleNamespace()

    result = runner._sampling_metadata_for_model_sampler(model_metadata)

    assert result is model_metadata
    assert getattr(result, module.NO_FREQUENCY_PRESENCE_PENALTIES) is True


@pytest.mark.parametrize(
    ("frequency_reqs", "presence_reqs"),
    [({"request-0"}, set()), (set(), {"request-0"})],
)
def test_gpu_ar_patch_reports_frequency_or_presence_penalties(
    monkeypatch: pytest.MonkeyPatch,
    frequency_reqs: set[str],
    presence_reqs: set[str],
) -> None:
    module, runner_cls = _load_gpu_ar_module(monkeypatch)
    assert module.install_sampling_metadata_patch()

    runner = runner_cls()
    runner.model_config = SimpleNamespace(model_arch="CosyVoice3Model")
    runner.input_batch = SimpleNamespace(
        frequency_penalties_reqs=frequency_reqs,
        presence_penalties_reqs=presence_reqs,
        temperature_cpu=[1.0],
        top_p_cpu=[0.8],
        top_k_cpu=[25],
    )
    result = runner._sampling_metadata_for_model_sampler(SimpleNamespace())

    assert getattr(result, module.NO_FREQUENCY_PRESENCE_PENALTIES) is False


def test_gpu_ar_patch_is_idempotent_and_fails_closed_on_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, runner_cls = _load_gpu_ar_module(monkeypatch)
    assert module.install_sampling_metadata_patch()
    patched_method = runner_cls._sampling_metadata_for_model_sampler
    assert module.install_sampling_metadata_patch()
    assert runner_cls._sampling_metadata_for_model_sampler is patched_method

    runner = runner_cls()
    runner.model_config = SimpleNamespace(model_arch="CosyVoice3Model")
    runner.input_batch = SimpleNamespace()
    result = runner._sampling_metadata_for_model_sampler(SimpleNamespace())

    assert not hasattr(result, module.NO_FREQUENCY_PRESENCE_PENALTIES)


def test_gpu_ar_patch_exports_cpu_sampling_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, runner_cls = _load_gpu_ar_module(monkeypatch)
    assert module.install_sampling_metadata_patch()

    runner = runner_cls()
    runner.model_config = SimpleNamespace(model_arch="CosyVoice3Model")
    runner.input_batch = SimpleNamespace(
        temperature_cpu=[1.0],
        top_p_cpu=[0.8],
        top_k_cpu=[25],
    )
    result = runner._sampling_metadata_for_model_sampler(SimpleNamespace())

    assert getattr(result, module.HOST_SAMPLING_SCALARS) == (
        [1.0],
        [0.8],
        [25],
    )


def test_gpu_ar_patch_does_not_touch_other_model_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, runner_cls = _load_gpu_ar_module(monkeypatch)
    assert module.install_sampling_metadata_patch()

    runner = runner_cls()
    runner.model_config = SimpleNamespace(model_arch="OtherModel")
    runner.input_batch = SimpleNamespace(
        frequency_penalties_reqs=set(),
        presence_penalties_reqs=set(),
        temperature_cpu=[1.0],
        top_p_cpu=[0.8],
        top_k_cpu=[25],
    )
    result = runner._sampling_metadata_for_model_sampler(SimpleNamespace())

    assert not hasattr(result, module.NO_FREQUENCY_PRESENCE_PENALTIES)
    assert not hasattr(result, module.HOST_SAMPLING_SCALARS)


def test_ras_eligibility_reuses_host_side_frequency_presence_flag() -> None:
    source = _source(MODEL)
    tree = ast.parse(source, MODEL)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CosyVoice3Model"
    )
    method_names = {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }

    assert {"_cosyvoice3_ras_enabled", "_ras_sample_one"} <= method_names
    assert "return bool(no_frequency_presence_penalties)" in source
    assert "def _req_scalar" in source
    assert "if param is None or param.numel() == 0" in source
    assert "HOST_SAMPLING_SCALARS" in source
    assert "return super()._cosyvoice3_ras_enabled(sampling_metadata)" in source
    assert "torch.any(" not in source
    assert "sampling_metadata.no_penalties" not in source


def test_cpu_history_count_preserves_ras_fallback_distribution() -> None:
    source = _source(MODEL)

    assert "sum(int(token) == top_id for token in recent)" in source
    assert "weighted_scores = weighted_scores.clone()" in source
    assert 'weighted_scores[top_id] = float("-inf")' in source
    assert "fallback_probs = weighted_scores.softmax(dim=0)" in source
    assert "cls._multinomial_sample(" in source
    assert "torch.as_tensor" not in source
