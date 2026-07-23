# SPDX-License-Identifier: Apache-2.0
"""Executable behavior tests for the optional CosyVoice3 Omni wrapper."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vllm_musa/omni/models/cosyvoice3.py"
HOST_SCALARS = "_vllm_musa_host_sampling_scalars"
NO_FREQUENCY_PRESENCE = "_vllm_musa_no_frequency_presence_penalties"


def _reference_ras_sample(
    weighted_scores: torch.Tensor,
    decoded_tokens: list[int],
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    generator: torch.Generator | None = None,
) -> int:
    sorted_scores, sorted_indices = torch.sort(
        weighted_scores, descending=True, stable=True
    )
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumulative_probs - sorted_probs) > top_p
    sorted_probs[cutoff] = 0
    if top_k > 0 and top_k < sorted_probs.numel():
        sorted_probs[top_k:] = 0
    sampled_idx = torch.multinomial(
        sorted_probs, num_samples=1, generator=generator
    ).item()
    top_id = int(sorted_indices[sampled_idx].item())

    recent = decoded_tokens[-win_size:]
    if recent.count(top_id) >= tau_r * win_size:
        weighted_scores[top_id] = -torch.inf
        top_id = int(
            torch.multinomial(
                torch.softmax(weighted_scores, dim=-1),
                num_samples=1,
                generator=generator,
            ).item()
        )
    return top_id


class _FakeOmniCosyVoice3Model:
    @staticmethod
    def _nucleus_sample_one(
        weighted_scores,
        *,
        top_p,
        top_k,
        generator,
    ):
        sorted_scores, sorted_indices = torch.sort(
            weighted_scores, descending=True, stable=True
        )
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumulative_probs - sorted_probs) > top_p
        sorted_probs[cutoff] = 0
        if top_k > 0 and top_k < sorted_probs.numel():
            sorted_probs[top_k:] = 0
        sampled_idx = torch.multinomial(
            sorted_probs, num_samples=1, generator=generator
        ).item()
        return int(sorted_indices[sampled_idx].item())

    @staticmethod
    def _multinomial_sample(probs, *, generator):
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @staticmethod
    def _req_scalar(param, req_idx: int, default):
        if param is None or param.numel() == 0:
            return default
        flat = param.reshape(-1)
        value = flat[min(req_idx, flat.numel() - 1)].item()
        return int(value) if isinstance(default, int) else float(value)

    def _cosyvoice3_ras_enabled(self, sampling_metadata) -> bool:
        return bool(getattr(self, "base_ras_enabled", False))

    @classmethod
    def _ras_sample_one(
        cls,
        weighted_scores,
        decoded_tokens,
        *,
        top_p,
        top_k,
        win_size,
        tau_r,
        generator,
    ):
        return _reference_ras_sample(
            weighted_scores,
            decoded_tokens,
            top_p,
            top_k,
            win_size,
            tau_r,
            generator,
        )


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_model_module(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    monkeypatch.setenv("VLLM_MUSA_COSYVOICE3_SAMPLER_MODE", mode)

    for package in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        "vllm_omni.model_executor.models.cosyvoice3",
        "vllm",
        "vllm.logger",
        "vllm_musa",
        "vllm_musa.omni",
    ):
        fake_package = _install_fake_module(monkeypatch, package)
        fake_package.__path__ = []

    base_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.cosyvoice3.cosyvoice3",
    )
    base_module.CosyVoice3Model = _FakeOmniCosyVoice3Model

    utils_module = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.cosyvoice3.utils",
    )
    utils_module.ras_sample = _reference_ras_sample

    logger_module = sys.modules["vllm.logger"]
    logger_module.init_logger = logging.getLogger

    gpu_ar_module = _install_fake_module(monkeypatch, "vllm_musa.omni.gpu_ar")
    gpu_ar_module.HOST_SAMPLING_SCALARS = HOST_SCALARS
    gpu_ar_module.NO_FREQUENCY_PRESENCE_PENALTIES = NO_FREQUENCY_PRESENCE

    module_name = f"test_cosyvoice3_model_{mode}_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "recent_tokens",
    [[], [0]],
    ids=["ordinary", "repeat_fallback"],
)
def test_sync_free_ras_matches_reference_and_generator_state(
    monkeypatch: pytest.MonkeyPatch,
    recent_tokens: list[int],
) -> None:
    module = _load_model_module(monkeypatch, "sync_free")
    scores = torch.tensor([10.0, 0.0, -1.0, -2.0])
    reference_generator = torch.Generator().manual_seed(20260723)
    candidate_generator = torch.Generator().manual_seed(20260723)

    expected = _reference_ras_sample(
        scores.clone(),
        recent_tokens,
        top_p=0.9,
        top_k=4,
        win_size=10,
        tau_r=0.1,
        generator=reference_generator,
    )
    actual = module.CosyVoice3Model._ras_sample_one(
        scores.clone(),
        recent_tokens,
        top_p=0.9,
        top_k=4,
        win_size=10,
        tau_r=0.1,
        generator=candidate_generator,
    )

    assert actual == expected
    assert torch.equal(
        torch.rand(8, generator=candidate_generator),
        torch.rand(8, generator=reference_generator),
    )


def test_sync_free_reads_host_scalars_and_preserves_empty_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_model_module(monkeypatch, "sync_free")
    model = module.CosyVoice3Model()
    temperature = torch.tensor([9.0])
    top_p = torch.tensor([9.0])
    top_k = torch.tensor([9])
    model._vllm_musa_sampling_metadata = SimpleNamespace(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        **{HOST_SCALARS: ([1.0], [0.8], [25])},
    )

    assert model._req_scalar(temperature, 0, 1.0) == 1.0
    assert model._req_scalar(top_p, 0, 1.0) == pytest.approx(0.8)
    assert model._req_scalar(top_k, 0, 0) == 25
    assert model._req_scalar(None, 0, 0.5) == 0.5
    assert model._req_scalar(torch.empty(0), 0, 7) == 7


def test_baseline_mode_retains_device_scalar_and_reference_ras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_model_module(monkeypatch, "baseline")
    model = module.CosyVoice3Model()
    device_value = torch.tensor([0.75])
    model._vllm_musa_sampling_metadata = SimpleNamespace(
        temperature=device_value,
        **{HOST_SCALARS: ([0.25], [0.8], [25])},
    )

    assert model._req_scalar(device_value, 0, 1.0) == pytest.approx(0.75)

    scores = torch.tensor([10.0, 0.0, -1.0])
    reference_generator = torch.Generator().manual_seed(17)
    baseline_generator = torch.Generator().manual_seed(17)
    expected = _reference_ras_sample(
        scores.clone(), [0], 0.9, 3, 10, 0.1, reference_generator
    )
    actual = module.CosyVoice3Model._ras_sample_one(
        scores.clone(),
        [0],
        top_p=0.9,
        top_k=3,
        win_size=10,
        tau_r=0.1,
        generator=baseline_generator,
    )

    assert actual == expected
    assert torch.equal(
        torch.rand(8, generator=baseline_generator),
        torch.rand(8, generator=reference_generator),
    )


def test_ras_eligibility_uses_host_flag_and_falls_back_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_model_module(monkeypatch, "sync_free")
    model = module.CosyVoice3Model()
    model.model_stage = "cosyvoice3_talker"
    base_metadata = {
        "max_num_logprobs": None,
        "temperature": torch.tensor([1.0]),
        "bad_words_token_ids": {},
    }

    eligible = SimpleNamespace(
        **base_metadata,
        **{NO_FREQUENCY_PRESENCE: True},
    )
    penalized = SimpleNamespace(
        **base_metadata,
        **{NO_FREQUENCY_PRESENCE: False},
    )
    bad_words = SimpleNamespace(
        **{**base_metadata, "bad_words_token_ids": {0: [[1]]}},
        **{NO_FREQUENCY_PRESENCE: True},
    )

    assert model._cosyvoice3_ras_enabled(eligible) is True
    assert model._cosyvoice3_ras_enabled(penalized) is False
    assert model._cosyvoice3_ras_enabled(bad_words) is False

    model.base_ras_enabled = True
    assert model._cosyvoice3_ras_enabled(SimpleNamespace()) is True


def test_invalid_mode_falls_back_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_model_module(monkeypatch, "unsupported")
    assert module._SAMPLER_MODE == "baseline"
