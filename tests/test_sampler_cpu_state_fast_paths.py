# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001
"""CPU-state contracts for MUSA V2 sampling fast paths."""

from types import SimpleNamespace

import numpy as np
import pytest
import torchada  # noqa: F401
import torch

from vllm_musa.v1.sample import topk_topp_sampler as sampler


requires_musa = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


def _state_array(
    values: list[int] | list[float], dtype: torch.dtype
) -> SimpleNamespace:
    numpy_dtype = np.int32 if dtype == torch.int32 else np.float32
    return SimpleNamespace(
        np=np.asarray(values, dtype=numpy_dtype),
        gpu=torch.tensor(values, dtype=dtype),
    )


def _gumbel_gate_sampler(
    rows: int,
    *,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    min_p: float = 0.05,
    all_seeded: bool = True,
    num_speculative_tokens: int = 1,
    use_fp64_gumbel: bool = False,
    logprobs_mode: str = "raw_logprobs",
) -> SimpleNamespace:
    states = SimpleNamespace(
        temperature=_state_array([temperature] * rows, torch.float32),
        top_k=_state_array([top_k] * rows, torch.int32),
        top_p=_state_array([top_p] * rows, torch.float32),
        min_p=_state_array([min_p] * rows, torch.float32),
        has_user_seed=np.full(rows, all_seeded, dtype=np.bool_),
    )
    return SimpleNamespace(
        sampling_states=states,
        num_speculative_tokens=num_speculative_tokens,
        use_fp64_gumbel=use_fp64_gumbel,
        logprobs_mode=logprobs_mode,
    )


class _FakeGenerator:
    def __init__(
        self,
        seed: int,
        offset: int = 0,
        device: torch.device | None = None,
        fail_offset: int | None = None,
    ) -> None:
        self.seed = seed
        self.offset = offset
        self.device = device or torch.device("cpu")
        self.fail_offset = fail_offset

    def initial_seed(self) -> int:
        return self.seed

    def get_offset(self) -> int:
        return self.offset

    def set_offset(self, offset: int) -> None:
        if offset == self.fail_offset:
            raise RuntimeError("injected set_offset failure")
        self.offset = offset


def _legacy_gumbel_metadata(rows: int, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_logprobs=kwargs.get("max_num_logprobs"),
        logprob_token_ids=kwargs.get("logprob_token_ids"),
        all_random=kwargs.get("all_random", True),
        top_k=kwargs.get("top_k", torch.full((rows,), 50, dtype=torch.int32)),
        top_p=kwargs.get("top_p"),
        spec_token_ids=kwargs.get("spec_token_ids", [[] for _ in range(rows)]),
        generators=kwargs.get(
            "generators", {row: _FakeGenerator(60043 + row) for row in range(rows)}
        ),
    )


def test_qwen_legacy_gumbel_gate_and_generator_handoff(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_QWEN_LEGACY_GUMBEL", "1")
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 4
    logits = torch.randn((rows, 248320))
    metadata = _legacy_gumbel_metadata(rows)

    assert sampler.can_use_qwen_legacy_gumbel(
        logits, metadata, "raw_logprobs", None, False
    )
    metadata.generators = dict(reversed(tuple(metadata.generators.items())))
    assert sampler.can_use_qwen_legacy_gumbel(
        logits, metadata, "raw_logprobs", None, False
    )
    state = sampler.get_qwen_legacy_generator_state(metadata.generators, rows)
    assert state == ([60043, 60044, 60045, 60046], [0, 0, 0, 0])

    captured = {}

    def fake_gumbel_sample(
        processed_logits,
        mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        captured.update(
            processed_logits=processed_logits,
            mapping=mapping,
            temperature=temperature,
            seeds=seeds,
            positions=positions,
            kwargs=kwargs,
        )
        return torch.arange(rows, dtype=torch.int64)

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", lambda tensor, *_args: tensor)
    monkeypatch.setattr(
        sampler.vllm_worker_sampler, "gumbel_sample", fake_gumbel_sample
    )
    sampled = sampler.sample_qwen_legacy_gumbel(
        logits, metadata.generators, metadata.top_k, state
    )

    assert sampled.tolist() == [0, 1, 2, 3]
    assert captured["mapping"].tolist() == [0, 1, 2, 3]
    assert captured["temperature"].tolist() == [1.0] * rows
    assert captured["seeds"].tolist() == [60043, 60044, 60045, 60046]
    assert captured["positions"].tolist() == [0] * rows
    assert captured["kwargs"] == {"apply_temperature": False, "use_fp64": False}
    assert [generator.get_offset() for generator in metadata.generators.values()] == [
        4
    ] * rows


def test_qwen_legacy_gumbel_gate_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_QWEN_LEGACY_GUMBEL", "1")
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    logits = torch.randn((4, 248320))

    assert sampler.can_use_qwen_legacy_gumbel(
        torch.randn((1, 248320)),
        _legacy_gumbel_metadata(1),
        "raw_logprobs",
        None,
        False,
    )
    assert not sampler.can_use_qwen_legacy_gumbel(
        torch.randn((4, 151936)),
        _legacy_gumbel_metadata(4),
        "raw_logprobs",
        None,
        False,
    )
    assert not sampler.can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, max_num_logprobs=0),
        "raw_logprobs",
        None,
        False,
    )
    assert not sampler.can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, generators={0: _FakeGenerator(1)}),
        "raw_logprobs",
        None,
        False,
    )
    assert not sampler.can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, spec_token_ids=[[1], [], [], []]),
        "raw_logprobs",
        None,
        False,
    )
    assert not sampler.can_use_qwen_legacy_gumbel(
        logits, _legacy_gumbel_metadata(4), "raw_logprobs", None, True
    )
    invalid = _legacy_gumbel_metadata(4)
    invalid.generators[0].offset = 2
    assert sampler.get_qwen_legacy_generator_state(invalid.generators, 4) is None


def test_qwen_legacy_gumbel_rolls_back_partial_generator_advance(
    monkeypatch,
) -> None:
    rows = 4
    logits = torch.randn((rows, 248320))
    generators = {
        row: _FakeGenerator(
            60043 + row,
            device=logits.device,
            fail_offset=4 if row == 1 else None,
        )
        for row in range(rows)
    }
    state = sampler.get_qwen_legacy_generator_state(generators, rows)
    monkeypatch.setattr(sampler, "_apply_top_k_top_p", lambda tensor, *_args: tensor)
    monkeypatch.setattr(
        sampler.vllm_worker_sampler,
        "gumbel_sample",
        lambda *_args, **_kwargs: torch.arange(rows, dtype=torch.int64),
    )

    with pytest.raises(RuntimeError, match="Failed to advance legacy MUSA generators"):
        sampler.sample_qwen_legacy_gumbel(
            logits,
            generators,
            torch.full((rows,), 50, dtype=torch.int32),
            state,
        )

    assert [generator.get_offset() for generator in generators.values()] == [0] * rows


def test_qwen_legacy_gumbel_gate_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MUSA_QWEN_LEGACY_GUMBEL", raising=False)
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)

    assert not sampler.can_use_qwen_legacy_gumbel(
        torch.randn((4, 248320)),
        _legacy_gumbel_metadata(4),
        "raw_logprobs",
        None,
        False,
    )


def test_qwen_v2_gumbel_gate_accepts_only_exact_contract(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_QWEN_V2_GUMBEL", "1")
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 4
    logits = torch.randn((rows, 151936))
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)
    pos = mapping.clone()

    assert sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows), logits, mapping, mapping_np, pos, False
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, top_p=0.9),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, all_seeded=False),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, num_speculative_tokens=2),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows), logits, mapping, mapping_np, pos, True
    )


def test_qwen_v2_gumbel_gate_rejects_non_qwen_vocab(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_QWEN_V2_GUMBEL", "1")
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 4
    mapping = torch.arange(rows, dtype=torch.int64)

    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows),
        torch.randn((rows, 131072)),
        mapping,
        np.arange(rows, dtype=np.int64),
        mapping,
        False,
    )


def test_qwen_v2_gumbel_gate_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MUSA_QWEN_V2_GUMBEL", raising=False)
    rows = 1
    mapping = torch.zeros(rows, dtype=torch.int64)

    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows),
        torch.randn((rows, 151936)),
        mapping,
        np.zeros(rows, dtype=np.int64),
        mapping,
        False,
    )


def test_uniform_active_min_p_uses_cpu_state() -> None:
    assert sampler._uniform_active_min_p(
        np.asarray([0.05], dtype=np.float32)
    ) == pytest.approx(0.05)
    assert sampler._uniform_active_min_p(
        np.asarray([0.05, 0.05], dtype=np.float32)
    ) == pytest.approx(0.05)
    assert sampler._uniform_active_min_p(np.asarray([], dtype=np.float32)) is None
    assert (
        sampler._uniform_active_min_p(np.asarray([0.0, 0.0], dtype=np.float32)) is None
    )
    assert (
        sampler._uniform_active_min_p(np.asarray([0.05, 0.1], dtype=np.float32)) is None
    )


def test_uniform_top_k_threshold_preserves_ties() -> None:
    logits = torch.arange(64, dtype=torch.float32).repeat(2, 1)
    logits[:, 12:17] = 14.0
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_only(
        logits.clone(), torch.full((2,), 50, dtype=torch.int32)
    )

    actual = sampler._apply_top_k_top_p(logits.clone(), 50, None)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())
    assert actual.isfinite().sum(dim=1).tolist() == [52, 52]


def test_tensor_top_k_top_p_prefilter_matches_upstream() -> None:
    torch.manual_seed(60046)
    logits = torch.randn((4, 256), dtype=torch.float32)
    top_k = torch.tensor([20, 50, 7, 49], dtype=torch.int32)
    top_p = torch.tensor([0.75, 0.8, 0.9, 0.95], dtype=torch.float32)
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.clone(), top_k, top_p
    )

    actual_input = logits.clone()
    actual = sampler._apply_top_k_top_p_musa_topk_prefilter(actual_input, top_k, top_p)

    assert actual.data_ptr() == actual_input.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())


def test_seeded_filters_reuse_processed_logits_only_when_allowed() -> None:
    sampling_states = SimpleNamespace(
        vocab_size=64,
        top_k=_state_array([50, 50], torch.int32),
        top_p=_state_array([1.0, 1.0], torch.float32),
        min_p=_state_array([0.0, 0.0], torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)
    processed = torch.randn((2, 64), dtype=torch.float32)
    original = processed.clone()

    reused = sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        processed,
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )
    assert reused.data_ptr() == processed.data_ptr()
    assert not torch.equal(processed, original)

    preserved_input = original.clone()
    copied = sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        preserved_input,
        mapping,
        mapping_np,
        preserve_processed_logits=True,
    )
    assert copied.data_ptr() != preserved_input.data_ptr()
    torch.testing.assert_close(preserved_input, original, rtol=0, atol=0)
    torch.testing.assert_close(copied, reused, rtol=0, atol=0)


def test_seeded_top_p_keeps_tensor_top_k_fallback(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    sampling_states = SimpleNamespace(
        vocab_size=64,
        top_k=_state_array([50, 50], torch.int32),
        top_p=_state_array([0.9, 0.9], torch.float32),
        min_p=_state_array([0.0, 0.0], torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((2, 64)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)
    assert isinstance(captured["top_p"], torch.Tensor)


def test_seeded_large_uniform_top_k_top_p_uses_cpu_scalar(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 16
    vocab_size = 248320
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([0.9] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert captured["top_k"] == 50
    assert isinstance(captured["top_p"], torch.Tensor)


def test_seeded_small_batch_qwen_vocab_uses_cpu_scalar(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 4
    vocab_size = 151936
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([1.0] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert captured == {"top_k": 50, "top_p": None}


def test_non_qwen_vocab_keeps_tensor_top_k_path(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 4
    vocab_size = 131072
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([1.0] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)


def test_seeded_bs1_active_top_p_keeps_tensor_path(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    vocab_size = 151936
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50], torch.int32),
        top_p=_state_array([0.9], torch.float32),
        min_p=_state_array([0.0], torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.zeros(1, dtype=torch.int64)
    mapping_np = np.zeros(1, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((1, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)
    assert isinstance(captured["top_p"], torch.Tensor)


def test_uniform_top_k_top_p_prefilter_matches_upstream() -> None:
    torch.manual_seed(60043)
    logits = torch.randn((4, 256), dtype=torch.float32)
    top_p = torch.tensor([0.75, 0.8, 0.9, 0.95], dtype=torch.float32)
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.clone(), torch.full((4,), 50, dtype=torch.int32), top_p
    )

    actual_input = logits.clone()
    actual = sampler._apply_top_k_top_p_musa_uniform_k_prefilter(
        actual_input, 50, top_p
    )

    assert actual.data_ptr() == actual_input.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())


def test_uniform_top_k_top_p_prefilter_falls_back_for_boundary_ties() -> None:
    logits = torch.arange(64, dtype=torch.float32).repeat(2, 1)
    logits[:, 12:17] = 14.0
    top_p = torch.tensor([0.9, 0.95], dtype=torch.float32)
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.clone(), torch.full((2,), 50, dtype=torch.int32), top_p
    )

    actual = sampler._apply_top_k_top_p_musa_uniform_k_prefilter(
        logits.clone(), 50, top_p
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())


def test_unseeded_uniform_min_p_reaches_sampler_as_scalar(monkeypatch) -> None:
    captured = {}

    def fake_sample_from_logits(logits, top_k, top_p, min_p):
        captured.update(top_k=top_k, top_p=top_p, min_p=min_p)
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(sampler, "sample_from_logits", fake_sample_from_logits)
    sampling_states = SimpleNamespace(
        vocab_size=128,
        top_k=_state_array([128, 128], torch.int32),
        top_p=_state_array([1.0, 1.0], torch.float32),
        min_p=_state_array([0.05, 0.05], torch.float32),
    )
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)

    sampler.sample_worker_logits(
        torch.randn((2, 128)), sampling_states, mapping, mapping_np
    )

    assert captured == {"top_k": None, "top_p": None, "min_p": pytest.approx(0.05)}


def test_unseeded_mixed_min_p_keeps_tensor_fallback(monkeypatch) -> None:
    captured = {}

    def fake_sample_from_logits(logits, top_k, top_p, min_p):
        captured["min_p"] = min_p
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(sampler, "sample_from_logits", fake_sample_from_logits)
    sampling_states = SimpleNamespace(
        vocab_size=128,
        top_k=_state_array([128, 128], torch.int32),
        top_p=_state_array([1.0, 1.0], torch.float32),
        min_p=_state_array([0.05, 0.1], torch.float32),
    )
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)

    sampler.sample_worker_logits(
        torch.randn((2, 128)), sampling_states, mapping, mapping_np
    )

    assert isinstance(captured["min_p"], torch.Tensor)
    torch.testing.assert_close(
        captured["min_p"], torch.tensor([0.05, 0.1]), rtol=0, atol=0
    )


@requires_musa
@pytest.mark.parametrize("vocab_size", [151936, 248320])
@pytest.mark.parametrize("rows", [1, 4, 16, 64])
def test_uniform_min_p_scalar_preserves_tokens_and_generator_state(
    vocab_size: int, rows: int
) -> None:
    torch.manual_seed(60043 + rows)
    probs = torch.softmax(
        torch.randn((rows, vocab_size), device="musa", dtype=torch.float32), dim=-1
    )
    probs = sampler._ops.top_k_renorm_probs(probs, 50)
    scalar_generator = torch.Generator(device="musa").manual_seed(5678)
    tensor_generator = torch.Generator(device="musa").manual_seed(5678)

    scalar_tokens = sampler._ops.min_p_sampling_from_probs(
        probs, 0.05, generator=scalar_generator
    )
    tensor_tokens = sampler._ops.min_p_sampling_from_probs(
        probs,
        torch.full((rows,), 0.05, device="musa", dtype=torch.float32),
        generator=tensor_generator,
    )
    torch.musa.synchronize()

    assert torch.equal(scalar_tokens, tensor_tokens)
    assert torch.equal(scalar_generator.get_state(), tensor_generator.get_state())
