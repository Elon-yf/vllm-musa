from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
logits_processor = pytest.importorskip("vllm.model_executor.layers.logits_processor")
custom_ar = pytest.importorskip(
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce"
)


def _processor(org_vocab_size: int):
    processor = object.__new__(logits_processor.LogitsProcessor)
    processor.org_vocab_size = org_vocab_size
    return processor


def _patch_tp(monkeypatch, world_size: int = 2, rank: int = 0):
    monkeypatch.setattr(
        logits_processor,
        "get_tensor_model_parallel_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(
        logits_processor, "get_tensor_model_parallel_rank", lambda: rank
    )


def test_tp1_logits_skip_collectives(monkeypatch):
    _patch_tp(monkeypatch, world_size=1)
    local_logits = torch.arange(32, dtype=torch.int32).to(torch.bfloat16).view(1, 32)
    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim: pytest.fail("TP1 must skip IPC gather"),
    )

    output = _processor(151936)._musa_all_reduce_logits(local_logits, SimpleNamespace())

    assert output is local_logits


@pytest.mark.parametrize(
    "org_vocab_size,shard_size", [(151936, 75968), (248320, 124160)]
)
def test_qwen_bf16_logits_use_ipc_gather(monkeypatch, org_vocab_size, shard_size):
    _patch_tp(monkeypatch)
    local_logits = (
        torch.arange(shard_size, dtype=torch.int32)
        .to(torch.bfloat16)
        .view(1, shard_size)
    )
    gathered = torch.empty((1, shard_size * 2), dtype=torch.bfloat16)
    calls = []

    def gather(input_tensor, dim):
        calls.append((input_tensor, dim))
        return gathered

    monkeypatch.setattr(custom_ar, "maybe_musa_jit_logits_all_gather", gather)
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_reduce",
        lambda input_tensor: pytest.fail("all-reduce fallback must not run"),
    )

    output = _processor(org_vocab_size)._musa_all_reduce_logits(
        local_logits, SimpleNamespace()
    )
    assert output is gathered
    assert len(calls) == 1
    assert calls[0][0] is local_logits
    assert calls[0][1] == -1


@pytest.mark.parametrize(
    "org_vocab_size,shape,dtype",
    [
        (32000, (4, 16000), torch.bfloat16),
        (248320, (65, 124160), torch.bfloat16),
        (248320, (4, 124160), torch.float32),
    ],
)
def test_unsupported_logits_preserve_all_reduce_fallback(
    monkeypatch, org_vocab_size, shape, dtype
):
    _patch_tp(monkeypatch)
    local_logits = (
        torch.arange(shape[0] * shape[1], dtype=torch.int32).to(dtype).view(shape)
    )
    fallback_inputs = []

    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim: pytest.fail("IPC gather gate must not run"),
    )

    def all_reduce(input_tensor):
        fallback_inputs.append(input_tensor.clone())
        return input_tensor

    monkeypatch.setattr(
        logits_processor, "tensor_model_parallel_all_reduce", all_reduce
    )
    output = _processor(org_vocab_size)._musa_all_reduce_logits(
        local_logits, SimpleNamespace()
    )

    assert output.shape == (*shape[:-1], shape[-1] * 2)
    assert torch.equal(output[..., : shape[-1]], local_logits)
    assert torch.count_nonzero(output[..., shape[-1] :]).item() == 0
    assert len(fallback_inputs) == 1


def test_tp6_logits_preserve_all_reduce_fallback(monkeypatch):
    _patch_tp(monkeypatch, world_size=6, rank=3)
    local_logits = torch.arange(32, dtype=torch.int32).to(torch.bfloat16).view(1, 32)
    monkeypatch.setattr(
        custom_ar,
        "maybe_musa_jit_logits_all_gather",
        lambda input_tensor, dim: pytest.fail("TP6 IPC gather must stay disabled"),
    )
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_reduce",
        lambda input_tensor: input_tensor,
    )

    output = _processor(248320)._musa_all_reduce_logits(local_logits, SimpleNamespace())

    assert output.shape == (1, 32 * 6)
    assert torch.equal(output[:, 32 * 3 : 32 * 4], local_logits)
