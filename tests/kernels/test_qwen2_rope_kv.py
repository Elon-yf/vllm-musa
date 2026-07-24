# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_musa.kernels import qwen2_rope_kv

requires_musa = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


def _make_exact_inputs(num_tokens: int = 1) -> dict[str, object]:
    packed_qkv = torch.empty((num_tokens, 1152), dtype=torch.bfloat16)
    query = packed_qkv[:, :896].view(num_tokens, 14, 64)
    key = packed_qkv[:, 896:1024].view(num_tokens, 2, 64)
    value = packed_qkv[:, 1024:].view(num_tokens, 2, 64)
    return {
        "query": query,
        "key": key,
        "value": value,
        "positions": torch.arange(num_tokens, dtype=torch.int64),
        "cos_sin_cache": torch.empty((2048, 64), dtype=torch.bfloat16),
        "is_neox": True,
        "key_cache": torch.empty((4, 64, 2, 64), dtype=torch.bfloat16),
        "value_cache": torch.empty((4, 64, 2, 64), dtype=torch.bfloat16),
        "slot_mapping": torch.arange(num_tokens, dtype=torch.int64),
    }


def _allow_cpu_as_musa(monkeypatch) -> None:
    monkeypatch.setattr(qwen2_rope_kv, "_is_musa_tensor", lambda _tensor: True)


def test_qwen2_rope_kv_exact_gate_accepts_packed_qkv_views(monkeypatch):
    _allow_cpu_as_musa(monkeypatch)
    inputs = _make_exact_inputs()

    assert qwen2_rope_kv.supports_qwen2_rope_kv_cache(**inputs)


def test_qwen2_rope_kv_gate_rejects_short_slot_mapping(monkeypatch):
    _allow_cpu_as_musa(monkeypatch)
    inputs = _make_exact_inputs()
    inputs["slot_mapping"] = torch.empty(0, dtype=torch.int64)

    assert not qwen2_rope_kv.supports_qwen2_rope_kv_cache(**inputs)


def test_qwen2_rope_kv_gate_rejects_non_neox_and_wrong_shape(monkeypatch):
    _allow_cpu_as_musa(monkeypatch)
    inputs = _make_exact_inputs()
    inputs["is_neox"] = False
    assert not qwen2_rope_kv.supports_qwen2_rope_kv_cache(**inputs)

    inputs = _make_exact_inputs()
    inputs["query"] = torch.empty((1, 16, 64), dtype=torch.bfloat16)
    assert not qwen2_rope_kv.supports_qwen2_rope_kv_cache(**inputs)


def test_qwen2_rope_kv_gate_rejects_non_nhd_cache(monkeypatch):
    _allow_cpu_as_musa(monkeypatch)
    inputs = _make_exact_inputs()
    inputs["key_cache"] = torch.empty((4, 2, 64, 64), dtype=torch.bfloat16).permute(
        0, 2, 1, 3
    )

    assert not qwen2_rope_kv.supports_qwen2_rope_kv_cache(**inputs)


def _cos_sin_cache(device: torch.device) -> torch.Tensor:
    positions = torch.arange(4096, device=device, dtype=torch.float32)[:, None]
    frequency = torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64
    angles = positions / (1_000_000.0**frequency)[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)


@requires_musa
@pytest.mark.parametrize("slot", [-1, 0, 63, 64, 191])
def test_qwen2_rope_kv_kernel_matches_native_path(slot: int):
    from vllm_musa import _custom_ops as musa_ops
    from vllm_musa.jit_kernel.csrc.rope import rotary_embedding

    device = torch.device("musa:0")
    torch.manual_seed(6002921)
    source = torch.randn((1, 1152), device=device, dtype=torch.bfloat16)
    positions = torch.tensor([317], device=device, dtype=torch.int64)
    cos_sin_cache = _cos_sin_cache(device)
    slot_mapping = torch.tensor([slot], device=device, dtype=torch.int64)
    cache_source = torch.randn((3, 64, 2, 64), device=device, dtype=torch.bfloat16)

    def make_case():
        packed = source.clone()
        query = packed[:, :896].view(1, 14, 64)
        key = packed[:, 896:1024].view(1, 2, 64)
        value = packed[:, 1024:].view(1, 2, 64)
        key_cache = cache_source.clone()
        value_cache = cache_source.clone()
        return packed, query, key, value, key_cache, value_cache

    reference = make_case()
    rotary_embedding(
        positions,
        reference[1],
        reference[2],
        64,
        cos_sin_cache,
        True,
    )
    musa_ops.musa_reshape_and_cache_flash_nhd(
        reference[2],
        reference[3],
        reference[4],
        reference[5],
        slot_mapping,
    )

    candidate = make_case()
    assert qwen2_rope_kv.try_qwen2_rope_kv_cache(
        candidate[1],
        candidate[2],
        candidate[3],
        positions,
        cos_sin_cache,
        True,
        candidate[4],
        candidate[5],
        slot_mapping,
    )
    torch.musa.synchronize()

    torch.testing.assert_close(candidate[0], reference[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(candidate[4], reference[4], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(candidate[5], reference[5], rtol=0, atol=0)
