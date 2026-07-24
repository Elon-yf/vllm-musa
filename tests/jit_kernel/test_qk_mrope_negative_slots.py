# SPDX-License-Identifier: Apache-2.0
"""MUSA checks for negative qk_mrope cache slots."""

import os
from dataclasses import dataclass

import pytest

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("VLLM_MUSA_JIT_CACHE_DIR", "/tmp/vllm_musa_pytest_jit_cache")
os.environ.setdefault("VLLM_MUSA_ARCH_LIST", "31")

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class QkMropeCase:
    label: str
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    mrope_sections: tuple[int, int, int]
    is_interleaved: bool
    gemma: bool


QK_MROPE_CASES = (
    QkMropeCase("qwen3_06b", 16, 8, 128, 128, (64, 0, 0), False, False),
    QkMropeCase("qwen3_8b", 32, 8, 128, 128, (64, 0, 0), False, False),
)


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> None:
    pytest.importorskip("torch_musa")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.musa.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def _qwen3_rmsnorm_rope_reference(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = tensor.float().square().mean(dim=-1, keepdim=True)
    normalized = tensor.float() * torch.rsqrt(variance + eps)
    normalized = (normalized * weight.float()).to(tensor.dtype)

    cos_sin = cos_sin_cache.index_select(0, positions[0])
    rotary_dim = cos_sin.shape[-1]
    half_dim = rotary_dim // 2
    cos = cos_sin[:, None, :half_dim].float()
    sin = cos_sin[:, None, half_dim:].float()
    x = normalized[..., :rotary_dim].float()
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    if rotary_dim == tensor.shape[-1]:
        return rotated.to(tensor.dtype)
    return torch.cat((rotated.to(tensor.dtype), normalized[..., rotary_dim:]), dim=-1)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("case", QK_MROPE_CASES, ids=lambda case: case.label)
def test_qk_mrope_negative_slot_skips_cache_but_preserves_outputs(
    index_dtype: torch.dtype,
    case: QkMropeCase,
) -> None:
    from vllm_musa.jit_kernel.csrc.norm import _qk_mrope_module

    torch.manual_seed(123)
    device = torch.device("musa")
    tokens = 2
    kv_heads = case.kv_heads
    cache_rows = 4
    sentinel = -7.0
    mrope_section_t, mrope_section_h, mrope_section_w = case.mrope_sections

    q = torch.randn(
        (tokens, case.q_heads, case.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (tokens, kv_heads, case.head_dim), device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        (tokens, kv_heads, case.head_dim), device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn((case.head_dim,), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((case.head_dim,), device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int64).repeat(3, 1)
    angles = torch.randn((8, case.rotary_dim // 2), device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    indices = torch.tensor([0, -1], device=device, dtype=index_dtype)

    module = _qk_mrope_module()
    q_ref = _qwen3_rmsnorm_rope_reference(
        q,
        q_weight,
        positions,
        cos_sin_cache,
        1e-6,
    )
    k_ref = _qwen3_rmsnorm_rope_reference(
        k,
        k_weight,
        positions,
        cos_sin_cache,
        1e-6,
    )

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    k_cache = torch.full(
        (cache_rows, kv_heads * case.head_dim),
        sentinel,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.full_like(k_cache, sentinel)
    module.sgl_musa_fused_qk_rmsnorm_mrope_cache_out(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        k_cache,
        v_cache,
        indices,
        True,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        case.is_interleaved,
        1e-6,
        case.gemma,
    )

    _assert_close(q_out, q_ref)
    _assert_close(k_out, k_ref)
    _assert_close(k_cache[0], k_out[0].reshape(-1))
    assert torch.equal(v_cache[0], v[0].reshape(-1))
    assert torch.equal(k_cache[1:], torch.full_like(k_cache[1:], sentinel))
    assert torch.equal(v_cache[1:], torch.full_like(v_cache[1:], sentinel))

    q_cache_only = torch.empty_like(q)
    k_cache_only = torch.full_like(k_cache, sentinel)
    v_cache_only = torch.full_like(v_cache, sentinel)
    module.sgl_musa_fused_qk_rmsnorm_mrope_cache(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_cache_only,
        k_cache_only,
        v_cache_only,
        indices,
        True,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        case.is_interleaved,
        1e-6,
        case.gemma,
    )

    _assert_close(q_cache_only, q_ref)
    _assert_close(k_cache_only[0], k_ref[0].reshape(-1))
    assert torch.equal(v_cache_only[0], v[0].reshape(-1))
    assert torch.equal(k_cache_only[1:], torch.full_like(k_cache_only[1:], sentinel))
    assert torch.equal(v_cache_only[1:], torch.full_like(v_cache_only[1:], sentinel))
