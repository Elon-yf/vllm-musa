"""Tests for MUSA csrc JIT kernels ported from SGLang."""

import os
from dataclasses import dataclass

import pytest
import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("VLLM_MUSA_JIT_CACHE_DIR", "/tmp/vllm_musa_pytest_jit_cache")
os.environ.setdefault("VLLM_MUSA_ARCH_LIST", "31")


def _require_musa() -> None:
    pytest.importorskip("torch_musa")
    pytest.importorskip("torchada")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)


@pytest.fixture(scope="module", autouse=True)
def _musa_device():
    _require_musa()


def _sync() -> None:
    torch.musa.synchronize()


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    _sync()
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), (
        (actual.float() - expected.float()).abs().max().item()
    )


def _rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gemma: bool = False,
) -> torch.Tensor:
    scale_weight = weight.float() + (1.0 if gemma else 0.0)
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * scale_weight).to(x.dtype)


def _fused_add_rmsnorm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_fp32 = x.float() + residual.float()
    residual_out = residual_fp32.to(x.dtype)
    scale_weight = weight.float() + (1.0 if gemma else 0.0)
    normalized = residual_fp32 * torch.rsqrt(
        residual_fp32.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * scale_weight).to(x.dtype), residual_out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_csrc_rmsnorm_kernels_match_reference(dtype: torch.dtype) -> None:
    from vllm_musa.jit_kernel.csrc.norm import (
        gemma_rmsnorm,
        rmsnorm,
    )

    torch.manual_seed(123)
    device = torch.device("musa")
    eps = 1e-6
    x = torch.randn((5, 128), device=device, dtype=dtype)
    weight = torch.randn((128,), device=device, dtype=dtype)

    _assert_close(rmsnorm(x, weight, eps), _rmsnorm_ref(x, weight, eps))
    _assert_close(
        gemma_rmsnorm(x, weight, eps),
        _rmsnorm_ref(x, weight, eps, gemma=True),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("shape", [(1, 2048), (17, 2048), (100, 5120)])
def test_csrc_fused_add_rmsnorm_matches_fp32_sum_reference(
    dtype: torch.dtype,
    gemma: bool,
    shape: tuple[int, int],
) -> None:
    from vllm_musa.jit_kernel.csrc.norm import fused_add_rmsnorm

    torch.manual_seed(321)
    device = torch.device("musa")
    eps = 1e-6
    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn((shape[-1],), device=device, dtype=dtype)
    expected, expected_residual = _fused_add_rmsnorm_ref(
        x, residual, weight, eps, gemma=gemma
    )

    output, residual_out = fused_add_rmsnorm(x, residual, weight, eps, gemma=gemma)

    assert output.data_ptr() == x.data_ptr()
    assert residual_out.data_ptr() == residual.data_ptr()
    _assert_close(output, expected)
    _assert_close(residual_out, expected_residual, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("rows", "hidden", "dtype", "expected"),
    [
        (1, 2048, torch.bfloat16, False),
        (1, 5120, torch.bfloat16, False),
        (63, 5120, torch.bfloat16, False),
        (64, 5120, torch.bfloat16, True),
        (128, 5120, torch.bfloat16, True),
        (0, 5120, torch.bfloat16, False),
        (64, 5120, torch.float16, False),
    ],
)
def test_gemma_fused_add_rmsnorm_dispatch_is_h5120_only(
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    expected: bool,
) -> None:
    from vllm_musa.model_executor.layers.layernorm import (
        _can_use_musa_jit_fused_add_rmsnorm,
    )

    device = torch.device("musa")
    x = torch.empty((rows, hidden), device=device, dtype=dtype)
    residual = torch.empty_like(x)
    weight = torch.empty((hidden,), device=device, dtype=dtype)

    assert _can_use_musa_jit_fused_add_rmsnorm(x, residual, weight) is expected


def test_gemma_fused_add_rmsnorm_dispatch_rejects_unsafe_layouts() -> None:
    from vllm_musa.model_executor.layers.layernorm import (
        _can_use_musa_jit_fused_add_rmsnorm,
    )

    device = torch.device("musa")
    weight = torch.empty((5120,), device=device, dtype=torch.bfloat16)
    noncontiguous = torch.empty((5120, 64), device=device, dtype=torch.bfloat16).t()
    contiguous = torch.empty((64, 5120), device=device, dtype=torch.bfloat16)

    assert not _can_use_musa_jit_fused_add_rmsnorm(noncontiguous, noncontiguous, weight)
    assert not _can_use_musa_jit_fused_add_rmsnorm(
        contiguous, contiguous.to(torch.float16), weight
    )


def test_gemma_fused_add_rmsnorm_requires_caller_inplace_contract() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_musa.model_executor.layers.layernorm import (
        MusaGemmaRMSNorm,
    )

    torch.manual_seed(321)
    device = torch.device("musa")
    dtype = torch.bfloat16
    eps = 1e-6
    x = torch.randn((64, 5120), device=device, dtype=dtype)
    residual = torch.randn_like(x)
    x_before = x.clone()
    residual_before = residual.clone()
    expected, expected_residual = _fused_add_rmsnorm_ref(
        x, residual, torch.zeros((5120,), device=device, dtype=dtype), eps, gemma=True
    )

    with set_current_vllm_config(VllmConfig()):
        functional = MusaGemmaRMSNorm(5120, eps=eps).to(device=device, dtype=dtype)
        inplace = MusaGemmaRMSNorm(5120, eps=eps, allow_inplace=True).to(
            device=device, dtype=dtype
        )

    output, residual_out = functional.forward_oot(x, residual)
    assert output.data_ptr() != x.data_ptr()
    assert residual_out.data_ptr() != residual.data_ptr()
    _assert_close(x, x_before, atol=0.0, rtol=0.0)
    _assert_close(residual, residual_before, atol=0.0, rtol=0.0)
    _assert_close(output, expected)
    _assert_close(residual_out, expected_residual, atol=0.0, rtol=0.0)

    output, residual_out = inplace.forward_oot(x, residual)
    assert output.data_ptr() == x.data_ptr()
    assert residual_out.data_ptr() == residual.data_ptr()
    _assert_close(output, expected)
    _assert_close(residual_out, expected_residual, atol=0.0, rtol=0.0)


def _topk_softmax_ref(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gating.float()
    if correction_bias is not None:
        logits = logits + correction_bias.float().unsqueeze(0)
    scores = torch.softmax(logits, dim=-1)
    _, ids = torch.topk(logits, topk, dim=-1)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.float(), ids.int()


def _topk_sigmoid_ref(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(gating.float())
    choice = scores
    if correction_bias is not None:
        choice = choice + correction_bias.float().unsqueeze(0)
    _, ids = torch.topk(choice, topk, dim=-1)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.float(), ids.int()


def _assert_ids_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    _sync()
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_csrc_topk_kernels_match_reference(
    dtype: torch.dtype,
    renormalize: bool,
) -> None:
    from vllm_musa.jit_kernel.csrc.topk import topk_sigmoid, topk_softmax

    torch.manual_seed(789)
    device = torch.device("musa")
    rows = 7
    experts = 128
    topk = 8
    values = torch.linspace(-4.0, 4.0, experts, device=device, dtype=torch.float32)
    perm = torch.randperm(experts, device=device)
    base = torch.empty((experts,), device=device, dtype=torch.float32)
    base[perm] = values
    row_offsets = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    gating = (base.unsqueeze(0) + row_offsets * 0.001).to(dtype)
    bias = torch.randn((experts,), device=device, dtype=torch.float32) * 0.01

    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    topk_softmax(weights, ids, gating, renormalize)
    ref_weights, ref_ids = _topk_softmax_ref(gating, topk, renormalize)
    _assert_ids_equal(ids, ref_ids)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)

    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    topk_softmax(weights, ids, gating, renormalize, correction_bias=bias)
    ref_weights, ref_ids = _topk_softmax_ref(gating, topk, renormalize, bias)
    _assert_ids_equal(ids, ref_ids)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)

    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    topk_sigmoid(weights, ids, gating, renormalize, correction_bias=bias)
    ref_weights, ref_ids = _topk_sigmoid_ref(gating, topk, renormalize, bias)
    _assert_ids_equal(ids, ref_ids)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)


def test_csrc_jit_integration_imports() -> None:
    import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as topk_router
    import vllm_musa.model_executor.layers.layernorm as layernorm
    from vllm_musa.jit_kernel.csrc import rope as rope_kernel

    assert hasattr(layernorm, "MusaRMSNorm")
    assert hasattr(topk_router, "_musa_jit_fused_topk")
    assert rope_kernel is not None


@dataclass(frozen=True)
class RopeShape:
    label: str
    num_heads_per_rank: int
    num_kv_heads_per_rank: int
    head_dim: int
    rotary_dim: int
    num_tokens: int
    is_neox: bool = True
    dtype: torch.dtype = torch.bfloat16


ROPE_SHAPES = [
    RopeShape("eagle3_draft_tp8_decode", 3, 1, 128, 128, 1),
    RopeShape("eagle3_draft_tp8_chain8", 3, 1, 128, 128, 8),
    RopeShape("m25_target_tp8_decode", 6, 1, 128, 128, 1),
    RopeShape("m25_target_tp8_prefill", 6, 1, 128, 128, 4096),
    RopeShape("qwen3_8b_tp8_decode", 4, 1, 128, 128, 1),
    RopeShape("qwen3_8b_tp8_prefill", 4, 1, 128, 128, 1024),
    RopeShape("qwen3_30b_tp2_decode", 16, 4, 128, 128, 1),
    RopeShape("min_heads_1_decode", 1, 1, 128, 128, 1),
    RopeShape("min_heads_2_decode", 2, 1, 128, 128, 1),
]


def _ensure_rope_op_registered() -> None:
    import vllm_musa  # noqa: F401
    from vllm_musa.jit_kernel.csrc import rope as _rope_module  # noqa: F401


def _make_rope_inputs(
    shape: RopeShape,
    max_pos: int = 8192,
    device: str = "cuda",
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    _ensure_rope_op_registered()
    torch.manual_seed(0)
    positions = torch.arange(shape.num_tokens, dtype=torch.int64, device=device)
    query = torch.randn(
        shape.num_tokens,
        shape.num_heads_per_rank * shape.head_dim,
        dtype=shape.dtype,
        device=device,
    )
    key = torch.randn(
        shape.num_tokens,
        shape.num_kv_heads_per_rank * shape.head_dim,
        dtype=shape.dtype,
        device=device,
    )

    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(
            head_size=shape.head_dim,
            rotary_dim=shape.rotary_dim,
            max_position_embeddings=max(max_pos, shape.num_tokens),
            base=10000.0,
            is_neox_style=shape.is_neox,
            dtype=shape.dtype,
            init_cache=True,
        ).to(device)
    return positions, query, key, rope


def _native_rope(rope, positions, query, key):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    q = query.clone()
    k = key.clone()
    out_q, out_k = RotaryEmbedding.forward_static(
        positions,
        q,
        k,
        rope.head_size,
        rope.rotary_dim,
        rope.cos_sin_cache,
        rope.is_neox_style,
    )
    return out_q, out_k


def _jit_rope(
    positions,
    query,
    key,
    head_size: int,
    cos_sin_cache,
    is_neox: bool,
):
    _ensure_rope_op_registered()
    q = query.clone()
    k = key.clone()
    csc = cos_sin_cache.to(query.device, dtype=query.dtype)
    torch.ops.vllm.musa_rotary_embedding(positions, q, k, head_size, csc, is_neox)
    return q, k


def _assert_rope_close(q_ref, k_ref, q_test, k_test) -> None:
    _assert_close(q_test, q_ref, atol=1e-2, rtol=1e-2)
    _assert_close(k_test, k_ref, atol=1e-2, rtol=1e-2)


def _check_rope_eager_parity(shape: RopeShape) -> None:
    positions, query, key, rope = _make_rope_inputs(shape)
    q_native, k_native = _native_rope(rope, positions, query, key)
    q_jit, k_jit = _jit_rope(
        positions,
        query,
        key,
        shape.head_dim,
        rope.cos_sin_cache,
        shape.is_neox,
    )
    _assert_rope_close(q_native, k_native, q_jit, k_jit)


def _check_rope_captured_jit_parity(shape: RopeShape) -> None:
    positions, query, key, rope = _make_rope_inputs(shape)
    q_native, k_native = _native_rope(rope, positions, query, key)
    q_buf = query.clone()
    k_buf = key.clone()
    csc = rope.cos_sin_cache.to(query.device, dtype=query.dtype)

    def fn() -> None:
        torch.ops.vllm.musa_rotary_embedding(
            positions,
            q_buf,
            k_buf,
            shape.head_dim,
            csc,
            shape.is_neox,
        )

    q_buf.copy_(query)
    k_buf.copy_(key)
    fn()
    q_buf.copy_(query)
    k_buf.copy_(key)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.current_stream().wait_stream(stream)

    q_buf.copy_(query)
    k_buf.copy_(key)
    graph.replay()
    torch.cuda.synchronize()
    _assert_rope_close(q_native, k_native, q_buf, k_buf)


def _check_rope_multi_replay(shape: RopeShape, n_replays: int = 32) -> None:
    positions, query, key, rope = _make_rope_inputs(shape)
    q_buf = query.clone()
    k_buf = key.clone()
    csc = rope.cos_sin_cache.to(query.device, dtype=query.dtype)

    def fn() -> None:
        torch.ops.vllm.musa_rotary_embedding(
            positions,
            q_buf,
            k_buf,
            shape.head_dim,
            csc,
            shape.is_neox,
        )

    fn()
    q_buf.copy_(query)
    k_buf.copy_(key)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.current_stream().wait_stream(stream)

    for _ in range(n_replays):
        q_buf.copy_(query)
        k_buf.copy_(key)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.isnan(q_buf).any()
        assert not torch.isnan(k_buf).any()
        assert not torch.isinf(q_buf).any()
        assert not torch.isinf(k_buf).any()


@pytest.mark.parametrize("shape", ROPE_SHAPES, ids=lambda shape: shape.label)
def test_csrc_rope_kernel_matches_native_and_cudagraph(shape: RopeShape) -> None:
    _check_rope_eager_parity(shape)
    _check_rope_captured_jit_parity(shape)
    _check_rope_multi_replay(shape)
