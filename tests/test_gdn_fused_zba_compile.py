# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001

import pytest

torchada = pytest.importorskip("torchada")
import torch  # noqa: E402

from vllm_musa.jit_kernel.tilelang.gdn_fused_proj import fused_zba  # noqa: E402
from vllm_musa.jit_kernel.tilelang.layernorm_gated import (  # noqa: E402
    rms_norm_gated,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA",
)


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 16, 64, 4104])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_zba_custom_op_matches_reference(num_tokens: int, dtype: torch.dtype):
    num_heads_qk = 4
    num_heads_v = 8
    head_qk = 128
    head_v = 128
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    generator = torch.Generator(device="cpu").manual_seed(767 + num_tokens)
    mixed_qkvz = torch.randn(
        (num_tokens, qkv_dim + num_heads_v * head_v),
        generator=generator,
        dtype=torch.float32,
    ).to(device="musa", dtype=dtype)
    mixed_ba = torch.randn(
        (num_tokens, num_heads_v * 2),
        generator=generator,
        dtype=torch.float32,
    ).to(device="musa", dtype=dtype)

    z, b, a = fused_zba(
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
    )
    torch.musa.synchronize()

    expected_z = mixed_qkvz[:, qkv_dim:].reshape(num_tokens, num_heads_v, head_v)
    expected_b, expected_a = mixed_ba.chunk(2, dim=-1)
    torch.testing.assert_close(z, expected_z, rtol=0, atol=0)
    torch.testing.assert_close(b, expected_b, rtol=0, atol=0)
    torch.testing.assert_close(a, expected_a, rtol=0, atol=0)
    assert z.is_contiguous()
    assert b.is_contiguous()
    assert a.is_contiguous()


def test_fused_zba_is_opaque_to_dynamo():
    num_heads_qk = 4
    num_heads_v = 8
    head_qk = 128
    head_v = 128
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkvz = torch.randn(
        (4, qkv_dim + num_heads_v * head_v),
        dtype=torch.bfloat16,
        device="musa",
    )
    mixed_ba = torch.randn(
        (4, num_heads_v * 2),
        dtype=torch.bfloat16,
        device="musa",
    )

    def call_fused_zba(x: torch.Tensor, ba: torch.Tensor):
        return fused_zba(
            x,
            ba,
            num_heads_qk,
            num_heads_v,
            head_qk,
            head_v,
        )

    exported = torch._dynamo.export(call_fused_zba)(mixed_qkvz, mixed_ba)
    graph = str(exported.graph_module.graph)
    assert "musa_fused_zba" in graph
    assert "_lru_cache_wrapper" not in graph


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 64, 4104])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_rms_norm_custom_op_matches_reference(
    num_tokens: int, dtype: torch.dtype
):
    hidden_size = 2048
    generator = torch.Generator(device="cpu").manual_seed(1767 + num_tokens)
    x = torch.randn(
        (num_tokens, hidden_size), generator=generator, dtype=torch.float32
    ).to(device="musa", dtype=dtype)
    z = torch.randn(
        (num_tokens, hidden_size), generator=generator, dtype=torch.float32
    ).to(device="musa", dtype=dtype)
    weight = torch.randn((hidden_size,), generator=generator, dtype=torch.float32).to(
        device="musa", dtype=dtype
    )

    actual = rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="silu",
    )
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight.float()
        * torch.nn.functional.silu(z.float())
    ).to(dtype)
    torch.musa.synchronize()

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert actual.is_contiguous()


def test_gated_rms_norm_is_opaque_to_dynamo():
    hidden_size = 2048
    x = torch.randn((4, hidden_size), dtype=torch.bfloat16, device="musa")
    z = torch.randn_like(x)
    weight = torch.randn((hidden_size,), dtype=torch.bfloat16, device="musa")

    def call_gated_norm(x_: torch.Tensor, z_: torch.Tensor, weight_: torch.Tensor):
        return rms_norm_gated(
            x=x_,
            weight=weight_,
            bias=None,
            z=z_,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="silu",
        )

    graphs: list[str] = []

    def capture_backend(graph_module, _inputs):
        graphs.append(str(graph_module.graph))
        # The MUSA export helper cannot execute a graph containing a literal
        # MUSA device in this torch build.  Capture the FX graph and return a
        # shape-compatible stand-in instead.
        return lambda *args: args[0]

    torch.compile(call_gated_norm, backend=capture_backend, fullgraph=True)(
        x, z, weight
    )
    assert graphs
    graph = graphs[0]
    assert "musa_rms_norm_gated" in graph
    assert "_lru_cache_wrapper" not in graph
