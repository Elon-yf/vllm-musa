# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S5000 validation for dense SwiGLU plus FP8 DeepGEMM fusion."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
pytest.importorskip("torch_musa")


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> Iterator[None]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)

    from vllm.platforms import current_platform

    import vllm_musa

    if not current_platform.is_device_capability((3, 1)):
        pytest.skip("the fused SiLU group-quant kernel requires S5000/mp31")
    vllm_musa.register_custom_ops()
    yield


def _quant_outputs(m: int, hidden: int):
    q = torch.empty((m, hidden), device="musa", dtype=torch.float8_e4m3fn)
    s = torch.empty((m, hidden // 128), device="musa", dtype=torch.float32)
    return q, s


@pytest.mark.parametrize("m", [1, 4, 4104])
def test_fused_quant_is_bit_exact_for_qwen_shapes(m: int) -> None:
    hidden = 12288
    torch.manual_seed(759000 + m)
    x = torch.randn((m, hidden * 2), device="musa", dtype=torch.bfloat16)
    reference_q, reference_s = _quant_outputs(m, hidden)
    fused_q, fused_s = _quant_outputs(m, hidden)

    activated = F.swish_glu(x)
    torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
        activated,
        reference_q,
        reference_s,
        128,
        1e-10,
        -448.0,
        448.0,
    )
    torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
        x,
        fused_q,
        fused_s,
        128,
        1e-10,
        -448.0,
        448.0,
    )
    torch.musa.synchronize()

    assert torch.equal(
        reference_q.view(torch.uint8).cpu(), fused_q.view(torch.uint8).cpu()
    )
    assert torch.equal(
        reference_s.view(torch.int32).cpu(), fused_s.view(torch.int32).cpu()
    )


def _small_deepgemm_inputs():
    torch.manual_seed(759100)
    x = torch.randn((4, 512), device="musa", dtype=torch.bfloat16)
    weight = torch.randn((256, 256), device="musa", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.ones((2, 2), device="musa", dtype=torch.float32)
    return x, weight, weight_scale


def test_pass_manager_qualname_resolves() -> None:
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_musa.compilation.passes import MusaPostGradPassManager
    from vllm_musa.platform import MUSAPlatformBase

    resolved = resolve_obj_by_qualname(MUSAPlatformBase.get_pass_manager_cls())
    assert resolved is MusaPostGradPassManager


@pytest.mark.parametrize("is_moe, expected", [(False, True), (True, False)])
def test_dense_model_gate_uses_model_config_api(is_moe: bool, expected: bool) -> None:
    from vllm_musa.compilation.passes.pass_manager import _is_dense_model

    config = SimpleNamespace(model_config=SimpleNamespace(is_model_moe=lambda: is_moe))
    assert _is_dense_model(config) is expected


def test_custom_op_schema_fake_and_aot_dynamic() -> None:
    x, weight, weight_scale = _small_deepgemm_inputs()
    op = torch.ops.vllm.musa_silu_deepgemm_fp8_op.default

    torch.library.opcheck(
        op,
        (x, weight, weight_scale, 128, False),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def test_rewrites_the_native_production_graph() -> None:
    from torch.fx.experimental.proxy_tensor import make_fx
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        PassConfig,
        VllmConfig,
        set_current_vllm_config,
    )

    from vllm_musa.compilation.passes.silu_deepgemm_fusion import (
        MusaSiluDeepGemmFusionPass,
    )

    x, weight, weight_scale = _small_deepgemm_inputs()

    def model(x, weight, weight_scale):
        d = x.shape[-1] // 2
        activated = F.silu(x[..., :d]) * x[..., d:]
        return torch.ops.vllm.musa_deepgemm_fp8_op(
            activated, weight, weight_scale, 128, False
        )

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="eager",
            custom_ops=["all"],
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        )
    )
    expected = model(x, weight, weight_scale)
    with set_current_vllm_config(config, check_compile=False):
        graph_module = make_fx(model, tracing_mode="fake")(x, weight, weight_scale)
        fusion = MusaSiluDeepGemmFusionPass(config)
        targets_before = {node.target for node in graph_module.graph.nodes}
        fusion(graph_module.graph)
        graph_module.recompile()

    targets_after = {node.target for node in graph_module.graph.nodes}
    assert torch.ops.vllm.musa_deepgemm_fp8_op.default in targets_before
    assert torch.ops.vllm.musa_silu_deepgemm_fp8_op.default not in targets_before
    assert torch.ops.vllm.musa_deepgemm_fp8_op.default not in targets_after
    assert torch.ops.vllm.musa_silu_deepgemm_fp8_op.default in targets_after
    assert fusion.matched_count == 1

    actual = graph_module(x, weight, weight_scale)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
