# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-contract checks for dense MUSA SwiGLU plus DeepGEMM fusion."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fused_deepgemm_op_uses_shipped_silu_group_quant_kernel():
    source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "kernels"
        / "linear"
        / "scaled_mm"
        / "deep_gemm.py"
    ).read_text()

    assert "def _musa_silu_deepgemm_fp8_op(" in source
    assert "silu_and_mul_per_token_group_fp8_quant(" in source
    assert "fp8_gemm_nt(" in source
    assert '"musa_silu_deepgemm_fp8_op"' in source
    assert "fake_impl=_musa_silu_deepgemm_fp8_op_fake" in source


def test_fusion_matches_the_production_native_swiglu_deepgemm_pair():
    source = (
        ROOT / "vllm_musa" / "compilation" / "passes" / "silu_deepgemm_fusion.py"
    ).read_text()

    assert "MatcherSiluAndMul(enabled=False)" in source
    assert "torch.ops.vllm.musa_deepgemm_fp8_op(" in source
    assert "torch.ops.vllm.musa_silu_deepgemm_fp8_op(" in source
    assert source.count("\n                128,") == 2
    assert source.count("\n                False,") == 2


def test_pass_manager_keeps_the_experimental_scope_narrow():
    source = (
        ROOT / "vllm_musa" / "compilation" / "passes" / "pass_manager.py"
    ).read_text()

    assert "VLLM_MUSA_SILU_DEEPGEMM_FUSION.get()" in source
    assert "VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()" in source
    assert "_is_dense_model(config)" in source
    assert "_use_row_major_activation_scales(False)" in source
    assert "self.pass_config.fuse_act_quant" in source
    assert "self.passes.append(MusaSiluDeepGemmFusionPass(config))" in source
