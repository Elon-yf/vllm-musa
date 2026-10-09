from pathlib import Path


ROOT = Path(__file__).parents[2]
PATCH = ROOT / "vllm_musa/patches/series/0169-MUSA-PaddleOCR-VL-model-local-FA-and-RoPE.patch"
HELPER = ROOT / "vllm_musa/optimization_contract/paddleocr_vl.py"


def test_paddle_fast_path_is_model_local_and_narrow():
    patch = PATCH.read_text(encoding="utf-8")
    helper = HELPER.read_text(encoding="utf-8")
    assert "vllm_musa.optimization_contract.paddleocr_vl" in patch
    assert "paddle_musa_fast_path(vllm_config)" in patch
    assert 'getattr(hf_config, "model_type", None) != "paddleocr_vl"' in helper
    assert 'list((rope_parameters or {}).get("mrope_section", ())) == [16, 24, 24]' in helper
    assert "tests/" not in patch


def test_paddle_path_reuses_existing_kernels_and_keeps_fp32_contract():
    helper = HELPER.read_text(encoding="utf-8")
    assert "vllm_musa.jit_kernel.csrc.rope" in helper
    assert "forward_cuda(positions, query, key)" in helper
    assert "x_work = x if x.dtype == torch.float32 else x.float()" in helper
    assert "output = x_work.clone()" in helper
    assert "rotary_dim = cos.shape[-1] * 2" in helper


def test_paddle_patch_does_not_register_global_rotary_ops():
    patch = PATCH.read_text(encoding="utf-8")
    helper = HELPER.read_text(encoding="utf-8")
    assert "register_oot" not in patch + helper
    assert "direct_register_custom_op" not in patch + helper
    assert "get_default_ir_op_priority" not in patch + helper
