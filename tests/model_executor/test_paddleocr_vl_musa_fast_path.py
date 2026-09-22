from pathlib import Path


MODEL = Path(__file__).parents[2] / "vllm/model_executor/models/paddleocr_vl.py"
HELPER = Path(__file__).parents[2] / "vllm/model_executor/models/paddleocr_vl_musa.py"


def test_paddle_fast_path_is_model_local_and_narrow():
    source = MODEL.read_text(encoding="utf-8")
    helper = HELPER.read_text(encoding="utf-8")
    assert "paddle_musa_fast_path(vllm_config)" in source
    assert "register_attention_backends" in source
    assert "@ApplyRotaryEmb.register_oot" not in source
    assert "model_type\", None) != \"paddleocr_vl\"" in helper
    assert "list((rope_parameters or {}).get(\"mrope_section\", ())) == [16, 24, 24]" in helper
    assert "get_default_ir_op_priority" not in source + helper


def test_paddle_path_reuses_existing_kernels_and_keeps_fp32_contract():
    helper = HELPER.read_text(encoding="utf-8")
    assert "vllm_musa.jit_kernel.csrc.rope" in helper
    assert "forward_cuda(positions, query, key)" in helper
    assert "enable_fp32_compute=True" in MODEL.read_text(encoding="utf-8")
    assert "x_work = x if x.dtype == torch.float32 else x.float()" in helper
    assert "output = x_work.clone()" in helper
    assert "rotary_dim = cos.shape[-1] * 2" in helper


def test_paddle_patch_does_not_register_global_rotary_ops():
    source = MODEL.read_text(encoding="utf-8")
    helper = HELPER.read_text(encoding="utf-8")
    assert "register_oot" not in source + helper
    assert "direct_register_custom_op" not in source + helper
    assert "get_default_ir_op_priority" not in source + helper
