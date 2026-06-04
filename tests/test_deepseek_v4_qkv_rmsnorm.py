from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def test_deepseek_v4_fused_qkv_rmsnorm_custom_op_is_registered():
    custom_ops = _read("vllm_musa/_custom_ops.py")
    bindings = _read("csrc/musa/torch_bindings.cpp")
    headers = _read("csrc/musa/musa_ops.h")
    setup = _read("setup.py")

    assert "def deepseek_v4_fused_q_kv_rmsnorm(" in custom_ops
    assert "torch.ops._C_musa_ops.deepseek_v4_fused_q_kv_rmsnorm" in custom_ops
    assert "deepseek_v4_fused_q_kv_rmsnorm(Tensor q" in bindings
    assert "&deepseek_v4_fused_q_kv_rmsnorm" in bindings
    assert "deepseek_v4_fused_q_kv_rmsnorm(" in headers
    assert "csrc/musa/attention/deepseek_v4_fused_qkv_rmsnorm.mu" in setup
