from pathlib import Path


ROOT = Path(__file__).parents[1]
PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0174-MUSA-use-direct-QK-rotary-for-MinerU-vision.patch"
)
HELPER = ROOT / "vllm_musa/models/mineru_qwen2_vl.py"


def test_direct_qk_series_patch_only_modifies_existing_upstream_file():
    source = PATCH.read_text(encoding="utf-8")
    target = "vllm/model_executor/models/qwen2_vl.py"
    headers = [line for line in source.splitlines() if line.startswith("diff --git ")]
    assert headers == [f"diff --git a/{target} b/{target}"]
    assert f"--- a/{target}\n+++ b/{target}\n" in source
    assert "new file mode" not in source
    assert "/dev/null" not in source
    assert "_musa_direct_qk_rotary" in source
    assert "forward_qk" in source


def test_direct_qk_helper_is_a_normal_source_file():
    helper = HELPER.read_text(encoding="utf-8")
    assert "def _musa_visual_rotary_qk(" in helper
    assert "def forward_qk(" in helper
    assert "rotary_embedding(" in helper
    assert "module._musa_direct_qk_rotary = True" in helper
