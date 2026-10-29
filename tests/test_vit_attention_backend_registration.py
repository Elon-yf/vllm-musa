"""Source-level regression test for the MUSA ViT backend timing fix."""

import ast
from pathlib import Path

PLATFORM = Path(__file__).parents[1] / "vllm_musa" / "platform.py"


def test_vit_selector_registers_musa_backends_before_candidate_walk() -> None:
    tree = ast.parse(PLATFORM.read_text(encoding="utf-8"))
    selector = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_vit_attn_backend"
    )
    first = selector.body[0]
    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name)
    assert first.value.func.id == "register_attention_backends"
