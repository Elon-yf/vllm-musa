# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from types import SimpleNamespace

import pytest
import torch
from torch import fx
from vllm.compilation.passes.fusion import rope_kvcache_fusion  # noqa: F401
from vllm.model_executor.layers.attention import attention  # noqa: F401

import vllm_musa.jit_kernel.csrc.rope  # noqa: F401
from vllm_musa.compilation.qwen2_rope_kv_presplit import (
    FUSED_SPLITTING_OP,
    KV_UPDATE_SPLITTING_OP,
    _arg,
    _is_call,
    apply_qwen2_rope_kv_presplit,
    plan_qwen2_rope_kv_presplit,
    qwen2_rope_kv_backend_supported,
)


def _set_tensor_meta(node: fx.Node, shape: tuple[int, ...], dtype=torch.bfloat16):
    node.meta["example_value"] = torch.empty(shape, dtype=dtype)


def _make_graph(
    sites: int,
    *,
    malformed_site: int | None = None,
    duplicate_layer_name: bool = False,
) -> fx.GraphModule:
    graph = fx.Graph()
    qkv = graph.placeholder("qkv")
    positions = graph.placeholder("positions")
    cos_sin_cache = graph.placeholder("cos_sin_cache")
    output = graph.placeholder("output")
    attention_nodes = []

    for index in range(sites):
        split = graph.call_method(
            "split",
            args=(qkv, [896, 128, 128]),
            kwargs={"dim": -1},
        )
        query_base = graph.call_function(operator.getitem, args=(split, 0))
        key_base = graph.call_function(operator.getitem, args=(split, 1))
        value_base = graph.call_function(operator.getitem, args=(split, 2))
        graph.call_function(
            torch.ops.vllm.musa_rotary_embedding.default,
            args=(positions, query_base, key_base, 64, cos_sin_cache, True),
        )
        query = graph.call_method("view", args=(query_base, -1, 14, 64))
        key = graph.call_method("view", args=(key_base, -1, 2, 64))
        value = graph.call_method("view", args=(value_base, -1, 2, 64))
        query_dtype = torch.float32 if index == malformed_site else torch.bfloat16
        _set_tensor_meta(query, (1, 14, 64), query_dtype)
        _set_tensor_meta(key, (1, 2, 64))
        _set_tensor_meta(value, (1, 2, 64))

        layer_index = 0 if duplicate_layer_name else index
        layer_name = f"model.layers.{layer_index}.self_attn.attn"
        kv_update = graph.call_function(
            torch.ops.vllm.unified_kv_cache_update.default,
            args=(key, value, layer_name),
        )
        attention_node = graph.call_function(
            torch.ops.vllm.unified_attention_with_output.default,
            args=(query, key, value, output, layer_name),
            kwargs={"kv_cache_dummy_dep": kv_update},
        )
        attention_nodes.append(attention_node)

    graph.output(tuple(attention_nodes))
    return fx.GraphModule({}, graph)


def _count_calls(graph_module: fx.GraphModule, qualified_name: str) -> int:
    return sum(_is_call(node, qualified_name) for node in graph_module.graph.nodes)


def test_qwen2_rope_kv_presplit_rewrites_all_24_sites():
    graph_module = _make_graph(24)
    candidates = plan_qwen2_rope_kv_presplit(graph_module)

    assert candidates is not None
    assert apply_qwen2_rope_kv_presplit(graph_module, candidates) == 24
    assert _count_calls(graph_module, "vllm::musa_rotary_embedding") == 0
    assert _count_calls(graph_module, KV_UPDATE_SPLITTING_OP) == 0
    assert _count_calls(graph_module, FUSED_SPLITTING_OP) == 24

    for node in graph_module.graph.nodes:
        if _is_call(node, "vllm::unified_attention_with_output"):
            assert _is_call(_arg(node, 7, "kv_cache_dummy_dep"), FUSED_SPLITTING_OP)
    graph_module.graph.lint()


@pytest.mark.parametrize(
    ("sites", "malformed_site", "duplicate_layer_name"),
    [
        (23, None, False),
        (25, None, False),
        (24, 7, False),
        (24, None, True),
    ],
)
def test_qwen2_rope_kv_presplit_is_atomic_on_mismatch(
    sites: int,
    malformed_site: int | None,
    duplicate_layer_name: bool,
):
    graph_module = _make_graph(
        sites,
        malformed_site=malformed_site,
        duplicate_layer_name=duplicate_layer_name,
    )
    graph_before = str(graph_module.graph)

    assert plan_qwen2_rope_kv_presplit(graph_module) is None
    assert str(graph_module.graph) == graph_before
    assert _count_calls(graph_module, "vllm::musa_rotary_embedding") == sites
    assert _count_calls(graph_module, KV_UPDATE_SPLITTING_OP) == sites
    assert _count_calls(graph_module, FUSED_SPLITTING_OP) == 0


def test_qwen2_rope_kv_backend_gate_requires_all_24_layers(monkeypatch):
    def set_layers(count: int, unsupported: int | None = None) -> None:
        layers = {
            f"model.layers.{index}.self_attn.attn": SimpleNamespace(
                impl=SimpleNamespace(
                    fused_rope_kvcache_supported=lambda index=index: (
                        index != unsupported
                    )
                )
            )
            for index in range(count)
        }
        monkeypatch.setattr(
            "vllm.config.get_layers_from_vllm_config",
            lambda _config, _layer_type: layers,
        )

    set_layers(24)
    assert qwen2_rope_kv_backend_supported(object())

    set_layers(23)
    assert not qwen2_rope_kv_backend_supported(object())

    set_layers(24, unsupported=7)
    assert not qwen2_rope_kv_backend_supported(object())
