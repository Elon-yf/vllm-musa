# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx

KV_UPDATE_SPLITTING_OP = "vllm::unified_kv_cache_update"
FUSED_SPLITTING_OP = "vllm::fused_rope_and_unified_kv_cache_update"
EXPECTED_FUSION_SITES = 24

_MISSING = object()


@dataclass(frozen=True)
class Qwen2RopeKVCandidate:
    rope: fx.Node
    kv_update: fx.Node
    attention: fx.Node
    query: fx.Node
    key: fx.Node
    value: fx.Node
    positions: fx.Node
    cos_sin_cache: fx.Node
    is_neox: bool
    layer_name: str


def _is_call(node: fx.Node, qualified_name: str) -> bool:
    if node.op != "call_function":
        return False
    target = str(node.target)
    expected = qualified_name.replace("::", ".")
    return target == expected or target == f"{expected}.default"


def _arg(
    node: fx.Node,
    index: int,
    name: str,
    default: Any = _MISSING,
) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > index:
        return node.args[index]
    if default is not _MISSING:
        return default
    return _MISSING


def _tensor_value(node: fx.Node) -> Any:
    for key in ("val", "example_value"):
        value = node.meta.get(key)
        if value is not None:
            return value
    return node.meta.get("tensor_meta")


def _shape_tail(node: fx.Node) -> tuple[Any, Any] | None:
    value = _tensor_value(node)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return shape[-2], shape[-1]


def _dtype(node: fx.Node) -> torch.dtype | None:
    return getattr(_tensor_value(node), "dtype", None)


def _view_base(node: fx.Node, heads: int) -> fx.Node | None:
    is_view = node.op == "call_method" and node.target in {"view", "reshape"}
    is_view = is_view or any(
        _is_call(node, target)
        for target in (
            "aten::view",
            "aten::reshape",
            "aten::_unsafe_view",
        )
    )
    if not is_view or not node.args or not isinstance(node.args[0], fx.Node):
        return None
    if _shape_tail(node) != (heads, 64):
        return None
    return node.args[0]


def _split_getitem(node: fx.Node, index: int) -> fx.Node | None:
    if (
        node.op != "call_function"
        or node.target is not operator.getitem
        or len(node.args) != 2
        or node.args[1] != index
        or not isinstance(node.args[0], fx.Node)
    ):
        return None
    return node.args[0]


def _is_exact_qkv_split(node: fx.Node) -> bool:
    if node.op == "call_method" and node.target == "split":
        sizes = _arg(node, 1, "split_size")
        dim = _arg(node, 2, "dim", 0)
    elif _is_call(node, "aten::split_with_sizes"):
        sizes = _arg(node, 1, "split_sizes")
        dim = _arg(node, 2, "dim", 0)
    else:
        return False
    return (
        isinstance(sizes, (list, tuple))
        and list(sizes) == [896, 128, 128]
        and dim in (-1, 1)
    )


def _only_users(node: fx.Node, expected: set[fx.Node]) -> bool:
    return set(node.users) == expected


def plan_qwen2_rope_kv_presplit(
    graph_module: fx.GraphModule,
    expected_sites: int = EXPECTED_FUSION_SITES,
) -> tuple[Qwen2RopeKVCandidate, ...] | None:
    """Plan an all-or-nothing raw-FX rewrite for the exact Qwen2 graph."""
    nodes = tuple(graph_module.graph.nodes)
    all_rope = [node for node in nodes if _is_call(node, "vllm::musa_rotary_embedding")]
    all_kv = [node for node in nodes if _is_call(node, KV_UPDATE_SPLITTING_OP)]
    if len(all_rope) != expected_sites or len(all_kv) != expected_sites:
        return None

    candidates: list[Qwen2RopeKVCandidate] = []
    seen_rope: set[fx.Node] = set()
    seen_layer_names: set[str] = set()

    for kv_update in all_kv:
        if len(kv_update.users) != 1:
            return None
        attention = next(iter(kv_update.users))
        if not _is_call(attention, "vllm::unified_attention_with_output"):
            return None

        query = _arg(attention, 0, "query")
        key = _arg(attention, 1, "key")
        value = _arg(attention, 2, "value")
        dummy = _arg(attention, 7, "kv_cache_dummy_dep", None)
        layer_name = _arg(attention, 4, "layer_name")
        if (
            not all(isinstance(node, fx.Node) for node in (query, key, value))
            or dummy is not kv_update
            or not isinstance(layer_name, str)
            or _arg(kv_update, 0, "key") is not key
            or _arg(kv_update, 1, "value") is not value
            or _arg(kv_update, 2, "layer_name") != layer_name
            or layer_name in seen_layer_names
        ):
            return None

        query_base = _view_base(query, 14)
        key_base = _view_base(key, 2)
        value_base = _view_base(value, 2)
        if not all(
            isinstance(node, fx.Node) for node in (query_base, key_base, value_base)
        ):
            return None

        query_split = _split_getitem(query_base, 0)
        key_split = _split_getitem(key_base, 1)
        value_split = _split_getitem(value_base, 2)
        if (
            query_split is None
            or query_split is not key_split
            or query_split is not value_split
            or not _is_exact_qkv_split(query_split)
            or any(_dtype(node) != torch.bfloat16 for node in (query, key, value))
        ):
            return None

        rope_candidates = {
            user
            for user in query_base.users
            if _is_call(user, "vllm::musa_rotary_embedding")
        } & {
            user
            for user in key_base.users
            if _is_call(user, "vllm::musa_rotary_embedding")
        }
        if len(rope_candidates) != 1:
            return None
        rope = rope_candidates.pop()
        positions = _arg(rope, 0, "positions")
        cos_sin_cache = _arg(rope, 4, "cos_sin_cache")
        is_neox = _arg(rope, 5, "is_neox")
        if (
            rope in seen_rope
            or rope.users
            or not isinstance(positions, fx.Node)
            or not isinstance(cos_sin_cache, fx.Node)
            or _arg(rope, 1, "query") is not query_base
            or _arg(rope, 2, "key") is not key_base
            or _arg(rope, 3, "head_size") != 64
            or is_neox is not True
            or not _only_users(query_base, {query, rope})
            or not _only_users(key_base, {key, rope})
            or not _only_users(value_base, {value})
            or not _only_users(query, {attention})
            or not _only_users(key, {kv_update, attention})
            or not _only_users(value, {kv_update, attention})
        ):
            return None

        candidates.append(
            Qwen2RopeKVCandidate(
                rope=rope,
                kv_update=kv_update,
                attention=attention,
                query=query,
                key=key,
                value=value,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=True,
                layer_name=layer_name,
            )
        )
        seen_rope.add(rope)
        seen_layer_names.add(layer_name)

    if len(candidates) != expected_sites or seen_rope != set(all_rope):
        return None
    return tuple(candidates)


def apply_qwen2_rope_kv_presplit(
    graph_module: fx.GraphModule,
    candidates: tuple[Qwen2RopeKVCandidate, ...],
) -> int:
    """Apply a previously validated plan; invariant failures are fatal."""
    # Importing the upstream pass module registers the fused outer custom op.
    from vllm.compilation.passes.fusion import rope_kvcache_fusion  # noqa: F401

    fused_op = torch.ops.vllm.fused_rope_and_unified_kv_cache_update.default
    graph = graph_module.graph
    for candidate in candidates:
        with graph.inserting_before(candidate.kv_update):
            fused = graph.call_function(
                fused_op,
                kwargs={
                    "query": candidate.query,
                    "key": candidate.key,
                    "value": candidate.value,
                    "positions": candidate.positions,
                    "cos_sin_cache": candidate.cos_sin_cache,
                    "is_neox": candidate.is_neox,
                    "layer_name": candidate.layer_name,
                },
            )
        fused.meta = dict(candidate.kv_update.meta)
        candidate.kv_update.replace_all_uses_with(fused)
        graph.erase_node(candidate.kv_update)
        if candidate.rope.users:
            raise RuntimeError("MUSA Qwen2 RoPE node gained an unexpected user")
        graph.erase_node(candidate.rope)

    graph.lint()
    graph_module.recompile()
    return len(candidates)


__all__ = [
    "EXPECTED_FUSION_SITES",
    "FUSED_SPLITTING_OP",
    "KV_UPDATE_SPLITTING_OP",
    "Qwen2RopeKVCandidate",
    "apply_qwen2_rope_kv_presplit",
    "plan_qwen2_rope_kv_presplit",
]
