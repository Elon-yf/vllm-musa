# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.caching.
"""

_GRAPH_PICKLER_COMPAT = """import torch.fx._graph_pickler as _vllm_graph_pickler
GraphPickler = _vllm_graph_pickler.GraphPickler
Options = getattr(_vllm_graph_pickler, "Options", None)


def _musa_graph_pickler_dumps(graph_module):
    if Options is None:
        return GraphPickler.dumps(graph_module)
    return GraphPickler.dumps(graph_module, Options(ops_filter=None))
"""

_BROKEN_GRAPH_PICKLER_COMPAT = """try:
    try:
    from torch.fx._graph_pickler import GraphPickler, Options
except ImportError:
    from torch.fx._graph_pickler import GraphPickler

    Options = None

except ImportError:
    from torch.fx._graph_pickler import GraphPickler

    Options = None
"""

_BROKEN_DUMPS_BLOCK = """            if Options is None:
                return GraphPickler.dumps(graph_module)
            if Options is None:
                return GraphPickler.dumps(graph_module)
            return GraphPickler.dumps(graph_module, Options(ops_filter=None))"""

PATCHES = [
    (
        _BROKEN_GRAPH_PICKLER_COMPAT,
        _GRAPH_PICKLER_COMPAT,
    ),
    (
        "from torch.fx._graph_pickler import GraphPickler, Options",
        _GRAPH_PICKLER_COMPAT,
    ),
    (
        _BROKEN_DUMPS_BLOCK,
        "            return _musa_graph_pickler_dumps(graph_module)",
    ),
    (
        "            return GraphPickler.dumps(graph_module, Options(ops_filter=None))",
        "            return _musa_graph_pickler_dumps(graph_module)",
    ),
]
