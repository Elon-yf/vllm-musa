# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.caching.
"""

PATCHES = [
    (
        "from torch.fx._graph_pickler import GraphPickler, Options",
        """try:
    from torch.fx._graph_pickler import GraphPickler, Options
except ImportError:
    from torch.fx._graph_pickler import GraphPickler

    Options = None
""",
    ),
    (
        "            return GraphPickler.dumps(graph_module, Options(ops_filter=None))",
        """            if Options is None:
                return GraphPickler.dumps(graph_module)
            return GraphPickler.dumps(graph_module, Options(ops_filter=None))""",
    ),
]
