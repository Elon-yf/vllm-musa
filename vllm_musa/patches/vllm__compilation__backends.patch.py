# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.backends.
"""

PATCHES = [
    (
        "from contextlib import contextmanager",
        "from contextlib import contextmanager, nullcontext",
    ),
    (
        "def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any:",
        "def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any) -> Any:",
    ),
    (
        """            with (
                # Graphs that are isometric (different node names but same
                # structure) should be treated as the same.
                torch._functorch.config.patch(autograd_cache_normalize_inputs=True),
                patch(
""",
        """            functorch_cache_key_ctx = (
                torch._functorch.config.patch(autograd_cache_normalize_inputs=True)
                if hasattr(
                    torch._functorch.config, "autograd_cache_normalize_inputs"
                )
                else nullcontext()
            )
            with (
                # Graphs that are isometric (different node names but same
                # structure) should be treated as the same.
                functorch_cache_key_ctx,
                patch(
""",
    ),
]
