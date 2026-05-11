# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.backends.
"""

PATCHES = [
    (
        "def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any:",
        "def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any) -> Any:",
    ),
]
