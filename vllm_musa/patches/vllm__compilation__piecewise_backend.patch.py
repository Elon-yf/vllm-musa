# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.piecewise_backend.
"""

PATCHES = [
    (
        "import pickle\nfrom collections.abc import Callable\nfrom pickle import Pickler",
        "import pickle\nfrom collections.abc import Callable\nfrom contextlib import nullcontext\nfrom pickle import Pickler",
    ),
    (
        """            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry = fn.serialize()
""",
        """            functorch_cache_ctx = (
                torch._functorch.config.patch("bundled_autograd_cache", True)
                if hasattr(torch._functorch.config, "bundled_autograd_cache")
                else nullcontext()
            )
            with functorch_cache_ctx:
                entry = fn.serialize()
""",
    ),
]
