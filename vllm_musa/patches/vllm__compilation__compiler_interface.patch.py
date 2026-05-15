# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.compiler_interface.
"""

PATCHES = [
    (
        """    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        cfg["bundled_autograd_cache"] = False
    return cfg
""",
        """    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        cfg["bundled_autograd_cache"] = False

    from torch._functorch import config as functorch_config

    return {
        key: value
        for key, value in cfg.items()
        if hasattr(functorch_config, key)
    }
""",
    ),
]
