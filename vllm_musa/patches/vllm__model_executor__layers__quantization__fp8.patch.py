# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.quantization.fp8.
"""

PATCHES = [
    # Patch get_fp8_moe_backend and Fp8LinearMethod where don't support use_marlin
    (
        "if current_platform.is_rocm() or current_platform.is_xpu():",
        "if current_platform.is_rocm() or current_platform.is_musa() or current_platform.is_xpu():",
    ),
    # Patch get_min_capability to support musa device
    (
        "return 75",
        "return 31",
    ),
]
