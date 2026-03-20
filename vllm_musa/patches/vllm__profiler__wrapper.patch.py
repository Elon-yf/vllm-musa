# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.profiler.wrapper to support MUSA device type.
"""

PATCHES = [
    # Patch global variable TorchProfilerActivity
    (
        'TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]',
        'TorchProfilerActivity = Literal["CPU", "MUSA"]',
    ),
    # Patch global variable TorchProfilerActivityMap
    (
        '"CUDA": torch.profiler.ProfilerActivity.CUDA,',
        '"MUSA": torch.profiler.ProfilerActivity.MUSA,',
    ),
]
