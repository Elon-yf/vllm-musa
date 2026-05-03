# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_MUSA_FP8_FALLBACK = """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None and current_platform.is_musa():
        from vllm_musa.fp8_linear import (
            MUSAFp8BlockScaledMMLinearKernel,
            MUSAFP8ScaledMMLinearKernel,
        )

        if possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS:
            platform_kernels = [MUSAFp8BlockScaledMMLinearKernel]
        else:
            platform_kernels = [MUSAFP8ScaledMMLinearKernel]
    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )

    for kernel in platform_kernels:
"""

PATCHES = [
    (
        """from vllm_musa.model_executor.layers.quantization.musa_fp8_linear import (
    MUSAFP8ScaledMMLinearKernel,
)
""",
        """""",
    ),
    (
        """    PlatformEnum.OOT: [
        MUSAFP8ScaledMMLinearKernel,
    ],
""",
        """""",
    ),
    (
        """    PlatformEnum.XPU: [
        XPUFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.OOT: [
        MUSAFP8ScaledMMLinearKernel,
    ],
}
""",
        """    PlatformEnum.XPU: [
        XPUFP8ScaledMMLinearKernel,
    ],
}
""",
    ),
    (
        """    for kernel in possible_kernels[current_platform._enum]:
""",
        _MUSA_FP8_FALLBACK,
    ),
    (
        """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )

    for kernel in platform_kernels:
""",
        _MUSA_FP8_FALLBACK,
    ),
]
