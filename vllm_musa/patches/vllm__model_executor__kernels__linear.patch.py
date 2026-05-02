# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

PATCHES = [
    (
        """    for kernel in possible_kernels[current_platform._enum]:
""",
        """    platform_kernels = possible_kernels.get(current_platform._enum)
    if platform_kernels is None:
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. No kernels are registered for "
            f"platform {current_platform._enum.name}."
        )

    for kernel in platform_kernels:
""",
    ),
]
