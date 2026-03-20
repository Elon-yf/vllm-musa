# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# """
# Patch for vllm.attention.ops.flashmla.
# """

PATCHES = [
    # Patch is_flashmla_dense_supported to support musa device type and update unsupported reason
    # For vLLM 0.13+
    (
        "if not current_platform.is_device_capability_family(90):",
        "if current_platform.get_device_capability()[0] != 3:",
    )
]
