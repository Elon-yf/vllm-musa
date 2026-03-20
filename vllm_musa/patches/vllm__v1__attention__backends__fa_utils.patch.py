# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.attention.backends.fa_utils.
"""

PATCHES = [
    # Patch get_flash_attn_version where vllm musa use fa2 by default
    (
        """def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    if current_platform.is_xpu():
""",
        """def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    if current_platform.is_musa():
        return 2
    if current_platform.is_xpu():
""",
    ),
]
