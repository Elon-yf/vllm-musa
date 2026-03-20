# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.attention.backends.mla.flashmla.
"""

PATCHES = [
    # Patch supports_compute_capability to support musa capability
    (
        "return capability.major in [9, 10]",
        "return capability.major in [3]",
    ),
    # Patch FlashMLAMetadataBuilder.reorder_batch_threshold
    ("reorder_batch_threshold: int = 128", "reorder_batch_threshold: int = 1"),
]
