# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# """
# Patch for vllm.v1.sample.ops.topk_topp_sampler.
# """

PATCHES = [
    # Patch to use apply_top_k_top_p_pytorch instead of triton.
    # For vLLM 0.13+
    (
        "if HAS_TRITON and logits.shape[0] >= 8:",
        "if HAS_TRITON and logits.shape[0] >= 8 and not current_platform.is_musa():",
    )
]
