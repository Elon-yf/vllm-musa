# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Patch for vllm.v1.sample.ops.topk_topp_triton

This patch fixes compatibility issues with MUSA's Triton version.

Changes:
--------
1. Remove tl.cast() calls for casting intermediate results to tl.int32
   - MUSA Triton handles type inference better without explicit casts
   - Reduces unnecessary type conversions

2. Refactor conditional logic for final_pivot
   - Simplify logic by separating assignment from conditional evaluation
   - Improves compatibility with MUSA Triton compiler

Affected functions:
-------------------
- Top-K/Top-P sampling kernel
- Flow python code format
"""

PATCHES = [
    (
        """                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )""",
        """                    search_iters = (search_range + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC""",
    ),
    (
        """                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
""",
        """                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = -float("inf")
                if num_finite_total > k:
                    final_pivot = k_pivot

                if num_finite_total <= k:
                    pass
                elif TOPP_ENABLED:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
""",
    ),
    (
        """                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # Third pass: Calculate exp logits and sum, gather outliers
""",
        """                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = (search_range + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC

                        # Third pass: Calculate exp logits and sum, gather outliers
""",
    ),
    (
        """                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
                                // BLOCK_SIZE_TRUNC,
                                tl.int32,
                            )

                            # Fourth pass: Calculate BUFFER and get outliers
""",
        """                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = (
                                search_range + BLOCK_SIZE_TRUNC - 1
                            ) // BLOCK_SIZE_TRUNC

                            # Fourth pass: Calculate BUFFER and get outliers
""",
    ),
    (
        """        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            p = tl.load(P + row_id)
            if p < 1.0:
""",
        """        if final_pivot != -float("inf"):
            pass
        elif TOPP_ENABLED:
            #### STANDALONE TOP-P SAMPLING ####
            p = tl.load(P + row_id)
            if p < 1.0:
""",
    ),
    (
        """                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
""",
        """                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = (search_range + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC

                    found_pivot = 0
""",
    ),
]
