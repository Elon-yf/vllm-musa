# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

PATCHES = [
    (
        "    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64)",
        """    local_max_dtype = (
        torch.float32
        if getattr(torch.version, "musa", None) is not None
        else torch.float64
    )
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)""",
    ),
]
