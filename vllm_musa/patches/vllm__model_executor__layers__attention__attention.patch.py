# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.attention.attention.
"""

PATCHES = [
    (
        """            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
""",
        """            output_shape = (num_tokens, self.num_heads * self.head_size_v)
""",
    ),
]
