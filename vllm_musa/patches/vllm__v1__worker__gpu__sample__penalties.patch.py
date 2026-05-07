# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

PATCHES = [
    (
        "    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty",
        """    use_penalty = use_rep_penalty or use_freq_penalty
    use_penalty = use_penalty or use_pres_penalty""",
    ),
]
