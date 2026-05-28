# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 — utils.py kernel safety.

Adds early-exit guard for ``query_len == 0`` rows in
``eagle_prepare_inputs_padded_kernel``. The padded-drafter batch can
contain empty padding rows whose ``query_start_loc[i] == query_start_loc[i+1]``.
Without this guard the kernel writes a stale ``q_last_tok_idx`` for
those rows and can corrupt downstream draft token sampling.

NOTE: this is the upstream ``vllm/v1/spec_decode/utils.py`` —
distinct from ``vllm_musa/v1/spec_decode/utils.py`` which lives at
the same path under vllm-musa and installs the MUSA-Triton-adapted
``eagle_prepare_next_token_padded_kernel``.
"""

# Hunk 1: insert query_len == 0 guard at top of kernel body
_OLD = """    if req_idx >= num_reqs:
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)"""

_NEW = """    if req_idx >= num_reqs:
        return

    q_start_idx = tl.load(query_start_loc_gpu_ptr + req_idx)
    q_end_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1)
    query_len = q_end_idx - q_start_idx

    if query_len == 0:
        tl.store(token_indices_to_sample_ptr + req_idx, q_end_idx - 1)
        tl.store(num_rejected_tokens_gpu_ptr + req_idx, 0)
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)"""

# Hunk 2: reuse q_end_idx instead of redundant tl.load
_OLD_LAST_TOK = """    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1"""

_NEW_LAST_TOK = """    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = q_end_idx - 1"""

PATCHES = [
    (_OLD, _NEW),
    (_OLD_LAST_TOK, _NEW_LAST_TOK),
]
