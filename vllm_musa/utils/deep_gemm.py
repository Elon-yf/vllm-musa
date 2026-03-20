# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import torch
from vllm.utils.deep_gemm import _lazy_init


@functools.cache
def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    # ==================== MUSA ADAPTATION ====================
    mk_align_size = 128
    # ========================== END ==========================
    return [mk_align_size, mk_align_size]


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight: torch.Tensor,
    supports_deep_gemm: bool | None = None,
):
    return False


import vllm.utils.deep_gemm

vllm.utils.deep_gemm.get_mk_alignment_for_contiguous_layout = (
    get_mk_alignment_for_contiguous_layout
)
vllm.utils.deep_gemm.should_use_deepgemm_for_fp8_linear = (
    should_use_deepgemm_for_fp8_linear
)
