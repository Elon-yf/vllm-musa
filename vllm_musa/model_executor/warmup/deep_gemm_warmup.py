# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def _fp8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    return False


import vllm.model_executor.warmup.deep_gemm_warmup

vllm.model_executor.warmup.deep_gemm_warmup._fp8_linear_may_use_deep_gemm = (
    _fp8_linear_may_use_deep_gemm
)
