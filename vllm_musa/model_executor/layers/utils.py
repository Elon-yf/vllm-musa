# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.model_executor.layers import utils as vllm_layer_utils
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers import vocab_parallel_embedding as vllm_vocab_embedding
from vllm.utils.torch_utils import direct_register_custom_op


def _musa_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = weight.data
    if bias is not None:
        bias = bias.data
    return torch.ops.vllm.musa_unquantized_gemm(x, weight, bias)


def _musa_unquantized_gemm_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.linear(x, weight, bias)


def _musa_unquantized_gemm_op_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        x.shape[:-1] + (weight.shape[0],),
        dtype=x.dtype,
        device=x.device,
    )


direct_register_custom_op(
    "musa_unquantized_gemm",
    _musa_unquantized_gemm_op,
    fake_impl=_musa_unquantized_gemm_op_fake,
)


def _dispatch_unquantized_gemm():
    return _musa_unquantized_gemm


if not getattr(
    vllm_layer_utils.dispatch_unquantized_gemm,
    "_musa_dispatches_unquantized_gemm",
    False,
):
    _dispatch_unquantized_gemm._musa_dispatches_unquantized_gemm = True
    vllm_layer_utils.dispatch_unquantized_gemm = _dispatch_unquantized_gemm
    vllm_linear.dispatch_unquantized_gemm = _dispatch_unquantized_gemm
    vllm_vocab_embedding.dispatch_unquantized_gemm = _dispatch_unquantized_gemm
