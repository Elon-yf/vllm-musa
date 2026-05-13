# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.parameter import BasevLLMParameter


def _materialize_plain_parameter(layer: torch.nn.Module) -> None:
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, BasevLLMParameter):
        return

    plain_weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
    plain_weight.__dict__.update(weight.__dict__)
    layer.weight = plain_weight


_ORIGINAL_UNQUANTIZED_LINEAR_PROCESS_WEIGHTS = (
    vllm_linear.UnquantizedLinearMethod.process_weights_after_loading
)


def _musa_process_unquantized_linear_weights_after_loading(
    self: vllm_linear.UnquantizedLinearMethod,
    layer: torch.nn.Module,
) -> None:
    from vllm.platforms import current_platform

    if current_platform.is_musa():
        _materialize_plain_parameter(layer)
        return

    _ORIGINAL_UNQUANTIZED_LINEAR_PROCESS_WEIGHTS(self, layer)


if not getattr(
    vllm_linear.UnquantizedLinearMethod.process_weights_after_loading,
    "_musa_materializes_plain_parameter",
    False,
):
    setattr(
        _musa_process_unquantized_linear_weights_after_loading,
        "_musa_materializes_plain_parameter",
        True,
    )
    vllm_linear.UnquantizedLinearMethod.process_weights_after_loading = (
        _musa_process_unquantized_linear_weights_after_loading
    )
