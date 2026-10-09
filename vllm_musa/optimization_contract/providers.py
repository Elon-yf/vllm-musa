from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .deepseek_v4 import resolve_deepseek_v4_contract
from .qwen import resolve_qwen_contract
from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)

ContractProvider = Callable[
    [ModelSignature, ExecutionSignature],
    MusaOptimizationContract | None,
]


def resolve_paddleocr_vl_contract(
    model: ModelSignature, execution: ExecutionSignature
) -> MusaOptimizationContract | None:
    if model.hf_model_type != "paddleocr_vl":
        return None
    model = replace(model, family=ModelFamily.PADDLEOCR_VL, role=ModelRole.TEXT)
    exact_geometry = (
        model.vision_hidden_size == 1152
        and (model.vision_num_hidden_layers == 27 or model.vision_depth == 27)
        and model.vision_num_attention_heads == 16
        and model.vision_patch_size == 14
        and model.vision_image_size == 384
        and model.hidden_size == 1024
        and model.num_attention_heads == 16
        and model.num_key_value_heads == 2
        and model.mrope_section == (16, 24, 24)
    )
    features = (
        frozenset({OptimizationFeature.PADDLEOCR_VL_ROTARY})
        if exact_geometry
        else frozenset()
    )
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile="paddleocr_vl",
        supported_features=features,
        preferred_features=features,
    )


# Keep provider registration explicit. Providers must fail closed when their
# exact family metadata is incomplete so one family cannot enable another's
# fast paths.
CONTRACT_PROVIDERS: tuple[ContractProvider, ...] = (
    resolve_deepseek_v4_contract,
    resolve_paddleocr_vl_contract,
    resolve_qwen_contract,
)
