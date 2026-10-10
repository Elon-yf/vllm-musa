from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from .deepseek_v4 import resolve_deepseek_v4_contract
from .glm import resolve_glm_contract
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
    geometry = model.paddleocr_vl_rotary_geometry
    if geometry is None:
        return None
    model = replace(model, family=ModelFamily.PADDLEOCR_VL, role=ModelRole.TEXT)
    enabled = (
        geometry[0] == 1152
        and 27 in geometry[1:3]
        and geometry[3:] == (16, 14, 384, 1024, 16, 2, (16, 24, 24))
        and getattr(torch.version, "musa", None) is not None
    )
    preferred = frozenset({OptimizationFeature.PADDLEOCR_VL_ROTARY} if enabled else ())
    return MusaOptimizationContract(
        model=model,
        execution=execution,
        profile="paddleocr_vl",
        supported_features=preferred,
        preferred_features=preferred,
    )


def install_paddleocr_vl_rotary(
    visual: torch.nn.Module, language_model: torch.nn.Module
) -> bool:
    """Install adapters only after the Paddle provider selected this contract."""
    from .rotary import MusaMRotaryEmbedding, install_vision_rotary

    encoder = getattr(getattr(visual, "vision_model", None), "encoder", None)
    layers = getattr(encoder, "layers", ())
    if not install_vision_rotary(
        (getattr(layer, "self_attn", None) for layer in layers), expected_blocks=27
    ):
        return False
    for layer in getattr(getattr(language_model, "model", None), "layers", ()):
        attention = getattr(layer, "self_attn", None)
        rotary = getattr(attention, "rotary_emb", None)
        if rotary is not None:
            rotary.is_neox_style = True
            attention.rotary_emb = MusaMRotaryEmbedding(rotary)
    return True


# Keep provider registration explicit. Providers must fail closed when their
# exact family metadata is incomplete so one family cannot enable another's
# fast paths.
CONTRACT_PROVIDERS: tuple[ContractProvider, ...] = (
    resolve_deepseek_v4_contract,
    resolve_glm_contract,
    resolve_paddleocr_vl_contract,
    resolve_qwen_contract,
)
