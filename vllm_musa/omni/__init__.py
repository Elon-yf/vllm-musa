# SPDX-License-Identifier: Apache-2.0
"""Optional vLLM-Omni integrations for the MUSA platform."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_COSYVOICE3_ARCH = "CosyVoice3Model"
_COSYVOICE3_CLASS = "vllm_musa.omni.models.cosyvoice3:CosyVoice3Model"


def _invalidate_cosyvoice3_model_cache() -> int:
    """Drop CosyVoice3 resolutions made before Omni plugins were loaded."""
    try:
        from vllm.model_executor.model_loader import utils as loader_utils
    except ImportError:
        return 0

    cache = getattr(loader_utils, "_MODEL_ARCH_BY_HASH", None)
    if not isinstance(cache, dict):
        return 0

    stale_keys = [
        key
        for key, value in cache.items()
        if isinstance(value, tuple) and len(value) == 2 and value[1] == _COSYVOICE3_ARCH
    ]
    for key in stale_keys:
        cache.pop(key, None)
    return len(stale_keys)


def register_omni_optimizations() -> None:
    """Register lazy MUSA wrappers for supported vLLM-Omni models."""
    from vllm.model_executor.models import ModelRegistry
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    from vllm_musa.omni.gpu_ar import install_sampling_metadata_patch

    install_sampling_metadata_patch()
    ModelRegistry.register_model(_COSYVOICE3_ARCH, _COSYVOICE3_CLASS)
    OmniModelRegistry.register_model(_COSYVOICE3_ARCH, _COSYVOICE3_CLASS)
    invalidated = _invalidate_cosyvoice3_model_cache()
    logger.info(
        "Registered MUSA CosyVoice3 sampler optimizations "
        "(invalidated architecture cache entries: %d)",
        invalidated,
    )


__all__ = ["register_omni_optimizations"]
