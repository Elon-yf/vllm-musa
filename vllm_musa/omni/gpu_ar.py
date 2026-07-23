# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific vLLM-Omni autoregressive runner integrations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

NO_FREQUENCY_PRESENCE_PENALTIES = "_vllm_musa_no_frequency_presence_penalties"
HOST_SAMPLING_SCALARS = "_vllm_musa_host_sampling_scalars"
_PATCH_MARKER = "_vllm_musa_cosyvoice3_penalty_metadata_patch"
_COSYVOICE3_ARCH = "CosyVoice3Model"


def install_sampling_metadata_patch() -> bool:
    """Expose frequency/presence eligibility without a device sync.

    vLLM's input batch already tracks the requests using each penalty in CPU
    sets.  CosyVoice3 only needs the frequency and presence state, whereas the
    generic ``SamplingMetadata.no_penalties`` flag also includes repetition
    penalties.  Attach a model-private boolean after Omni has built its model
    sampler metadata so the model can preserve that distinction.

    Returns ``False`` when the expected Omni runner is unavailable.  The model
    wrapper treats a missing attribute as a compatibility fallback and calls
    the original synchronization-based implementation.
    """
    try:
        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
    except (ImportError, AttributeError):
        logger.warning(
            "vLLM-Omni GPUARModelRunner is unavailable; "
            "CosyVoice3 will retain synchronized penalty checks"
        )
        return False

    original: Callable[..., Any] = GPUARModelRunner._sampling_metadata_for_model_sampler
    if getattr(original, _PATCH_MARKER, False):
        return True

    @wraps(original)
    def _sampling_metadata_for_model_sampler(self, sampling_metadata):
        model_metadata = original(self, sampling_metadata)
        model_config = getattr(self, "model_config", None)
        if getattr(model_config, "model_arch", None) != _COSYVOICE3_ARCH:
            return model_metadata

        input_batch = getattr(self, "input_batch", None)
        frequency_reqs = getattr(input_batch, "frequency_penalties_reqs", None)
        presence_reqs = getattr(input_batch, "presence_penalties_reqs", None)
        host_scalars = tuple(
            getattr(input_batch, name, None)
            for name in ("temperature_cpu", "top_p_cpu", "top_k_cpu")
        )
        if all(value is not None for value in host_scalars):
            setattr(model_metadata, HOST_SAMPLING_SCALARS, host_scalars)
        if frequency_reqs is None or presence_reqs is None:
            return model_metadata

        setattr(
            model_metadata,
            NO_FREQUENCY_PRESENCE_PENALTIES,
            not frequency_reqs and not presence_reqs,
        )
        return model_metadata

    setattr(_sampling_metadata_for_model_sampler, _PATCH_MARKER, True)
    GPUARModelRunner._sampling_metadata_for_model_sampler = (
        _sampling_metadata_for_model_sampler
    )
    logger.info("Installed MUSA CosyVoice3 host-side penalty metadata integration")
    return True


__all__ = [
    "HOST_SAMPLING_SCALARS",
    "NO_FREQUENCY_PRESENCE_PENALTIES",
    "install_sampling_metadata_patch",
]
