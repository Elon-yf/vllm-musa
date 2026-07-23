# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific CosyVoice3 Talker sampling optimizations."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import (
    CosyVoice3Model as _OmniCosyVoice3Model,
)

from vllm_musa.omni.gpu_ar import (
    HOST_SAMPLING_SCALARS,
    NO_FREQUENCY_PRESENCE_PENALTIES,
)

logger = logging.getLogger(__name__)

_SAMPLER_MODE_ENV = "VLLM_MUSA_COSYVOICE3_SAMPLER_MODE"
_VALID_SAMPLER_MODES = {"baseline", "sync_free"}


def _sampler_mode() -> str:
    value = os.getenv(_SAMPLER_MODE_ENV, "sync_free").strip().lower()
    if value not in _VALID_SAMPLER_MODES:
        logger.warning(
            "Invalid %s=%r; falling back to baseline sampling",
            _SAMPLER_MODE_ENV,
            value,
        )
        return "baseline"
    return value


_SAMPLER_MODE = _sampler_mode()


class CosyVoice3Model(_OmniCosyVoice3Model):
    """CosyVoice3 wrapper that removes decode-time sampling stalls."""

    def _req_scalar(self, param, req_idx: int, default):
        """Read per-request scalars from CPU arrays when available."""
        if param is None or param.numel() == 0:
            return default
        if _SAMPLER_MODE != "baseline":
            metadata = getattr(self, "_vllm_musa_sampling_metadata", None)
            host_scalars = getattr(metadata, HOST_SAMPLING_SCALARS, None)
            if host_scalars is not None:
                for device_param, host_param in zip(
                    (
                        getattr(metadata, "temperature", None),
                        getattr(metadata, "top_p", None),
                        getattr(metadata, "top_k", None),
                    ),
                    host_scalars,
                    strict=True,
                ):
                    if param is not device_param or host_param is None:
                        continue
                    if len(host_param) == 0:
                        return default
                    index = min(req_idx, len(host_param) - 1)
                    value = host_param[index]
                    return int(value) if isinstance(default, int) else float(value)
        return super()._req_scalar(param, req_idx, default)

    def _cosyvoice3_ras_enabled(self, sampling_metadata) -> bool:
        if _SAMPLER_MODE == "baseline":
            return super()._cosyvoice3_ras_enabled(sampling_metadata)
        self._vllm_musa_sampling_metadata = sampling_metadata
        no_frequency_presence_penalties = getattr(
            sampling_metadata,
            NO_FREQUENCY_PRESENCE_PENALTIES,
            None,
        )
        if no_frequency_presence_penalties is None:
            return super()._cosyvoice3_ras_enabled(sampling_metadata)
        if self.model_stage != "cosyvoice3_talker":
            return False
        if sampling_metadata.max_num_logprobs is not None:
            return False
        if sampling_metadata.temperature is None:
            return False
        if bool(sampling_metadata.bad_words_token_ids):
            return False
        return bool(no_frequency_presence_penalties)

    @classmethod
    def _ras_sample_one(
        cls,
        weighted_scores: torch.Tensor,
        decoded_tokens: Sequence[int],
        *,
        top_p: float,
        top_k: int,
        win_size: int,
        tau_r: float,
        generator: torch.Generator | None,
    ) -> int:
        if _SAMPLER_MODE == "baseline":
            return super()._ras_sample_one(
                weighted_scores,
                decoded_tokens,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
                generator=generator,
            )

        top_id = cls._nucleus_sample_one(
            weighted_scores,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        if win_size > 0 and decoded_tokens:
            # ``top_id`` is already host-visible.  Counting a ten-token Python
            # history avoids a tensor allocation, H2D copy, kernel launch, and
            # a second D2H synchronization on every autoregressive step.
            recent = decoded_tokens[-win_size:]
            rep_num = sum(int(token) == top_id for token in recent)
            if rep_num >= win_size * tau_r:
                weighted_scores = weighted_scores.clone()
                weighted_scores[top_id] = float("-inf")
                fallback_probs = weighted_scores.softmax(dim=0)
                top_id = int(
                    cls._multinomial_sample(
                        fallback_probs,
                        generator=generator,
                    ).item()
                )
        return top_id


logger.info("CosyVoice3 MUSA sampler mode: %s", _SAMPLER_MODE)

__all__ = ["CosyVoice3Model"]
