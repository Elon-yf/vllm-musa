# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: fuse Qwen3 dense-decode q-norm + neox RoPE into one
TileLang kernel (opt-in via ``VLLM_MUSA_QWEN3_FUSED_QKNORM=1``).

Dormant unless the env flag is set. When enabled it registers the opaque fused
custom op and rebinds ``Qwen3Attention.__init__``/``forward``; the attention
backend keeps ``reshape_and_cache`` (no backend surgery)."""

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

PATCHES: list = []


def apply() -> None:
    if os.environ.get("VLLM_MUSA_QWEN3_FUSED_QKNORM", "0") != "1":
        return
    # MUSA: the fused op is a MUSA-only TileLang kernel; ignore the flag on any
    # non-MUSA runtime so an accidentally-set env var cannot break CPU/CUDA runs.
    if not (hasattr(torch.version, "musa") and torch.version.musa is not None):
        logger.debug("Skipping Qwen3 fused qk-norm patch: MUSA unavailable")
        return
    try:
        from vllm.model_executor.models.qwen3 import Qwen3Attention

        from vllm_musa.qwen3_jit.qk_norm_rope_kv import register_and_patch
    except Exception as e:
        logger.debug("Skipping Qwen3 fused qk-norm patch: %s", e)
        return

    logger.info("Enabling Qwen3 fused qk-norm+RoPE decode kernel (opt-in)")
    register_and_patch(Qwen3Attention)
