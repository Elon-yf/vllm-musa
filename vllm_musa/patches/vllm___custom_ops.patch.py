# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: route direct ``vllm._custom_ops`` rms_norm /
rotary_embedding calls through MUSA-safe fallbacks for dflash paths (
was inline ``_patch_vllm_custom_ops_dflash_fallbacks``)."""

from vllm.logger import init_logger

from vllm_musa.patches._shared import musa_safe_rms_norm, musa_safe_rotary_embedding

logger = init_logger(__name__)

PATCHES: list = []


def apply() -> None:
    try:
        from vllm import _custom_ops as vllm_custom_ops
    except Exception:
        return

    current = getattr(vllm_custom_ops, "rms_norm", None)
    if not getattr(current, "_musa_safe_rms_norm", False):
        setattr(musa_safe_rms_norm, "_musa_safe_rms_norm", True)
        vllm_custom_ops.rms_norm = musa_safe_rms_norm

    current = getattr(vllm_custom_ops, "rotary_embedding", None)
    if not getattr(current, "_musa_safe_rotary_embedding", False):
        setattr(musa_safe_rotary_embedding, "_musa_safe_rotary_embedding", True)
        vllm_custom_ops.rotary_embedding = musa_safe_rotary_embedding
