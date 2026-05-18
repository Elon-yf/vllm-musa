# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-specific kernel providers for vllm.ir ops.

Imported from vllm_musa.platform.MUSAPlatformBase.import_ir_kernels()
after super().import_ir_kernels() so upstream providers (native /
vllm_c / oink / aiter / xpu_kernels) register first; MUSA providers
appear alongside.
"""

from . import musa_ops  # noqa: F401

__all__ = ["musa_ops"]
