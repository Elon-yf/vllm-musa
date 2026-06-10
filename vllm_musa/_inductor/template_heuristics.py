# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inductor template heuristics for device_type='musa'.

Without this registration Inductor finds no Triton GEMM template
heuristic for ``device_type='musa'``, logs ``No template heuristic
found ...`` once per GEMM op, and lowers GEMMs to the ATen backend
(muBLAS) — the fast path on MUSA today.

Default-off: enabling the Triton GEMM templates has shown no runtime
win over the ATen lowering, increases autotune compile time, and can
destabilize spec-decode draft compiles. Opt in with
``VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1`` to experiment, e.g. when
tuning MUSA-specific tile configs.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

# init_logger provides info_once/warning_once (plain logging.getLogger
# does not).
logger = init_logger(__name__)

# Idempotency guard
_REGISTERED = False


def maybe_register_musa_template_heuristics() -> None:
    """Idempotent registration of MUSA Inductor template heuristics.

    Safe to call multiple times. Opportunistic — silently no-ops on old
    torch versions or when the opt-in env var is unset (the default).
    Set ``VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1`` to enable.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # Default-off: no measured win over the ATen GEMM lowering, slower
    # autotune compiles, and known spec-decode draft-compile instability.
    if os.getenv("VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS", "0") != "1":
        logger.debug(
            "MUSA Inductor GEMM template heuristics are disabled "
            "(default). Set VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1 to "
            "experiment with Triton GEMM autotuning."
        )
        _REGISTERED = True
        return

    try:
        from torch._inductor.kernel.bmm import bmm_template
        from torch._inductor.kernel.mm import mm_template
        from torch._inductor.template_heuristics.registry import (
            register_template_heuristic,
        )
        from torch._inductor.template_heuristics.triton import (
            AddMMConfigMixin,
            CUDAConfigHeuristic,
            MMTemplateConfigMixin,
        )
    except ImportError as exc:
        logger.warning_once(
            "torch._inductor template-heuristics API not "
            "available (%s); skipping MUSA registration. Compiled `mm` "
            "ops on MUSA will fall through to ATen via torchada.",
            exc,
        )
        _REGISTERED = True
        return

    try:
        from torch._inductor.template_heuristics.triton import GemmConfig
    except ImportError:
        GemmConfig = None

    class MUSAConfigHeuristic(CUDAConfigHeuristic):
        """Template config heuristic for MUSA (MTT S5000).

        Tile configs swept on S5000 against the ATen lowering across
        transformer linear-layer shapes. Software pipelining beyond
        ``num_stages=1`` and the small CUDA-default tiles both lose on
        this hardware, so the list favors single-stage, deep-K tiles
        and stays within the ~128 KB/block shared-memory budget.
        """

        def __init__(self) -> None:
            super().__init__()
            if GemmConfig is None:
                return
            # (block_m, block_n, block_k, num_stages, num_warps)
            self.mm_configs = [
                # large-M tiles
                GemmConfig(256, 64, 256, 1, 16),
                GemmConfig(256, 128, 64, 1, 8),
                GemmConfig(128, 128, 64, 1, 8),
                # mid-M tiles
                GemmConfig(64, 64, 256, 1, 4),
                GemmConfig(64, 128, 128, 1, 8),
                GemmConfig(64, 128, 64, 2, 4),
                # small-M tiles
                GemmConfig(32, 128, 64, 1, 4),
                GemmConfig(16, 128, 64, 1, 4),
                GemmConfig(16, 64, 64, 1, 2),
            ]

    # Register the four common (template, device, op) combinations.
    # The decorator-based pattern matches CUDAMMTemplateConfigHeuristic
    # registration in torch._inductor.template_heuristics.triton.
    @register_template_heuristic(mm_template.uid, "musa")
    @register_template_heuristic(bmm_template.uid, "musa")
    class MUSAMMTemplateConfigHeuristic(  # noqa: F841 — register-only
        MMTemplateConfigMixin, MUSAConfigHeuristic
    ):
        """Standard MM template heuristic for MUSA."""

    @register_template_heuristic(mm_template.uid, "musa", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "musa", op_name="baddbmm")
    class MUSAAddMMTemplateConfigHeuristic(  # noqa: F841 — register-only
        AddMMConfigMixin, MUSAConfigHeuristic
    ):
        """Addmm/baddbmm template heuristic for MUSA."""

    # scaled_mm (FP8 linear layers).
    @register_template_heuristic(mm_template.uid, "musa", op_name="scaled_mm")
    class MUSAScaledMMTemplateConfigHeuristic(  # noqa: F841 — register-only
        MMTemplateConfigMixin, MUSAConfigHeuristic
    ):
        """Scaled MM (FP8) template heuristic for MUSA."""

    _REGISTERED = True
    logger.info_once(
        "Inductor template heuristics registered for "
        "device_type='musa' (mm + bmm + addmm + baddbmm + scaled_mm)."
    )
