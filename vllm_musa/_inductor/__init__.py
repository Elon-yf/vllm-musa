# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inductor integration hooks for MUSA device type.

Registers Inductor template heuristics for ``device_type="musa"`` so
compiled ``mm``/``bmm``/``addmm``/``baddbmm``/``scaled_mm`` ops can
attempt Triton autotune instead of falling through to the empty
fallback heuristic (which Inductor maps to the ATen path).

The registration is **opportunistic**: it is wrapped in try/except so
old torch versions or future API changes don't crash the platform.
"""

from vllm_musa._inductor.template_heuristics import (
    maybe_register_musa_template_heuristics,
)

__all__ = ["maybe_register_musa_template_heuristics"]
