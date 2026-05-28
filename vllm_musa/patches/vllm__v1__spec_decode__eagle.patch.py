# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA spec-decode kernel monkey-patch shim.

Loaded by ``vllm_musa.patches`` as ``vllm.v1.spec_decode.eagle`` — the
filename's underscore-encoded module path is the patch system's lookup
key.

This file used to install ``EagleFullLoopRunner`` — a 912-line MUSA-only
custom runner that captured the full N-step Eagle3 draft loop as ONE
giant CUDAGraph. On yeahdongcn70 (mcc 5.1.0) the giant-chain capture
corrupted the draft model's MCCL barrier state at replay (accept rate
collapsed to 3.5% with positions 1-4 at 0%, GPU 4+6 firmware reboots
when override removed). See ticket MUSA-0203 for the full bisect.

The runner was deleted in favor of the upstream pattern from
https://github.com/vllm-project/vllm/pull/34880 — wrap the draft model
with the standard ``CUDAGraphWrapper`` + per-step ``CudagraphDispatcher``
keyed by ``BatchDescriptor`` (one captured graph per step, dispatched
fresh per call, structurally identical to the target's PIECEWISE
captures that already work on MUSA).

This file is now the minimal shim that ensures the MUSA-Triton-adapted
``eagle_prepare_next_token_padded_kernel`` (in ``vllm_musa/v1/spec_decode/utils.py``)
is imported BEFORE upstream's ``vllm.v1.spec_decode.llm_base_proposer``
binds its own copy of the (broken-on-MUSA) kernel. Without this prime
the proposer's Triton compile fails with ``mismatched type for
valid_count``.
"""

PATCHES: list = []  # No file mutation; the monkey-patch install is the side effect.

# CRITICAL ORDER: prime the MUSA-Triton-adapted kernel before upstream binds it.
import vllm_musa.v1.spec_decode.utils  # noqa: F401
