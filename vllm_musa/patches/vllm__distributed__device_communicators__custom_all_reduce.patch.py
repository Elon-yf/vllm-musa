# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.distributed.device.communicators.custom_all_reduce.
"""

PATCHES = [
    # Patch CustomAllreduce.max_size
    (
        "max_size=8192 * 1024,",
        "max_size=16 * 8192 * 1024,",
    ),
    # Use ray lead to the env MUSA_VISIBLE_DEVICES has some problem, and the patch can be deleted after fixed
    (
        "if cuda_visible_devices:",
        "if cuda_visible_devices and current_platform.is_cuda():",
    ),
    # Patch CustomAllreduce enable musa's custom_allreduce
    (
        "if not current_platform.is_rocm() and not _can_p2p(rank, world_size):",
        "if not current_platform.is_rocm() and not current_platform.is_musa() and not _can_p2p(rank, world_size):",
    ),
    # Upgrade the previous MUSA patch if it was already persisted on disk.
    (
        "if ( not current_platform.is_rocm() or not current_platform.is_musa() ) and not _can_p2p(rank, world_size):",
        "if not current_platform.is_rocm() and not current_platform.is_musa() and not _can_p2p(rank, world_size):",
    ),
    # MUSA-0062: the MUSA-0052 `world_size > 2` CAR gate has been
    # removed to re-enable custom_all_reduce on MUSA for TP>2. The
    # compile-path safety question (Inductor lowering `_C.all_reduce`
    # past the Python alignment gate) is handled at the kernel level —
    # see generated/musa0062/ for the diagnosis and fix.
]
