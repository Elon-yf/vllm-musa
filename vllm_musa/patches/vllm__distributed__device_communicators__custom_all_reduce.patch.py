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
    # MUSA-0052: gate CAR on MUSA to world_size <= 2.
    # The C++ kernel (custom_all_reduce.cuh:614) asserts
    # `numel % (16/sizeof(T)) == 0`. Python's `should_custom_ar` enforces
    # this via `inp_size % 16 != 0`, so eager dispatch is safe. But
    # Inductor-compiled FX graphs lower `torch.ops._C.all_reduce`
    # directly, bypassing the Python gate; the kernel then sees
    # misaligned inputs on the MiniMax-M2.7 hot path and throws
    # (MUSA-0046 smoke #7). Until the MUSA kernel handles unaligned
    # tails (true MUSA-0052 follow-up subtask), restrict CAR on MUSA to
    # world_size <= 2 where the practical hot path is aligned and
    # tested. For TP>2, fall through to NCCL/equivalent.
    (
        """        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False""",
        """        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        # MUSA-0052: Inductor compile-mode lowers _C.all_reduce
        # bypassing this gate. Until the MUSA kernel handles unaligned
        # tails, restrict CAR on MUSA to world_size <= 2.
        from vllm.platforms import current_platform as _cp
        if _cp.is_musa() and self.world_size > 2:
            return False""",
    ),
]
