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
    # MUSA-0062 (torch 2.7.1): removed MUSA-0052's `world_size > 2` CAR
    # gate, re-enabling custom_all_reduce on MUSA for TP>2. The
    # compile-path safety (Inductor lowering past the Python alignment
    # gate) was handled at the kernel level — see generated/musa0062/.
    #
    # MUSA-0069 (torch >= 2.9): torch 2.9's Inductor lowering produces
    # buffer shapes the MUSA CAR kernel rejects ("input length must be
    # multiple of 4"). Re-instate the MUSA-0052-style `world_size > 2`
    # disable, but only for torch >= 2.9 + MUSA, so MUSA-0062's
    # re-enablement on torch 2.7.1 still stands. Recovering CAR perf on
    # torch 2.9 is a possible kernel/lowering follow-up. See
    # generated/musa0069/.
    (
        """        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False""",
        """        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        # MUSA-0069: on torch >= 2.9, the Inductor compile-path produces
        # buffer shapes the MUSA CAR kernel rejects ("input length must
        # be multiple of 4"). MUSA-0062's gate-removal was torch-2.7.1-
        # only; for torch >= 2.9 + MUSA + world_size > 2, restore the
        # MUSA-0052 disable. Net perf is the same as the
        # `--disable-custom-all-reduce` CLI workaround; a kernel-level
        # fix to recover CAR perf on torch 2.9 is a possible follow-up.
        import torch as _torch
        from vllm.platforms import current_platform as _cp
        try:
            _tv = _torch.__version__.split("+")[0].split(".")
            _torch_29plus = (int(_tv[0]), int(_tv[1])) >= (2, 9)
        except (ValueError, IndexError):
            _torch_29plus = False
        if _cp.is_musa() and self.world_size > 2 and _torch_29plus:
            return False""",
    ),
]
