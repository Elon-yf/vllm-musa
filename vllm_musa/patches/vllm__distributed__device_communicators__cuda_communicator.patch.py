# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0088 iter-2: extend the upstream CudaCommunicator qr_comm gate to
also initialize QuickAllReduce on MUSA.

Upstream code at vllm/vllm/distributed/device_communicators/cuda_communicator.py:

    if current_platform.is_rocm():
        # Initialize a custom quick all-reduce implementation for AMD.
        ...
        self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

This patch extends the gate to also cover MUSA. The QuickAllReduce
implementation used is vllm-musa's local one (mirrors the upstream API
shape: __init__(group, device)).

The patch is a no-op until vllm-musa/csrc/quick_all_reduce.cu compiles
successfully — the vllm_musa QuickAllReduce class detects missing
torch.ops._C_quick_ar.* and sets `disabled = True`, so the dispatch
in cuda_communicator.all_reduce() falls through to the next path.
"""

_OLD = """            if current_platform.is_rocm():
                # Initialize a custom quick all-reduce implementation for AMD.
                # Quick reduce is designed as a complement to custom allreduce.
                # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                # If it's a rocm, 'use_custom_allreduce==True' means it must
                # currently be an MI300 series.
                self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)
"""

_NEW = """            if current_platform.is_rocm():
                # Initialize a custom quick all-reduce implementation for AMD.
                # Quick reduce is designed as a complement to custom allreduce.
                # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                # If it's a rocm, 'use_custom_allreduce==True' means it must
                # currently be an MI300 series.
                self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)
            elif current_platform.is_musa():
                # MUSA-0088: extend qr_comm to MUSA. Uses vllm-musa's local
                # QuickAllReduce (csrc/quick_all_reduce.cu ported from SGLang).
                # Falls back to disabled if the kernel hasn't compiled yet —
                # cuda_communicator.all_reduce() then routes to custom_all_reduce
                # / mccl as before.
                from vllm_musa.distributed.device_communicators.quick_all_reduce import (
                    QuickAllReduce as MUSAQuickAllReduce,
                )

                # API shape: MUSAQuickAllReduce(group, world_size, rank, max_size).
                # Derive world_size + rank from the cpu_group.
                import torch.distributed as _dist

                _ws = _dist.get_world_size(group=self.cpu_group)
                _rank = _dist.get_rank(group=self.cpu_group)
                self.qr_comm = MUSAQuickAllReduce(
                    group=self.cpu_group, world_size=_ws, rank=_rank
                )
"""

PATCHES = [(_OLD, _NEW)]
