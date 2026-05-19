# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _pin_to_numa_local_cpus(local_rank: int) -> None:
    """MUSA-0122: pin worker to NUMA-local CPUs based on local_rank.

    On the 8xS5000 host (yeahdongcn60), per ``mthreads-gmi topo -m`` the
    CPU Affinity column maps:
      - GPUs 0-3 -> NUMA node 0 (CPUs 0-31, 64-95)
      - GPUs 4-7 -> NUMA node 1 (CPUs 32-63, 96-127)
    Without pinning, all 8 workers default to ``Cpus_allowed_list: 0-127``
    and the OS can schedule them cross-socket. This causes variable
    host-side memory access latency, which amplifies the AR barrier
    busy-wait time (each AR pays the max-of-N divergence cost).

    Opt out via ``VLLM_MUSA_DISABLE_NUMA_PIN=1``.
    """
    if os.environ.get("VLLM_MUSA_DISABLE_NUMA_PIN", "") == "1":
        return
    if not hasattr(os, "sched_setaffinity"):
        return
    if local_rank < 4:
        cpus = set(range(0, 32)) | set(range(64, 96))
        node = 0
    else:
        cpus = set(range(32, 64)) | set(range(96, 128))
        node = 1
    try:
        os.sched_setaffinity(0, cpus)
        logger.info(
            "MUSA-0122: worker local_rank=%d pinned to NUMA node %d (%d CPUs)",
            local_rank,
            node,
            len(cpus),
        )
    except Exception as exc:
        logger.warning(
            "MUSA-0122: failed to pin worker local_rank=%d to NUMA %d: %s",
            local_rank,
            node,
            exc,
        )


class MTGPUWorker(Worker):
    """A worker class that executes (a partition of) the model on a MTGPU.
    Each worker is associated with a single MTGPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        # MUSA-0122: pin BEFORE the parent constructor runs anything that
        # touches MUSA / memory allocation, so first allocations land on
        # the NUMA-local node.
        _pin_to_numa_local_cpus(local_rank)
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(
            self.model_runner.uniform_decode_query_len,
            uniform_decode=True,
        )
