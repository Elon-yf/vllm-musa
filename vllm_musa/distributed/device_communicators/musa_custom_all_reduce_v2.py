from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import ClassVar, NamedTuple

import torch
import torch.distributed as dist

# vllm-musa 侧对应物（名称差异见 MUSA-100062 §2 映射表）
from vllm_musa.jit_kernel.csrc.allreduce_algo import AllReduceAlgo
from vllm_musa.distributed.device_communicators.musa_rt_v2 import (
    _MusaRTLibraryV2 as CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm_musa.distributed.device_communicators import musa_ar_v2_launcher as jit_ar
from vllm_musa.distributed.device_communicators.musa_allreduce_rmsnorm_v2 import (
    MusaCustomAllReduceV2RMSNorm,
)

cudaMemcpyHostToDevice = 1  # musaMemcpyKind; 3.3 runtime 无 cuda* 别名，见 §2.1(6)

requires_cuda_workspace_preflight = False


class AllReduceFusionPattern(IntEnum):
    kARResidualRMSNorm = 0


class MusaAllReduceFusionWorkspace:
    backend = "musa_custom_ar_v2"

    def __init__(
        self,
        coordinator,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.coordinator = coordinator
        self.world_size = world_size
        self.rank = rank
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype

    def is_buffer_size_sufficient(
        self,
        *,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: bool | None = None,
    ) -> bool:
        del use_oneshot
        ca_comm = self.coordinator.ca_comm
        return (
            ca_comm is not None
            and not ca_comm.disabled
            and tp_size == self.world_size
            and num_tokens <= self.max_token_num
            and hidden_dim == self.hidden_dim
            and dtype == self.dtype
            and num_tokens * hidden_dim * dtype.itemsize <= ca_comm.max_size
        )

    def destroy(self) -> None:
        # The GroupCoordinator owns the communicator and its IPC allocation.
        self.coordinator = None


def _find_group_coordinator(device_group, world_size: int, rank: int):
    from sglang.srt.distributed import (
        get_attn_tp_group,
        get_moe_ep_group,
        get_moe_tp_group,
        get_tp_group,
    )

    getters = (get_attn_tp_group, get_moe_ep_group, get_moe_tp_group, get_tp_group)
    for getter in getters:
        try:
            coordinator = getter()
        except AssertionError:
            continue
        if (
            coordinator.device_group is device_group
            and coordinator.world_size == world_size
            and coordinator.rank_in_group == rank
        ):
            return coordinator
    raise RuntimeError(
        "MUSA allreduce-fusion workspace could not find the matching "
        f"GroupCoordinator for rank={rank}, world_size={world_size}"
    )


def create_allreduce_fusion_workspace(
    *,
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    force_oneshot_support: bool = False,
    group=None,
    **kwargs,
) -> MusaAllReduceFusionWorkspace:
    del force_oneshot_support, kwargs
    if backend != "trtllm":
        raise ValueError(f"Unsupported MUSA allreduce-fusion backend: {backend}")
    if group is None:
        raise ValueError("MUSA allreduce-fusion workspace requires a device group")
    coordinator = _find_group_coordinator(group, world_size, rank)
    ca_comm = coordinator.ca_comm
    if ca_comm is None or ca_comm.disabled:
        raise RuntimeError("MUSA custom allreduce v2 communicator is unavailable")
    return MusaAllReduceFusionWorkspace(
        coordinator, world_size, rank, max_token_num, hidden_dim, dtype
    )


def allreduce_fusion(
    *,
    input: torch.Tensor,
    workspace: MusaAllReduceFusionWorkspace,
    pattern: AllReduceFusionPattern,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    launch_with_pdl: bool = True,
    trigger_completion_at_end: bool = False,
    use_oneshot: bool | None = None,
    fp32_acc: bool = False,
) -> bool:
    del launch_with_pdl, trigger_completion_at_end, use_oneshot, fp32_acc
    if pattern != AllReduceFusionPattern.kARResidualRMSNorm:
        raise ValueError(f"Unsupported MUSA allreduce-fusion pattern: {pattern}")
    ca_comm = workspace.coordinator.ca_comm
    result = ca_comm.fused_allreduce_rmsnorm(
        input,
        residual_in,
        rms_gamma,
        rms_eps,
        norm_out=norm_out,
        residual_out=residual_out,
    )
    return result is not None


class ConfigResult(NamedTuple):
    num_blocks: int
    num_threads: int


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _handle_to_list(handle: cudaIpcMemHandle_t) -> list[int]:
    return list(bytes(handle))


def _handle_from_list(values: list[int]) -> cudaIpcMemHandle_t:
    raw = bytes(int(value) & 0xFF for value in values)
    if len(raw) != ctypes.sizeof(cudaIpcMemHandle_t):
        raise ValueError(
            "Invalid MUSA IPC handle size: "
            f"{len(raw)} != {ctypes.sizeof(cudaIpcMemHandle_t)}"
        )
    return cudaIpcMemHandle_t.from_buffer_copy(raw)


class _AllReduceData(ctypes.Structure):
    """Host-side rank descriptor matching CUDA's ``AllReduceData`` layout."""

    _fields_ = [("input", ctypes.c_void_p * 8)]


class MusaCustomAllReduceV2Obj:
    """MUSA implementation of CUDA custom-all-reduce v2's operator ABI.

    The public ``CustomAllReduceV2`` Python control plane owns process-group
    coordination and calls this object through the same methods as the CUDA
    TVM-FFI object. Storage, MUSA IPC, and kernel selection remain backend
    details here. CUDA Graph input registration follows the shared v2 ABI.
    """

    supported_world_sizes = (2, 4, 6, 8)
    # Consumed by the shared CustomAllReduceV2._determine_algo() control plane.
    # S5000 thresholds are selected from eager-mode crossover measurements.
    # Pull covers TP2/TP4 payloads; TP4/TP6 switch to two-shot for larger
    # payloads. Push remains available through an explicit algorithm override.
    mode_configs: ClassVar[dict[int, tuple[int, int]]] = {
        2: (0, 1 << 60),
        4: (0, 256 * 1024),
        6: (0, 256 * 1024),
        8: (0, 0),
    }

    def __init__(
        self,
        rank: int,
        world_size: int,
        pull_buffer_bytes: int,
        push_buffer_bytes: int,
        graph_input_count: int,
        *,
        max_pull_blocks: int | None = None,
        max_push_blocks: int | None = None,
    ) -> None:
        del max_pull_blocks, max_push_blocks
        if world_size not in self.supported_world_sizes:
            raise ValueError(
                f"MUSA custom all-reduce v2 does not support world_size={world_size}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid rank={rank} for world_size={world_size}")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self._pull_buffer_bytes = int(pull_buffer_bytes)
        self._push_buffer_bytes = int(push_buffer_bytes)
        self._opened_ipc_ptrs: dict[bytes, int] = {}
        self._graph_input_count = int(graph_input_count)
        self._is_graph_capturing = False
        self._graph_capture_inputs: list[int] = []
        self._cum_registered_count = 0
        self._push_launchers = {}
        self._registered_launchers = {}
        self._unregistered_launchers = {}
        self._shot_cache: dict[tuple[int, AllReduceAlgo], int] = {}
        self._closed = False
        self.disabled = True

        # MUSA push uses two staging epochs and one slot per source rank.
        self._push_storage_bytes = jit_ar.push_buffer_bytes(
            self._push_buffer_bytes, self.world_size
        )
        ar_signal_bytes = jit_ar.allreduce_v2_meta_size()
        fused_signal_bytes = jit_ar.meta_size(self.world_size)
        # Match CUDA v2's controller layout: push counters are independent,
        # while one-shot and two-shot share one pull semaphore protocol. The
        # fused protocol gets a disjoint signal/scratch region in this same IPC
        # allocation because its barrier generations have a different lifetime.
        self._push_signal_offset = 0
        self._pull_signal_offset = _align(self._push_signal_offset + ar_signal_bytes)
        # Registered two-shot and fused kernels address scratch as
        # ``signal + 1``. Keep both reservations disjoint from each other and
        # from staging data.
        self._scratch_bytes = self._pull_buffer_bytes
        self._fused_signal_offset = _align(
            self._pull_signal_offset + ar_signal_bytes + self._scratch_bytes
        )
        self._pull_buffer_offset = _align(
            self._fused_signal_offset + fused_signal_bytes + self._scratch_bytes
        )
        # CUDA v2 keeps pull payload and push sentinel-ring storage disjoint.
        # Pull writes are arbitrary nonzero values and would otherwise look
        # like already-published push packets after an algorithm transition.
        self._push_buffer_offset = _align(
            self._pull_buffer_offset + self._pull_buffer_bytes
        )
        storage_bytes = self._push_buffer_offset + self._push_storage_bytes

        # This module is MUSA-only.  Keep the runtime-neutral current-device
        # lookup, but use the MUSA device type so allocations do not dispatch
        # to CUDA's ``aten::empty`` backend.
        device = torch.device("musa", torch.get_device_module().current_device())
        self._device = device
        self._runtime = CudaRTLibrary()
        # share_storage() has no offset field, so this must be a runtime
        # allocation whose data pointer is also its IPC allocation base.  A
        # caching-allocator tensor may be a suballocation and is not valid for
        # this ABI.
        self._storage_allocation = self._runtime.cudaMalloc(storage_bytes)
        self._runtime.cudaMemset(self._storage_allocation, 0, storage_bytes)
        self._graph_rank_data = torch.zeros(
            (self._graph_input_count, 8), dtype=torch.int64, device=device
        )
        self._storage_ptr = int(self._storage_allocation.value)
        self._self_push_signal_ptr = self._storage_ptr + self._push_signal_offset
        self._self_pull_signal_ptr = self._storage_ptr + self._pull_signal_offset
        self._self_fused_signal_ptr = self._storage_ptr + self._fused_signal_offset
        self._self_buffer_ptr = self._storage_ptr + self._pull_buffer_offset
        self._self_push_buffer_ptr = self._storage_ptr + self._push_buffer_offset
        self._rank_data_cpu: torch.Tensor | None = None
        self._rank_data_device: torch.Tensor | None = None
        self._push_rank_data_device: torch.Tensor | None = None
        self._push_signal_ptrs_cpu: torch.Tensor | None = None
        self._pull_signal_ptrs_cpu: torch.Tensor | None = None
        self._pull_config = ConfigResult(0, 0)
        # Fused RMSNorm shares the v2 pull workspace and owns only its protocol
        # state inside the same allocation. Its capacity therefore follows the
        # pull workspace even if a caller configures a larger push workspace.
        self.max_size = self._pull_buffer_bytes
        self._fused_signal_ptrs_cpu: torch.Tensor | None = None
        self._fused_signal_ptrs: list[int] = []
        self._pull_buffer_ptrs: list[int] = []
        self._fused_rmsnorm = MusaCustomAllReduceV2RMSNorm(self)
        jit_ar.ensure_compiled(self.world_size)

    def share_storage(self) -> list[int]:
        handle = self._runtime.cudaIpcGetMemHandle(ctypes.c_void_p(self._storage_ptr))
        return _handle_to_list(handle)

    def post_init(self, handles: list[list[int]]) -> None:
        if len(handles) != self.world_size:
            raise ValueError(
                f"Expected {self.world_size} storage handles, got {len(handles)}"
            )
        storage_ptrs: list[int] = []
        for peer_rank, values in enumerate(handles):
            if peer_rank == self.rank:
                storage_ptrs.append(self._storage_ptr)
                continue
            key = bytes(int(value) & 0xFF for value in values)
            opened = self._opened_ipc_ptrs.get(key)
            if opened is None:
                opened = int(
                    self._runtime.cudaIpcOpenMemHandle(_handle_from_list(values)).value
                )
                self._opened_ipc_ptrs[key] = opened
            storage_ptrs.append(opened)

        buffer_ptrs = [ptr + self._pull_buffer_offset for ptr in storage_ptrs]
        push_buffer_ptrs = [ptr + self._push_buffer_offset for ptr in storage_ptrs]
        buffer_ptrs.extend([0] * (8 - self.world_size))
        push_buffer_ptrs.extend([0] * (8 - self.world_size))
        push_signal_ptrs = [ptr + self._push_signal_offset for ptr in storage_ptrs]
        pull_signal_ptrs = [ptr + self._pull_signal_offset for ptr in storage_ptrs]
        fused_signal_ptrs = [ptr + self._fused_signal_offset for ptr in storage_ptrs]
        self._rank_data_cpu = torch.tensor(buffer_ptrs, dtype=torch.int64)
        self._rank_data_device = torch.tensor(
            buffer_ptrs, dtype=torch.int64, device=self._device
        )
        # Polling push consumes a stable device descriptor. Keeping this in
        # the communicator workspace avoids passing RankData by value to every
        # kernel (64 bytes/thread of private spill on MP31).
        self._push_rank_data_device = torch.tensor(
            push_buffer_ptrs, dtype=torch.int64, device=self._device
        )
        self._push_signal_ptrs_cpu = torch.tensor(push_signal_ptrs, dtype=torch.int64)
        self._pull_signal_ptrs_cpu = torch.tensor(pull_signal_ptrs, dtype=torch.int64)
        self._fused_signal_ptrs_cpu = torch.tensor(fused_signal_ptrs, dtype=torch.int64)
        self._fused_signal_ptrs = fused_signal_ptrs
        self._pull_buffer_ptrs = buffer_ptrs[: self.world_size]
        self.disabled = False

    def set_cuda_graph_capture(self, is_capturing: bool) -> None:
        self._is_graph_capturing = bool(is_capturing)

    def _allocation_base_and_offset(self, ptr: int) -> tuple[int, int]:
        lib = ctypes.CDLL("libmusa.so")
        get_attribute = lib.muPointerGetAttribute
        get_attribute.restype = ctypes.c_int
        get_attribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        base_ptr = ctypes.c_void_p()
        error = get_attribute(
            ctypes.byref(base_ptr),
            11,  # MU_POINTER_ATTRIBUTE_RANGE_START_ADDR
            ctypes.c_ulonglong(ptr),
        )
        if error != 0 or base_ptr.value is None:
            raise RuntimeError(f"muPointerGetAttribute failed: {error}")
        base = int(base_ptr.value)
        return base, ptr - base

    def share_graph_inputs(self):
        result = []
        handle_cache: dict[int, list[int]] = {}
        for ptr in self._graph_capture_inputs[self._cum_registered_count :]:
            base, offset = self._allocation_base_and_offset(ptr)
            handle = handle_cache.get(base)
            if handle is None:
                handle = _handle_to_list(
                    self._runtime.cudaIpcGetMemHandle(ctypes.c_void_p(base))
                )
                handle_cache[base] = handle
            result.append((offset, handle))
        return result

    def register_inputs(self, rank_inputs) -> None:
        if len(rank_inputs) != self.world_size:
            raise ValueError(
                f"Expected graph inputs from {self.world_size} ranks, "
                f"got {len(rank_inputs)}"
            )
        count = len(self._graph_capture_inputs) - self._cum_registered_count
        if count == 0:
            return
        if any(len(inputs) != count for inputs in rank_inputs):
            raise ValueError("MUSA custom AR v2 graph input counts differ across ranks")

        data = (_AllReduceData * count)()
        for index in range(count):
            row = data[index].input
            for peer_rank, inputs in enumerate(rank_inputs):
                offset, handle_values = inputs[index]
                if peer_rank == self.rank:
                    ptr = self._graph_capture_inputs[self._cum_registered_count + index]
                else:
                    key = bytes(int(value) & 0xFF for value in handle_values)
                    base = self._opened_ipc_ptrs.get(key)
                    if base is None:
                        base = int(
                            self._runtime.cudaIpcOpenMemHandle(
                                _handle_from_list(handle_values)
                            ).value
                        )
                        self._opened_ipc_ptrs[key] = base
                    ptr = base + int(offset)
                row[peer_rank] = ctypes.c_void_p(ptr)

        begin = self._cum_registered_count
        end = begin + count
        # Match CUDA's native register_graph_buffers contract.  The rank
        # table is a tiny host-to-device descriptor update; using a
        # synchronous runtime memcpy makes this copy complete before the
        # next graph replay without a device-wide synchronize.
        dst = ctypes.c_void_p(
            self._graph_rank_data.data_ptr() + begin * ctypes.sizeof(_AllReduceData)
        )
        self._runtime.cudaMemcpy(
            dst,
            ctypes.cast(data, ctypes.c_void_p),
            ctypes.sizeof(data),
            cudaMemcpyHostToDevice,
        )
        self._cum_registered_count = end

    def _check_ready(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rank_data_cpu is None
            or self._rank_data_device is None
            or self._push_rank_data_device is None
            or self._push_signal_ptrs_cpu is None
            or self._pull_signal_ptrs_cpu is None
        ):
            raise RuntimeError("MUSA custom all-reduce v2 post_init() was not called")
        return self._rank_data_cpu, self._pull_signal_ptrs_cpu

    def _select_shot(self, nbytes: int, algo: AllReduceAlgo) -> int:
        if not isinstance(algo, AllReduceAlgo):
            raise TypeError(f"Unsupported custom all-reduce v2 algorithm: {algo}")
        key = (int(nbytes), algo)
        cached = self._shot_cache.get(key)
        if cached is not None:
            return cached

        if jit_ar.is_shot_forced():
            shot = jit_ar.preferred_shot(self.world_size, nbytes)
        elif algo == AllReduceAlgo.ONE_SHOT_PUSH:
            shot = jit_ar.SHOT_PUSH
        elif algo == AllReduceAlgo.ONE_SHOT_PULL:
            shot = jit_ar.SHOT_ONE_STAGE
        else:
            # SHOT_TWO_STAGE_512 is a launch-shape/kernel variant of the v2
            # TWO_SHOT_PULL algorithm, not a separate algorithm selector.
            if self.world_size == 8 or (self.world_size == 4 and nbytes <= 512 * 1024):
                shot = jit_ar.SHOT_TWO_STAGE_512
            else:
                shot = jit_ar.SHOT_TWO_STAGE
        is_push = shot in (jit_ar.SHOT_PUSH, jit_ar.SHOT_PUSH_WIDE)
        required = (
            jit_ar.push_buffer_bytes(nbytes, self.world_size) if is_push else nbytes
        )
        # Pull payload and push ring have independent capacities and storage.
        capacity = self._push_storage_bytes if is_push else self._pull_buffer_bytes
        if required > capacity:
            raise RuntimeError(
                f"MUSA custom AR v2 algorithm {algo.name} needs {required} bytes, "
                f"but its workspace has {capacity}"
            )
        self._shot_cache[key] = shot
        return shot

    def all_reduce(self, input: torch.Tensor, algo: AllReduceAlgo) -> torch.Tensor:
        rank_data, signal_ptrs = self._check_ready()
        if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported MUSA custom AR dtype: {input.dtype}")
        if input.device != self._device:
            raise ValueError(
                f"MUSA custom AR v2 expected input on {self._device}, "
                f"got {input.device}"
            )
        if not input.is_contiguous():
            raise ValueError("MUSA custom AR v2 input must be contiguous")
        nbytes = input.numel() * input.element_size()
        if nbytes % 16 != 0:
            raise ValueError(
                f"MUSA custom AR v2 input bytes must be a multiple of 16: {nbytes}"
            )
        if input.data_ptr() % 16 != 0:
            raise ValueError("MUSA custom AR v2 input pointer must be 16-byte aligned")
        if input.numel() > (1 << 31) - 1:
            raise ValueError("MUSA custom AR v2 input exceeds its int32 item limit")
        if (
            self._is_graph_capturing
            and self.world_size == 4
            and 256 * 1024 < nbytes <= 512 * 1024
        ):
            # TP4 graph replay uses the registered two-shot-512 kernel only
            # for the middle payload range; smaller payloads stay one-shot
            # pull to avoid forcing a slower two-shot decode path.
            algo = AllReduceAlgo.TWO_SHOT_PULL
        shot = self._select_shot(nbytes, algo)
        graph_launch = (
            self._is_graph_capturing
            and torch.get_device_module().is_current_stream_capturing()
        )
        # Match CUDA v2: registered one-shot pull writes a distinct output,
        # while registered two-shot broadcasts reduced shards back into each
        # rank's graph input and therefore remains in-place.
        output = (
            torch.empty_like(input)
            if graph_launch and shot == jit_ar.SHOT_ONE_STAGE
            else input
        )
        if shot in (jit_ar.SHOT_PUSH, jit_ar.SHOT_PUSH_WIDE):
            assert self._push_signal_ptrs_cpu is not None
            launch_signal_ptrs = self._push_signal_ptrs_cpu
            self_signal_ptr = self._self_push_signal_ptr
            self_buffer_ptr = self._self_push_buffer_ptr
        else:
            launch_signal_ptrs = signal_ptrs
            self_signal_ptr = self._self_pull_signal_ptr
            self_buffer_ptr = self._self_buffer_ptr

        if shot in (jit_ar.SHOT_PUSH, jit_ar.SHOT_PUSH_WIDE):
            assert self._push_rank_data_device is not None
            launcher = self._push_launchers.get(shot)
            if launcher is None:
                launcher = jit_ar.launch_device_descriptor_func(self.world_size, shot)
                self._push_launchers[shot] = launcher
            launcher(
                self._push_rank_data_device,
                launch_signal_ptrs,
                output,
                self_signal_ptr,
                self.rank,
                self.world_size,
                shot,
            )
        elif graph_launch:
            index = len(self._graph_capture_inputs)
            if index >= self._graph_input_count:
                raise RuntimeError(
                    "MUSA custom AR v2 graph pointer table overflow: "
                    f"{index} >= {self._graph_input_count}"
                )
            self._graph_capture_inputs.append(int(input.data_ptr()))
            launcher = self._registered_launchers.get(shot)
            if launcher is None:
                launcher = jit_ar.launch_registered_pybind_func(self.world_size, shot)
                self._registered_launchers[shot] = launcher
            launcher(
                self._graph_rank_data[index],
                launch_signal_ptrs,
                output,
                self_signal_ptr,
                self.rank,
                self.world_size,
                shot,
            )
        else:
            launcher = self._unregistered_launchers.get(shot)
            if launcher is None:
                launcher = jit_ar.launch_unregistered_pybind_func(self.world_size, shot)
                self._unregistered_launchers[shot] = launcher
            launcher(
                rank_data,
                launch_signal_ptrs,
                input,
                output,
                self_signal_ptr,
                self_buffer_ptr,
                self._push_buffer_bytes,
                self.rank,
                self.world_size,
                shot,
            )
        return output

    def fused_allreduce_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """FlashInfer-compatible AR + residual + RMSNorm operator ABI."""
        return self._fused_rmsnorm.fused_allreduce_rmsnorm(
            input,
            residual,
            weight,
            eps,
            norm_out=norm_out,
            residual_out=residual_out,
        )

    def fused_allreduce_gemma_dual_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_: torch.Tensor,
        post_weight_: torch.Tensor,
        next_weight_: torch.Tensor,
        post_eps: float,
        next_eps: float,
        *,
        attention: bool,
        scalar_: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._fused_rmsnorm.fused_allreduce_gemma_dual_rmsnorm(
            input_,
            residual_,
            post_weight_,
            next_weight_,
            post_eps,
            next_eps,
            attention=attention,
            scalar_=scalar_,
        )

    def config_pull(self, num_blocks: int = -1, num_threads: int = -1) -> ConfigResult:
        old = self._pull_config
        self._pull_config = ConfigResult(
            old.num_blocks if num_blocks == -1 else num_blocks,
            old.num_threads if num_threads == -1 else num_threads,
        )
        return old

    def free(self, group: torch.distributed.ProcessGroup) -> None:
        if self._closed:
            return
        if dist.is_initialized():
            dist.barrier(group=group)
        for ptr in self._opened_ipc_ptrs.values():
            self._runtime.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
        self._opened_ipc_ptrs.clear()
        self._runtime.cudaFree(self._storage_allocation)
        self._rank_data_device = None
        self._push_rank_data_device = None
        self._graph_rank_data = None
        self._push_launchers.clear()
        self._registered_launchers.clear()
        self._unregistered_launchers.clear()
        self._shot_cache.clear()
        self.disabled = True
        self._closed = True


def get_musa_custom_all_reduce_cls():
    return MusaCustomAllReduceV2Obj
