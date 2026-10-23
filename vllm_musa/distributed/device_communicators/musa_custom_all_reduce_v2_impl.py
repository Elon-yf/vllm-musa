"""MUSA custom all-reduce v2 implementation, selected by an environment switch.

SGLang MR 577 selects its all-reduce implementation once, at construction time,
by returning a different class from ``dispatch_custom_allreduce()``
(``SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2``); the callers keep calling the same six
public methods and never learn which implementation is behind them.

This module provides the same thing for vllm-musa: ``_MusaCustomAllreduceV2Impl``
implements the surface that ``MusaJitCustomAllreduce`` already delegates to, so
``VLLM_MUSA_CUSTOM_AR_V2=1`` swaps the implementation at construction and the
hot path keeps calling a single bound method.  There is no per-call decision
between v1 and v2 and no fallback: if v2 cannot be brought up, construction
raises.

What comes from the MR and what is local glue:

* kernel and workspace/objects: ``custom_all_reduce_v2.mu`` and
  ``musa_custom_all_reduce_v2.py``, both ported verbatim (the kernel is
  byte-identical after the symbol rename).
* algorithm choice: the MR's ``ModeConfig`` table, taken from the ported
  object's ``mode_configs`` exactly as the MR's ``_init_config`` does.  On TP2
  the push threshold is 0, so payloads take ONE_SHOT_PULL / TWO_SHOT_PULL; push
  is reachable only through an explicit algorithm override, matching upstream.
* this class: the local equivalent of the MR's ``CustomAllReduceV2`` control
  plane, reduced to the members vllm-musa's wrapper actually consumes.  Fused
  all-reduce + RMSNorm and custom all-gather are not part of the ported kernel
  surface, so they report "unsupported" here instead of silently running the v1
  kernels (which use a different barrier protocol).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from vllm_musa.jit_kernel.csrc.allreduce_algo import AllReduceAlgo
from vllm_musa.distributed.device_communicators.musa_custom_all_reduce_v2 import (
    MusaCustomAllReduceV2Obj,
)
from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
    _register_comm,
)

logger = init_logger(__name__)

# Upstream's MUSA default pull workspace (``_init_config`` path in MR 577).
_DEFAULT_PULL_BYTES = 512 * 1024 * 1024
# Upstream hard-codes this for the graph input pointer table.
_GRAPH_INPUT_COUNT = 131072


def _is_torch_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False


def _is_current_stream_capturing() -> bool:
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        check = getattr(module, "is_current_stream_capturing", None)
        if check is None:
            continue
        try:
            return bool(check())
        except Exception:
            continue
    return False


def _is_weak_contiguous(tensor: torch.Tensor) -> bool:
    """Same contract as upstream's ``is_weak_contiguous``."""
    if tensor.is_contiguous():
        return True
    storage = tensor.untyped_storage()
    return tensor.storage_offset() + tensor.numel() == (
        storage.nbytes() // tensor.element_size()
    )


class _MusaCustomAllreduceV2Impl:
    """v2 all-reduce with the interface ``MusaJitCustomAllreduce`` delegates to."""

    _MAX_CAR_SIZE = _DEFAULT_PULL_BYTES

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = _DEFAULT_PULL_BYTES,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = device
        self._IS_CAPTURING = False
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        mode_config = MusaCustomAllReduceV2Obj.mode_configs.get(self.world_size)
        if mode_config is None:
            raise RuntimeError(
                "MUSA custom all-reduce v2 supports world sizes "
                f"{sorted(MusaCustomAllReduceV2Obj.mode_configs)} only, "
                f"got {self.world_size}"
            )
        self._push_threshold, self._pull_threshold = (
            int(mode_config[0]),
            int(mode_config[1]),
        )
        # The MR sizes the push ring from the push threshold, which is 0 for the
        # world sizes here; push therefore needs an explicit override to run.
        self.max_push_size = self._push_threshold
        self.max_pull_size = int(max_size)
        self.max_size = max(self.max_pull_size, self.max_push_size)
        self._algo_override: AllReduceAlgo | None = None

        self.obj = MusaCustomAllReduceV2Obj(
            rank=self.rank,
            world_size=self.world_size,
            pull_buffer_bytes=self.max_pull_size,
            push_buffer_bytes=self.max_push_size,
            graph_input_count=_GRAPH_INPUT_COUNT,
        )
        handles: list[Any] = [None] * self.world_size
        dist.all_gather_object(handles, self.obj.share_storage(), group=group)
        self.obj.post_init(handles)
        # 复用现有 CAR 的自定义 op：`vllm.musa_jit_custom_all_reduce` 通过
        # comm_id 回调 `_custom_all_reduce_impl`，Dynamo 只看到一个不透明算子，
        # 不会去 trace data_ptr()/ctypes 这些无法编译的东西。
        self._comm_id = _register_comm(self)
        self.disabled = False
        logger.info_once(
            "MUSA custom all-reduce v2 enabled: world_size=%d pull_bytes=%d "
            "push_bytes=%d push_threshold=%d pull_threshold=%d "
            "(payloads above the push threshold take ONE_SHOT_PULL)",
            self.world_size,
            self.max_pull_size,
            self.max_push_size,
            self._push_threshold,
            self._pull_threshold,
        )

    # -- algorithm selection -------------------------------------------------
    def override_algo(self, algo: Optional[AllReduceAlgo]) -> None:
        """Explicit algorithm override, mirroring upstream's ``override_algo``."""
        self._algo_override = algo

    def _pick_algo(self, nbytes: int) -> AllReduceAlgo:
        if self._algo_override is not None:
            return self._algo_override
        push_threshold = min(self._push_threshold, self.max_push_size)
        pull_threshold = min(self._pull_threshold, self.max_pull_size)
        if nbytes <= push_threshold:
            return AllReduceAlgo.ONE_SHOT_PUSH
        if nbytes <= pull_threshold:
            return AllReduceAlgo.ONE_SHOT_PULL
        return AllReduceAlgo.TWO_SHOT_PULL

    # -- public surface used by MusaJitCustomAllreduce -----------------------
    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not _is_weak_contiguous(inp):
            return False
        return inp_size <= self.max_size

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        if self._IS_CAPTURING:
            if not _is_current_stream_capturing():
                # 图预热阶段不应真的通信，与 v1/vLLM 的 warmup 语义一致。
                return torch.empty_like(input)
            return torch.ops.vllm.musa_jit_custom_all_reduce(input, self._comm_id)
        if _is_torch_compiling():
            # 让 Dynamo 在图里看到一个已注册算子，而不是 Python/ctypes 实现。
            return torch.ops.vllm.musa_jit_custom_all_reduce(input, self._comm_id)
        return self._custom_all_reduce_impl(input)

    def _custom_all_reduce_impl(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        algo = self._pick_algo(nbytes)
        output = self.obj.all_reduce(input, algo)
        return output if isinstance(output, torch.Tensor) else torch.from_dlpack(output)

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        self._IS_CAPTURING = True
        try:
            self.obj.set_cuda_graph_capture(True)
            yield
        finally:
            self.obj.set_cuda_graph_capture(False)
            self._IS_CAPTURING = False
        if _is_current_stream_capturing():
            raise RuntimeError(
                "Cannot register MUSA custom AR v2 graph inputs while capturing"
            )
        pairs = self.obj.share_graph_inputs()
        handles = [handle for _, handle in pairs]
        offsets = [offset for offset, _ in pairs]
        handles_all = self._share_list(handles)
        offsets_all = self._share_list(offsets)
        self.obj.register_inputs(
            [list(zip(o, h)) for o, h in zip(offsets_all, handles_all)]
        )
        logger.info("Registering %d MUSA custom AR v2 graph addresses", len(pairs))

    def _share_list(self, values: list[int]) -> list[list[int]]:
        tensor = torch.tensor(values, dtype=torch.int64, device="cpu")
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self.group)
        return [g.tolist() for g in gathered]

    def close(self) -> None:
        if not getattr(self, "disabled", True) and hasattr(self, "obj"):
            self.obj.free(self.group)

    # -- features the ported kernel surface does not provide -----------------
    # Fused AR+RMSNorm v2 exists upstream, but its kernel path is not part of
    # this port; running the v1 fused kernels here would mix barrier protocols.
    def should_fused_allreduce_rmsnorm(
        self, inp: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return False

    def should_fused_allreduce_residual_rmsnorm(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return False

    def _reject_fused_allreduce_residual_rmsnorm_reason(
        self, inp: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> str:
        return "fused AR+RMSNorm is not implemented for the v2 kernel path"

    def fused_allreduce_rmsnorm(
        self, input: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError(
            "fused all-reduce + RMSNorm is not implemented for MUSA custom AR v2"
        )

    def fused_allreduce_residual_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError(
            "fused all-reduce + residual + RMSNorm is not implemented for "
            "MUSA custom AR v2"
        )

    def fused_allreduce_residual_rmsnorm_no_raw(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError(
            "fused all-reduce + residual + RMSNorm is not implemented for "
            "MUSA custom AR v2"
        )

    def should_custom_all_gather(
        self,
        input: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> bool:
        return False

    def custom_all_gather(
        self,
        input: torch.Tensor,
        dim: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "custom all-gather is not implemented for MUSA custom AR v2"
        )
