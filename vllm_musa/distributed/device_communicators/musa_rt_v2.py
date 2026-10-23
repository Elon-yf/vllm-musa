"""v2 用的 MUSA runtime 适配层：在 v1 的 `_MusaRTLibrary` 上补齐上游 API 名。

移植时发现的一处**接口差异**（不是 bug，是命名风格差异）：

- vllm-musa v1 的 `_MusaRTLibrary` 暴露的是 snake_case 包装
  （`malloc` / `memset` / `ipc_get_mem_handle` / `ipc_open_mem_handle` /
  `ipc_close_mem_handle` / `free`），没有 `cudaMemcpy`；
- 上游 v2 对象（SGLang MR 577 `custom_all_reduce_v2.py`）直接按 CUDA 原名牌
  调用（`cudaMalloc` / `cudaMemset` / `cudaIpcGetMemHandle` / `cudaMemcpy` …）。

这里不改 v1 的类（1926 行生产文件、v1 路径正在跑 A/B，不动它），而是用子类
把上游名字映射到 v1 已有的包装，并额外绑一个 `cudaMemcpy`（v1 用
`musaMemcpyAsync` 在 C++ 里做 staging，Python 层此前不需要 host 侧 memcpy）。

为什么必须用 `libmusart.so` 的裸 `cudaMalloc` 而不是 `torch.musa` 分配：
`share_storage()` 的 IPC handle 协议没有 offset 字段，所以分配基址必须等于
`data_ptr`；caching allocator 的张量可能是 suballocation，不能用在 v2 的
一块式（signal/scratch/buffer/ring）IPC 布局上。
"""

from __future__ import annotations

import ctypes

from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
    _Function,
    _MusaRTLibrary,
    cudaError_t,
    cudaIpcMemHandle_t,
)

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToDevice = 3


def _as_void_p(value):
    """Return ``value`` as a ``ctypes.c_void_p`` without double-wrapping."""
    if isinstance(value, ctypes.c_void_p):
        return value
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return ctypes.c_void_p(int(data_ptr()))
    return ctypes.cast(value, ctypes.c_void_p)


class _MusaRTLibraryV2(_MusaRTLibrary):
    """给 v2 对象提供 CUDA 原名牌的 runtime 访问（复用 v1 的 ctypes 绑定）。"""

    def __init__(self, so_file: str = "libmusart.so") -> None:
        super().__init__(so_file)
        if "cudaMemcpy" not in self.funcs:
            spec = _Function(
                "cudaMemcpy",
                cudaError_t,
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_uint,
                ],
            )
            func = getattr(self.lib, spec.name)
            func.restype = spec.restype
            func.argtypes = spec.argtypes
            self.funcs[spec.name] = func

    # ---- 上游 (SGLang MR 577) 使用的 API 名 -------------------------------
    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        return self.malloc(int(size))

    def cudaFree(self, ptr) -> None:
        self.free(ptr)

    def cudaMemset(self, ptr, value: int, count: int) -> None:
        self.memset(ptr, int(value), int(count))

    def cudaIpcGetMemHandle(self, ptr) -> cudaIpcMemHandle_t:
        return self.ipc_get_mem_handle(ptr)

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        return self.ipc_open_mem_handle(handle)

    def cudaIpcCloseMemHandle(self, ptr) -> None:
        self.ipc_close_mem_handle(ptr)

    def cudaMemcpy(self, dst, src, count: int, kind: int = cudaMemcpyHostToDevice) -> None:
        # The v2 object passes ``ctypes.c_void_p`` for both ends (it builds the
        # host staging buffer itself), while tensors and plain ints also reach
        # this shim.  Normalise instead of re-wrapping, because wrapping an
        # existing c_void_p raises "cannot be converted to pointer".
        self.check(
            self.funcs["cudaMemcpy"](
                _as_void_p(dst),
                _as_void_p(src),
                int(count),
                int(kind),
            )
        )


__all__ = [
    "_MusaRTLibraryV2",
    "cudaMemcpyHostToDevice",
    "cudaMemcpyDeviceToDevice",
    "cudaIpcMemHandle_t",
]
