"""v2 发射层适配（P3）：把 v2 对象要的启动器名字接到 vllm-musa 的 jit_kernel 上。

P2 阶段本文件只做名字与 ABI 常量的适配；P3 落地了 v2 的 device-descriptor
发射路径：`distributed/custom_all_reduce_v2.mu`（移植自 SGLang MR 577）+ 
`vllm_musa/jit_kernel/csrc/allreduce_v2.py`（JIT 构建与 FFI 包装）。

仍然保持"不许静默回退"的原则：开 v2 就只用 v2 的 kernel。
- push（`SHOT_PUSH` / `SHOT_PUSH_WIDE`）走 v2 的 `launch_device_descriptor`；
- pull（`SHOT_ONE_STAGE` / `SHOT_TWO_STAGE` / `SHOT_TWO_STAGE_512`）走 v2 自己的
  `launch_unregistered` / `launch_registered`——即 `.mu` 里导出的两个
  ``extern "C"`` 入口，用 ctypes 绑定（上游用 pybind11 绑同一对符号）；
- 融合 AR+RMSNorm 的 v2 变体尚未移植，调用方会拿到明确的 NotImplementedError。

来源：SGLang MR 577 `python/sglang/srt/hardware_backend/musa/jit_kernel/csrc/allreduce.py`
（`allreduce_v2_meta_size` / `launch_device_descriptor_func` / SHOT_* 常量）。
"""

from __future__ import annotations

# ---- 与 MR 577 逐字对齐的常量 -------------------------------------------
SHOT_PUSH = 0
SHOT_ONE_STAGE = 1
SHOT_TWO_STAGE = 2
SHOT_PUSH_WIDE = 3
SHOT_TWO_STAGE_512 = 4

# .mu 里 Signal 的固定形状：kSignalMaxBlocks=512 个 128B 槽位，pull+push 两份
SIGNAL_MAX_BLOCKS = 512
SEMAPHORE_BYTES = 128

_PUSH_SHOTS = (SHOT_PUSH, SHOT_PUSH_WIDE)


def meta_size(world_size: int = 8) -> int:
    """固定 ABI：2 * 512 * 128 = 131072 B。

    v2 把 meta 大小钉死，形状调优不得改变 IPC/共享存储 ABI（MR 577 注释原话）。
    v1 的 meta_size 来自 C++ `vllm_musa_custom_ar_meta_size()`，两者不相等，
    移植时凡是按 v1 大小分配的地方都要跟着改。

    v2 对象用它算"融合路径预留区"大小（`fused_signal_bytes`）。融合 kernel
    尚未移植，但预留区仍按这个 ABI 尺寸留出，保证同一块 IPC 里的
    push/pull/scratch 偏移在将来接入融合路径时不需要重算。
    """
    del world_size
    return 2 * SIGNAL_MAX_BLOCKS * SEMAPHORE_BYTES


def allreduce_v2_meta_size() -> int:
    """v2 纯 all-reduce 的 Signal 区大小（v2 对象按此分配 push/pull 控制器）。"""
    return 2 * SIGNAL_MAX_BLOCKS * SEMAPHORE_BYTES


def push_buffer_bytes(nbytes: int, world_size: int) -> int:
    """push 模式每个 rank 的存储字节数（逐字移植自 MR 577 allreduce.py）。

    Push 用一个 v2 风格的 2 段环：stages(2) x 源 rank 槽位(world_size)。
    """
    # Push data uses a v2-style 2-stage ring: stages * source-rank slots.
    return int(nbytes) * (2 * int(world_size))


def is_shot_forced() -> bool:
    """是否由环境变量强制了 shot（移植自 MR 577；无参数）。"""
    return forced_shot() is not None


def forced_shot() -> int | None:
    """按 MR 的语义支持强制 shot，环境变量改成 vLLM_MUSA_ 前缀。

    MR 577 用 `SGLANG_CUSTOM_AR_FORCE_SHOT`；vllm-musa 不允许沿用 SGLANG_ 前缀，
    因此这里读 `VLLM_MUSA_CUSTOM_AR_FORCE_SHOT`。未设置时返回 None，让
    `_select_shot()` 走算法语义分支（默认 ONE_SHOT_PUSH 等）。
    """
    import os

    raw = os.environ.get("VLLM_MUSA_CUSTOM_AR_FORCE_SHOT")
    if raw is None or raw == "":
        return None
    shot = int(raw)
    if shot not in (
        SHOT_PUSH,
        SHOT_ONE_STAGE,
        SHOT_TWO_STAGE,
        SHOT_PUSH_WIDE,
        SHOT_TWO_STAGE_512,
    ):
        raise ValueError(
            f"VLLM_MUSA_CUSTOM_AR_FORCE_SHOT={shot} 不是合法 shot（0..4）"
        )
    return shot


def preferred_shot(world_size: int, nbytes: int) -> int:
    """沿用 vllm-musa v1 的选择逻辑，v2 只在其上加 config 覆盖。"""
    from vllm_musa.jit_kernel.csrc import allreduce as _v1_ar

    return int(_v1_ar.preferred_shot(int(world_size), int(nbytes)))


def ensure_compiled(world_size: int) -> None:
    """预热 v2 的 pull 模块（TP2 默认算法），避免首个 decode 步触发 JIT。

    push 只在显式 override 时才会用到，按需编译即可。
    """
    from vllm_musa.jit_kernel.csrc import allreduce_v2 as _v2_ar

    _v2_ar.ensure_compiled(int(world_size), SHOT_ONE_STAGE)


def launch_unregistered_pybind_func(world_size: int, shot: int):
    """v2 的未注册发射入口（对齐上游 pybind 的 ``launch_unregistered``）。

    上游从 pybind11 模块里取这个函数；vllm-musa 用 ctypes 绑同一对
    ``extern "C"`` 符号，kernel 与参数完全一致。
    """
    from vllm_musa.jit_kernel.csrc import allreduce_v2 as _v2_ar

    world_size = int(world_size)
    shot = int(shot)
    _v2_ar.ensure_compiled(world_size, shot)
    launch = _v2_ar.launch_unregistered_func(world_size, shot)

    def _launch(
        rank_data,
        signal_ptrs_cpu,
        input,
        out,
        self_signal_ptr: int,
        self_buffer_ptr: int,
        push_buffer_bytes: int,
        rank: int,
        world_size_arg: int,
        shot_arg: int,
    ) -> None:
        if int(world_size_arg) != world_size or int(shot_arg) != shot:
            raise RuntimeError(
                "MUSA custom AR v2 unregistered launch 绑定的是 "
                f"(world_size={world_size}, shot={shot})，调用时传入 "
                f"(world_size={world_size_arg}, shot={shot_arg})"
            )
        launch(
            rank_data,
            signal_ptrs_cpu,
            input,
            out,
            int(self_signal_ptr),
            int(self_buffer_ptr),
            int(push_buffer_bytes),
            int(rank),
            world_size,
            shot,
        )

    return _launch


def launch_registered_pybind_func(world_size: int, shot: int):
    """v2 的已注册（图捕获）发射入口，对齐上游 ``launch_registered``。"""
    from vllm_musa.jit_kernel.csrc import allreduce_v2 as _v2_ar

    world_size = int(world_size)
    shot = int(shot)
    _v2_ar.ensure_compiled(world_size, shot)
    launch = _v2_ar.launch_registered_func(world_size, shot)

    def _launch(
        rank_data,
        signal_ptrs_cpu,
        out,
        self_signal_ptr: int,
        rank: int,
        world_size_arg: int,
        shot_arg: int,
    ) -> None:
        if int(world_size_arg) != world_size or int(shot_arg) != shot:
            raise RuntimeError(
                "MUSA custom AR v2 registered launch 绑定的是 "
                f"(world_size={world_size}, shot={shot})，调用时传入 "
                f"(world_size={world_size_arg}, shot={shot_arg})"
            )
        launch(
            rank_data,
            signal_ptrs_cpu,
            out,
            int(self_signal_ptr),
            int(rank),
            world_size,
            shot,
        )

    return _launch


def launch_device_descriptor_func(world_size: int, shot: int):
    """v2 新入口：从 communicator 自己的 device rank 指针发射（省掉每步 H2D）。

    返回的闭包签名与 v2 对象调用点一致：
    ``(rank_data_device, signal_ptrs_cpu, out, self_signal_ptr, rank,
    world_size, shot)``。

    目前 v2 kernel 只对 push 系列导出 device-descriptor 入口；对 pull shot
    这里**显式报错**而不是退回 v1，否则"开了 v2"会变成"实际跑 v1"。
    """
    shot = int(shot)
    if shot not in _PUSH_SHOTS:
        raise NotImplementedError(
            f"MUSA custom AR v2 device-descriptor launch 只覆盖 push shot "
            f"({SHOT_PUSH}={SHOT_PUSH}, {SHOT_PUSH_WIDE}={SHOT_PUSH_WIDE})，"
            f"收到 shot={shot}"
        )

    from vllm_musa.jit_kernel.csrc import allreduce_v2 as _v2_ar

    world_size = int(world_size)
    _v2_ar.ensure_compiled(world_size, shot)

    def _launch(
        rank_data_device,
        signal_ptrs_cpu,
        out,
        self_signal_ptr: int,
        rank: int,
        world_size_arg: int,
        shot_arg: int,
    ) -> None:
        if int(world_size_arg) != world_size or int(shot_arg) != shot:
            raise RuntimeError(
                "MUSA custom AR v2 launch closure 绑定的是 "
                f"(world_size={world_size}, shot={shot})，调用时传入 "
                f"(world_size={world_size_arg}, shot={shot_arg})"
            )
        _v2_ar.launch_device_descriptor(
            rank_data_device,
            signal_ptrs_cpu,
            out,
            int(self_signal_ptr),
            int(rank),
            world_size,
            shot,
        )

    return _launch
