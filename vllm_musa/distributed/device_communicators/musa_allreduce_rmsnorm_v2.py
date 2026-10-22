"""v2 融合算子（AR + Residual + RMSNorm）占位实现 —— **故意未移植**。

MR 577 的融合路径由 `distributed/custom_all_reduce_rmsnorm.mu`（3 个实活 kernel
+ 4 个导出）与 `custom_all_reduce_rmsnorm.py`（12 KB 包装）承担，其中还叠加了
`custom_ar_dispatch.py` 里 56/60 SM 的机器相关常量与 `cache_weight` 模板分支。

MUSA-100062 的第一阶段只移植**纯 all-reduce**（decode 步长的主要通信开销），
融合路径明确留到下一阶段。这里保留同名类，是为了让 v2 对象能构造出来，
同时在真正调用融合算子时**大声失败**，而不是静默回退到 v1 融合 kernel
（那会让"v2 已使能"变成假象，measurement 也不可解释）。
"""

from __future__ import annotations


class MusaCustomAllReduceV2RMSNorm:
    """融合 AR+RMSNorm 的 v2 后端（未移植，调用即抛错）。"""

    def __init__(self, owner) -> None:
        self._owner = owner

    def fused_allreduce_rmsnorm(self, *args, **kwargs):
        raise NotImplementedError(
            "v2 fused AR+residual+RMSNorm 尚未移植（需要 "
            "custom_all_reduce_rmsnorm.mu + custom_all_reduce_rmsnorm.py，"
            "见 MUSA-100062 §4 后续阶段）；本阶段只支持纯 all-reduce"
        )

    def fused_allreduce_gemma_dual_rmsnorm(self, *args, **kwargs):
        raise NotImplementedError(
            "v2 fused Gemma dual RMSNorm 尚未移植（MUSA-100062 §4 后续阶段）"
        )
