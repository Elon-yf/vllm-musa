"""Custom all-reduce 算法枚举（移植自 SGLang MR 577 `sglang/jit_kernel/all_reduce.py`）。

vllm-musa 原本没有这个枚举：v1 用 `shot` 整数在 C++ 侧分发，没有 Python 层
"算法"概念。v2 的 Python 对象（`musa_custom_all_reduce_v2.py`）按算法选 shot、
按算法决定 push/pull，所以需要一个显式枚举，否则 `_select_shot()` 无法表达
"同一个 payload 可以走三种算法"这件事。

成员与取值语义逐字对齐上游：`shot` 属性只区分 1shot/2shot，
push 与 pull 的区别由 `is_push()` 表达（v2 kernel 的 shot 常量另有定义）。
"""

from __future__ import annotations

import enum


class AllReduceAlgo(enum.Enum):
    ONE_SHOT_PUSH = enum.auto()
    ONE_SHOT_PULL = enum.auto()
    TWO_SHOT_PULL = enum.auto()

    def is_push(self) -> bool:
        return self == AllReduceAlgo.ONE_SHOT_PUSH

    @property
    def shot(self) -> int:
        return 2 if self == AllReduceAlgo.TWO_SHOT_PULL else 1
