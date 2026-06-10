# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: mark torch-wrap rewritten Tensor methods cache-safe.

vLLM's IR torch-wrap (``ir_enable_torch_wrap``) emits unbound ``torch.Tensor.*``
call_function nodes (e.g. ``torch.Tensor.split`` for QKV). torch 2.9's AOT
autograd cache safelist predates these nodes, so every compiled graph bypasses
the cache and ``InductorStandaloneAdaptor.compile`` raises "The compiled
artifact is not serializable" unless ``VLLM_DISABLE_COMPILE_CACHE=1``.

torch exposes ``unsafe_marked_cacheable_functions`` exactly for this; the dict
value is a version salt mixed into cache keys. torch >= 2.10 safelists tensor
methods natively, where ``setdefault`` keeps this a no-op. With the entry,
default-mode boots save and reload compile artifacts on torch 2.9.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

PATCHES: list = []

_SAFE_TENSOR_METHODS = ("torch._tensor.split",)
_CACHE_SALT = "vllm-musa-aot-cache-safelist-v1"


def apply() -> None:
    try:
        safelist = torch._inductor.config.unsafe_marked_cacheable_functions
    except Exception as e:
        logger.debug("Skipping AOT-cache safelist patch: %s", e)
        return
    for name in _SAFE_TENSOR_METHODS:
        safelist.setdefault(name, _CACHE_SALT)
