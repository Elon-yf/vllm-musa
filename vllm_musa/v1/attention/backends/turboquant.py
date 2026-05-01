# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA TurboQuant attention backend."""

from typing import ClassVar

import torchada  # noqa: F401
import torch
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import turboquant_attn as _turboquant_attn
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_musa.v1.attention.backends import fa_utils as _musa_fa_utils

_turboquant_attn.flash_attn_varlen_func = _musa_fa_utils.flash_attn_varlen_func
_turboquant_attn.get_flash_attn_version = _musa_fa_utils.get_flash_attn_version
_turboquant_attn.is_flash_attn_varlen_func_available = (
    _musa_fa_utils.is_flash_attn_varlen_func_available
)
_turboquant_attn._HAS_FLASH_ATTN = _musa_fa_utils.is_flash_attn_varlen_func_available()


class MUSATurboQuantAttentionImpl(_turboquant_attn.TurboQuantAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fa_version = _musa_fa_utils.get_flash_attn_version(head_size=self.head_size)


@register_backend(AttentionBackendEnum.TURBOQUANT)
class MUSATurboQuantAttentionBackend(_turboquant_attn.TurboQuantAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_impl_cls() -> type["MUSATurboQuantAttentionImpl"]:
        return MUSATurboQuantAttentionImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 3

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if use_mla:
            return "TurboQuant is not supported for MLA attention on MUSA"
        if has_sink:
            return "TurboQuant is not supported with attention sinks on MUSA"
        if use_sparse:
            return "TurboQuant is not supported for sparse attention on MUSA"
        if kv_cache_dtype == "turboquant_k8v4":
            return (
                "TurboQuant k8v4 uses FP8 key storage, which requires Triton float8 "
                "conversions that are not supported on MUSA"
            )
        return super().supports_combination(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            device_capability=device_capability,
        )
