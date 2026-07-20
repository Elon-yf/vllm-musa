# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger

from vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    _use_row_major_activation_scales,
)
from vllm_musa.utils.environ import envs

from .rms_deepgemm_fusion import MusaRMSDeepGemmFusionPass
from .silu_deepgemm_fusion import MusaSiluDeepGemmFusionPass

logger = init_logger(__name__)


@cache
def _has_validated_musa_device_capability() -> bool:
    """Query the current logical device; MTML capability IDs are physical."""
    try:
        device_id = torch.musa.current_device()
        return tuple(torch.musa.get_device_capability(device_id)) == (3, 1)
    except Exception:
        return False


def _is_dense_model(config: VllmConfig) -> bool:
    model_config = config.model_config
    if model_config is None:
        return False
    is_model_moe = getattr(model_config, "is_model_moe", None)
    if callable(is_model_moe):
        return not is_model_moe()
    return not bool(getattr(model_config, "is_moe", False))


def _silu_deepgemm_fusion_requested(config: VllmConfig) -> bool:
    from vllm_musa.platform import _is_validated_qwen3_8b_fp8_single_gpu

    return _is_validated_qwen3_8b_fp8_single_gpu(config)


def _rms_deepgemm_fusion_requested(config: VllmConfig) -> bool:
    from vllm_musa.platform import _is_validated_qwen3_8b_fp8_single_gpu

    return _is_validated_qwen3_8b_fp8_single_gpu(config)


class MusaPostGradPassManager(PostGradPassManager):
    """Add MUSA-only graph fusions to vLLM's standard post-grad pipeline."""

    def configure(self, config: VllmConfig) -> None:
        super().configure(config)
        if (
            _silu_deepgemm_fusion_requested(config)
            and envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
            and _is_dense_model(config)
            and _has_validated_musa_device_capability()
            and _use_row_major_activation_scales(False)
            and self.pass_config.fuse_act_quant
        ):
            with set_current_vllm_config(config, check_compile=False):
                self.passes.append(MusaSiluDeepGemmFusionPass(config))
            logger.info("Enabled MUSA dense SwiGLU+DeepGEMM fusion pass")
        if (
            _rms_deepgemm_fusion_requested(config)
            and envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
            and _is_dense_model(config)
            and current_platform.is_device_capability((3, 1))
            and _use_row_major_activation_scales(False)
            and self.pass_config.fuse_act_quant
        ):
            with set_current_vllm_config(config, check_compile=False):
                self.passes.append(MusaRMSDeepGemmFusionPass(config))
            logger.info("Enabled MUSA dense residual RMSNorm+DeepGEMM fusion pass")
