# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import fx
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternReplacement,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MusaRMSDeepGemmPattern(VllmPatternReplacement):
    """Fuse residual RMSNorm with the following opaque MUSA DeepGEMM call."""

    def get_inputs(self) -> list[torch.Tensor]:
        device = current_platform.device_type
        return [
            torch.empty((5, 256), dtype=torch.bfloat16, device=device),
            torch.empty((5, 256), dtype=torch.bfloat16, device=device),
            torch.empty((256,), dtype=torch.bfloat16, device=device),
            torch.empty((64, 256), dtype=current_platform.fp8_dtype(), device=device),
            torch.empty((2, 2), dtype=torch.float32, device=device),
        ]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor,
            residual: torch.Tensor,
            norm_weight: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normalized, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
                input, residual, norm_weight, 1e-6, None
            )
            output = torch.ops.vllm.musa_deepgemm_fp8_op(
                normalized,
                weight,
                weight_scale,
                128,
                False,
            )
            return output, residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor,
            residual: torch.Tensor,
            norm_weight: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.vllm.musa_fused_add_rms_deepgemm_fp8_op(
                input,
                residual,
                norm_weight,
                weight,
                weight_scale,
                128,
                False,
                1e-6,
            )

        return _replacement


class MusaRMSDeepGemmFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "musa_rms_deepgemm_fusion_pass")
        self.register(MusaRMSDeepGemmPattern())
        self.dump_patterns(config, self.pm_pass)

    def __call__(self, graph: fx.Graph) -> None:
        super().__call__(graph)
        if self.matched_count:
            logger.info(
                "MUSA residual RMSNorm+DeepGEMM fusion matched %d pattern(s)",
                self.matched_count,
            )
