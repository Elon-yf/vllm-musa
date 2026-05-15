# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.platforms import current_platform


class MUSATorchFP8ScaledMMKernelMixin:
    """Enable torch._scaled_mm-backed FP8 kernels on MUSA.

    The upstream Torch FP8 kernels are written for CUDA/ROCm/CPU and reject low
    CUDA compute capabilities. S5000 exposes FP8 scaled-mm through torch_musa, so
    MUSA needs only a platform-specific support gate; the implementation can use
    the upstream apply/process logic unchanged.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_musa():
            return False, "requires MUSA."
        return True, None

    def get_output_padding(self) -> int | None:
        # torch_musa/muDNN cannot currently pad FP8 tensors via torch.nn.functional.pad.
        # Padding is only a torch._scaled_mm performance hint, so skip it on MUSA.
        return None


class MUSAPerTensorTorchFP8ScaledMMLinearKernel(
    MUSATorchFP8ScaledMMKernelMixin,
    PerTensorTorchFP8ScaledMMLinearKernel,
):
    pass


class MUSAChannelWiseTorchFP8ScaledMMLinearKernel(
    MUSATorchFP8ScaledMMKernelMixin,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
):
    pass
