"""Numerical smoke for Paddle's FP32, out-of-place vision RoPE path."""

import torchada  # noqa: F401
import torch

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm_musa.model_executor.layers.rotary_embedding.base import (
    MusaVisionApplyRotaryEmb,
)


def main() -> None:
    device = torch.device("cuda")
    shape = (2, 4888, 16, 72)
    base = torch.randn((2, 4888, 72, 16), device=device, dtype=torch.bfloat16)
    x = base.permute(0, 1, 3, 2)
    cos = torch.randn((4888, 36), device=device, dtype=torch.float32)
    sin = torch.randn((4888, 36), device=device, dtype=torch.float32)
    before = x.clone()
    got = MusaVisionApplyRotaryEmb(enable_fp32_compute=True)(x, cos, sin)
    expected = ApplyRotaryEmb.forward_static(
        x, cos, sin, enable_fp32_compute=True
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert got.shape == shape
    assert got.dtype == torch.bfloat16
    assert got.data_ptr() != x.data_ptr()


if __name__ == "__main__":
    main()
