"""Broker-container correctness smoke for Paddle's model-local MUSA path."""

import torchada  # noqa: F401
import torch

from vllm_musa.model_executor.models.paddleocr_vl_musa import paddle_apply_rotary


def main() -> None:
    device = torch.device("cuda")
    shape = (2, 4888, 16, 72)
    base = torch.randn((2, 4888, 72, 16), device=device, dtype=torch.bfloat16)
    x = base.permute(0, 1, 3, 2)
    cos = torch.randn((4888, 36), device=device, dtype=torch.float32)
    sin = torch.randn((4888, 36), device=device, dtype=torch.float32)
    before = x.clone()
    got = paddle_apply_rotary(x, cos, sin)
    x_float = x.float()
    x1, x2 = x_float.chunk(2, dim=-1)
    expected = torch.cat(
        (x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :],
         x2 * cos[None, :, None, :] + x1 * sin[None, :, None, :]),
        dim=-1,
    ).to(torch.bfloat16)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert got.shape == shape
    assert got.dtype == torch.bfloat16
    assert got.data_ptr() != x.data_ptr()
    print("PASS paddle_musa_rotary_d72_stride_noninplace")


if __name__ == "__main__":
    main()
