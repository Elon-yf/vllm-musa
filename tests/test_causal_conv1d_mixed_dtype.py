import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")
causal_conv1d = pytest.importorskip("vllm_musa.jit_kernel.tilelang.causal_conv1d")


def test_decode_kernel_keeps_supported_mixed_dtypes(monkeypatch):
    captured = {}

    def fake_kernel(*kernel_config):
        captured["kernel_config"] = kernel_config

        def run(x, weight, bias, state, indices, mapping, has_init, out, *scalars):
            captured["x"] = x
            captured["weight"] = weight
            captured["state"] = state
            captured["out"] = out
            captured["scalars"] = scalars
            out.copy_(x)

        return run

    monkeypatch.setattr(
        causal_conv1d,
        "_causal_conv1d_decode_width4_batched_kernel",
        fake_kernel,
    )
    monkeypatch.setattr(causal_conv1d, "_DECODE_HAS_INIT_BUF", {})

    x = torch.randn(4, 16, dtype=torch.bfloat16)
    state = torch.randn(4, 16, 3, dtype=torch.float32)
    weight = torch.randn(16, 4, dtype=torch.float32)
    indices = torch.arange(4, dtype=torch.int32)

    output = causal_conv1d.musa_tilelang_causal_conv1d_update(
        x,
        state,
        weight,
        activation="silu",
        conv_state_indices=indices,
    )

    assert output is not None
    assert output.dtype == torch.bfloat16
    assert captured["x"].dtype == torch.bfloat16
    assert captured["weight"].dtype == torch.float32
    assert captured["state"].dtype == torch.float32
    assert captured["out"].dtype == torch.bfloat16
    assert captured["kernel_config"][:5] == (
        "bfloat16",
        "float32",
        "bfloat16",
        "float32",
        "bfloat16",
    )
    assert captured["kernel_config"][13] is True
    assert captured["kernel_config"][16] is True
    assert captured["scalars"][-1] == causal_conv1d.NULL_BLOCK_ID


def test_decode_kernel_preserves_same_dtype_path(monkeypatch):
    captured = {}

    def fake_kernel(*kernel_config):
        captured["kernel_config"] = kernel_config

        def run(x, weight, bias, state, indices, mapping, has_init, out, *scalars):
            captured["x"] = x
            captured["scalars"] = scalars
            out.copy_(x)

        return run

    monkeypatch.setattr(
        causal_conv1d,
        "_causal_conv1d_decode_width4_batched_kernel",
        fake_kernel,
    )
    monkeypatch.setattr(causal_conv1d, "_DECODE_HAS_INIT_BUF", {})

    x = torch.randn(2, 8, dtype=torch.float32)
    state = torch.randn(2, 8, 3, dtype=torch.float32)
    weight = torch.randn(8, 4, dtype=torch.float32)

    output = causal_conv1d.musa_tilelang_causal_conv1d_update(
        x,
        state,
        weight,
    )

    assert output is not None
    assert output.dtype == torch.float32
    assert captured["x"] is not None
    assert captured["x"].dtype == torch.float32
    assert captured["kernel_config"][:5] == ("float32",) * 5
    assert captured["kernel_config"][13] is False
    assert captured["kernel_config"][16] is False
    assert captured["scalars"][-1] == causal_conv1d.PAD_SLOT_ID


def test_decode_kernel_rejects_unverified_mixed_dtype_tuple():
    x = torch.randn(2, 8, dtype=torch.float32)
    state = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)

    assert (
        causal_conv1d.musa_tilelang_causal_conv1d_update(x, state, weight) is None
    )


def test_decode_kernel_rejects_noncontiguous_indices():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    state = torch.randn(4, 8, 3, dtype=torch.float32)
    weight = torch.randn(8, 4, dtype=torch.float32)
    indices = torch.arange(4, dtype=torch.int32)[::2]

    assert not indices.is_contiguous()
    assert (
        causal_conv1d.musa_tilelang_causal_conv1d_update(
            x, state, weight, conv_state_indices=indices
        )
        is None
    )
