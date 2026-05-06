# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA Platform implementation."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestMUSAPlatformBase:
    """Tests for MUSAPlatformBase class."""

    def test_device_name(self):
        """Test that device_name is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.device_name == "musa"

    def test_device_type(self):
        """Test that device_type is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.device_type == "musa"

    def test_dispatch_key(self):
        """Test that dispatch_key uses MUSA."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.dispatch_key == "MUSA"

    def test_dist_backend(self):
        """Test that dist_backend uses mccl."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.dist_backend == "mccl"

    def test_device_control_env_var(self):
        """Test that device_control_env_var is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.device_control_env_var == "MUSA_VISIBLE_DEVICES"

    def test_ray_device_key(self):
        """Test that ray_device_key is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.ray_device_key == "GPU"

    def test_is_cuda_alike_returns_true(self):
        """Test that is_cuda_alike returns True for MUSA."""
        from vllm_musa.platform import MUSAPlatformBase

        platform = MUSAPlatformBase()
        assert platform.is_cuda_alike() is True

    def test_is_sleep_mode_available_returns_true(self):
        """Test that is_sleep_mode_available returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        platform = MUSAPlatformBase()
        assert platform.is_sleep_mode_available() is True

    def test_supported_dtypes(self):
        """Test that supported_dtypes includes bf16, fp16, and fp32."""
        import torch

        from vllm_musa.platform import MUSAPlatformBase

        platform = MUSAPlatformBase()
        dtypes = platform.supported_dtypes

        assert torch.bfloat16 in dtypes
        assert torch.float16 in dtypes
        assert torch.float32 in dtypes

    def test_opaque_attention_op_returns_true(self):
        """Test that opaque_attention_op returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.opaque_attention_op() is True

    def test_use_custom_allreduce_returns_true(self):
        """Test that use_custom_allreduce returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.use_custom_allreduce() is True

    def test_support_hybrid_kv_cache(self):
        """Test that support_hybrid_kv_cache returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.support_hybrid_kv_cache() is True

    def test_supports_fp8_for_musa_3_1(self):
        """Test that FP8 is supported on MUSA capability 3.1."""
        from vllm_musa.platform import MUSAPlatformBase
        from vllm.platforms.interface import DeviceCapability

        with patch.object(
            MUSAPlatformBase,
            "get_device_capability",
            return_value=DeviceCapability(3, 1),
        ):
            assert MUSAPlatformBase.supports_fp8() is True

    def test_supports_fp8_rejects_pre_3_1(self):
        """Test that pre-3.1 MUSA capability does not support FP8."""
        from vllm_musa.platform import MUSAPlatformBase
        from vllm.platforms.interface import DeviceCapability

        with patch.object(
            MUSAPlatformBase,
            "get_device_capability",
            return_value=DeviceCapability(3, 0),
        ):
            assert MUSAPlatformBase.supports_fp8() is False

    def test_support_static_graph_mode(self):
        """Test that support_static_graph_mode returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.support_static_graph_mode() is True

    def test_get_punica_wrapper(self):
        """Test get_punica_wrapper returns correct path."""
        from vllm_musa.platform import MUSAPlatformBase

        result = MUSAPlatformBase.get_punica_wrapper()
        assert result == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    def test_get_device_communicator_cls(self):
        """Test get_device_communicator_cls returns CUDA communicator."""
        from vllm_musa.platform import MUSAPlatformBase

        result = MUSAPlatformBase.get_device_communicator_cls()
        expected = (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        )
        assert result == expected

    def test_get_static_graph_wrapper_cls(self):
        """Test get_static_graph_wrapper_cls returns CUDA graph wrapper."""
        from vllm_musa.platform import MUSAPlatformBase

        result = MUSAPlatformBase.get_static_graph_wrapper_cls()
        assert result == "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    def test_register_attention_backends_overrides_turboquant(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm_musa.platform import register_attention_backends

        AttentionBackendEnum.TURBOQUANT.clear_override()
        register_attention_backends()

        assert (
            AttentionBackendEnum.TURBOQUANT.get_path()
            == "vllm_musa.v1.attention.backends.turboquant.MUSATurboQuantAttentionBackend"
        )

    def test_get_valid_backends_includes_turboquant_for_non_mla(self):
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm_musa.platform import _get_backend_priorities

        priorities = _get_backend_priorities(
            use_mla=False,
            device_capability=DeviceCapability(3, 1),
        )

        assert AttentionBackendEnum.TURBOQUANT in priorities
        assert AttentionBackendEnum.TURBOQUANT not in _get_backend_priorities(
            use_mla=True,
            device_capability=DeviceCapability(3, 1),
        )

    def test_turboquant_rejects_k8v4_on_musa(self):
        import torch
        from vllm.platforms.interface import DeviceCapability
        from vllm_musa.v1.attention.backends.turboquant import (
            MUSATurboQuantAttentionBackend,
        )

        reason = MUSATurboQuantAttentionBackend.supports_combination(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="turboquant_k8v4",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            device_capability=DeviceCapability(3, 1),
        )

        assert reason is not None
        assert "Triton float8 conversions" in reason

    def test_flash_attention_rejects_fp8_kv_cache_dtype(self):
        import vllm_musa.platform  # noqa: F401
        from vllm_musa.v1.attention.backends.flash_attn import (
            MUSAFlashAttentionBackend,
        )

        assert MUSAFlashAttentionBackend.supports_kv_cache_dtype("fp8") is False
        assert MUSAFlashAttentionBackend.supports_kv_cache_dtype("fp8_e4m3") is False

    def test_flash_attention_fp8_dtype_error_is_clear(self):
        import vllm_musa.platform  # noqa: F401
        from vllm_musa.v1.attention.backends.flash_attn import (
            MUSAFlashAttentionBackend,
        )

        with pytest.raises(NotImplementedError, match="FP8 dtype is not supported"):
            MUSAFlashAttentionBackend.get_fp8_dtype_for_flashattn("fp8")

    def test_flashmla_fp8_decode_metadata_uses_dense_fp8_helper(self, monkeypatch):
        import torch
        import vllm_musa.v1.attention.backends.mla.flashmla as flashmla

        dense_tile_metadata = torch.ones((1, 8), dtype=torch.int32)
        dense_num_splits = torch.ones((3,), dtype=torch.int32)
        dense_calls = []

        def get_dense_fp8_metadata(seq_lens, num_q_tokens_per_head_k, num_heads_k):
            dense_calls.append((seq_lens, num_q_tokens_per_head_k, num_heads_k))
            return dense_tile_metadata, dense_num_splits

        def get_generic_metadata(*args, **kwargs):
            raise AssertionError("FP8 decode metadata must not call generic helper")

        monkeypatch.setattr(flashmla, "get_mla_metadata_dense_fp8", get_dense_fp8_metadata)
        monkeypatch.setattr(flashmla, "get_mla_metadata", get_generic_metadata)

        builder = object.__new__(flashmla.FlashMLAMetadataBuilder)
        builder.num_q_heads = 4
        builder.is_fp8_kvcache = True
        builder.compilation_config = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        )
        builder.cg_buf_tile_scheduler_metadata = None
        builder.cg_buf_num_splits = None

        seq_lens = torch.tensor([4, 7], dtype=torch.int32)
        metadata = builder._build_decode(
            block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
            seq_lens_device=seq_lens,
            max_seq_len=7,
            query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
            query_start_loc_device=torch.tensor([0, 2, 4], dtype=torch.int32),
            num_decode_tokens=4,
            dcp_tot_seq_lens_device=None,
        )

        assert dense_calls == [(seq_lens, 8, 1)]
        assert metadata.scheduler_metadata.tile_scheduler_metadata is dense_tile_metadata
        assert metadata.scheduler_metadata.num_splits is dense_num_splits

    def test_flashmla_non_fp8_decode_metadata_uses_generic_helper(self, monkeypatch):
        import torch
        import vllm_musa.v1.attention.backends.mla.flashmla as flashmla

        generic_tile_metadata = torch.zeros((1, 8), dtype=torch.int32)
        generic_num_splits = torch.zeros((2,), dtype=torch.int32)
        generic_calls = []

        def get_dense_fp8_metadata(*args, **kwargs):
            raise AssertionError("non-FP8 decode metadata must not call FP8 helper")

        def get_generic_metadata(*args, **kwargs):
            generic_calls.append((args, kwargs))
            return generic_tile_metadata, generic_num_splits

        monkeypatch.setattr(flashmla, "get_mla_metadata_dense_fp8", get_dense_fp8_metadata)
        monkeypatch.setattr(flashmla, "get_mla_metadata", get_generic_metadata)

        builder = object.__new__(flashmla.FlashMLAMetadataBuilder)
        builder.num_q_heads = 2
        builder.is_fp8_kvcache = False
        builder.compilation_config = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        )
        builder.cg_buf_tile_scheduler_metadata = None
        builder.cg_buf_num_splits = None

        seq_lens = torch.tensor([3], dtype=torch.int32)
        metadata = builder._build_decode(
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            seq_lens_device=seq_lens,
            max_seq_len=3,
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_device=torch.tensor([0, 1], dtype=torch.int32),
            num_decode_tokens=1,
            dcp_tot_seq_lens_device=None,
        )

        assert len(generic_calls) == 1
        args, kwargs = generic_calls[0]
        assert args == (seq_lens, 2, 1)
        assert kwargs == {"is_fp8_kvcache": False}
        assert metadata.scheduler_metadata.tile_scheduler_metadata is generic_tile_metadata
        assert metadata.scheduler_metadata.num_splits is generic_num_splits

    def test_flashmla_fp8_forward_mqa_uses_fp8_kernel_and_descales(self, monkeypatch):
        import torch
        import vllm_musa.v1.attention.backends.mla.flashmla as flashmla

        monkeypatch.setattr(flashmla, "reshape_query_for_spec_decode", lambda q, _: q)
        monkeypatch.setattr(flashmla, "reshape_attn_output_for_spec_decode", lambda out: out)
        fp8_calls = []

        def flash_mla_fp8(**kwargs):
            fp8_calls.append(kwargs)
            return torch.full_like(kwargs["q"], 2), torch.tensor([1.0])

        def flash_mla_generic(*args, **kwargs):
            raise AssertionError("FP8 KV cache must use the FP8 FlashMLA kernel")

        monkeypatch.setattr(flashmla, "flash_mla_with_kvcache_fp8", flash_mla_fp8)
        monkeypatch.setattr(flashmla, "flash_mla_with_kvcache", flash_mla_generic)

        impl = object.__new__(flashmla.FlashMLAImpl)
        impl.kv_cache_dtype = "fp8"
        impl.kv_lora_rank = 64
        impl.scale = 0.125
        scheduler_metadata = SimpleNamespace(
            tile_scheduler_metadata=torch.ones((1, 8), dtype=torch.int32),
            num_splits=torch.ones((2,), dtype=torch.int32),
        )
        decode = SimpleNamespace(
            scheduler_metadata=scheduler_metadata,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
        )
        metadata = SimpleNamespace(decode=decode, num_decodes=1)
        layer = SimpleNamespace(_q_scale=torch.tensor(0.25), _k_scale=torch.tensor(0.5))
        q = torch.ones((1, 2, 4), dtype=torch.float32)
        kv_cache = torch.ones((1, 3, 4), dtype=torch.float32)

        output, lse = impl.forward_mqa(q, kv_cache, metadata, layer)

        assert len(fp8_calls) == 1
        call = fp8_calls[0]
        assert call["q"] is q
        assert torch.equal(call["k_cache"], kv_cache.unsqueeze(-2))
        assert call["block_table"] is decode.block_table
        assert call["cache_seqlens"] is decode.seq_lens
        assert call["head_dim_v"] == 64
        assert call["tile_scheduler_metadata"] is scheduler_metadata.tile_scheduler_metadata
        assert call["num_splits"] is scheduler_metadata.num_splits
        assert call["softmax_scale"] == 0.125
        assert call["causal"] is True
        assert torch.equal(call["descale_q"], layer._q_scale.reshape(1))
        assert torch.equal(call["descale_k"], layer._k_scale.reshape(1))
        assert torch.equal(output, torch.full_like(q, 2))
        assert torch.equal(lse, torch.tensor([1.0]))

    def test_flashmla_fp8_forward_mqa_moves_descales_to_query_device(self, monkeypatch):
        import torch
        import vllm_musa.v1.attention.backends.mla.flashmla as flashmla

        monkeypatch.setattr(flashmla, "reshape_query_for_spec_decode", lambda q, _: q)
        monkeypatch.setattr(flashmla, "reshape_attn_output_for_spec_decode", lambda out: out)
        fp8_calls = []

        def flash_mla_fp8(**kwargs):
            fp8_calls.append(kwargs)
            return torch.empty_like(kwargs["q"]), torch.tensor([1.0])

        monkeypatch.setattr(flashmla, "flash_mla_with_kvcache_fp8", flash_mla_fp8)

        impl = object.__new__(flashmla.FlashMLAImpl)
        impl.kv_cache_dtype = "fp8"
        impl.kv_lora_rank = 64
        impl.scale = 0.125
        scheduler_metadata = SimpleNamespace(
            tile_scheduler_metadata=torch.ones((1, 8), dtype=torch.int32),
            num_splits=torch.ones((2,), dtype=torch.int32),
        )
        decode = SimpleNamespace(
            scheduler_metadata=scheduler_metadata,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
        )
        metadata = SimpleNamespace(decode=decode, num_decodes=1)
        layer = SimpleNamespace(_q_scale=0.25, _k_scale=0.5)
        q = torch.ones((1, 2, 4), dtype=torch.float32, device="meta")
        kv_cache = torch.ones((1, 3, 4), dtype=torch.float32, device="meta")

        output, lse = impl.forward_mqa(q, kv_cache, metadata, layer)

        assert len(fp8_calls) == 1
        call = fp8_calls[0]
        assert call["descale_q"].device == q.device
        assert call["descale_k"].device == q.device
        assert output.device == q.device
        assert torch.equal(lse, torch.tensor([1.0]))

    @pytest.mark.parametrize(
        ("q_scale", "k_scale", "match"),
        [
            pytest.param(None, 0.5, r"layer\._q_scale", id="missing-q-scale"),
            pytest.param(0.25, None, r"layer\._k_scale", id="missing-k-scale"),
            pytest.param([0.25, 0.5], 0.5, r"scalar layer\._q_scale", id="vector-q-scale"),
            pytest.param(0.25, [0.5, 1.0], r"scalar layer\._k_scale", id="vector-k-scale"),
        ],
    )
    def test_flashmla_fp8_forward_mqa_validates_descales(
        self, monkeypatch, q_scale, k_scale, match
    ):
        import torch
        import vllm_musa.v1.attention.backends.mla.flashmla as flashmla

        layer = SimpleNamespace()
        if q_scale is not None:
            layer._q_scale = q_scale
        if k_scale is not None:
            layer._k_scale = k_scale

        monkeypatch.setattr(flashmla, "reshape_query_for_spec_decode", lambda q, _: q)

        def flash_mla_fp8(**kwargs):
            raise AssertionError("invalid descale tensors should fail before kernel call")

        monkeypatch.setattr(flashmla, "flash_mla_with_kvcache_fp8", flash_mla_fp8)

        impl = object.__new__(flashmla.FlashMLAImpl)
        impl.kv_cache_dtype = "fp8"
        impl.kv_lora_rank = 64
        impl.scale = 0.125
        scheduler_metadata = SimpleNamespace(
            tile_scheduler_metadata=torch.ones((1, 8), dtype=torch.int32),
            num_splits=torch.ones((2,), dtype=torch.int32),
        )
        decode = SimpleNamespace(
            scheduler_metadata=scheduler_metadata,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
        )
        metadata = SimpleNamespace(decode=decode, num_decodes=1)
        q = torch.ones((1, 2, 4), dtype=torch.float32)
        kv_cache = torch.ones((1, 3, 4), dtype=torch.float32)

        with pytest.raises(ValueError, match=match):
            impl.forward_mqa(q, kv_cache, metadata, layer)

    def test_flashmla_non_fp8_forward_mqa_uses_generic_kernel(self, monkeypatch):
        import torch
        import vllm_musa.v1.attention.backends.mla.flashmla as flashmla

        monkeypatch.setattr(flashmla.envs, "VLLM_BATCH_INVARIANT", False)
        monkeypatch.setattr(flashmla, "reshape_query_for_spec_decode", lambda q, _: q)
        monkeypatch.setattr(flashmla, "reshape_attn_output_for_spec_decode", lambda out: out)
        generic_calls = []

        def flash_mla_generic(**kwargs):
            generic_calls.append(kwargs)
            return torch.full_like(kwargs["q"], 3), torch.tensor([2.0])

        def flash_mla_fp8(*args, **kwargs):
            raise AssertionError("non-FP8 KV cache must use the generic FlashMLA kernel")

        monkeypatch.setattr(flashmla, "flash_mla_with_kvcache", flash_mla_generic)
        monkeypatch.setattr(flashmla, "flash_mla_with_kvcache_fp8", flash_mla_fp8)

        impl = object.__new__(flashmla.FlashMLAImpl)
        impl.kv_cache_dtype = "auto"
        impl.kv_lora_rank = 32
        impl.scale = 0.25
        scheduler_metadata = SimpleNamespace(
            tile_scheduler_metadata=torch.zeros((1, 8), dtype=torch.int32),
            num_splits=torch.zeros((2,), dtype=torch.int32),
        )
        decode = SimpleNamespace(
            scheduler_metadata=scheduler_metadata,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
        )
        metadata = SimpleNamespace(decode=decode, num_decodes=1)
        q = torch.ones((1, 2, 4), dtype=torch.float32)
        kv_cache = torch.ones((1, 3, 4), dtype=torch.float32)

        output, lse = impl.forward_mqa(q, kv_cache, metadata, SimpleNamespace())

        assert len(generic_calls) == 1
        call = generic_calls[0]
        assert call["q"] is q
        assert torch.equal(call["k_cache"], kv_cache.unsqueeze(-2))
        assert call["block_table"] is decode.block_table
        assert call["cache_seqlens"] is decode.seq_lens
        assert call["head_dim_v"] == 32
        assert call["tile_scheduler_metadata"] is scheduler_metadata.tile_scheduler_metadata
        assert call["num_splits"] is scheduler_metadata.num_splits
        assert call["softmax_scale"] == 0.25
        assert call["causal"] is True
        assert torch.equal(output, torch.full_like(q, 3))
        assert torch.equal(lse, torch.tensor([2.0]))

    def test_fp8_scaled_mm_oot_registers_musa_kernel(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()
        import vllm_musa.fp8_linear  # noqa: F401

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum
        from vllm_musa.fp8_linear import MUSAFP8ScaledMMLinearKernel

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(16, 16),
                module_name="musa_fp8_test",
            )

        assert isinstance(kernel, MUSAFP8ScaledMMLinearKernel)

    def test_fp8_scaled_mm_uses_weight_scale(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()
        import vllm_musa.fp8_linear  # noqa: F401

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
            set_current_vllm_config(VllmConfig()),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(16, 16),
                module_name="musa_fp8_test",
            )

        layer = torch.nn.Module()
        layer.weight = torch.empty(16, 16)
        layer.weight_scale = torch.ones(1, 1)
        layer.input_scale = torch.ones(1)
        layer.input_scale_ub = torch.ones(1)

        weight, weight_scale, input_scale, input_scale_ub = kernel._get_layer_params(
            layer
        )

        assert weight is layer.weight
        assert weight_scale is layer.weight_scale
        assert input_scale is layer.input_scale
        assert input_scale_ub is layer.input_scale_ub

    def test_fp8_scaled_mm_falls_back_to_weight_scale_inv(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()
        import vllm_musa.fp8_linear  # noqa: F401

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
            set_current_vllm_config(VllmConfig()),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(16, 16),
                module_name="musa_fp8_test",
            )

        layer = torch.nn.Module()
        layer.weight = torch.empty(16, 16)
        layer.weight_scale_inv = torch.ones(1, 1)
        layer.input_scale = torch.ones(1)
        layer.input_scale_ub = torch.ones(1)

        weight, weight_scale, input_scale, input_scale_ub = kernel._get_layer_params(
            layer
        )

        assert weight is layer.weight
        assert weight_scale is layer.weight_scale_inv
        assert input_scale is layer.input_scale
        assert input_scale_ub is layer.input_scale_ub

    def test_fp8_scaled_mm_accepts_upstream_transposed_weight(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(576, 2048),
                module_name="musa_fp8_shape_test",
            )

        B = torch.empty(2048, 576, dtype=torch.float8_e4m3fn).contiguous()
        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_shape"] = qweight.shape
            captured["qweight_is_contiguous"] = qweight.is_contiguous()
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 2048, dtype=torch.float8_e4m3fn),
                B=B,
                out_dtype=torch.float16,
                As=torch.ones(1, 16),
                Bs=torch.ones(5, 16),
                bias=None,
                output_shape=[1, 2048],
            )

        assert captured["qweight_shape"] == (576, 2048)
        assert captured["qweight_is_contiguous"]
        assert output.shape == (1, 576)

    def test_fp8_scaled_mm_preserves_square_out_in_weight_orientation(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(256, 256),
                module_name="musa_fp8_square_orientation_test",
            )

        B = torch.empty(256, 256, dtype=torch.float8_e4m3fn).contiguous()
        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_data_ptr"] = qweight.data_ptr()
            captured["qweight_shape"] = qweight.shape
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 256, dtype=torch.float8_e4m3fn),
                B=B,
                out_dtype=torch.float16,
                As=torch.ones(1, 2),
                Bs=torch.ones(2, 2),
                bias=None,
                output_shape=[1, 256],
            )

        assert captured["qweight_data_ptr"] == B.data_ptr()
        assert captured["qweight_shape"] == (256, 256)
        assert output.shape == (1, 256)

    def test_fp8_scaled_mm_preserves_square_non_contiguous_out_in_weight_orientation(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(16, 16),
                module_name="musa_fp8_square_non_contiguous_orientation_test",
            )

        backing = torch.arange(16 * 17, dtype=torch.float32).reshape(16, 17)
        B = backing[:, :16].to(torch.float8_e4m3fn)
        assert not B.is_contiguous()
        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight"] = qweight.float().clone()
            captured["qweight_is_contiguous"] = qweight.is_contiguous()
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 16, dtype=torch.float8_e4m3fn),
                B=B,
                out_dtype=torch.float16,
                As=torch.ones(1, 1),
                Bs=torch.ones(1, 1),
                bias=None,
                output_shape=[1, 16],
            )

        assert torch.equal(captured["qweight"], B.contiguous().float())
        assert captured["qweight_is_contiguous"]
        assert output.shape == (1, 16)

    def test_fp8_scaled_mm_transposes_square_upstream_weight_view(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(256, 256),
                module_name="musa_fp8_square_transposed_test",
            )

        original_qweight = torch.empty(256, 256, dtype=torch.float8_e4m3fn).contiguous()
        B = original_qweight.t()
        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_data_ptr"] = qweight.data_ptr()
            captured["qweight_shape"] = qweight.shape
            captured["qweight_is_contiguous"] = qweight.is_contiguous()
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 256, dtype=torch.float8_e4m3fn),
                B=B,
                out_dtype=torch.float16,
                As=torch.ones(1, 2),
                Bs=torch.ones(2, 2),
                bias=None,
                output_shape=[1, 256],
            )

        assert captured["qweight_data_ptr"] == original_qweight.data_ptr()
        assert captured["qweight_shape"] == (256, 256)
        assert captured["qweight_is_contiguous"]
        assert output.shape == (1, 256)

    def test_fp8_scaled_mm_expands_qkv_scalar_weight_scale(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(576, 2048),
                module_name="musa_fp8_qkv_scale_test",
            )

        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_scales_shape"] = qweight_scales.shape
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 2048, dtype=torch.float8_e4m3fn),
                B=torch.empty(2048, 576, dtype=torch.float8_e4m3fn),
                out_dtype=torch.float16,
                As=torch.ones(1, 16),
                Bs=torch.tensor(1.0),
                bias=None,
                output_shape=[1, 2048],
            )

        assert captured["qweight_scales_shape"] == (5, 16)
        assert output.shape == (1, 576)

    def test_fp8_scaled_mm_expands_qkv_per_shard_weight_scales(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(768, 2048),
                module_name="musa_fp8_qkv_per_shard_scale_test",
            )

        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_scales"] = qweight_scales
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        Bs = torch.arange(48, dtype=torch.float32).reshape(3, 16)
        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 2048, dtype=torch.float8_e4m3fn),
                B=torch.empty(2048, 768, dtype=torch.float8_e4m3fn),
                out_dtype=torch.float16,
                As=torch.ones(1, 16),
                Bs=Bs,
                bias=None,
                output_shape=[1, 2048],
            )

        expected_scales = torch.cat(
            [
                Bs[0:1].expand(2, -1),
                Bs[1:2].expand(2, -1),
                Bs[2:3].expand(2, -1),
            ]
        )
        assert torch.equal(captured["qweight_scales"], expected_scales)
        assert captured["qweight_scales"].shape == (6, 16)
        assert output.shape == (1, 768)

    def test_fp8_scaled_mm_expands_non_uniform_logical_shard_weight_scales(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(1024, 2048),
                module_name="musa_fp8_non_uniform_qkv_scale_test",
            )

        layer = SimpleNamespace(logical_widths=[512, 256, 256])
        kernel.process_weights_after_loading(layer)
        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_scales"] = qweight_scales
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        Bs = torch.arange(48, dtype=torch.float32).reshape(3, 16)
        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 2048, dtype=torch.float8_e4m3fn),
                B=torch.empty(2048, 1024, dtype=torch.float8_e4m3fn),
                out_dtype=torch.float16,
                As=torch.ones(1, 16),
                Bs=Bs,
                bias=None,
                output_shape=[1, 2048],
            )

        expected_scales = torch.cat(
            [
                Bs[0:1].expand(4, -1),
                Bs[1:2].expand(2, -1),
                Bs[2:3].expand(2, -1),
            ]
        )
        assert torch.equal(captured["qweight_scales"], expected_scales)
        assert captured["qweight_scales"].shape == (8, 16)
        assert output.shape == (1, 1024)

    def test_fp8_scaled_mm_expands_scalar_weight_scale(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(256, 256),
                module_name="musa_fp8_aligned_scale_test",
            )

        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["x_scales_shape"] = x_scales.shape
            captured["qweight_scales_shape"] = qweight_scales.shape
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 256, dtype=torch.float8_e4m3fn),
                B=torch.empty(256, 256, dtype=torch.float8_e4m3fn),
                out_dtype=torch.float16,
                As=torch.tensor(1.0),
                Bs=torch.tensor(1.0),
                bias=None,
                output_shape=[1, 256],
            )

        assert captured["x_scales_shape"] == (1, 1)
        assert captured["qweight_scales_shape"] == (2, 2)
        assert output.shape == (1, 256)

    def test_fp8_scaled_mm_rejects_invalid_weight_scale_rows(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTensorSym,
            kFp8StaticTensorSym,
        )
        from vllm.platforms.interface import PlatformEnum

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTensorSym,
                weight_quant_key=kFp8StaticTensorSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(256, 128),
                module_name="musa_fp8_invalid_scale_test",
            )

        A = torch.ones((1, 128), dtype=torch.bfloat16)
        B = torch.ones((256, 128), dtype=torch.float8_e4m3fn)
        As = torch.ones((1, 1), dtype=torch.float32)
        Bs = torch.ones((3, 1), dtype=torch.float32)

        with pytest.raises(ValueError, match="one weight scale row per 128 output channels"):
            kernel.apply_scaled_mm(
                A=A,
                B=B,
                out_dtype=torch.bfloat16,
                As=As,
                Bs=Bs,
                bias=None,
                output_shape=[1, 256],
            )

    def test_fp8_block_scaled_mm_oot_registers_musa_kernel(self):
        import torch
        import vllm_musa

        vllm_musa._apply_vllm_patches()
        import vllm_musa.fp8_linear  # noqa: F401

        from vllm.model_executor.kernels import linear
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Dynamic128Sym,
            kFp8Static128BlockSym,
        )
        from vllm.platforms.interface import PlatformEnum
        from vllm_musa.fp8_linear import MUSAFp8BlockScaledMMLinearKernel

        with (
            patch.object(linear.current_platform, "_enum", PlatformEnum.OOT),
            patch.object(linear.current_platform, "is_musa", return_value=True),
        ):
            kernel = linear.init_fp8_linear_kernel(
                activation_quant_key=kFp8Dynamic128Sym,
                weight_quant_key=kFp8Static128BlockSym,
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(256, 256),
                module_name="musa_fp8_block_test",
            )

        assert isinstance(kernel, MUSAFp8BlockScaledMMLinearKernel)

    def test_fp8_moe_passes_activation_scales_to_musa_gemv(self):
        import torch
        from vllm_musa.model_executor.layers.fused_moe import fused_moe

        hidden_states = torch.ones(2, 4, dtype=torch.float32)
        w1 = torch.empty(3, 8, 4, dtype=torch.float8_e4m3fn)
        w2 = torch.empty(3, 4, 4, dtype=torch.float8_e4m3fn)
        topk_weights = torch.ones(2, 2, dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        w1_scale = torch.ones(3, 1, 1, dtype=torch.float32)
        w2_scale = torch.ones(3, 1, 1, dtype=torch.float32)
        a1_scale = torch.ones(1, dtype=torch.float32)
        a2_scale = torch.ones(1, dtype=torch.float32) * 2
        a1q_scale = torch.ones(2, 1, dtype=torch.float32)
        a2q_scale = torch.ones(4, 1, dtype=torch.float32) * 2

        quant_calls = []
        native_calls = []
        qintermediate = {}

        def fake_quantize(A, A_scale, **kwargs):
            quant_calls.append({"A": A, "A_scale": A_scale, "kwargs": kwargs})
            if len(quant_calls) == 1:
                return A.clone(), a1q_scale
            qintermediate["tensor"] = A.clone()
            return qintermediate["tensor"], a2q_scale

        def fake_musa_gemv_moe(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            topk_ids,
            mul_routed_weight,
            topk,
            use_int4_w4a16,
            use_swigelu,
        ):
            native_calls.append(
                {
                    "A": A,
                    "B": B,
                    "C": C,
                    "A_scale": A_scale,
                    "B_scale": B_scale,
                    "topk": topk,
                    "use_swigelu": use_swigelu,
                }
            )
            C.zero_()

        def fake_moe_sum(moe_output, output):
            output.copy_(moe_output.sum(dim=1))

        with (
            patch.object(
                fused_moe,
                "_get_config_quant_dtype",
                return_value=torch.float8_e4m3fn,
            ),
            patch.object(fused_moe, "try_get_optimal_moe_config", return_value={}),
            patch.object(fused_moe, "moe_kernel_quantize_input", fake_quantize),
            patch.object(fused_moe.musa_ops, "musa_fused_gemv_moe", fake_musa_gemv_moe),
            patch.object(fused_moe.ops, "moe_sum", fake_moe_sum),
        ):
            output = fused_moe.fused_experts_impl(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                use_fp8_w8a8=True,
                per_channel_quant=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=[128, 128],
            )

        assert output.shape == hidden_states.shape
        assert quant_calls[0]["A"] is hidden_states
        assert quant_calls[0]["A_scale"] is a1_scale
        assert quant_calls[1]["A"] is native_calls[0]["C"]
        assert quant_calls[1]["A_scale"] is a2_scale
        assert native_calls[0]["A_scale"] is a1q_scale
        assert native_calls[0]["B_scale"] is w1_scale
        assert native_calls[1]["A"] is qintermediate["tensor"]
        assert native_calls[1]["A_scale"] is a2q_scale
        assert native_calls[1]["B_scale"] is w2_scale


class TestNonMtmlMUSAPlatform:
    """Tests for NonMtmlMUSAPlatform class."""

    def test_get_device_capability(self):
        """Test get_device_capability returns DeviceCapability."""
        with patch("torch.cuda.get_device_capability") as mock_cap:
            mock_cap.return_value = (3, 1)

            from vllm_musa.platform import NonMtmlMUSAPlatform

            # Clear cache to allow re-testing
            NonMtmlMUSAPlatform.get_device_capability.cache_clear()

            cap = NonMtmlMUSAPlatform.get_device_capability(0)

            assert cap.major == 3
            assert cap.minor == 1

    def test_get_device_name(self):
        """Test get_device_name returns device name."""
        with patch("torch.cuda.get_device_name") as mock_name:
            mock_name.return_value = "MTT S80"

            from vllm_musa.platform import NonMtmlMUSAPlatform

            name = NonMtmlMUSAPlatform.get_device_name(0)

            assert name == "MTT S80"

    def test_get_device_total_memory(self):
        """Test get_device_total_memory returns memory size."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 * 1024 * 1024  # 80GB

        with patch("torch.cuda.get_device_properties") as mock_get_props:
            mock_get_props.return_value = mock_props

            from vllm_musa.platform import NonMtmlMUSAPlatform

            memory = NonMtmlMUSAPlatform.get_device_total_memory(0)

            assert memory == 80 * 1024 * 1024 * 1024

    def test_is_fully_connected_returns_false_with_warning(self):
        """Test is_fully_connected returns False without MTML."""
        from vllm_musa.platform import NonMtmlMUSAPlatform

        result = NonMtmlMUSAPlatform.is_fully_connected([0, 1])

        assert result is False


class TestWithMtmlContext:
    """Tests for the with_mtml_context decorator."""

    def test_decorator_returns_function_result(self):
        """Test that the decorator returns the wrapped function's result."""
        from vllm_musa.platform import mtml_available, with_mtml_context

        if not mtml_available:
            pytest.skip("MTML not available")

        @with_mtml_context
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_preserves_function_name(self):
        """Test that the decorator preserves the wrapped function's name."""
        from vllm_musa.platform import with_mtml_context

        @with_mtml_context
        def my_test_function():
            return "test"

        assert my_test_function.__name__ == "my_test_function"


class TestMtmlMUSAPlatform:
    """Tests for MtmlMUSAPlatform class."""

    def test_get_device_capability_returns_3_1(self, mock_pymtml):
        """Test get_device_capability returns (3, 1) for FP8 support."""
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        # Clear cache
        MtmlMUSAPlatform.get_device_capability.cache_clear()

        cap = MtmlMUSAPlatform.get_device_capability(0)

        assert cap.major == 3
        assert cap.minor == 1

    def test_get_device_name(self):
        """Test get_device_name returns a string."""
        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        name = MtmlMUSAPlatform.get_device_name(0)

        assert isinstance(name, str)
        assert len(name) > 0
        # MUSA device names typically start with "MTT"
        assert "MTT" in name or len(name) > 0

    def test_get_device_uuid(self):
        """Test get_device_uuid returns a valid UUID string."""
        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        uuid = MtmlMUSAPlatform.get_device_uuid(0)

        assert isinstance(uuid, str)
        # UUIDs have a specific format with dashes
        assert "-" in uuid
        assert len(uuid) >= 32  # Minimum UUID length

    def test_get_device_total_memory(self):
        """Test get_device_total_memory returns a positive integer."""
        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        memory = MtmlMUSAPlatform.get_device_total_memory(0)

        assert isinstance(memory, int)
        assert memory > 0
        # Typical GPU memory is at least 4GB
        assert memory >= 4 * 1024 * 1024 * 1024


class TestPlatformSelection:
    """Tests for platform autodetection."""

    def test_musa_platform_is_one_of_two_options(self):
        """Test that MUSAPlatform is either MtmlMUSAPlatform or NonMtmlMUSAPlatform."""
        from vllm_musa.platform import (
            MtmlMUSAPlatform,
            MUSAPlatform,
            NonMtmlMUSAPlatform,
        )

        assert MUSAPlatform in (MtmlMUSAPlatform, NonMtmlMUSAPlatform)

    def test_platform_selection_based_on_mtml_availability(self):
        """Test that platform selection is correct based on MTML availability."""
        from vllm_musa.platform import (
            MtmlMUSAPlatform,
            MUSAPlatform,
            NonMtmlMUSAPlatform,
            mtml_available,
        )

        if mtml_available:
            assert MUSAPlatform is MtmlMUSAPlatform
        else:
            assert MUSAPlatform is NonMtmlMUSAPlatform


class TestImportTorchada:
    """Tests for torchada import handling."""

    def test_torchada_is_imported(self):
        """Test that torchada is imported when musa module loads."""
        # torchada should be available in sys.modules after importing musa
        import vllm_musa.platform  # noqa: F401

        assert "torchada" in sys.modules


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from vllm_musa import platform

        assert hasattr(platform, "__all__")

        expected_exports = [
            "MUSAPlatform",
            "MUSAPlatformBase",
            "MtmlMUSAPlatform",
            "NonMtmlMUSAPlatform",
            "with_mtml_context",
            "mtml_available",
        ]

        for export in expected_exports:
            assert export in platform.__all__, f"{export} not in __all__"
            assert hasattr(platform, export), f"{export} not defined in module"

    def test_musa_platform_plugin_function_exists(self):
        """Test that musa_platform_plugin function exists for entry point."""
        from vllm_musa import musa_platform_plugin

        assert callable(musa_platform_plugin)
