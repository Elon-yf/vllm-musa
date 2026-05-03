# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA Platform implementation."""

import sys
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

    def test_fp8_scaled_mm_uses_weight_scale_inv_fallback(self):
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

    def test_fp8_scaled_mm_accepts_loaded_out_in_weight(self):
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
                weight_shape=(2048, 576),
                module_name="musa_fp8_shape_test",
            )

        captured = {}

        def fake_gemv(x, qweight, x_scales, qweight_scales):
            captured["qweight_shape"] = qweight.shape
            return torch.zeros(x.shape[0], qweight.shape[0], dtype=torch.bfloat16)

        with patch("vllm_musa.fp8_linear.musa_ops.musa_fused_gemv", fake_gemv):
            output = kernel.apply_scaled_mm(
                A=torch.empty(1, 2048, dtype=torch.float8_e4m3fn),
                B=torch.empty(576, 2048, dtype=torch.float8_e4m3fn),
                out_dtype=torch.float16,
                As=torch.ones(1, 16),
                Bs=torch.ones(5, 16),
                bias=None,
                output_shape=[1, 2048],
            )

        assert captured["qweight_shape"] == (576, 2048)
        assert output.shape == (1, 576)

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
