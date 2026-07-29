"""Source-level guards for fused CAR-RMSNorm and its Graph input path."""

from __future__ import annotations

import ast
import os
import runpy
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
COMM = ROOT / (
    "vllm_musa/distributed/device_communicators/"
    "musa_jit_custom_all_reduce.py"
)
LAUNCHER = ROOT / "vllm_musa/jit_kernel/csrc/allreduce.py"
KERNEL = ROOT / (
    "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce.mu"
)
ENVIRON = ROOT / "vllm_musa/utils/environ.py"
FUSED_OPS = ROOT / "vllm_musa/fused_allreduce_rmsnorm_ops.py"
FUSION_PASS = ROOT / "vllm_musa/_inductor/musa_allreduce_rms_fusion.py"


def _ffi_function_source(source: str, function_name: str) -> str:
    start = source.index(f"void {function_name}(")
    next_start = source.find("\nvoid vllm_musa_", start + 1)
    return source[start:] if next_start == -1 else source[start:next_start]


def test_fused_path_gates_can_be_disabled_independently() -> None:
    namespace = runpy.run_path(str(ENVIRON))
    envs = namespace["envs"]
    fused = envs.VLLM_MUSA_FUSED_AR_RMSNORM
    registered = envs.VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT

    with patch.dict(os.environ, {}, clear=True):
        assert fused.get() is True
        assert registered.get() is True
        with fused.override(False):
            assert fused.get() is False
            assert registered.get() is True
        with registered.override(False):
            assert fused.get() is True
            assert registered.get() is False


def test_fusion_uses_one_canonical_feature_gate() -> None:
    env_source = ENVIRON.read_text()
    comm_source = COMM.read_text()
    fusion_source = FUSION_PASS.read_text()
    canonical = "VLLM_MUSA_FUSED_AR_RMSNORM"
    assert canonical in env_source
    assert canonical in comm_source
    assert canonical in fusion_source


def test_python_registered_path_is_syntax_valid_and_dependency_free() -> None:
    comm_source = COMM.read_text()
    launcher_source = LAUNCHER.read_text()
    ast.parse(comm_source)
    ast.parse(launcher_source)
    assert "_register_graph_buffers()" in comm_source
    assert "capture_succeeded" in comm_source
    assert "def _graph_rank_data_for_input" in comm_source
    assert "_pending_graph_inputs" in comm_source
    assert "_GraphInputRegistration" not in comm_source
    assert "torch.get_device_module().is_current_stream_capturing()" in comm_source
    assert "torch.musa.synchronize(self.device)" in comm_source
    assert "_graph_registered_input_eligible" in comm_source
    assert comm_source.count("_use_registered_graph_input(input)") == 3
    # Ordinary CAR and the three fused variants share the same graph slots.
    assert comm_source.count("_graph_rank_data_for_input(input)") == 4
    assert launcher_source.count("_check_registered_rank_data(rank_data)") == 3


def test_fused_kernel_copy_is_cpu_rank_data_only() -> None:
    source = KERNEL.read_text()
    for function_name in (
        "vllm_musa_fused_ar_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered",
    ):
        function_source = _ffi_function_source(source, function_name)
        assert "const bool rank_data_on_cpu" in function_source
        assert "const bool rank_data_on_musa" in function_source
        assert (
            "rank_data.device().device_type == inp.device().device_type"
            in function_source
        )
        assert "const RankData* device_data = nullptr;" in function_source
        assert "if (rank_data_on_cpu) {\n    const musaError_t copy_err" in (
            function_source
        )
        assert function_source.count("musaMemcpyAsync(") == 1


def test_tp2_specialized_fast_paths_preserve_vllm_abis() -> None:
    source = KERNEL.read_text()
    assert "fused_ar_rmsnorm_tp2_specialized_kernel" in source
    assert source.count("launch_fused_ar_rmsnorm_tp2_specialized<T, WT,") == 3
    assert "hidden == 5120 && rows <= 128" in source
    assert "hidden == 2048 && rows <= 128" in source
    assert "load_weight_scalar<WT>" in source
    assert "if constexpr (WriteReduced)" in source
    assert source.count(
        'TVM_FFI_ICHECK(shot == 1 || shot == 2) << "shot must be 1 or 2";'
    ) == 3
    assert source.count(
        "if (shot == 1) {\n    if constexpr (nranks == 2)"
    ) == 3


def test_fused_ops_reuse_the_lifecycle_managed_jit_registry() -> None:
    comm_source = COMM.read_text()
    fused_source = FUSED_OPS.read_text()
    fusion_source = FUSION_PASS.read_text()
    ast.parse(fused_source)
    assert "def get_musa_jit_custom_allreduce_comm" in comm_source
    assert "get_musa_jit_custom_allreduce_comm" in fused_source
    assert "self.comm_id = self.jit_comm_id" in fusion_source


def test_manual_rewrite_is_scoped_to_comm_and_full_variance() -> None:
    source = FUSION_PASS.read_text()
    assert "def _is_target_musa_car_node" in source
    assert "self._car_comm_id(node) == self.jit_comm_id" in source
    assert "if not self._is_target_musa_car_node(car):" in source
    assert source.count("variance_size is not None") == 3


def test_fusion_runtime_state_participates_in_cache_uuid() -> None:
    source = FUSION_PASS.read_text()
    assert "def uuid(self) -> str:" in source
    assert '"enabled": not self.disabled' in source
    assert '"comm_id": self.comm_id' in source
    assert '"jit_comm_id": self.jit_comm_id' in source
    assert '"group_name": self.group_name' in source


def test_fused_kernel_validates_world_size_before_fixed_registry_arrays() -> None:
    source = KERNEL.read_text()
    for function_name in (
        "vllm_musa_fused_ar_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered",
        "vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered",
    ):
        function_source = _ffi_function_source(source, function_name)
        assert function_source.count("validate_world_size(world_size);") == 1
        assert function_source.index("validate_world_size(world_size);") < (
            function_source.index("RankSignals sg{};")
        )


def test_no_raw_one_shot_matches_raw_dtype_rounding_order() -> None:
    source = KERNEL.read_text()
    assert "const T reduced_value = from_float<T>(acc[i]);" in source
    assert (
        "to_float(reduced_value) + to_float(residual_pack.data[i])"
        in source
    )


def test_tp4_rows64_two_shot_uses_bounded_block_count() -> None:
    kernel_source = KERNEL.read_text()
    assert "kTp4Rows64TwoShotBlockLimit = 32" in kernel_source
    assert "if constexpr (nranks == 4)" in kernel_source
    assert "if (rows == 64 && hidden == 2048)" in kernel_source
    assert (
        kernel_source.count(
            "fused_ar_rmsnorm_2shot_blocks<nranks>(rows, hidden)"
        )
        == 3
    )


def test_registered_graph_input_has_a_fixed_staging_boundary() -> None:
    comm_source = COMM.read_text()
    assert "_GRAPH_REGISTERED_INPUT_MAX_BYTES = 512 * 1024" in comm_source
    assert (
        "tensor.numel() * tensor.element_size()\n"
        "            <= self._GRAPH_REGISTERED_INPUT_MAX_BYTES"
    ) in comm_source
