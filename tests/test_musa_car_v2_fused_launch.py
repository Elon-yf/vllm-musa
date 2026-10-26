"""CPU checks for real v2 fused factory configuration and source ABI routing.

The JIT loader is stubbed so these tests exercise factory/cache behavior without
claiming MUSA compilation, numerical accuracy, or graph replay correctness.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLREDUCE = ROOT / "vllm_musa/jit_kernel/csrc/allreduce.py"
POLICY = ROOT / "vllm_musa/optimization_contract/Fuse_CAR_RMSnorm.py"
V2_SOURCE = ROOT / "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce_v2.mu"


def _load(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def jit(monkeypatch):
    for name in (
        "vllm_musa", "vllm_musa.jit_kernel", "vllm_musa.jit_kernel.csrc",
        "vllm_musa.optimization_contract",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    policy = _load(POLICY, "vllm_musa.optimization_contract.Fuse_CAR_RMSnorm", monkeypatch)
    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16, torch.float32 = "fp16", "bf16", "fp32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    builds, calls = [], []

    def load_jit(name, sources, *, extra_musa_cflags):
        builds.append((name, sources, extra_musa_cflags))
        result = types.SimpleNamespace()
        for suffix in ("unregistered", "registered", "row_unregistered", "row_registered"):
            symbol = f"sgl_musa_custom_ar_fused_allreduce_rmsnorm_{suffix}"
            def invoke(*args, _symbol=symbol):
                calls.append((_symbol, args))
            setattr(result, symbol, invoke)
        return result

    loader = types.ModuleType("vllm_musa.jit_kernel.csrc.jit")
    loader.load_musa_jit = load_jit
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    module = _load(ALLREDUCE, "vllm_musa.jit_kernel.csrc.allreduce", monkeypatch)
    return module, policy, builds, calls


@pytest.mark.parametrize("row,registered", [(False, False), (False, True), (True, False), (True, True)])
def test_fixed_abi_routes_all_four_source_entries(jit, row, registered):
    module, _, builds, calls = jit
    launcher = module.v2_fused_allreduce_rmsnorm_func(
        4, 2048, 1, mp_count=60, row_kernel=row, graph_rank_data=registered,
    )
    tensors = [object() for _ in range(7)]
    rd, sg, inp, residual, rout, norm, weight = tensors
    launcher(rd, sg, inp, residual, rout, norm, weight, None, 11, 12, 1024, 0, 4, 1e-5)
    suffix = ("row_" if row else "") + ("registered" if registered else "unregistered")
    symbol, args = calls.pop()
    assert symbol.endswith(suffix)
    if registered:
        expected = (rd, sg, residual, rout, norm, weight, 11, 0, 4)
    elif row:
        expected = (rd, sg, inp, residual, rout, norm, weight, 11, 12, 1024, 0, 4)
    else:
        expected = (rd, sg, inp, norm, residual, rout, weight, 11, 12, 1024, 0, 4)
    assert args == expected + ((2048,) if row else ()) + (1e-5,)
    assert builds[0][1] == ("distributed/custom_all_reduce_v2.mu",)
    assert "-DSGL_CUSTOM_AR_FUSED_RMSNORM=1" in builds[0][2]
    assert launcher.kernel_symbol == symbol
    assert launcher.jit_cflags == builds[0][2]
    assert f"-DSGL_CUSTOM_AR_THREADS={launcher.launch_config.threads}" in launcher.jit_cflags


def test_policy_launch_fields_reach_compiler(jit):
    module, _, builds, _ = jit
    cfg = module.v2_fused_compile_config(4, 2048, 1, mp_count=60)
    module.v2_fused_allreduce_rmsnorm_func(4, 2048, 1, mp_count=60)
    flags = builds[0][2]
    assert f"-DSGL_CUSTOM_AR_THREADS={cfg.threads}" in flags
    assert f"-DSGL_CUSTOM_AR_BLOCKS={cfg.blocks}" in flags
    assert f"-DSGL_CUSTOM_AR_MAX_BLOCKS={cfg.max_blocks}" in flags
    assert f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={cfg.dynamic_blocks}" in flags
    assert set(cfg.cflags()).issubset(flags)


def test_complete_effective_config_changes_memory_and_disk_cache(jit, monkeypatch):
    module, _, builds, _ = jit
    cfg = module.v2_fused_compile_config(4, 2048, 1, mp_count=60)
    monkeypatch.setattr(module, "v2_fused_compile_config", lambda *args, **kwargs: cfg)
    factory = module.v2_fused_allreduce_rmsnorm_func
    factory(4, 2048, 1, mp_count=60)
    factory(4, 2048, 1, mp_count=60)
    assert len(builds) == 1
    cfg = replace(cfg, max_blocks=cfg.max_blocks + 1)
    factory(4, 2048, 1, mp_count=60)
    assert len(builds) == 2 and builds[0][0] != builds[1][0]
    factory(4, 2048, 1, mp_count=60, cache_policy=1)
    assert len(builds) == 3 and builds[1][0] != builds[2][0]
    monkeypatch.setenv("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "1")
    factory(4, 2048, 1, mp_count=60, cache_policy=1)
    assert len(builds) == 4 and builds[2][0] != builds[3][0]


@pytest.mark.parametrize("options", [{"need_raw": True}, {"has_residual": False}])
def test_unsupported_output_abi_fails_before_compile(jit, options):
    module, _, builds, _ = jit
    with pytest.raises(NotImplementedError, match="explicit kernel output support"):
        module.v2_fused_allreduce_rmsnorm_func(4, 2048, 1, mp_count=60, **options)
    assert not builds


def test_pure_v2_does_not_select_fused_translation_unit(jit):
    module, _, builds, _ = jit
    module._custom_ar_v2_module(4, module.SHOT_PUSH)
    assert not any("FUSED_RMSNORM" in flag for flag in builds[0][2])


def test_fused_translation_unit_has_four_typed_ffi_exports():
    source = V2_SOURCE.read_text()
    assert "#if !defined(SGL_CUSTOM_AR_FUSED_RMSNORM)" in source
    fused = source.split("#else  // SGL_CUSTOM_AR_FUSED_RMSNORM", 1)[1]
    assert fused.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC(") == 4
