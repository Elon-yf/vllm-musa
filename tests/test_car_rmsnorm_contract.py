# SPDX-License-Identifier: Apache-2.0
"""Pure-Python coverage for the shared CAR-RMSNorm routing contract."""

from types import SimpleNamespace

import torch

from vllm_musa.optimization_contract.car_rmsnorm import (
    FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    can_use_fused_allreduce_rmsnorm,
    fused_allreduce_rmsnorm_compile_endpoints,
    fused_allreduce_rmsnorm_compile_reject_reason,
    can_enable_fused_allreduce_rmsnorm,
    can_use_registered_graph_input_for_generic_car,
)


def _allowed(
    *,
    tp: int,
    hidden: int,
    rows: int | None = None,
    start: int | None = None,
    end: int | None = None,
    quantized: bool | None = False,
    family: str | None = FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    phase: str | None = "decode",
) -> bool:
    compile_range = None
    if start is not None or end is not None:
        compile_range = SimpleNamespace(start=start, end=end)
    return can_use_fused_allreduce_rmsnorm(
        tp_size=tp,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        rows=rows,
        compile_range=compile_range,
        raw_needed=False,
        registered=False,
        model_family=family,
        quantized=quantized,
        phase=phase,
        path="staging",
    )


def test_generic_graph_registered_input_policy_is_narrow() -> None:
    policy = can_use_registered_graph_input_for_generic_car
    common = {"model_family": FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY}

    assert not policy(tp_size=4, hidden_size=2048, quantized=False, **common)
    assert policy(tp_size=4, hidden_size=2048, quantized=True, **common)
    assert policy(tp_size=2, hidden_size=5120, quantized=False, **common)
    assert policy(tp_size=2, hidden_size=5120, quantized=True, **common)
    assert policy(tp_size=8, hidden_size=4096, quantized=False, **common)
    assert policy(
        tp_size=None,
        hidden_size=None,
        quantized=None,
        model_family=None,
    )


def test_tp2_bf16_and_fp8_weight_deny_rows() -> None:
    assert _allowed(tp=2, hidden=5120, rows=1)
    assert not _allowed(tp=2, hidden=5120, rows=16)
    assert _allowed(tp=2, hidden=5120, rows=64)

    assert _allowed(tp=2, hidden=5120, rows=1, quantized=True)
    assert not _allowed(tp=2, hidden=5120, rows=4, quantized=True)
    assert not _allowed(tp=2, hidden=5120, rows=16, quantized=True)
    assert _allowed(tp=2, hidden=5120, rows=64, quantized=True)


def test_tp4_bf16_and_fp8_weight_deny_rows() -> None:
    assert _allowed(tp=4, hidden=2048, rows=1)
    assert not _allowed(tp=4, hidden=2048, rows=16)
    assert not _allowed(tp=4, hidden=2048, rows=64)

    assert _allowed(tp=4, hidden=2048, rows=16, quantized=True)
    assert not _allowed(tp=4, hidden=2048, rows=64, quantized=True)


def test_compile_ranges_must_not_cross_a_native_row() -> None:
    assert _allowed(tp=2, hidden=5120, start=1, end=15)
    assert not _allowed(tp=2, hidden=5120, start=15, end=16)
    assert _allowed(tp=2, hidden=5120, start=17, end=63)
    assert _allowed(tp=2, hidden=5120, start=64, end=4096)

    assert _allowed(tp=2, hidden=5120, start=1, end=3, quantized=True)
    assert not _allowed(tp=2, hidden=5120, start=3, end=4, quantized=True)
    assert _allowed(tp=2, hidden=5120, start=5, end=15, quantized=True)

    assert _allowed(tp=4, hidden=2048, start=1, end=15)
    assert not _allowed(tp=4, hidden=2048, start=1, end=16)
    assert not _allowed(tp=4, hidden=2048, start=15, end=16)
    assert not _allowed(tp=4, hidden=2048, start=16, end=16)
    assert not _allowed(tp=4, hidden=2048, start=17, end=63)
    assert not _allowed(tp=4, hidden=2048, start=64, end=64)
    assert not _allowed(tp=4, hidden=2048, start=65, end=4096)

    assert _allowed(tp=4, hidden=2048, start=1, end=63, quantized=True)
    assert not _allowed(tp=4, hidden=2048, start=64, end=64, quantized=True)


def test_tp4_bf16_rows_outside_fused_range_fail_closed() -> None:
    assert _allowed(tp=4, hidden=2048, rows=4)
    assert _allowed(tp=4, hidden=2048, rows=15)
    assert not _allowed(tp=4, hidden=2048, rows=16)
    assert not _allowed(tp=4, hidden=2048, rows=17)
    assert not _allowed(tp=4, hidden=2048, rows=64)
    assert not _allowed(tp=4, hidden=2048, rows=4096)

    # A concrete row must not be hidden inside an inconsistent symbolic range.
    assert not _allowed(tp=4, hidden=2048, rows=16, start=1, end=15)
    assert _allowed(tp=4, hidden=2048, rows=4, start=1, end=15)


def test_endpoints_are_shared_for_bf16_and_fp8_weight_cells() -> None:
    assert fused_allreduce_rmsnorm_compile_endpoints(tp_size=2, hidden_size=5120) == (
        3,
        4,
        15,
        16,
        63,
    )
    assert fused_allreduce_rmsnorm_compile_endpoints(tp_size=4, hidden_size=2048) == (
        15,
        16,
        63,
        64,
    )
    assert fused_allreduce_rmsnorm_compile_endpoints(tp_size=8, hidden_size=2048) == ()


def test_default_enablement_is_bounded_to_known_car_signatures() -> None:
    common = {
        "pp_size": 1,
        "dtype": torch.bfloat16,
        "model_family": FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    }

    assert can_enable_fused_allreduce_rmsnorm(
        tp_size=2, hidden_size=5120, **common
    )
    assert can_enable_fused_allreduce_rmsnorm(
        tp_size=4, hidden_size=2048, **common
    )
    assert not can_enable_fused_allreduce_rmsnorm(
        tp_size=8, hidden_size=2048, **common
    )
    assert not can_enable_fused_allreduce_rmsnorm(
        tp_size=2, hidden_size=4096, **common
    )
    assert not can_enable_fused_allreduce_rmsnorm(
        tp_size=2, pp_size=2, hidden_size=5120, dtype=torch.bfloat16,
        model_family=FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    )


def test_unknown_metadata_fails_closed_but_known_phases_share_the_contract() -> None:
    assert not _allowed(tp=2, hidden=5120, rows=1, family=None)
    assert not _allowed(tp=2, hidden=5120, rows=1, quantized=None)
    assert _allowed(tp=2, hidden=5120, rows=1, phase="mixed")
    assert not _allowed(tp=2, hidden=5120, rows=None)
    assert fused_allreduce_rmsnorm_compile_reject_reason(
        tp_size=2,
        hidden_size=5120,
        dtype=torch.float32,
        rows=1,
        model_family=FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
        quantized=False,
    )


def test_tp1_keeps_non_car_provider_compatibility() -> None:
    assert _allowed(tp=1, hidden=5120, rows=16)


def test_compile_endpoints_isolate_every_native_row() -> None:
    """Guard the hand-listed endpoint tuples against policy-table edits.

    ``fused_allreduce_rmsnorm_compile_endpoints`` spells its cut points out
    longhand while the deny rows live in ``CAR_RMSNORM_POLICY_TABLE``. Adding a
    native row without adding its two cuts would silently leave that row inside
    a fused bucket, so assert the correspondence instead of trusting a comment.
    """
    from vllm_musa.optimization_contract.car_rmsnorm import CAR_RMSNORM_POLICY_TABLE

    for rule in CAR_RMSNORM_POLICY_TABLE:
        endpoints = set(
            fused_allreduce_rmsnorm_compile_endpoints(
                tp_size=rule["tp_size"], hidden_size=rule["hidden_size"]
            )
        )
        assert endpoints, rule
        for row in rule["native_rows"]:
            # Isolating ``row`` needs the cut below it and the cut that closes it.
            assert row - 1 in endpoints, (rule, row)
            assert row in endpoints, (rule, row)


def _car_config(
    *,
    optimization_level: int = 2,
    tp_size: int = 2,
    hidden_size: int = 5120,
    pp_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    pass_value: bool | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        optimization_level=optimization_level,
        model_config=SimpleNamespace(dtype=dtype, get_hidden_size=lambda: hidden_size),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_allreduce_rms=pass_value)
        ),
    )


def _pin_family(monkeypatch) -> None:
    from vllm_musa.optimization_contract import car_rmsnorm as contract

    monkeypatch.setattr(
        contract,
        "infer_car_rmsnorm_model_family",
        lambda _config: FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    )


def test_default_on_rule_is_owned_by_the_contract(monkeypatch) -> None:
    """The contract, not each call site, decides when CAR-RMSNorm defaults on."""
    from vllm_musa.optimization_contract import car_rmsnorm as contract

    _pin_family(monkeypatch)

    assert contract.car_rmsnorm_default_on(_car_config())
    assert contract.car_rmsnorm_default_on(_car_config(optimization_level=3))
    # Below the measured optimization level the feature stays off.
    assert not contract.car_rmsnorm_default_on(_car_config(optimization_level=1))
    assert not contract.car_rmsnorm_default_on(_car_config(optimization_level=0))
    # Outside a policy cell nothing defaults on.
    assert not contract.car_rmsnorm_default_on(_car_config(tp_size=8))
    assert not contract.car_rmsnorm_default_on(_car_config(hidden_size=4096))
    assert not contract.car_rmsnorm_default_on(_car_config(pp_size=2))
    assert not contract.car_rmsnorm_default_on(_car_config(dtype=torch.float32))


def test_explicit_pass_value_overrides_the_default(monkeypatch) -> None:
    """An explicit setting stays authoritative in both directions."""
    from vllm_musa.optimization_contract import car_rmsnorm as contract

    _pin_family(monkeypatch)

    assert contract.resolve_car_rmsnorm_enabled(
        _car_config(pass_value=True, optimization_level=0)
    )
    assert not contract.resolve_car_rmsnorm_enabled(
        _car_config(pass_value=False, optimization_level=3)
    )
    # Unset falls back to exactly the rule the platform writes out.
    unset = _car_config(pass_value=None)
    assert contract.resolve_car_rmsnorm_enabled(unset) is (
        contract.car_rmsnorm_default_on(unset)
    )
