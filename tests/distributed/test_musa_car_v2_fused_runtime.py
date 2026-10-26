"""Run with torchrun on TP2/TP4; --help and --list-cases need no GPU imports.

Example: torchrun --standalone --nproc_per_node=2 \
    tests/distributed/test_musa_car_v2_fused_runtime.py --output-dir /tmp/car-fused-tp2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    impl: str
    dtype: str
    weight_dtype: str
    hidden: int
    rows: int
    variant: str


def _csv(value: str) -> list[str]:
    values = value.split(",")
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return values


def _positive_csv(value: str) -> list[int]:
    try:
        values = [int(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected positive integers") from error
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impls", type=_csv, default=["v2", "v1"])
    parser.add_argument("--dtypes", type=_csv, default=["bfloat16", "float16"])
    parser.add_argument("--weight-dtypes", type=_csv, default=["same"])
    parser.add_argument("--hidden", type=_positive_csv, default=[2048, 5120])
    parser.add_argument("--rows", type=_positive_csv, default=[1, 16, 128, 256])
    parser.add_argument("--variants", type=_csv, default=["norm", "residual", "no_raw"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--max-size-mib", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fused-dispatch", choices=["0", "1"], default="1")
    parser.add_argument("--output-dir", type=Path, default=Path("car-v2-fused-results"))
    parser.add_argument(
        "--profile", action="store_true", help="export per-rank device traces"
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="print plan without execution"
    )
    return parser


def _cases(args: argparse.Namespace) -> list[Case]:
    allowed = {
        "impls": {"v1", "v2"},
        "dtypes": {"float16", "bfloat16"},
        "weight_dtypes": {"same", "float32"},
        "variants": {"norm", "residual", "no_raw"},
    }
    for name, choices in allowed.items():
        if not set(getattr(args, name)) <= choices:
            raise ValueError(f"unsupported {name}: {getattr(args, name)}")
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            raise ValueError(f"duplicate {name} entries")
    if (
        min(
            args.iterations, args.graph_replays, args.max_size_mib, args.timeout_seconds
        )
        < 1
    ):
        raise ValueError(
            "iteration, replay, buffer and timeout values must be positive"
        )
    if args.eps <= 0:
        raise ValueError("eps must be positive")
    if max(args.rows) * max(args.hidden) * 2 > args.max_size_mib * 1024**2:
        raise ValueError("target shapes exceed --max-size-mib")
    return [
        Case(*values)
        for values in itertools.product(
            args.impls,
            args.dtypes,
            args.weight_dtypes,
            args.hidden,
            args.rows,
            args.variants,
        )
    ]


def _runtime() -> tuple[Any, Any, Any]:
    # isort: off
    import torchada  # noqa: F401
    import torch
    import torch.distributed as dist

    # isort: on

    if getattr(torch.version, "musa", None) is None or not hasattr(torch, "musa"):
        raise RuntimeError("real MUSA torch runtime is required; no GPU checks ran")
    if not torch.musa.is_available():
        raise RuntimeError("MUSA devices are unavailable; no GPU checks ran")
    from vllm_musa.distributed.device_communicators import musa_jit_custom_all_reduce

    return torch, dist, musa_jit_custom_all_reduce


def _fp32_weight_cases(args: argparse.Namespace) -> list[Case]:
    if args.skip_fallback or "v2" not in args.impls or "float32" in args.weight_dtypes:
        return []
    return [
        Case("v2", "bfloat16", "float32", 2048, 1, variant)
        for variant in ("norm", "residual", "no_raw")
    ]


def _inputs(torch: Any, case: Case, rank: int, step: int, seed: int) -> tuple[Any, ...]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed + rank * 1009 + step * 37
    )
    dtype = getattr(torch, case.dtype)
    shape = (case.rows, case.hidden)
    x = torch.randn(shape, generator=generator).to(dtype)
    residual = torch.randn(shape, generator=generator).to(dtype)
    # All ranks use the same normalization weights.
    generator.manual_seed(seed + step * 53)
    weight_dtype = dtype if case.weight_dtype == "same" else torch.float32
    weight = (1 + 0.1 * torch.randn((case.hidden,), generator=generator)).to(
        weight_dtype
    )
    return x, residual, weight


def _reference(
    torch: Any, dist: Any, group: Any, values: tuple[Any, ...], variant: str, eps: float
) -> tuple[Any, ...]:
    x, residual, weight = values
    reduced = x.float().clone()
    dist.all_reduce(reduced, group=group)
    reduced = reduced.to(x.dtype)
    residual_out = (reduced.float() + residual.float()).to(x.dtype)
    normalized = reduced if variant == "norm" else residual_out
    fp32 = normalized.float()
    norm = (
        fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(x.dtype)
    if variant == "norm":
        return norm, reduced
    if variant == "residual":
        return norm, residual_out, reduced
    return norm, residual_out


def _call(comm: Any, values: tuple[Any, ...], variant: str, eps: float) -> Any:
    x, residual, weight = values
    if variant == "norm":
        return comm.fused_allreduce_rmsnorm(x, weight, eps)
    if variant == "residual":
        return comm.fused_allreduce_residual_rmsnorm(x, residual, weight, eps)
    return comm.fused_allreduce_residual_rmsnorm_no_raw(x, residual, weight, eps)


def _eligible(comm: Any, values: tuple[Any, ...], variant: str) -> bool:
    x, residual, weight = values
    if variant == "norm":
        return comm.should_fused_allreduce_rmsnorm(x, weight)
    return comm.should_fused_allreduce_residual_rmsnorm(x, residual, weight)


def _validate(
    torch: Any,
    outputs: Any,
    expected: tuple[Any, ...],
    inputs: tuple[Any, ...],
    original: tuple[Any, ...],
) -> float:
    if not isinstance(outputs, tuple) or len(outputs) != len(expected):
        raise AssertionError(
            f"fused output ABI mismatch: {type(outputs)}, expected {len(expected)}"
        )
    storage = [value.untyped_storage().data_ptr() for value in (*inputs, *outputs)]
    if len(storage) != len(set(storage)):
        raise AssertionError("fused outputs must not alias inputs or each other")
    for value, snapshot in zip(inputs, original):
        torch.testing.assert_close(value.cpu(), snapshot, rtol=0, atol=0)
    rtol, atol = (0.016, 1e-3) if inputs[0].dtype == torch.bfloat16 else (0.001, 1e-4)
    max_error = 0.0
    for value, target in zip(outputs, expected):
        actual = value.cpu()
        torch.testing.assert_close(actual, target.cpu(), rtol=rtol, atol=atol)
        max_error = max(
            max_error, (actual.float() - target.cpu().float()).abs().max().item()
        )
    return max_error


def _consensus(dist: Any, group: Any, error: str | None) -> None:
    errors: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failures = [
        f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
    ]
    if failures:
        raise AssertionError("; ".join(failures))


def _checked(
    torch: Any,
    dist: Any,
    group: Any,
    outputs: Any,
    expected: tuple[Any, ...],
    device_inputs: tuple[Any, ...],
    original: tuple[Any, ...],
) -> float:
    error = None
    max_error = 0.0
    try:
        torch.musa.synchronize()
        max_error = _validate(torch, outputs, expected, device_inputs, original)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _consensus(dist, group, error)
    return max_error


def _execution_mode(
    dist: Any, group: Any, comm: Any, impl: str, expected: str | None = None
) -> str:
    mode = getattr(comm, "_last_fused_execution", None) if impl == "v2" else "fused"
    modes: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(modes, mode, group=group)
    if any(value not in ("fused", "fallback") for value in modes):
        raise AssertionError(f"missing actual fused execution mode: {modes}")
    if any(value != modes[0] for value in modes):
        raise AssertionError(f"execution modes differ between ranks: {modes}")
    if expected is not None and modes[0] != expected:
        raise AssertionError(f"expected {expected} execution, got {modes[0]}")
    return str(modes[0])


def _run_case(
    torch: Any,
    dist: Any,
    group: Any,
    comm: Any,
    device: Any,
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rank = dist.get_rank(group)
    errors: list[float] = []
    eager_modes: list[str] = []
    eligible_call = True
    for step in range(args.iterations):
        original = _inputs(torch, case, rank, step, args.seed)
        expected = _reference(torch, dist, group, original, case.variant, args.eps)
        inputs = tuple(value.to(device) for value in original)
        eligible: list[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(
            eligible, _eligible(comm, inputs, case.variant), group=group
        )
        if len(set(eligible)) != 1:
            raise AssertionError(
                f"communicator eligibility differs between ranks: {eligible}"
            )
        eligible_call = bool(eligible[0])
        expected_fallback = case.impl == "v2" and case.weight_dtype == "float32"
        if case.impl == "v2" and not eligible_call:
            raise AssertionError(
                f"v2 eligibility violates supported shape contract: {case}"
            )
        if eligible_call:
            if case.impl == "v2":
                comm._last_fused_execution = None
            outputs = _call(comm, inputs, case.variant, args.eps)
            execution = _execution_mode(
                dist, group, comm, case.impl, "fallback" if expected_fallback else None
            )
            eager_modes.append(execution)
        else:
            result = _call(comm, inputs, case.variant, args.eps)
            _consensus(
                dist,
                group,
                None if result is None else "expected None fallback sentinel",
            )
            outputs = _unfused(torch, dist, inputs, case.variant, args.eps)
            eager_modes.append("fallback")
        errors.append(_checked(torch, dist, group, outputs, expected, inputs, original))

    replays = 0
    graph_mode = None
    if not args.skip_graph and eligible_call:
        graph = torch.musa.MUSAGraph()
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with comm.capture():
            with torch.musa.stream(stream):
                # Prepare graph launch variants without active stream capture.
                _call(comm, inputs, case.variant, args.eps)
            torch.musa.current_stream().wait_stream(stream)
            torch.musa.synchronize()
            dist.barrier(group=group)
            if case.impl == "v2":
                comm._last_fused_execution = None
            with torch.musa.graph(graph, stream=stream):
                outputs = _call(comm, inputs, case.variant, args.eps)
        graph_mode = _execution_mode(
            dist, group, comm, case.impl, "fallback" if expected_fallback else None
        )
        for replay in range(args.graph_replays):
            original = _inputs(torch, case, rank, args.iterations + replay, args.seed)
            expected = _reference(torch, dist, group, original, case.variant, args.eps)
            for destination, source in zip(inputs, original):
                destination.copy_(source)
            torch.musa.synchronize()
            dist.barrier(group=group)
            graph.replay()
            errors.append(
                _checked(torch, dist, group, outputs, expected, inputs, original)
            )
            replays += 1
        torch.musa.synchronize()
    executed_modes = set(eager_modes)
    if graph_mode is not None:
        executed_modes.add(graph_mode)
    return {
        **asdict(case),
        "status": "PASS",
        "kind": (
            next(iter(executed_modes)).upper() if len(executed_modes) == 1 else "MIXED"
        ),
        "eager_calls": args.iterations,
        "eager_fused_calls": eager_modes.count("fused"),
        "eager_fallback_calls": eager_modes.count("fallback"),
        "graph_replays": replays,
        "graph_execution": graph_mode,
        "graph_fused_replays": replays if graph_mode == "fused" else 0,
        "graph_fallback_replays": replays if graph_mode == "fallback" else 0,
        "max_abs_error": max(errors),
    }


def _unfused(
    torch: Any, dist: Any, values: tuple[Any, ...], variant: str, eps: float
) -> tuple[Any, ...]:
    x, residual, weight = values
    reduced = x.float().clone()
    dist.all_reduce(reduced)
    reduced = reduced.to(x.dtype)
    residual_out = (reduced.float() + residual.float()).to(x.dtype)
    value = reduced if variant == "norm" else residual_out
    fp32 = value.float()
    norm = (
        fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(x.dtype)
    if variant == "norm":
        return norm, reduced
    return (
        (norm, residual_out, reduced) if variant == "residual" else (norm, residual_out)
    )


def _fallback_cases(
    torch: Any, device: Any, impl: str, args: argparse.Namespace
) -> list[tuple[str, tuple[Any, ...]]]:
    x = torch.ones((2, 2048), device=device, dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    weight = torch.ones((2048,), device=device, dtype=x.dtype)
    odd = torch.ones((2, 17), device=device, dtype=x.dtype)
    oversized = torch.ones(
        (args.max_size_mib * 1024**2 // (2048 * 2) + 1, 2048),
        device=device,
        dtype=x.dtype,
    )
    cases = [
        ("float32-input", (x.float(), residual.float(), weight.float())),
        (
            "unaligned-hidden",
            (odd, odd.clone(), torch.ones(17, device=device, dtype=x.dtype)),
        ),
        ("noncontiguous-input", (x.t().contiguous().t(), residual, weight)),
        ("oversized-input", (oversized, oversized.clone(), weight)),
    ]
    return cases


def _run_fallbacks(
    torch: Any,
    dist: Any,
    group: Any,
    comm: Any,
    device: Any,
    impl: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results = []
    for name, inputs in _fallback_cases(torch, device, impl, args):
        for variant in args.variants:
            original = tuple(value.cpu().clone() for value in inputs)
            expected = _reference(torch, dist, group, original, variant, args.eps)
            _consensus(
                dist,
                group,
                (
                    "unsupported input accepted by fused gate"
                    if _eligible(comm, inputs, variant)
                    else None
                ),
            )
            result = _call(comm, inputs, variant, args.eps)
            _consensus(
                dist,
                group,
                None if result is None else "expected None fallback sentinel",
            )
            outputs = _unfused(torch, dist, inputs, variant, args.eps)
            error = _checked(torch, dist, group, outputs, expected, inputs, original)
            results.append(
                {
                    "impl": impl,
                    "case": name,
                    "variant": variant,
                    "status": "PASS",
                    "kind": "FALLBACK",
                    "max_abs_error": error,
                }
            )
    return results


def _profiler(torch: Any, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    available = torch.profiler.supported_activities()
    device_activities = [
        activity
        for activity in available
        if activity != torch.profiler.ProfilerActivity.CPU
    ]
    if not device_activities:
        raise RuntimeError(
            "device profiler activity unavailable; refusing a CPU-only trace"
        )
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, *device_activities],
        record_shapes=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = _cases(args)
    fp32_weight_cases = _fp32_weight_cases(args)
    if args.list_cases:
        print(
            json.dumps(
                {
                    "status": "NOT_RUN",
                    "planned_cases": len(cases),
                    "cases": [asdict(case) for case in cases],
                    "fp32_weight_fallback_cases": [
                        asdict(case) for case in fp32_weight_cases
                    ],
                },
                indent=2,
            )
        )
        return 0
    os.environ["VLLM_MUSA_CUSTOM_AR_V2"] = "1"
    os.environ["VLLM_MUSA_CAR_FUSED_DISPATCH"] = args.fused_dispatch
    torch, dist, custom_ar = _runtime()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4):
        raise RuntimeError("launch using torchrun with TP2 or TP4; no checks ran")
    device = torch.device("musa", int(os.environ["LOCAL_RANK"]))
    torch.musa.set_device(device)
    dist.init_process_group(
        backend="cpu:gloo,musa:mccl", timeout=timedelta(seconds=args.timeout_seconds)
    )
    group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    rank = dist.get_rank()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"rank-{rank}.json"
    report: dict[str, Any] = {
        "status": "RUNNING",
        "rank": rank,
        "world_size": world_size,
        "torch_version": torch.__version__,
        "musa_version": torch.version.musa,
        "fused_dispatch": args.fused_dispatch,
        "cases": [],
        "fallbacks": [],
        "fp32_weight_fallbacks": [],
        "planned_cases": len(cases),
        "planned_fp32_weight_fallbacks": len(fp32_weight_cases),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "tolerances": {
            "bfloat16": {"rtol": 0.016, "atol": 1e-3},
            "float16": {"rtol": 0.001, "atol": 1e-4},
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with _profiler(torch, args.profile) as profiler:
            for impl in args.impls:
                cls = (
                    custom_ar._MusaCustomAllreduceV2Impl
                    if impl == "v2"
                    else custom_ar._MusaJitCustomAllreduceImpl
                )
                comm = cls(
                    group=group, device=device, max_size=args.max_size_mib * 1024**2
                )
                _consensus(
                    dist,
                    group,
                    "communicator initialization disabled" if comm.disabled else None,
                )
                for case in (case for case in cases if case.impl == impl):
                    report["active_case"] = asdict(case)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    row = _run_case(torch, dist, group, comm, device, case, args)
                    report["cases"].append(row)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                for case in (case for case in fp32_weight_cases if case.impl == impl):
                    report["active_case"] = asdict(case)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    row = _run_case(torch, dist, group, comm, device, case, args)
                    report["fp32_weight_fallbacks"].append(row)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                if not args.skip_fallback:
                    report["fallbacks"].extend(
                        _run_fallbacks(torch, dist, group, comm, device, impl, args)
                    )
                torch.musa.synchronize()
                dist.barrier(group=group)
                comm.close()
            torch.musa.synchronize()
        if args.profile:
            trace_path = args.output_dir / f"rank-{rank}.trace.json"
            profiler.export_chrome_trace(str(trace_path))
            report["trace"] = str(trace_path)
        report["status"] = "PASS"
        report.pop("active_case", None)
        report["passed_cases"] = len(report["cases"])
        report["passed_fp32_weight_fallbacks"] = len(report["fp32_weight_fallbacks"])
        execution_rows = report["cases"] + report["fp32_weight_fallbacks"]
        report["passed_fused_cases"] = sum(
            row["kind"] == "FUSED" for row in report["cases"]
        )
        report["passed_fallbacks"] = len(report["fallbacks"]) + sum(
            row["kind"] == "FALLBACK" for row in execution_rows
        )
        report["checked_fused_calls"] = sum(
            row["eager_fused_calls"] + row["graph_fused_replays"]
            for row in execution_rows
        )
        report["checked_fallback_calls"] = len(report["fallbacks"]) + sum(
            row["eager_fallback_calls"] + row["graph_fallback_replays"]
            for row in execution_rows
        )
        checked_replays = sum(row["graph_replays"] for row in execution_rows)
        report["graph_status"] = "PASS" if checked_replays else "NOT_RUN"
        report["fallback_status"] = "NOT_RUN" if args.skip_fallback else "PASS"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "status",
                            "world_size",
                            "passed_fused_cases",
                            "passed_fallbacks",
                            "checked_fused_calls",
                            "graph_status",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    except Exception as exc:
        report.update(
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
            passed_cases=len(report["cases"]),
            passed_fallbacks=len(report["fallbacks"]),
            passed_fp32_weight_fallbacks=len(report["fp32_weight_fallbacks"]),
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        dist.destroy_process_group()


def test_musa_car_v2_fused_runtime() -> None:
    """Pytest entry requires explicit distributed execution authorization."""
    import pytest

    if os.environ.get("RUN_MUSA_CAR_V2_FUSED_RUNTIME") != "1":
        pytest.skip("run this file with torchrun; GPU runtime verification not run")
    assert main([]) == 0


def test_execution_mode_uses_actual_v2_dispatch() -> None:
    from types import SimpleNamespace

    import pytest

    class LocalGroup:
        @staticmethod
        def get_world_size(group: Any) -> int:
            return 2

        @staticmethod
        def all_gather_object(results: list[Any], value: Any, group: Any) -> None:
            results[:] = [value, value]

    comm = SimpleNamespace(supports_fused_rmsnorm=True)
    with pytest.raises(AssertionError, match="missing actual fused execution mode"):
        _execution_mode(LocalGroup, None, comm, "v2")
    comm._last_fused_execution = "fallback"
    assert _execution_mode(LocalGroup, None, comm, "v2", "fallback") == "fallback"
    comm._last_fused_execution = "fused"
    assert _execution_mode(LocalGroup, None, comm, "v2") == "fused"
    with pytest.raises(AssertionError, match="expected fallback execution"):
        _execution_mode(LocalGroup, None, comm, "v2", "fallback")
    assert _execution_mode(LocalGroup, None, SimpleNamespace(), "v1") == "fused"


def test_fp32_weight_fallback_plan_keeps_target_matrix() -> None:
    args = _parser().parse_args([])
    assert len(_cases(args)) == 96
    extras = _fp32_weight_cases(args)
    assert len(extras) == 3
    assert {case.variant for case in extras} == {"norm", "residual", "no_raw"}
    assert all(case.impl == "v2" and case.weight_dtype == "float32" for case in extras)
    args.skip_fallback = True
    assert not _fp32_weight_cases(args)


if __name__ == "__main__":
    raise SystemExit(main())
