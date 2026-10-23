"""Decision counters for the MUSA custom all-reduce (rules A and B).

A paired v1/v2 measurement is only meaningful when both arms take the *same*
all-reduce calls (rule A: custom all-reduce or fall back) and differ only in the
algorithm chosen inside the custom path (rule B).  Both implementations record
their decisions here so a run can prove that instead of assuming it.

Enabled with ``VLLM_MUSA_CAR_TRACE=1``; the counters are dumped to stderr at
interpreter exit.  The counters are deliberately cheap: one dictionary lookup
per all-reduce call, and nothing at all when the switch is off.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading

TRACE_ENV = "VLLM_MUSA_CAR_TRACE"
_ENABLED = os.environ.get(TRACE_ENV, "").strip().lower() in ("1", "true", "yes", "on")
_LOCK = threading.Lock()
_COUNTS: dict[tuple[str, str, str], int] = {}


def enabled() -> bool:
    return _ENABLED


def _size_bucket(nbytes: int) -> str:
    """Power-of-two bucket, so payload classes stay readable in the dump."""
    if nbytes <= 0:
        return "0B"
    return f"{1 << (int(nbytes).bit_length() - 1)}B"


def record(kind: str, arm: str, decision: str, nbytes: int) -> None:
    if not _ENABLED:
        return
    key = (kind, arm, f"{decision}@{_size_bucket(nbytes)}")
    with _LOCK:
        _COUNTS[key] = _COUNTS.get(key, 0) + 1


def counts() -> dict[tuple[str, str, str], int]:
    with _LOCK:
        return dict(_COUNTS)


def reset() -> None:
    with _LOCK:
        _COUNTS.clear()


def _dump() -> None:
    counters = counts()
    if not counters:
        return
    print("[car-trace] decision counts", file=sys.stderr)
    for (kind, arm, decision), count in sorted(counters.items()):
        print(f"[car-trace] {kind:5s} {arm:3s} {decision:30s} {count}", file=sys.stderr)


atexit.register(_dump)
