# SPDX-License-Identifier: Apache-2.0
"""Cross-repository contract checks for torchada's in-place source porter."""

import sys
from pathlib import Path

import pytest

from build_utils import dependencies

ROOT = Path(__file__).resolve().parent.parent


def test_torchada_floor_is_consistent():
    requirement = dependencies.TORCHADA_REQUIREMENT
    assert requirement == "torchada>=0.1.71"
    assert f'"{requirement}"' in (ROOT / "pyproject.toml").read_text()
    assert "TORCHADA_REQUIREMENT," in (ROOT / "setup.py").read_text()


def test_stale_torchada_is_upgraded_before_import(monkeypatch):
    observed = []
    versions = iter(["0.1.70", "0.1.71"])

    monkeypatch.setattr(
        dependencies, "_installed_version", lambda _name: next(versions)
    )
    monkeypatch.setattr(
        dependencies,
        "_install",
        lambda requirement: observed.append(("install", requirement)),
    )
    monkeypatch.setattr(
        dependencies.importlib,
        "import_module",
        lambda name: observed.append(("import", name)) or object(),
    )
    monkeypatch.delitem(sys.modules, "torchada", raising=False)

    dependencies.ensure_torchada_installed()

    assert observed == [
        ("install", "torchada>=0.1.71"),
        ("import", "torchada"),
    ]


def test_loaded_stale_torchada_fails_before_mixing_versions(monkeypatch):
    monkeypatch.setattr(dependencies, "_installed_version", lambda _name: "0.1.70")
    monkeypatch.setitem(sys.modules, "torchada", object())

    with pytest.raises(RuntimeError, match="restart the build process"):
        dependencies.ensure_torchada_installed()


def test_no_legacy_mirror_contract_remains():
    legacy_tokens = (
        "csrc_musa",
        "libtorch_stable_musa",
        "attention_musa",
        "quantization_musa",
        "per-file _musa",
    )
    paths = [ROOT / ".gitignore", ROOT / "third_party" / "PINS"]
    paths.extend(sorted((ROOT / "vllm_musa" / "patches" / "series").glob("*.patch")))

    offenders = []
    for path in paths:
        text = path.read_text(errors="replace")
        for token in legacy_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    assert not offenders, offenders


def test_native_sampler_includes_flashinfer_header_by_real_name():
    sampler = (ROOT / "csrc" / "musa" / "sampler.mu").read_text()
    assert "#include <flashinfer/sampling.cuh>" in sampler
    assert "#include <flashinfer/sampling.muh>" not in sampler
