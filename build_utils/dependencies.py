# SPDX-License-Identifier: Apache-2.0
"""Build-time dependency checks used before setuptools imports extensions."""

import importlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from packaging.specifiers import SpecifierSet

TORCHADA_MIN_VERSION = "0.1.71"
TORCHADA_REQUIREMENT = f"torchada>={TORCHADA_MIN_VERSION}"


def _installed_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _satisfies(installed: Optional[str], requirement: str) -> bool:
    if installed is None:
        return False
    _, specifier = requirement.split("torchada", 1)
    return SpecifierSet(specifier).contains(installed, prereleases=True)


def _install(requirement: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", requirement, "--upgrade", "-q"]
    )


def ensure_torchada_installed():
    """Import a torchada new enough to define this checkout's build contract.

    This runs before importing ``torch.utils.cpp_extension``.  Merely declaring
    an ``install_requires`` floor is too late for direct ``build_ext`` and for
    metadata/build hooks that execute setup.py before resolving project deps.
    """

    installed = _installed_version("torchada")
    if not _satisfies(installed, TORCHADA_REQUIREMENT):
        if "torchada" in sys.modules:
            raise RuntimeError(
                f"torchada {installed or 'unknown'} is already loaded, but "
                f"{TORCHADA_REQUIREMENT} is required; restart the build process"
            )
        print(f"Installing {TORCHADA_REQUIREMENT}...")
        _install(TORCHADA_REQUIREMENT)
        installed = _installed_version("torchada")
        if not _satisfies(installed, TORCHADA_REQUIREMENT):
            raise RuntimeError(
                f"installed torchada {installed or 'unknown'}, but "
                f"{TORCHADA_REQUIREMENT} is required"
            )

    importlib.invalidate_caches()
    return importlib.import_module("torchada")
