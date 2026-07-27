# SPDX-License-Identifier: Apache-2.0

"""Small, import-order-safe Mooncake compatibility helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_LEGACY_DEVICE_ENV = "MOONCAKE_RDMA_DEVICES"
_FILTER_ENV = "MC_TE_FILTERS"


def configure_legacy_device_filter() -> None:
    """Map the old vLLM-MUSA HCA variable to Mooncake's official filter.

    The helper intentionally imports only the Python standard library.  It is
    called at the start of the MUSA general-plugin entry point, before vLLM can
    construct a Mooncake worker, and never replaces or monkeypatches upstream
    connector code.
    """

    legacy_value = os.environ.get(_LEGACY_DEVICE_ENV)
    if legacy_value is None:
        return

    # Presence, rather than truthiness, is intentional: MC_TE_FILTERS=""
    # explicitly requests the official all-devices/auto-discovery behavior.
    if _FILTER_ENV in os.environ:
        if os.environ[_FILTER_ENV] == legacy_value:
            logger.warning(
                "%s is deprecated and has the same value as %s, including "
                "after a prior compatibility mapping. Use %s directly.",
                _LEGACY_DEVICE_ENV,
                _FILTER_ENV,
                _FILTER_ENV,
            )
            return
        logger.warning(
            "%s is deprecated and ignored because %s is already set; use %s "
            "for Mooncake HCA filtering.",
            _LEGACY_DEVICE_ENV,
            _FILTER_ENV,
            _FILTER_ENV,
        )
        return

    if not legacy_value.strip():
        logger.warning(
            "%s is deprecated and empty; leaving Mooncake HCA auto-discovery "
            "enabled. Use %s for an explicit allow-list.",
            _LEGACY_DEVICE_ENV,
            _FILTER_ENV,
        )
        return

    os.environ[_FILTER_ENV] = legacy_value
    logger.warning(
        "%s is deprecated; mapped to %s for Mooncake Transfer Engine. "
        "Use %s directly for an explicit HCA allow-list.",
        _LEGACY_DEVICE_ENV,
        _FILTER_ENV,
        _FILTER_ENV,
    )


__all__ = ["configure_legacy_device_filter"]
