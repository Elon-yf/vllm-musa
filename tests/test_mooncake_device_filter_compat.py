# SPDX-License-Identifier: Apache-2.0

import logging
import os

from vllm_musa.distributed.mooncake_compat import configure_legacy_device_filter


def test_missing_legacy_value_leaves_environment_unchanged(monkeypatch):
    monkeypatch.delenv("MOONCAKE_RDMA_DEVICES", raising=False)
    monkeypatch.delenv("MC_TE_FILTERS", raising=False)

    configure_legacy_device_filter()

    assert "MOONCAKE_RDMA_DEVICES" not in os.environ
    assert "MC_TE_FILTERS" not in os.environ


def test_legacy_device_list_maps_to_official_filter(monkeypatch, caplog):
    monkeypatch.setenv("MOONCAKE_RDMA_DEVICES", "mlx5_2,mlx5_3")
    monkeypatch.delenv("MC_TE_FILTERS", raising=False)

    with caplog.at_level(logging.WARNING):
        configure_legacy_device_filter()

    assert "MC_TE_FILTERS" in caplog.text
    assert "deprecated" in caplog.text
    assert os.environ["MC_TE_FILTERS"] == "mlx5_2,mlx5_3"


def test_official_filter_wins_over_legacy_value(monkeypatch, caplog):
    monkeypatch.setenv("MOONCAKE_RDMA_DEVICES", "mlx5_2")
    monkeypatch.setenv("MC_TE_FILTERS", "mlx5_3")

    with caplog.at_level(logging.WARNING):
        configure_legacy_device_filter()

    assert os.environ["MC_TE_FILTERS"] == "mlx5_3"
    assert "ignored because MC_TE_FILTERS is already set" in caplog.text


def test_empty_legacy_value_keeps_auto_discovery(monkeypatch, caplog):
    monkeypatch.setenv("MOONCAKE_RDMA_DEVICES", "   ")
    monkeypatch.delenv("MC_TE_FILTERS", raising=False)

    with caplog.at_level(logging.WARNING):
        configure_legacy_device_filter()

    assert "MC_TE_FILTERS" not in os.environ
    assert "auto-discovery" in caplog.text


def test_official_empty_filter_wins_over_legacy_value(monkeypatch, caplog):
    monkeypatch.setenv("MOONCAKE_RDMA_DEVICES", "mlx5_2")
    monkeypatch.setenv("MC_TE_FILTERS", "")

    with caplog.at_level(logging.WARNING):
        configure_legacy_device_filter()

    assert os.environ["MC_TE_FILTERS"] == ""
    assert "ignored because MC_TE_FILTERS is already set" in caplog.text


def test_official_filter_wins_when_legacy_value_is_blank(monkeypatch, caplog):
    monkeypatch.setenv("MOONCAKE_RDMA_DEVICES", "   ")
    monkeypatch.setenv("MC_TE_FILTERS", "mlx5_3")

    with caplog.at_level(logging.WARNING):
        configure_legacy_device_filter()

    assert os.environ["MC_TE_FILTERS"] == "mlx5_3"
    assert "ignored because MC_TE_FILTERS is already set" in caplog.text
    assert "auto-discovery" not in caplog.text


def test_compatibility_helper_is_idempotent(monkeypatch, caplog):
    monkeypatch.setenv("MOONCAKE_RDMA_DEVICES", "mlx5_2")
    monkeypatch.delenv("MC_TE_FILTERS", raising=False)

    configure_legacy_device_filter()
    caplog.clear()
    configure_legacy_device_filter()

    assert os.environ["MC_TE_FILTERS"] == "mlx5_2"
    assert "same value as MC_TE_FILTERS" in caplog.text
    assert "ignored because" not in caplog.text
