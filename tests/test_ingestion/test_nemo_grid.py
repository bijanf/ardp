"""Tests for NEMO grid construction."""

from __future__ import annotations

import xarray as xr

from ardp.ingestion.nemo_grid import build_nemo_grid


def test_build_nemo_grid_returns_grid(nemo_grid_ds: xr.Dataset) -> None:
    grid = build_nemo_grid(nemo_grid_ds)
    assert grid is not None
    # Check axes are present
    assert "X" in grid.axes
    assert "Y" in grid.axes
    assert "Z" in grid.axes


def test_build_nemo_grid_periodic_x(nemo_grid_ds: xr.Dataset) -> None:
    grid = build_nemo_grid(nemo_grid_ds, periodic=["X"])
    assert "X" in grid._periodic


def test_build_nemo_grid_no_periodic(nemo_grid_ds: xr.Dataset) -> None:
    grid = build_nemo_grid(nemo_grid_ds, periodic=[])
    assert len(grid._periodic) == 0


def test_build_nemo_grid_dimensions(nemo_grid_ds: xr.Dataset) -> None:
    grid = build_nemo_grid(nemo_grid_ds)
    # Verify the dataset dimensions match expected sizes
    assert nemo_grid_ds.sizes["x"] == 60
    assert nemo_grid_ds.sizes["y"] == 40
    assert nemo_grid_ds.sizes["z"] == 10
