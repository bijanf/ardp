"""Tests for region masking."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.spatial.regions import atlantic_basin_mask, create_region_mask, extract_section_at_latitude


def test_create_region_mask_basic() -> None:
    lon = xr.DataArray(np.linspace(-80, 0, 20), dims="x")
    lat = xr.DataArray(np.linspace(-60, 70, 20), dims="y")
    lon2d, lat2d = xr.broadcast(lon, lat)

    mask = create_region_mask(lon2d, lat2d, -60, 20, -35, -15)
    assert mask.dtype == bool
    assert mask.any()
    # Points outside should be False
    assert not mask.sel(y=0, x=0)  # lat=-60, lon=-80


def test_atlantic_basin_mask(nemo_grid_ds: xr.Dataset) -> None:
    mask = atlantic_basin_mask(nemo_grid_ds["nav_lon"], nemo_grid_ds["nav_lat"])
    assert mask.dtype == bool
    assert mask.any()


def test_extract_section_nearest_j(nemo_grid_ds: xr.Dataset) -> None:
    section = extract_section_at_latitude(nemo_grid_ds, 0.0)
    # Should have removed the y dimension
    assert "y" not in section.dims
    # Check that the latitude is close to target
    mean_lat = float(section["nav_lat"].mean())
    assert abs(mean_lat) < 5.0
