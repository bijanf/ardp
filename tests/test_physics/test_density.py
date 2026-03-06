"""Tests for density computations."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.physics.density import depth_to_pressure, potential_density


def test_potential_density_shape(nemo_grid_ds: xr.Dataset) -> None:
    sigma = potential_density(
        nemo_grid_ds["salinity"].isel(time=0),
        nemo_grid_ds["temperature"].isel(time=0),
    )
    assert sigma.dims == ("z", "y", "x")
    assert sigma.shape == (10, 40, 60)


def test_potential_density_range(nemo_grid_ds: xr.Dataset) -> None:
    sigma = potential_density(
        nemo_grid_ds["salinity"].isel(time=0),
        nemo_grid_ds["temperature"].isel(time=0),
    )
    # Sigma-0 for seawater should be roughly 20-28 kg/m^3
    assert float(sigma.min()) > 15.0
    assert float(sigma.max()) < 32.0


def test_potential_density_dask(nemo_grid_ds_dask: xr.Dataset) -> None:
    sigma = potential_density(
        nemo_grid_ds_dask["salinity"].isel(time=0),
        nemo_grid_ds_dask["temperature"].isel(time=0),
    )
    # Should be a dask array
    assert sigma.chunks is not None
    # Compute and check it's valid
    result = sigma.compute()
    assert not np.isnan(result).all()


def test_depth_to_pressure() -> None:
    depth = xr.DataArray([0.0, 100.0, 1000.0], dims="z")
    lat = xr.DataArray([45.0, 45.0, 45.0], dims="z")
    p = depth_to_pressure(depth, lat)
    # Pressure at surface should be ~0, at 1000m should be ~1000 dbar
    assert abs(float(p.isel(z=0))) < 1.0
    assert abs(float(p.isel(z=2)) - 1008.0) < 20.0
