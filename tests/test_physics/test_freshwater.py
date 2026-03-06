"""Tests for freshwater transport computations."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.physics.freshwater import (
    freshwater_transport_gyre,
    freshwater_transport_overturning,
)


def test_f_ov_shape(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)  # Single latitude section
    f_ov = freshwater_transport_overturning(
        ds["v_velocity"],
        ds["salinity"],
        ds["e1t"],
        ds["e3t"],
    )
    # Should be a scalar (all spatial dims summed out)
    assert f_ov.size == 1


def test_f_ov_two_layer(two_layer_ocean: xr.Dataset) -> None:
    """Verify F_ov on an analytical two-layer ocean."""
    ds = two_layer_ocean.isel(time=0)
    f_ov = freshwater_transport_overturning(
        ds["v_velocity"],
        ds["salinity"],
        ds["e1t"],
        ds["e3t"],
    )
    # The two-layer has uniform fields in x, so F_ov should be finite and negative
    # (northward upper flow with S > S0 exports freshwater)
    val = float(f_ov)
    assert np.isfinite(val)
    assert val < 0  # Freshwater divergence (export from south)


def test_f_az_zero_for_uniform_x(two_layer_ocean: xr.Dataset) -> None:
    """F_az should be zero when v and S are zonally uniform."""
    ds = two_layer_ocean.isel(time=0)
    f_az = freshwater_transport_gyre(
        ds["v_velocity"],
        ds["salinity"],
        ds["e1t"],
        ds["e3t"],
    )
    assert abs(float(f_az)) < 1e-10


def test_f_ov_dask(nemo_grid_ds_dask: xr.Dataset) -> None:
    ds = nemo_grid_ds_dask.isel(time=0, y=20)
    f_ov = freshwater_transport_overturning(
        ds["v_velocity"],
        ds["salinity"],
        ds["e1t"],
        ds["e3t"],
    )
    result = f_ov.compute()
    assert np.isfinite(float(result))
