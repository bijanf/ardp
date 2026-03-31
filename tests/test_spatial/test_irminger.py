"""Tests for Irminger Sea diagnostics."""

from __future__ import annotations

import xarray as xr

from ardp.spatial.irminger import irminger_mask, mixed_layer_depth


def test_irminger_mask_within_bounds() -> None:
    lon = xr.DataArray([-36.0], dims="x")
    lat = xr.DataArray([62.0], dims="y")
    lon2d, lat2d = xr.broadcast(lon, lat)

    mask = irminger_mask(lon2d, lat2d)
    assert bool(mask.values)


def test_irminger_mask_outside_bounds() -> None:
    lon = xr.DataArray([0.0], dims="x")
    lat = xr.DataArray([62.0], dims="y")
    lon2d, lat2d = xr.broadcast(lon, lat)

    mask = irminger_mask(lon2d, lat2d)
    assert not bool(mask.values)


def test_mixed_layer_depth_uniform_profile() -> None:
    """Uniform density => MLD = max depth."""
    depth = xr.DataArray([10.0, 50.0, 100.0, 500.0], dims="z")
    sigma = xr.DataArray([27.0, 27.0, 27.0, 27.0], dims="z")

    mld = mixed_layer_depth(sigma, depth)
    assert float(mld) == 500.0


def test_mixed_layer_depth_stratified() -> None:
    """Stratified profile => MLD at first exceedance."""
    depth = xr.DataArray([10.0, 50.0, 100.0, 500.0], dims="z")
    sigma = xr.DataArray([27.0, 27.0, 27.05, 27.5], dims="z")

    mld = mixed_layer_depth(sigma, depth, density_threshold=0.03)
    assert float(mld) == 100.0
