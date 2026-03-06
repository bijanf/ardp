"""Tests for MOC streamfunction."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.physics.streamfunction import moc_depth_space


def test_moc_depth_space_shape(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)
    psi = moc_depth_space(
        ds["v_velocity"],
        ds["e1t"],
        ds["e3t"],
    )
    assert "z" in psi.dims


def test_moc_depth_space_units(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)
    psi = moc_depth_space(
        ds["v_velocity"],
        ds["e1t"],
        ds["e3t"],
    )
    assert psi.attrs["units"] == "Sv"


def test_moc_depth_space_finite(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)
    psi = moc_depth_space(
        ds["v_velocity"],
        ds["e1t"],
        ds["e3t"],
    )
    assert np.all(np.isfinite(psi.values))
