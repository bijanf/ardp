"""Tests for meridional heat transport."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.physics.mht import meridional_heat_transport


def test_mht_shape(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)
    mht = meridional_heat_transport(
        ds["v_velocity"],
        ds["temperature"],
        ds["e1t"],
        ds["e3t"],
    )
    assert mht.size == 1


def test_mht_units(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)
    mht = meridional_heat_transport(
        ds["v_velocity"],
        ds["temperature"],
        ds["e1t"],
        ds["e3t"],
    )
    assert mht.attrs["units"] == "PW"


def test_mht_finite(nemo_grid_ds: xr.Dataset) -> None:
    ds = nemo_grid_ds.isel(time=0, y=20)
    mht = meridional_heat_transport(
        ds["v_velocity"],
        ds["temperature"],
        ds["e1t"],
        ds["e3t"],
    )
    assert np.isfinite(float(mht))
