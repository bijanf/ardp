"""Tests for North Atlantic Warming Hole index."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.fingerprints.nawh import compute_nawh_index


def test_nawh_index_shape(nemo_grid_ds: xr.Dataset) -> None:
    sst = nemo_grid_ds["temperature"].isel(z=0)
    result = compute_nawh_index(
        sst,
        nemo_grid_ds["nav_lon"],
        nemo_grid_ds["nav_lat"],
        nemo_grid_ds["e1t"],
        nemo_grid_ds["e2t"],
    )
    assert "time" in result.dims
    assert result.sizes["time"] == 12


def test_nawh_index_units(nemo_grid_ds: xr.Dataset) -> None:
    sst = nemo_grid_ds["temperature"].isel(z=0)
    result = compute_nawh_index(
        sst,
        nemo_grid_ds["nav_lon"],
        nemo_grid_ds["nav_lat"],
        nemo_grid_ds["e1t"],
        nemo_grid_ds["e2t"],
    )
    assert result.attrs["units"] == "degC"
