"""Tests for F_ovS fingerprint."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.fingerprints.f_ovs import compute_f_ovs, validate_trend
from ardp.spatial.regions import extract_section_at_latitude


def test_extract_section_at_latitude(nemo_grid_ds: xr.Dataset) -> None:
    section = extract_section_at_latitude(nemo_grid_ds, -34.5)
    assert "y" not in section.dims


def test_compute_f_ovs_returns_scalar_per_time(nemo_grid_ds: xr.Dataset) -> None:
    f_ovs = compute_f_ovs(nemo_grid_ds)
    assert "time" in f_ovs.dims
    assert f_ovs.sizes["time"] == 12


def test_compute_f_ovs_finite(nemo_grid_ds: xr.Dataset) -> None:
    f_ovs = compute_f_ovs(nemo_grid_ds)
    assert np.all(np.isfinite(f_ovs.values))


def test_compute_f_ovs_two_layer(two_layer_ocean: xr.Dataset) -> None:
    """F_ovS on the two-layer ocean should be negative (freshwater export)."""
    f_ovs = compute_f_ovs(
        two_layer_ocean,
        lat=-34.5,
        v_var="v_velocity",
        s_var="salinity",
        mask_var="vmask",
    )
    assert float(f_ovs) < 0


def test_compute_f_ovs_dask(nemo_grid_ds_dask: xr.Dataset) -> None:
    f_ovs = compute_f_ovs(nemo_grid_ds_dask)
    result = f_ovs.compute()
    assert np.all(np.isfinite(result.values))


def test_validate_trend() -> None:
    """Validate trend detection with a synthetic linear time series."""
    time = xr.cftime_range("1980-01", periods=480, freq="MS")
    years = np.arange(480) / 12.0
    # -1.2 mSv/yr = -1.2e-3 Sv/yr
    values = 0.5 + (-1.2e-3) * years + np.random.default_rng(42).normal(0, 1e-4, 480)
    ts = xr.DataArray(values, dims="time", coords={"time": time})
    ts.name = "F_ovS"

    result = validate_trend(ts)
    assert result["is_valid"]
    assert abs(result["trend_msv_per_year"] - (-1.2)) < 0.1
