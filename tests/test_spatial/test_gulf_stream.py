"""Tests for Gulf Stream destabilization tracking."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.spatial.gulf_stream import compute_ssh_gradient, find_destabilization_point


def test_ssh_gradient_shape(nemo_grid_ds: xr.Dataset) -> None:
    ssh = nemo_grid_ds["ssh"].isel(time=0)
    grad = compute_ssh_gradient(ssh, nemo_grid_ds["e1t"], nemo_grid_ds["e2t"])
    # Should be smaller than input by 1 in each spatial dim
    assert grad.sizes["x"] == 59
    assert grad.sizes["y"] == 39


def test_ssh_gradient_positive(nemo_grid_ds: xr.Dataset) -> None:
    ssh = nemo_grid_ds["ssh"].isel(time=0)
    grad = compute_ssh_gradient(ssh, nemo_grid_ds["e1t"], nemo_grid_ds["e2t"])
    assert (grad >= 0).all()


def test_destabilization_point_synthetic() -> None:
    """Synthetic jet with known destabilization point."""
    nx = 50
    x = np.arange(nx)
    lon = xr.DataArray(np.linspace(-80, -45, nx), dims="x")

    # Gradient: strong until x=30, then drops off
    grad_vals = np.ones(nx) * 0.001
    grad_vals[:30] = 0.01
    jet_grad = xr.DataArray(grad_vals, dims="x")

    destab = find_destabilization_point(jet_grad, lon, threshold_fraction=0.5)
    # Should be near the longitude at x=29 (last point above threshold)
    expected_lon = float(lon.isel(x=29))
    assert abs(float(destab) - expected_lon) < 2.0
