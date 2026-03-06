"""Synthetic NEMO dataset fixtures for testing."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def nemo_grid_ds() -> xr.Dataset:
    """Create a synthetic NEMO C-grid dataset.

    Grid: 60 lon x 40 lat (-80E to -20E, -60N to 70N), 10 depth levels, 12 months.
    Includes physically plausible T, S, velocity, and metric fields.
    """
    nx, ny, nz, nt = 60, 40, 10, 12

    lon_vals = np.linspace(-80, -20, nx)
    lat_vals = np.linspace(-60, 70, ny)
    depth_vals = np.array([5, 15, 30, 50, 100, 200, 500, 1000, 2000, 4000], dtype=float)
    time_vals = xr.cftime_range("2000-01", periods=nt, freq="MS")

    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

    # Metrics (approximate, in meters)
    dx = 111_000.0 * np.cos(np.deg2rad(lat2d)) * (lon_vals[1] - lon_vals[0])
    dy = np.full_like(lat2d, 111_000.0 * (lat_vals[1] - lat_vals[0]))
    dz = np.diff(
        np.concatenate([[0], (depth_vals[:-1] + depth_vals[1:]) / 2, [5500]]),
    )

    # Temperature: warm at surface/tropics, cold at depth/poles
    lat_norm = (lat2d[np.newaxis, np.newaxis, :, :] - lat_vals.mean()) / 40.0
    depth_norm = depth_vals[np.newaxis, :, np.newaxis, np.newaxis] / 4000.0
    temperature = 25.0 * (1 - depth_norm) * np.exp(-lat_norm**2) + 2.0
    temperature = np.broadcast_to(temperature, (nt, nz, ny, nx)).copy()

    # Salinity: higher in subtropics, lower at depth
    salinity = 35.0 + 1.5 * np.exp(-lat_norm**2) * (1 - 0.5 * depth_norm)
    salinity = np.broadcast_to(salinity, (nt, nz, ny, nx)).copy()

    # Meridional velocity: northward at surface, southward at depth (simplified MOC)
    v_velocity = 0.05 * (1 - 2 * depth_norm) * np.exp(-lat_norm**2)
    v_velocity = np.broadcast_to(v_velocity, (nt, nz, ny, nx)).copy()

    # Zonal velocity: weak westward flow
    u_velocity = -0.02 * np.ones((nt, nz, ny, nx))

    # SSH: higher in subtropics
    ssh = 0.5 * np.exp(-lat_norm[0, 0, :, :] ** 2)
    ssh = np.broadcast_to(ssh[np.newaxis, :, :], (nt, ny, nx)).copy()

    # Masks (1 = ocean, 0 = land) — all ocean for simplicity
    tmask = np.ones((nz, ny, nx), dtype=np.int8)
    umask = np.ones((nz, ny, nx), dtype=np.int8)
    vmask = np.ones((nz, ny, nx), dtype=np.int8)

    ds = xr.Dataset(
        {
            "temperature": (["time", "z", "y", "x"], temperature),
            "salinity": (["time", "z", "y", "x"], salinity),
            "u_velocity": (["time", "z", "y", "x"], u_velocity),
            "v_velocity": (["time", "z", "y", "x"], v_velocity),
            "ssh": (["time", "y", "x"], ssh),
            "e1t": (["y", "x"], dx),
            "e2t": (["y", "x"], dy),
            "e3t": (["z"], dz),
            "tmask": (["z", "y", "x"], tmask),
            "umask": (["z", "y", "x"], umask),
            "vmask": (["z", "y", "x"], vmask),
            "nav_lat": (["y", "x"], lat2d),
            "nav_lon": (["y", "x"], lon2d),
        },
        coords={
            "x": np.arange(nx),
            "y": np.arange(ny),
            "z": np.arange(nz),
            "depth": ("z", depth_vals),
            "time": time_vals,
        },
    )
    return ds


@pytest.fixture
def nemo_grid_ds_dask(nemo_grid_ds: xr.Dataset) -> xr.Dataset:
    """Dask-backed version of the synthetic NEMO dataset."""
    return nemo_grid_ds.chunk({"time": 1})


@pytest.fixture
def two_layer_ocean() -> xr.Dataset:
    """Minimal two-layer ocean for analytical F_ovS verification.

    Upper layer (0-1000m): v=+0.1 m/s, S=35.5
    Lower layer (1000-4000m): v=-0.033 m/s (mass balance), S=34.8
    At 34.5S, basin width = 60 degrees * cos(34.5) * 111km ~ 5.5e6 m

    F_ov = -(1/S0) * integral of <v>(z) * (<S>(z) - S0) dz
    Upper: -(1/35) * 0.1 * (35.5-35) * 1000 = -(1/35)*50 = -1.4286
    Lower: -(1/35) * (-0.033) * (34.8-35) * 3000 = -(1/35)*(-0.033*(-0.2)*3000) = -(1/35)*19.8 = -0.5657
    F_ov = -1.4286 - 0.5657 ≈ -1.994 Sv (not multiplied by width; per-meter)
    With width: F_ov integrated in x is handled by zonal mean.
    """
    nx, nz = 30, 2
    lat_section = -34.5
    lon_vals = np.linspace(-60, 20, nx)
    depth_vals = np.array([500.0, 2500.0])
    dz_vals = np.array([1000.0, 3000.0])
    cos_lat = np.cos(np.deg2rad(lat_section))
    dx = 111_000.0 * cos_lat * (lon_vals[1] - lon_vals[0])

    v = np.zeros((1, nz, 1, nx))
    s = np.zeros((1, nz, 1, nx))

    # Upper layer
    v[0, 0, 0, :] = 0.1
    s[0, 0, 0, :] = 35.5

    # Lower layer: mass-balanced
    v[0, 1, 0, :] = -0.1 * dz_vals[0] / dz_vals[1]
    s[0, 1, 0, :] = 34.8

    lat2d = np.full((1, nx), lat_section)
    lon2d = lon_vals[np.newaxis, :]
    dx2d = np.full((1, nx), dx)
    dy2d = np.full((1, nx), 111_000.0)

    ds = xr.Dataset(
        {
            "v_velocity": (["time", "z", "y", "x"], v),
            "salinity": (["time", "z", "y", "x"], s),
            "e1t": (["y", "x"], dx2d),
            "e2t": (["y", "x"], dy2d),
            "e3t": (["z"], dz_vals),
            "nav_lat": (["y", "x"], lat2d),
            "nav_lon": (["y", "x"], lon2d),
            "vmask": (["z", "y", "x"], np.ones((nz, 1, nx), dtype=np.int8)),
        },
        coords={
            "x": np.arange(nx),
            "y": np.arange(1),
            "z": np.arange(nz),
            "depth": ("z", depth_vals),
            "time": xr.cftime_range("2000-01", periods=1, freq="MS"),
        },
    )
    return ds
