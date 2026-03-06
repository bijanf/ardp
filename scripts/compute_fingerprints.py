#!/usr/bin/env python3
"""Load reanalysis data and compute all AMOC fingerprints, saving as NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import SAMBA_LAT
from ardp.fingerprints.f_ovs import compute_f_ovs_timeseries, validate_trend
from ardp.fingerprints.nawh import compute_nawh_index
from ardp.fingerprints.salinity_pileup import compute_salinity_pileup
from ardp.spatial.gulf_stream import track_destabilization_timeseries


def _load_glorys12(data_dir: Path) -> xr.Dataset:
    """Load GLORYS12 data and prepare grid info for fingerprint computation."""
    ds = xr.open_mfdataset(str(data_dir / "*.nc"), chunks={"time": 1})

    # Rename to canonical variable names
    rename = {}
    for src, dst in [("thetao", "temperature"), ("so", "salinity"),
                     ("uo", "u_velocity"), ("vo", "v_velocity"), ("zos", "ssh")]:
        if src in ds:
            rename[src] = dst
    if rename:
        ds = ds.rename(rename)

    # Rename dims to y/x/z for compatibility with fingerprint code
    dim_rename = {}
    if "latitude" in ds.dims:
        dim_rename["latitude"] = "y"
    if "longitude" in ds.dims:
        dim_rename["longitude"] = "x"
    if "depth" in ds.dims:
        dim_rename["depth"] = "z"
    if dim_rename:
        ds = ds.rename(dim_rename)

    # Build 2D lon/lat arrays from 1D coordinates
    lon_1d = ds["x"]
    lat_1d = ds["y"]
    lon2d, lat2d = xr.broadcast(lon_1d, lat_1d)
    # Ensure dims are (y, x)
    lon2d = lon2d.transpose("y", "x")
    lat2d = lat2d.transpose("y", "x")

    ds["nav_lon"] = lon2d
    ds["nav_lat"] = lat2d

    # Approximate grid spacings [m] for the regular grid
    dlon = float(np.abs(lon_1d.values[1] - lon_1d.values[0]))
    dlat = float(np.abs(lat_1d.values[1] - lat_1d.values[0]))

    # e1t ~ dx = dlon * 111km * cos(lat)
    e1t = dlon * 111000.0 * np.cos(np.deg2rad(lat2d))
    e1t = e1t.transpose("y", "x")
    ds["e1t"] = e1t

    # e2t ~ dy = dlat * 111km (constant)
    e2t = xr.DataArray(
        np.full_like(lat2d.values, dlat * 111000.0),
        dims=("y", "x"),
    )
    ds["e2t"] = e2t

    # Approximate vertical cell thickness from depth bounds
    depth = ds["z"].values
    e3t_vals = np.diff(depth, prepend=0)
    ds["e3t"] = xr.DataArray(e3t_vals, dims=("z",))

    return ds


def _load_oras5(data_dir: Path) -> xr.Dataset:
    """Load ORAS5 data."""
    from ardp.ingestion.oras5 import ORAS5Loader
    loader = ORAS5Loader(data_dir)
    return loader.load_dataset()


def _load_cglors(data_dir: Path) -> xr.Dataset:
    """Load C-GLORS data."""
    from ardp.ingestion.cglors import CGLORSLoader
    loader = CGLORSLoader(data_dir)
    return loader.load_dataset()


LOADERS = {
    "glorys12": _load_glorys12,
    "oras5": _load_oras5,
    "cglors": _load_cglors,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AMOC fingerprints from downloaded reanalysis data."
    )
    parser.add_argument(
        "--product",
        choices=["glorys12", "oras5", "cglors"],
        required=True,
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--results-dir", default="data/results",
        help="Output directory for computed results (default: data/results).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    product_dir = Path(args.data_dir) / args.product
    print(f"Loading {args.product} data from {product_dir}...")
    ds = LOADERS[args.product](product_dir)
    print(f"  Dataset: {list(ds.dims.items())}")

    # --- F_ovS ---
    print("Computing F_ovS...")
    try:
        f_ovs = compute_f_ovs_timeseries(
            ds, lat=SAMBA_LAT,
            v_var="v_velocity", s_var="salinity",
            e1_var="e1t", e3_var="e3t",
            mask_var=None,
            lat_var="nav_lat",
            y_dim="y", x_dim="x", z_dim="z",
        )
        f_ovs.to_netcdf(results_dir / "f_ovs.nc")
        print(f"  Saved: {results_dir / 'f_ovs.nc'}")

        trend_info = validate_trend(f_ovs)
        print(f"  Trend: {trend_info['trend_msv_per_year']:.2f} mSv/yr "
              f"(expected: {trend_info['expected']:.2f} mSv/yr, "
              f"valid: {trend_info['is_valid']})")
    except Exception as e:
        print(f"  F_ovS failed: {e}")

    # Grid arrays for spatial fingerprints
    lon2d = ds["nav_lon"]
    lat2d = ds["nav_lat"]
    e1 = ds["e1t"]
    e2 = ds["e2t"]

    # --- Salinity pile-up ---
    print("Computing salinity pile-up...")
    try:
        sss = ds["salinity"].isel(z=0) if "z" in ds["salinity"].dims else ds["salinity"]
        pileup = compute_salinity_pileup(sss, lon2d, lat2d, e1, e2)
        pileup.to_netcdf(results_dir / "salinity_pileup.nc")
        print(f"  Saved: {results_dir / 'salinity_pileup.nc'}")
    except Exception as e:
        print(f"  Salinity pile-up failed: {e}")

    # --- NAWH ---
    print("Computing NAWH index...")
    try:
        sst = ds["temperature"].isel(z=0) if "z" in ds["temperature"].dims else ds["temperature"]
        nawh = compute_nawh_index(sst, lon2d, lat2d, e1, e2)
        nawh.to_netcdf(results_dir / "nawh.nc")
        print(f"  Saved: {results_dir / 'nawh.nc'}")
    except Exception as e:
        print(f"  NAWH failed: {e}")

    # --- Gulf Stream destabilization ---
    print("Computing Gulf Stream destabilization...")
    try:
        ssh = ds["ssh"]
        destab = track_destabilization_timeseries(
            ssh, lon2d, lat2d, e1, e2,
            x_dim="x", y_dim="y",
        )
        destab.to_netcdf(results_dir / "gulf_stream_destab.nc")
        print(f"  Saved: {results_dir / 'gulf_stream_destab.nc'}")
    except Exception as e:
        print(f"  Gulf Stream destabilization failed: {e}")

    print("Fingerprint computation complete.")


if __name__ == "__main__":
    main()
