#!/usr/bin/env python3
"""Download full-latitude vo from Pangeo, zonally integrate on-the-fly.

Computes v_zonal[time, depth, lat] = sum_x(vo * dx * atlantic_mask)
for each model, saving only the 2D zonal transport (~300 MB per model)
instead of the full 4D field (~27 GB).

Output: data/cmip6_fullfield/{model}_{experiment}_vo_zonal.nc
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import warnings
from pathlib import Path

print = functools.partial(print, flush=True)

import intake
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

PANGEO_CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
TARGET_MODELS = [
    # Bistable (F_ovS < 0)
    "NESM3", "IPSL-CM6A-LR", "CNRM-CM6-1", "MIROC6",
    "MPI-ESM1-2-HR", "CanESM5",
    # Monostable (F_ovS > 0)
    "UKESM1-0-LL", "CMCC-CM2-SR5", "GFDL-CM4", "ACCESS-CM2",
    "MPI-ESM1-2-LR", "HadGEM3-GC31-LL", "CESM2", "FIO-ESM-2-0",
    "GISS-E2-1-G", "FGOALS-g3",
]
EXPERIMENTS = ["historical", "ssp585"]
BATCH_SIZE = 12  # months per batch (~400 MB peak memory)


def atlantic_lon_bounds(lat: float) -> tuple[float, float]:
    """Return (lon_min, lon_max) for Atlantic at given latitude."""
    if lat < -34:
        return (-70.0, 20.0)
    elif lat < 0:
        return (-70.0, 20.0)
    elif lat < 10:
        return (-90.0, 10.0)
    elif lat < 30:
        return (-100.0, 0.0)
    elif lat < 45:
        return (-82.0, -5.0)
    elif lat < 65:
        return (-70.0, 0.0)
    else:
        return (-60.0, 10.0)


def find_zstore(cat_df, source_id: str, experiment_id: str, variable_id: str) -> str | None:
    """Find best zarr store URL for a given model/experiment/variable."""
    mask = (
        (cat_df["source_id"] == source_id)
        & (cat_df["experiment_id"] == experiment_id)
        & (cat_df["variable_id"] == variable_id)
        & (cat_df["table_id"] == "Omon")
    )
    df = cat_df[mask]
    if len(df) == 0:
        return None
    for grid in ["gn", "gr"]:
        subset = df[df["grid_label"] == grid]
        if len(subset) > 0:
            df = subset
            break
    for member in ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3"]:
        subset = df[df["member_id"] == member]
        if len(subset) > 0:
            df = subset
            break
    return df.iloc[0]["zstore"]


def get_lat_lon_depth(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, str, str, str, str]:
    """Extract 1D lat, 1D lon, depth arrays and dimension names.

    Returns (lat_1d, lon_1d, depth, j_dim, x_dim, z_dim, lat_name, lon_name).
    """
    # Find latitude
    lat_name = None
    for name in ["lat", "latitude", "nav_lat"]:
        if name in ds.coords:
            lat_name = name
            break
    if lat_name is None:
        for name in ds.coords:
            if "lat" in name.lower():
                lat_name = name
                break
    if lat_name is None:
        raise ValueError(f"Cannot find latitude in {list(ds.coords)}")

    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 1:
        lat_1d = lat_vals
        j_dim = lat_name
    elif lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_dim = ds[lat_name].dims[0]
    else:
        raise ValueError(f"Unexpected lat shape: {lat_vals.shape}")

    # Find longitude
    lon_name = None
    for name in ["longitude", "lon", "nav_lon"]:
        if name in ds.coords:
            lon_name = name
            break
    if lon_name is None:
        for name in ds.coords:
            if "lon" in name.lower():
                lon_name = name
                break
    if lon_name is None:
        raise ValueError(f"Cannot find longitude in {list(ds.coords)}")

    lon_vals = ds[lon_name].values
    # Return full lon array (1D or 2D) — compute_grid_metrics handles both
    lon_full = lon_vals

    # Find x dimension
    x_dim = None
    vo_dims = list(ds["vo"].dims)
    for d in vo_dims:
        if d not in ["time", j_dim] and d not in ["lev", "depth", "olevel", "deptht"]:
            x_dim = d
            break
    if x_dim is None:
        raise ValueError(f"Cannot identify x dimension in {vo_dims}")

    # Find depth
    z_dim = None
    for name in ["lev", "depth", "olevel", "deptht"]:
        if name in ds.coords or name in ds.dims:
            z_dim = name
            break
    if z_dim is None:
        raise ValueError(f"Cannot find depth in {list(ds.coords)}")

    if z_dim in ds.coords:
        depth = ds[z_dim].values.astype(float)
        units = ds[z_dim].attrs.get("units", "m")
        if units in ("centimeters", "cm"):
            depth = depth / 100.0
    else:
        depth = np.arange(ds.sizes[z_dim], dtype=float)

    return lat_1d, lon_full, depth, j_dim, x_dim, z_dim


def compute_grid_metrics(
    lat_1d: np.ndarray, lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dx(lat, lon) and Atlantic mask(lat, lon).

    Handles both 1D lon (regular grids) and 2D lon (curvilinear NEMO/POP).
    Returns (dx_2d, atl_mask_2d) both shape (ny, nx).
    """
    ny = len(lat_1d)

    if lon.ndim == 1:
        # Regular grid: broadcast 1D lon to 2D
        nx = len(lon)
        lon_2d = np.broadcast_to(lon[np.newaxis, :], (ny, nx)).copy()
    elif lon.ndim == 2:
        # Curvilinear grid: use actual 2D lon at each row
        nx = lon.shape[1]
        lon_2d = lon.copy()
    else:
        raise ValueError(f"Unexpected lon shape: {lon.shape}")

    # Wrap to [-180, 180]
    lon_wrapped = np.where(lon_2d > 180, lon_2d - 360, lon_2d)

    # dx from per-row longitude differences
    dlon = np.diff(lon_2d, axis=1)
    dlon = np.concatenate([dlon, dlon[:, -1:]], axis=1)
    dlon = (dlon + 180) % 360 - 180
    cos_lat = np.cos(np.deg2rad(lat_1d))[:, np.newaxis]
    dx_2d = np.abs(dlon) * 111000.0 * cos_lat
    dx_2d = np.clip(dx_2d, 1.0, None)

    # Atlantic mask using per-row longitude
    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        if lat_1d[j] < -55 or lat_1d[j] > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(lat_1d[j])
        atl_mask[j, :] = (lon_wrapped[j, :] >= lon_min) & (lon_wrapped[j, :] <= lon_max)

    return dx_2d, atl_mask


def download_and_integrate(
    cat_df, model: str, experiment: str, output_dir: Path,
    start_year: int | None = None, force: bool = False,
) -> bool:
    """Download vo from Pangeo and zonally integrate on-the-fly.

    Returns True on success.
    """
    outfile = output_dir / f"{model}_{experiment}_vo_zonal.nc"
    if outfile.exists() and not force:
        print(f"  Already exists: {outfile.name}")
        return True

    vo_url = find_zstore(cat_df, model, experiment, "vo")
    if vo_url is None:
        print(f"  NOT FOUND: vo for {model}/{experiment}")
        return False

    print(f"  Opening zarr: {model}/{experiment}")
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_zarr(vo_url, consolidated=True, decode_times=time_coder)
    except Exception as e:
        print(f"  FAILED to open zarr: {e}")
        return False

    # Filter by start_year if specified
    if start_year is not None:
        times = ds.time.values
        years = np.array([t.year for t in times])
        mask = years >= start_year
        if mask.sum() == 0:
            print(f"  SKIP: no data after {start_year}")
            return False
        n_before = len(times)
        ds = ds.isel(time=mask)
        print(f"  Filtered: {n_before} -> {mask.sum()} timesteps (>= {start_year})")

    print(f"  Dataset dims: {dict(ds.sizes)}")

    try:
        lat_1d, lon_1d, depth, j_dim, x_dim, z_dim = get_lat_lon_depth(ds)
    except ValueError as e:
        print(f"  FAILED: {e}")
        return False

    ny = len(lat_1d)
    nz = len(depth)
    nx = lon_1d.shape[-1]  # works for both 1D and 2D lon
    print(f"  Grid: {ny} lat x {nx} lon x {nz} depth, lon {'2D' if lon_1d.ndim == 2 else '1D'}")
    print(f"  Lat range: {lat_1d.min():.1f} to {lat_1d.max():.1f}")
    print(f"  Depth range: {depth[0]:.1f} to {depth[-1]:.1f} m")

    dx_2d, atl_mask = compute_grid_metrics(lat_1d, lon_1d)
    n_atl_total = atl_mask.sum()
    print(f"  Atlantic points: {n_atl_total}/{ny*nx}")

    vo_da = ds["vo"]
    n_times = ds.sizes["time"]
    n_batches = (n_times + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Total timesteps: {n_times} ({n_batches} batches of {BATCH_SIZE})")

    # Pre-allocate output
    v_zonal_all = np.zeros((n_times, nz, ny), dtype=np.float32)
    all_times = []
    t0 = time.time()

    for b in range(n_batches):
        i_start = b * BATCH_SIZE
        i_end = min(i_start + BATCH_SIZE, n_times)

        try:
            batch = vo_da.isel(time=slice(i_start, i_end)).load()
            batch_times = batch.time.values
            all_times.append(batch_times)

            # vo shape: (batch_t, nz, ny, nx)
            v = batch.values
            v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)

            # Zonally integrate: v * dx * atl_mask, sum over x
            # dx_2d: (ny, nx), atl_mask: (ny, nx)
            weight = (dx_2d * atl_mask)[np.newaxis, np.newaxis, :, :]  # (1, 1, ny, nx)
            v_zonal = np.nansum(v * weight, axis=3)  # (batch_t, nz, ny)
            v_zonal_all[i_start:i_end] = v_zonal.astype(np.float32)

            elapsed = time.time() - t0
            rate = i_end / elapsed if elapsed > 0 else 0
            eta = (n_times - i_end) / rate if rate > 0 else 0
            print(
                f"    Batch {b+1}/{n_batches}: t[{i_start}:{i_end}] "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )
        except Exception as e:
            print(f"    Batch {b+1}/{n_batches}: FAILED ({e})")
            if not all_times:
                return False
            # Truncate to what we have
            n_times = i_start
            v_zonal_all = v_zonal_all[:n_times]
            break

    times = np.concatenate(all_times)
    elapsed = time.time() - t0
    print(f"  Download complete: {len(times)} timesteps in {elapsed/60:.1f} min")

    # Save as NetCDF
    ds_out = xr.Dataset(
        {
            "v_zonal": xr.DataArray(
                v_zonal_all[:len(times)],
                dims=("time", "depth_idx", "lat_idx"),
                attrs={
                    "units": "m^2/s (zonally integrated)",
                    "long_name": "Zonally integrated meridional velocity * dx over Atlantic",
                    "source_id": model,
                    "experiment_id": experiment,
                },
            ),
        },
        coords={
            "time": times,
            "depth": ("depth_idx", depth),
            "lat": ("lat_idx", lat_1d),
        },
    )
    ds_out.to_netcdf(outfile)
    print(f"  Saved: {outfile} ({outfile.stat().st_size / 1e6:.0f} MB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CMIP6 vo and zonally integrate on-the-fly from Pangeo."
    )
    parser.add_argument(
        "--models", nargs="+", default=TARGET_MODELS,
        help=f"Models to download. Default: {TARGET_MODELS}",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=EXPERIMENTS,
    )
    parser.add_argument("--output-dir", default="data/cmip6_fullfield")
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Skip timesteps before this year (e.g. 1950 to cut download time).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Opening Pangeo CMIP6 catalog...")
    try:
        cat = intake.open_esm_datastore(PANGEO_CATALOG_URL)
    except Exception as e:
        print(f"FAILED to open catalog: {e}")
        sys.exit(1)
    cat_df = cat.df
    print(f"Catalog: {len(cat_df)} entries")

    successes, failures = 0, 0
    t_start = time.time()

    for model in args.models:
        print(f"\n=== {model} ===")
        for experiment in args.experiments:
            print(f"  [{experiment}]")
            if download_and_integrate(cat_df, model, experiment, output_dir,
                                     start_year=args.start_year, force=args.force):
                successes += 1
            else:
                failures += 1

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
