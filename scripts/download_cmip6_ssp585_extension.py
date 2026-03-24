#!/usr/bin/env python3
"""Download CMIP6 SSP585 extensions (2101-2300) from ESGF via OPeNDAP.

For 5 models with data beyond 2100, downloads vo, zonally integrates
on-the-fly, and saves {model}_ssp585ext_vo_zonal.nc.

Uses OPeNDAP to avoid storing the full 3D field (~195 GB total).
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import requests
import xarray as xr

BATCH_SIZE = 12  # months per batch

MODELS = ["ACCESS-CM2", "ACCESS-ESM1-5", "MRI-ESM2-0", "CESM2-WACCM", "IPSL-CM6A-LR"]
MEMBER = "r1i1p1f1"
ESGF_NODE = "https://esgf-data.dkrz.de/esg-search/search"


def find_extension_files(model: str) -> list[str]:
    """Query ESGF to find OPeNDAP URLs for post-2100 vo files."""
    r = requests.get(ESGF_NODE,
        params={
            "type": "File",
            "source_id": model,
            "experiment_id": "ssp585",
            "variable_id": "vo",
            "table_id": "Omon",
            "member_id": MEMBER,
            "format": "application/solr+json",
            "limit": 200,
            "fields": "title,url",
        },
        timeout=60)
    data = r.json()
    docs = data.get("response", {}).get("docs", [])

    urls = []
    seen = set()
    for d in docs:
        title = d.get("title", "")
        if title in seen:
            continue
        # Check if file contains post-2100 data
        parts = title.replace(".nc", "").split("_")
        for p in parts:
            if "-" in p and len(p) >= 13:
                start_yr = p.split("-")[0][:4]
                if start_yr.isdigit() and int(start_yr) >= 2101:
                    # Find CEDA OPeNDAP URL (most reliable)
                    file_urls = d.get("url", [])
                    opendap = [u.split("|")[0].replace(".html", "")
                               for u in file_urls if "OPENDAP" in u and "ceda" in u]
                    if opendap:
                        urls.append(opendap[0])
                        seen.add(title)
                    break
    return sorted(urls)


def atlantic_lon_bounds(lat: float) -> tuple[float, float]:
    if lat < -34: return (-70.0, 20.0)
    elif lat < 0: return (-70.0, 20.0)
    elif lat < 10: return (-90.0, 10.0)
    elif lat < 30: return (-100.0, 0.0)
    elif lat < 45: return (-82.0, -5.0)
    elif lat < 65: return (-70.0, 0.0)
    else: return (-60.0, 10.0)


def get_lat_lon_depth(ds):
    """Extract 1D lat, lon, depth from dataset (handles various naming conventions)."""
    # Latitude
    lat_name = None
    for name in ["lat", "latitude", "nav_lat"]:
        if name in ds.coords:
            lat_name = name; break
    if lat_name is None:
        for name in ds.coords:
            if "lat" in name.lower():
                lat_name = name; break
    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_dim = ds[lat_name].dims[0]
    else:
        lat_1d = lat_vals
        j_dim = lat_name

    # Longitude
    lon_name = None
    for name in ["longitude", "lon", "nav_lon"]:
        if name in ds.coords:
            lon_name = name; break
    if lon_name is None:
        for name in ds.coords:
            if "lon" in name.lower():
                lon_name = name; break
    lon_vals = ds[lon_name].values

    # X dimension
    vo_dims = list(ds["vo"].dims)
    x_dim = None
    for d in vo_dims:
        if d not in ["time", j_dim] and d not in ["lev", "depth", "olevel", "deptht"]:
            x_dim = d; break

    # Depth
    z_dim = None
    for name in ["lev", "depth", "olevel", "deptht"]:
        if name in ds.coords or name in ds.dims:
            z_dim = name; break
    depth = ds[z_dim].values.astype(float) if z_dim in ds.coords else np.arange(ds.sizes[z_dim], dtype=float)
    units = ds[z_dim].attrs.get("units", "m") if z_dim in ds.coords else "m"
    if units in ("centimeters", "cm"):
        depth = depth / 100.0

    return lat_1d, lon_vals, depth, j_dim, x_dim, z_dim


def compute_grid_metrics(lat_1d, lon):
    """Compute dx and Atlantic mask."""
    ny = len(lat_1d)
    if lon.ndim == 1:
        nx = len(lon)
        lon_2d = np.broadcast_to(lon[np.newaxis, :], (ny, nx)).copy()
    elif lon.ndim == 2:
        nx = lon.shape[1]
        lon_2d = lon.copy()
    else:
        raise ValueError(f"Unexpected lon shape: {lon.shape}")

    lon_wrapped = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
    dlon = np.diff(lon_2d, axis=1)
    dlon = np.concatenate([dlon, dlon[:, -1:]], axis=1)
    dlon = (dlon + 180) % 360 - 180
    cos_lat = np.cos(np.deg2rad(lat_1d))[:, np.newaxis]
    dx_2d = np.abs(dlon) * 111000.0 * cos_lat
    dx_2d = np.clip(dx_2d, 1.0, None)

    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        if lat_1d[j] < -55 or lat_1d[j] > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(lat_1d[j])
        atl_mask[j, :] = (lon_wrapped[j, :] >= lon_min) & (lon_wrapped[j, :] <= lon_max)

    return dx_2d, atl_mask


def process_model(model: str, output_dir: Path, force: bool = False) -> bool:
    """Download and zonally integrate one model's extension data."""
    outfile = output_dir / f"{model}_ssp585ext_vo_zonal.nc"

    if outfile.exists() and not force:
        print(f"  Already exists: {outfile.name}")
        return True

    print(f"  Querying ESGF for post-2100 files...")
    file_urls = find_extension_files(model)
    if not file_urls:
        print(f"  No extension files found for {model}")
        return False
    print(f"  Found {len(file_urls)} files")
    for u in file_urls:
        print(f"    {u.rsplit('/', 1)[1]}")

    # Open first file to get grid info
    url0 = file_urls[0]
    print(f"  Opening first file via OPeNDAP...")
    try:
        ds0 = xr.open_dataset(url0, decode_times=True)
    except Exception as e:
        print(f"  FAILED to open: {e}")
        # Try with cftime
        try:
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            ds0 = xr.open_dataset(url0, decode_times=time_coder)
        except Exception as e2:
            print(f"  FAILED with cftime too: {e2}")
            return False

    lat_1d, lon, depth, j_dim, x_dim, z_dim = get_lat_lon_depth(ds0)
    ny = len(lat_1d)
    nz = len(depth)
    nx = lon.shape[-1] if lon.ndim >= 1 else lon.shape[0]
    print(f"  Grid: {ny} lat x {nx} lon x {nz} depth")
    ds0.close()

    dx_2d, atl_mask = compute_grid_metrics(lat_1d, lon)
    weight = (dx_2d * atl_mask)[np.newaxis, np.newaxis, :, :]  # (1, 1, ny, nx)

    all_v_zonal = []
    all_times = []
    t_start = time.time()

    for fi, url in enumerate(file_urls):
        fname = url.rsplit("/", 1)[1]
        print(f"\n  File {fi+1}/{len(file_urls)}: {fname}")

        try:
            try:
                ds = xr.open_dataset(url, decode_times=True)
            except Exception:
                time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                ds = xr.open_dataset(url, decode_times=time_coder)

            vo_da = ds["vo"]
            n_times = ds.sizes["time"]
            n_batches = (n_times + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"    {n_times} timesteps ({n_batches} batches)")

            v_zonal_file = np.zeros((n_times, nz, ny), dtype=np.float32)
            file_times = []

            for b in range(n_batches):
                i_start = b * BATCH_SIZE
                i_end = min(i_start + BATCH_SIZE, n_times)

                try:
                    batch = vo_da.isel(time=slice(i_start, i_end)).load()
                    batch_times = batch.time.values
                    file_times.append(batch_times)

                    v = batch.values
                    v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)
                    v_zonal = np.nansum(v * weight, axis=3)
                    v_zonal_file[i_start:i_end] = v_zonal.astype(np.float32)

                    elapsed = time.time() - t_start
                    print(f"    Batch {b+1}/{n_batches}: t[{i_start}:{i_end}] ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"    Batch {b+1}/{n_batches}: FAILED ({e})")
                    if not file_times:
                        break
                    v_zonal_file = v_zonal_file[:i_start]
                    break

            if file_times:
                times = np.concatenate(file_times)
                all_v_zonal.append(v_zonal_file[:len(times)])
                all_times.append(times)
                print(f"    Done: {len(times)} timesteps")
            ds.close()

        except Exception as e:
            print(f"    FAILED to process: {e}")
            continue

    if not all_times:
        print(f"  No data processed for {model}")
        return False

    # Concatenate all files
    v_zonal_all = np.concatenate(all_v_zonal, axis=0)
    times_all = np.concatenate(all_times)
    elapsed = time.time() - t_start
    print(f"\n  Total: {len(times_all)} timesteps in {elapsed/60:.1f} min")

    # Normalize time coordinate to avoid mixed cftime types
    # Convert all times to simple numeric (days since 1850-01-01)
    import cftime
    time_nums = []
    for t in times_all:
        if hasattr(t, 'year'):
            # cftime or datetime object
            days = (t.year - 1850) * 365.25 + (t.month - 1) * 30.44 + t.day
            time_nums.append(days)
        else:
            time_nums.append(float(t))
    times_uniform = xr.DataArray(
        np.array(time_nums, dtype=np.float64),
        dims="time",
        attrs={"units": "days since 1850-01-01", "calendar": "proleptic_gregorian"},
    )

    # Save
    ds_out = xr.Dataset(
        {"v_zonal": xr.DataArray(
            v_zonal_all, dims=("time", "depth_idx", "lat_idx"),
            attrs={"units": "m^2/s (zonally integrated)",
                   "source_id": model, "experiment_id": "ssp585ext"},
        )},
        coords={"time": times_uniform, "depth": ("depth_idx", depth), "lat": ("lat_idx", lat_1d)},
    )
    ds_out.to_netcdf(outfile)
    print(f"  Saved: {outfile} ({outfile.stat().st_size / 1e6:.0f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download CMIP6 SSP585 extensions (2101-2300) from ESGF."
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cmip6_fullfield"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    successes, failures = 0, 0
    for model in args.models:
        if model not in MODELS:
            print(f"Unknown model: {model}")
            continue
        print(f"\n{'='*60}")
        print(f"=== {model} (ssp585 extension to 2300) ===")
        print(f"{'='*60}")
        if process_model(model, args.output_dir, args.force):
            successes += 1
        else:
            failures += 1

    print(f"\nDone: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
