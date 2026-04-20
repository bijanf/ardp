#!/usr/bin/env python3
"""Compute F_ovS from ORAS5 depth data (velocity + salinity at 34.5S).

Memory-efficient: processes one file pair at a time, extracting only the
j-row at SAMBA_LAT (~430 KB per timestep) instead of loading all files
into a single dataset (~91 GB).

Handles ORAS5-specific issues:
- Velocity files use 'depthv' dim, salinity uses 'deptht' (same values)
- No mesh mask: grid spacings approximated from 2D nav_lon/nav_lat
- Files are per-variable, per-month: vomecrty_*_YYYYMM_*.nc, vosaline_*_YYYYMM_*.nc
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0, SAMBA_LAT
from ardp.physics.fovs import compute_fovs_from_section


def find_j_index(
    data_dir: Path, target_lat: float = SAMBA_LAT
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Open ONE file to find nearest j-index to target_lat and compute grid metrics.

    Returns (j_idx, e1t_atlantic, e3t, atlantic_mask, actual_lat) where:
    - j_idx: integer y-index for the SAMBA latitude
    - e1t_atlantic: 1D array of zonal grid spacings [m] for Atlantic x-points only
    - e3t: 1D array of vertical cell thicknesses [m] (length z)
    - atlantic_mask: boolean mask for Atlantic x-points (length x)
    - actual_lat: the latitude actually used
    """
    v_file = sorted(data_dir.glob("vomecrty_*.nc"))[0]
    ds = xr.open_dataset(v_file)

    nav_lat = ds["nav_lat"].values  # (y, x)
    nav_lon = ds["nav_lon"].values  # (y, x)

    # Mean latitude per j-row
    lat_1d = np.nanmean(nav_lat, axis=1)
    j_idx = int(np.abs(lat_1d - target_lat).argmin())
    actual_lat = float(lat_1d[j_idx])

    # Atlantic basin mask at this latitude
    lon_row = nav_lon[j_idx, :]  # (x,)
    atlantic_mask = (lon_row >= ATLANTIC_LON_MIN) & (lon_row <= ATLANTIC_LON_MAX)

    # e1t at the section: zonal spacing from longitude diffs
    lat_row = nav_lat[j_idx, :]
    dlon = np.diff(lon_row)
    # Handle wrap-around
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    # Pad last value
    dlon = np.append(dlon, dlon[-1])
    cos_lat = np.cos(np.deg2rad(lat_row))
    e1t = np.abs(dlon) * 111000.0 * cos_lat
    e1t = np.clip(e1t, 1.0, None)

    # e3t from depth coordinate
    depth = ds["depthv"].values
    e3t = np.diff(depth, prepend=0.0)

    ds.close()
    return j_idx, e1t[atlantic_mask], e3t, atlantic_mask, actual_lat


def pair_files(data_dir: Path) -> list[tuple[Path, Path, str]]:
    """Match velocity/salinity files by YYYYMM, return (v_file, s_file, yyyymm) triples."""
    v_files = {_extract_yyyymm(f): f for f in sorted(data_dir.glob("vomecrty_*_3D_*.nc"))}
    s_files = {_extract_yyyymm(f): f for f in sorted(data_dir.glob("vosaline_*_3D_*.nc"))}

    common = sorted(set(v_files) & set(s_files))
    if not common:
        raise FileNotFoundError(f"No matched v/s file pairs in {data_dir}")
    return [(v_files[k], s_files[k], k) for k in common]


def _extract_yyyymm(path: Path) -> str:
    """Extract YYYYMM from filename like vomecrty_..._3D_195801_CONS_v0.1.nc."""
    m = re.search(r"_3D_(\d{6})_", path.name)
    if not m:
        raise ValueError(f"Cannot extract YYYYMM from {path.name}")
    return m.group(1)


def process_one_month(
    v_file: Path,
    s_file: Path,
    j_idx: int,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    atlantic_mask: np.ndarray,
) -> tuple[np.datetime64, float]:
    """Open 2 files, extract j-slice, compute F_ovS, return (timestamp, scalar).

    Delegates to ardp.physics.fovs.compute_fovs_from_section which
    applies the de Vries & Weber (2005) formula WITH the mandatory
    barotropic (section-mean) velocity subtraction required for
    non-mass-conserving Boussinesq reanalysis products.

    Peak memory: ~900 KB (two 2D slices of shape (75, 1442)).
    """
    # Open velocity — extract only j-slice and Atlantic x-points
    ds_v = xr.open_dataset(v_file)
    v_section = ds_v["vomecrty"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]
    timestamp = ds_v["time_counter"].values[0]
    ds_v.close()

    # Open salinity — same extraction
    ds_s = xr.open_dataset(s_file)
    s_section = ds_s["vosaline"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]
    ds_s.close()

    f_ov = compute_fovs_from_section(
        v_section, s_section, e1t_atl, e3t, s0=S0,
    )
    return (timestamp, float(f_ov))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ORAS5 F_ovS from depth data.")
    parser.add_argument(
        "--data-dir", default="data/oras5", help="Directory with ORAS5 NetCDF files."
    )
    parser.add_argument(
        "--results-dir", default="data/results", help="Output directory for results."
    )
    parser.add_argument(
        "--lat", type=float, default=SAMBA_LAT, help=f"Target latitude (default: {SAMBA_LAT})."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Number of parallel workers (default: 14).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find j-index and grid metrics from one file
    print("Finding j-index and grid metrics...")
    j_idx, e1t_atl, e3t, atlantic_mask, actual_lat = find_j_index(data_dir, target_lat=args.lat)
    print(f"  Section at j={j_idx}, lat={actual_lat:.2f} (target: {args.lat})")
    print(f"  Atlantic x-points: {atlantic_mask.sum()}")

    # Step 2: Pair velocity/salinity files by YYYYMM
    pairs = pair_files(data_dir)
    print(f"  Found {len(pairs)} matched file pairs")

    # Step 3: Process all months in parallel
    print(f"Computing F_ovS ({args.workers} workers)...")
    results: list[tuple[np.datetime64, float]] = []
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_month, v_file, s_file, j_idx, e1t_atl, e3t, atlantic_mask): yyyymm
            for v_file, s_file, yyyymm in pairs
        }
        for i, future in enumerate(as_completed(futures), 1):
            yyyymm = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                errors.append(f"{yyyymm}: {e}")
                print(f"  WARNING: {yyyymm} failed: {e}")
            if i % 20 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)} done")

    if errors:
        print(f"  {len(errors)} months failed")

    # Step 4: Assemble results
    results.sort(key=lambda x: x[0])
    timestamps = np.array([r[0] for r in results])
    values = np.array([r[1] for r in results])

    f_ovs = xr.DataArray(
        values,
        dims=("time",),
        coords={"time": timestamps},
        name="F_ovS",
        attrs={
            "units": "Sv",
            "long_name": f"Overturning freshwater transport at {actual_lat:.1f}S",
            "section_latitude": actual_lat,
            "section_j_index": j_idx,
        },
    )

    print(f"\n  Got {len(f_ovs)} timesteps")
    print(f"  F_ovS range: {float(f_ovs.min()):.4f} to {float(f_ovs.max()):.4f} Sv")
    print(f"  F_ovS mean: {float(f_ovs.mean()):.4f} Sv")

    # Trend analysis
    import pandas as pd

    ts = pd.DatetimeIndex(timestamps)
    years = np.array([t.year + (t.month - 1) / 12.0 for t in ts])
    valid = np.isfinite(values)

    if valid.sum() >= 2:
        coeffs = np.polyfit(years[valid], values[valid], 1)
        trend_msv = coeffs[0] * 1e3
        print(f"\n  Trend: {trend_msv:.2f} mSv/yr (expected: -1.20 mSv/yr)")
        print(f"  Mean F_ovS: {float(np.mean(values[valid])):.4f} Sv")
        print(f"  {'NEGATIVE (bistable regime)' if np.mean(values[valid]) < 0 else 'POSITIVE'}")

    # Save
    outfile = results_dir / "oras5_f_ovs.nc"
    f_ovs.to_netcdf(outfile)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
