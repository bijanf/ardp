#!/usr/bin/env python3
"""Compute F_ovS from GLORYS12V1 data (velocity + salinity at 34.5S).

GLORYS12 has a regular 1D lat/lon grid (unlike ORAS5's curvilinear ORCA),
so grid metrics are simpler. Each yearly file contains 12 monthly timesteps
with variables: vo (meridional velocity), so (salinity).

Uses the de Vries & Weber (2005) formula:
  F_ov = -(1/S0) * integral_z { V_int(z) * (S_mean(z) - S0) * dz }
where V_int = integral_x(v * dx) is zonally INTEGRATED velocity [m²/s]
and S_mean is zonally AVERAGED salinity [PSU], both over Atlantic only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import S0, SAMBA_LAT
from ardp.physics.fovs import compute_fovs_from_section

# Atlantic basin longitude bounds at ~34.5S
ATLANTIC_LON_MIN = -70.0
ATLANTIC_LON_MAX = 20.0


def get_grid_metrics(
    data_dir: Path, target_lat: float = SAMBA_LAT
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Get j-index, grid spacings, and Atlantic mask from one GLORYS12 file.

    Returns (j_idx, e1t_atlantic, e3t, atlantic_mask, actual_lat) where:
    - j_idx: integer latitude index for the SAMBA latitude
    - e1t_atlantic: 1D array of zonal grid spacings [m] for Atlantic points
    - e3t: 1D array of vertical cell thicknesses [m] (length = n_depth)
    - atlantic_mask: boolean mask for Atlantic longitude points
    - actual_lat: the latitude actually used
    """
    sample_file = sorted(data_dir.glob("glorys12_*.nc"))[0]
    ds = xr.open_dataset(sample_file)

    lat = ds["latitude"].values  # 1D
    lon = ds["longitude"].values  # 1D

    # Find nearest j-index
    j_idx = int(np.abs(lat - target_lat).argmin())
    actual_lat = float(lat[j_idx])

    # Atlantic mask at this latitude
    atlantic_mask = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)

    # e1t: zonal grid spacing from longitude differences
    # For a regular grid, dlon is constant, but compute per-point for safety
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])  # pad last value
    cos_lat = np.cos(np.deg2rad(actual_lat))
    e1t = np.abs(dlon) * 111000.0 * cos_lat  # [m]
    e1t = np.clip(e1t, 1.0, None)

    # e3t: vertical cell thickness from depth coordinate
    depth = ds["depth"].values
    e3t = np.diff(depth, prepend=0.0)

    ds.close()
    return j_idx, e1t[atlantic_mask], e3t, atlantic_mask, actual_lat


def process_one_year(
    filepath: Path,
    j_idx: int,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    atlantic_mask: np.ndarray,
) -> list[tuple[np.datetime64, float]]:
    """Process one yearly GLORYS12 file, return list of (timestamp, F_ovS) tuples.

    Each file has 12 monthly timesteps. We extract only the j-slice at
    the SAMBA latitude and Atlantic longitudes (~1 MB per timestep).
    """
    ds = xr.open_dataset(filepath)
    results = []

    n_times = ds.sizes["time"]
    for t in range(n_times):
        timestamp = ds["time"].values[t]

        # Extract j-slice: (depth, lon_atlantic)
        v_section = ds["vo"].isel(time=t, latitude=j_idx).values[:, atlantic_mask]
        s_section = ds["so"].isel(time=t, latitude=j_idx).values[:, atlantic_mask]

        # Shared kernel applies the de Vries & Weber (2005) formula
        # with the mandatory barotropic-velocity subtraction.
        f_ov = compute_fovs_from_section(
            v_section, s_section, e1t_atl, e3t, s0=S0,
        )
        results.append((timestamp, float(f_ov)))

    ds.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GLORYS12 F_ovS.")
    parser.add_argument(
        "--data-dir", default="data/glorys12", help="Directory with GLORYS12 NetCDF files."
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
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get grid metrics from one file
    print("Finding j-index and grid metrics...")
    j_idx, e1t_atl, e3t, atlantic_mask, actual_lat = get_grid_metrics(
        data_dir, target_lat=args.lat
    )
    print(f"  Section at j={j_idx}, lat={actual_lat:.2f} (target: {args.lat})")
    print(f"  Atlantic x-points: {atlantic_mask.sum()}")

    # Step 2: Find all yearly files
    yearly_files = sorted(data_dir.glob("glorys12_*.nc"))
    print(f"  Found {len(yearly_files)} yearly files")

    # Step 3: Process all years in parallel
    print(f"Computing F_ovS ({args.workers} workers)...")
    all_results: list[tuple[np.datetime64, float]] = []
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_year, f, j_idx, e1t_atl, e3t, atlantic_mask): f.stem
            for f in yearly_files
        }
        for i, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                year_results = future.result()
                all_results.extend(year_results)
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"  WARNING: {name} failed: {e}")
            if i % 5 == 0 or i == len(yearly_files):
                print(f"  {i}/{len(yearly_files)} files done")

    if errors:
        print(f"  {len(errors)} files failed")

    # Step 4: Assemble results
    all_results.sort(key=lambda x: x[0])
    timestamps = np.array([r[0] for r in all_results])
    values = np.array([r[1] for r in all_results])

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
            "product": "GLORYS12V1",
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
        print(f"\n  Trend: {trend_msv:.2f} mSv/yr")
        print(f"  Mean F_ovS: {float(np.mean(values[valid])):.4f} Sv")
        print(f"  {'NEGATIVE (bistable regime)' if np.mean(values[valid]) < 0 else 'POSITIVE'}")

    # Save
    outfile = results_dir / "glorys12_f_ovs.nc"
    f_ovs.to_netcdf(outfile)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
