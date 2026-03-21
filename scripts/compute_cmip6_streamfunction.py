#!/usr/bin/env python3
"""Compute CMIP6 AMOC streamfunction snapshots using 30-year running means.

Takes the zonally-integrated v_zonal from download_cmip6_vo_fullfield.py and
computes Ψ(lat, depth) at 5-year intervals for animation.

Output: data/results/cmip6/streamfunction_{model}_ssp585.npz
  - psi: (n_frames, nz, ny) streamfunction in Sv
  - lat, depth: coordinate arrays
  - center_years: center year of each 30-year window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


TARGET_MODELS = [
    "NESM3", "IPSL-CM6A-LR", "CNRM-CM6-1", "MIROC6",
    "MPI-ESM1-2-HR", "CanESM5",
    "UKESM1-0-LL", "CMCC-CM2-SR5", "GFDL-CM4", "ACCESS-CM2",
    "MPI-ESM1-2-LR", "HadGEM3-GC31-LL", "CESM2", "FIO-ESM-2-0",
    "GISS-E2-1-G", "FGOALS-g3",
]
WINDOW_YEARS = 30
STEP_YEARS = 5


def load_and_concatenate(
    data_dir: Path, model: str,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Load historical + ssp585 v_zonal files and concatenate.

    Returns (v_zonal DataArray, depth, lat).
    """
    hist_file = data_dir / f"{model}_historical_vo_zonal.nc"
    ssp_file = data_dir / f"{model}_ssp585_vo_zonal.nc"

    datasets = []
    for f in [hist_file, ssp_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
        datasets.append(xr.open_dataset(f))

    # Concatenate along time, removing duplicates
    combined = xr.concat([ds["v_zonal"] for ds in datasets], dim="time")
    _, idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=sorted(idx))

    depth = datasets[0]["depth"].values
    lat = datasets[0]["lat"].values

    return combined, depth, lat


def compute_streamfunction_snapshots(
    v_zonal: xr.DataArray,
    depth: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Ψ(lat, depth) for 30-year windows at 5-year steps.

    Returns (psi, center_years) where psi is (n_frames, nz, ny) in Sv.
    """
    # Extract years from time coordinate
    times = v_zonal.time.values
    try:
        years = np.array([t.year for t in times])
    except AttributeError:
        # numpy.datetime64 — convert via pandas
        import pandas as pd
        years = pd.DatetimeIndex(times).year.values

    year_min = int(years.min())
    year_max = int(years.max())
    half_w = WINDOW_YEARS // 2

    # Generate center years at STEP_YEARS intervals
    # Allow partial windows at the end so we reach ~2095
    center_years = np.arange(
        year_min + half_w, year_max - STEP_YEARS + 1, STEP_YEARS
    )
    print(f"  Windows: {len(center_years)} frames, {center_years[0]}–{center_years[-1]}")

    dz = np.diff(depth, prepend=0.0)
    nz = len(depth)
    ny = len(lat)
    psi_all = np.zeros((len(center_years), nz, ny), dtype=np.float32)

    for i, cy in enumerate(center_years):
        mask = (years >= cy - half_w) & (years < cy + half_w)
        if mask.sum() == 0:
            continue

        # Time-mean zonally-integrated velocity: (nz, ny)
        v_mean = v_zonal.values[mask].mean(axis=0)

        # Replace NaN/inf with 0
        v_mean = np.where(np.isfinite(v_mean), v_mean, 0.0)

        # Transport per level: v_mean * dz
        transport = v_mean * dz[:, np.newaxis]

        # Streamfunction: cumulative from surface
        psi = np.cumsum(transport, axis=0) / 1e6  # Sv
        psi_all[i] = psi.astype(np.float32)

    return psi_all, center_years


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CMIP6 AMOC streamfunction snapshots."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmip6_fullfield"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/cmip6"))
    parser.add_argument("--models", nargs="+", default=TARGET_MODELS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        print(f"=== {model} ===")
        try:
            v_zonal, depth, lat = load_and_concatenate(args.data_dir, model)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  v_zonal: {dict(v_zonal.sizes)}")
        psi, center_years = compute_streamfunction_snapshots(v_zonal, depth, lat)

        # Sanity checks
        lat_mask = (lat >= 0) & (lat <= 60)
        depth_mask = depth >= 500
        for frame_i in [0, len(center_years) // 2, -1]:
            psi_interior = psi[frame_i][np.ix_(depth_mask, lat_mask)]
            psi_max = np.nanmax(psi_interior)
            print(f"  Year {center_years[frame_i]}: AMOC max = {psi_max:.1f} Sv")

        outfile = args.output_dir / f"streamfunction_{model}_ssp585.npz"
        np.savez_compressed(
            outfile,
            psi=psi, lat=lat, depth=depth, center_years=center_years,
        )
        print(f"  Saved: {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")
        print()


if __name__ == "__main__":
    main()
