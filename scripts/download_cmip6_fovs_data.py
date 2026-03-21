#!/usr/bin/env python3
"""Download CMIP6 34.5S latitude slices for F_ovS computation.

Queries the Pangeo CMIP6 catalog for monthly ocean model output (Omon table),
extracts only the single latitude row nearest -34.5S, and saves small NetCDF
files locally. Downloads in time batches to avoid memory/timeout issues with
large piControl runs.

Target models span bistable (F_ovS < 0), near-zero, and monostable regimes.

Requirements: pip install intake-esm gcsfs
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import intake
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# Target models: broad spread of AMOC regimes
TARGET_MODELS = [
    "CESM2",
    "MPI-ESM1-2-LR",
    "MPI-ESM1-2-HR",
    "UKESM1-0-LL",
    "CNRM-CM6-1",
    "EC-Earth3",
    "GFDL-ESM4",
    "CanESM5",
]

TARGET_EXPERIMENTS = ["historical", "piControl"]
TARGET_VARIABLES = ["vo", "so"]
TARGET_LAT = -34.5
BATCH_SIZE = 120  # timesteps per batch (~10 years of monthly data)

PANGEO_CATALOG_URL = (
    "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
)


def open_pangeo_catalog() -> intake.Catalog:
    """Open the Pangeo CMIP6 intake-esm catalog."""
    return intake.open_esm_datastore(PANGEO_CATALOG_URL)


def find_nearest_lat_idx(lat: np.ndarray, target: float) -> int:
    """Find index of nearest latitude to target."""
    return int(np.abs(lat - target).argmin())


def _find_lat_and_j(
    ds: xr.Dataset, variable_id: str
) -> tuple[str, int, float, str | None]:
    """Find latitude coordinate, j-index, actual lat, and j-dim name.

    Returns (lat_name, j_idx, actual_lat, j_dim_for_isel).
    j_dim_for_isel is the dimension name to use with isel, or lat_name if 1D.
    """
    # Find latitude coordinate
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
        raise ValueError(f"Cannot find latitude coordinate in {list(ds.coords)}")

    lat_vals = ds[lat_name].values

    if lat_vals.ndim == 1:
        j_idx = find_nearest_lat_idx(lat_vals, TARGET_LAT)
        actual_lat = float(lat_vals[j_idx])
        return lat_name, j_idx, actual_lat, lat_name
    elif lat_vals.ndim == 2:
        # Curvilinear grid — take mean latitude per j-row
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_idx = find_nearest_lat_idx(lat_1d, TARGET_LAT)
        actual_lat = float(lat_1d[j_idx])
        j_dim = ds[lat_name].dims[0]
        return lat_name, j_idx, actual_lat, j_dim
    else:
        raise ValueError(f"Unexpected lat shape {lat_vals.shape}")


def extract_section(
    cat: intake.Catalog,
    source_id: str,
    experiment_id: str,
    variable_id: str,
    output_dir: Path,
) -> bool:
    """Extract 34.5S section for one model/experiment/variable combination.

    Downloads in time batches to handle large datasets efficiently.
    Returns True on success, False on failure.
    """
    outfile = output_dir / f"{source_id}_{experiment_id}_{variable_id}.nc"
    if outfile.exists():
        print(f"  Already exists: {outfile.name}")
        return True

    # Search catalog — prefer regridded (gr) over native (gn)
    results = cat.search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        table_id="Omon",
    )

    if len(results.df) == 0:
        print(f"  NOT FOUND: {source_id}/{experiment_id}/{variable_id}")
        return False

    # Prefer 'gr' grid label, fall back to 'gn'
    df = results.df
    for grid_pref in ["gr", "gn"]:
        subset = df[df["grid_label"] == grid_pref]
        if len(subset) > 0:
            df = subset
            break

    # Prefer r1i1p1f1 member
    for member_pref in ["r1i1p1f1", "r1i1p1f2"]:
        subset = df[df["member_id"] == member_pref]
        if len(subset) > 0:
            df = subset
            break

    key = df.iloc[0]["zstore"]
    grid_label = df.iloc[0]["grid_label"]
    member_id = df.iloc[0]["member_id"]
    print(
        f"  Opening: {source_id}/{experiment_id}/{variable_id} "
        f"({grid_label}, {member_id})"
    )

    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_zarr(key, consolidated=True, decode_times=time_coder)
    except Exception as e:
        print(f"  FAILED to open zarr: {e}")
        return False

    # Find latitude and j-index
    try:
        lat_name, j_idx, actual_lat, j_dim = _find_lat_and_j(ds, variable_id)
    except ValueError as e:
        print(f"  FAILED: {e}")
        return False

    curv = "curvilinear" if ds[lat_name].values.ndim == 2 else "regular"
    print(f"  Nearest lat: {actual_lat:.2f} (target: {TARGET_LAT}, {curv})")

    # Select the lat slice (lazy)
    da = ds[variable_id]
    section_lazy = da.isel({j_dim: j_idx})
    n_times = section_lazy.sizes["time"]
    print(f"  Section shape: {dict(section_lazy.sizes)} ({n_times} timesteps)")

    # Download in batches
    t0 = time.time()
    batches = []
    n_batches = (n_times + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, n_times, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        end = min(i + BATCH_SIZE, n_times)
        try:
            batch = section_lazy.isel(time=slice(i, end)).load()
            batches.append(batch)
            elapsed = time.time() - t0
            rate = (end) / elapsed if elapsed > 0 else 0
            eta = (n_times - end) / rate if rate > 0 else 0
            print(
                f"    Batch {batch_num}/{n_batches}: "
                f"t[{i}:{end}] loaded ({elapsed:.0f}s elapsed, "
                f"~{eta:.0f}s remaining)"
            )
        except Exception as e:
            print(f"    Batch {batch_num}/{n_batches}: FAILED ({e})")
            # Try to salvage what we have
            if batches:
                print(f"    Continuing with {len(batches)} successful batches")
            break

    if not batches:
        print(f"  FAILED: no data loaded")
        return False

    # Concatenate batches
    section = xr.concat(batches, dim="time")

    # Also save lon coordinates for grid metric computation
    # Find and include longitude coordinate
    lon_data = {}
    for name in ds.coords:
        if "lon" in name.lower():
            lon_vals = ds[name]
            if lon_vals.ndim == 1:
                lon_data[name] = lon_vals
            elif lon_vals.ndim == 2:
                # Save the lon row at j_idx
                lon_data[name] = lon_vals.isel({j_dim: j_idx}).load()

    # Build output dataset with metadata
    section.attrs["source_id"] = source_id
    section.attrs["experiment_id"] = experiment_id
    section.attrs["grid_label"] = grid_label
    section.attrs["member_id"] = member_id
    section.attrs["section_latitude"] = actual_lat
    section.attrs["section_j_index"] = j_idx

    out_ds = section.to_dataset(name=variable_id)
    for lon_name, lon_arr in lon_data.items():
        out_ds[lon_name] = lon_arr

    out_ds.to_netcdf(outfile)
    elapsed = time.time() - t0
    size_mb = outfile.stat().st_size / 1e6
    print(f"  Saved: {outfile.name} ({size_mb:.1f} MB, {elapsed:.0f}s)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CMIP6 34.5S sections for F_ovS computation."
    )
    parser.add_argument(
        "--output-dir",
        default="data/cmip6_sections",
        help="Output directory for section files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=TARGET_MODELS,
        help="CMIP6 source_id(s) to download.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=TARGET_EXPERIMENTS,
        help="Experiment IDs to download.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Opening Pangeo CMIP6 catalog...")
    try:
        cat = open_pangeo_catalog()
    except Exception as e:
        print(f"FAILED to open Pangeo catalog: {e}")
        sys.exit(1)

    print(f"Catalog has {len(cat.df)} entries")
    print(f"Target models: {args.models}")
    print(f"Target experiments: {args.experiments}")
    print(f"Batch size: {BATCH_SIZE} timesteps")
    print()

    successes = 0
    failures = 0
    t_start = time.time()

    for model in args.models:
        for experiment in args.experiments:
            for variable in TARGET_VARIABLES:
                print(f"[{model}] {experiment}/{variable}:")
                ok = extract_section(cat, model, experiment, variable, output_dir)
                if ok:
                    successes += 1
                else:
                    failures += 1
                print()

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.1f} min: {successes} successes, {failures} failures")
    if failures > 0:
        print("Some models/experiments were not found or failed. This is expected —")
        print("not all models are available in the Pangeo catalog.")


if __name__ == "__main__":
    main()
