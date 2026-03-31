#!/usr/bin/env python3
"""Compute CMIP6 F_ovS directly from Pangeo cloud zarr stores.

Replaces the two-step download→compute pipeline with a single script that:
  1. Opens CMIP6 zarr stores lazily via intake-esm (no download)
  2. Extracts the 34.5S section (lazy isel)
  3. Computes grid metrics from coordinates
  4. Applies Atlantic mask
  5. Computes F_ovS per timestep (NaN-aware)
  6. Only the final 1D time series is materialized

Output: data/results/cmip6/fovs_{model}_{experiment}.nc
        data/results/cmip6/fovs_{model}_hist_{ssp}.nc  (concatenated)
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import warnings
from pathlib import Path

# Force unbuffered print
print = functools.partial(print, flush=True)

import intake
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

from ardp.constants import (
    ATLANTIC_LON_MAX,
    ATLANTIC_LON_MIN,
    PANGEO_CATALOG_URL,
    S0,
    SAMBA_LAT,
)
from ardp.models import CMIP6_CLOUD_MODELS as TARGET_MODELS

EXPERIMENTS = ["historical", "ssp245", "ssp585"]
BATCH_SIZE = 120  # timesteps per .compute() batch to limit memory

# Note on performance: Pangeo CMIP6 zarr stores are chunked with full spatial
# extent per chunk (e.g., CESM2: 6×60×384×320). An isel on a single j-index
# still downloads the entire spatial chunk. This means cloud-direct compute
# transfers the same volume as downloading the full field — the advantage is
# avoiding intermediate storage and simplifying the pipeline.


def find_nearest_j(ds: xr.Dataset) -> tuple[int, float, str, str]:
    """Find j-index nearest to SAMBA_LAT, handling 1D and 2D lat grids.

    Returns (j_idx, actual_lat, j_dim, lat_name).
    """
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
        j_idx = int(np.abs(lat_vals - SAMBA_LAT).argmin())
        actual_lat = float(lat_vals[j_idx])
        return j_idx, actual_lat, lat_name, lat_name
    elif lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_idx = int(np.abs(lat_1d - SAMBA_LAT).argmin())
        actual_lat = float(lat_1d[j_idx])
        j_dim = ds[lat_name].dims[0]
        return j_idx, actual_lat, j_dim, lat_name
    else:
        raise ValueError(f"Unexpected lat shape: {lat_vals.shape}")


def _identify_dims(da: xr.DataArray) -> tuple[str, str, str]:
    """Identify time, depth (z), and spatial (x) dimension names.

    Should be called on the SLICED section (3D: time, z, x) — NOT the full
    4D array, to avoid picking up the j-dimension as x.
    """
    time_dim = "time"
    depth_names = {"lev", "depth", "olevel", "deptht"}
    non_time = [d for d in da.dims if d != time_dim]

    if len(non_time) < 2:
        raise ValueError(f"Expected at least 2 non-time dims, got {da.dims}")

    z_dim = None
    for d in non_time:
        if d in depth_names:
            z_dim = d
            break
    if z_dim is None:
        z_dim = non_time[0]  # fallback: first non-time dim

    x_dim = None
    for d in non_time:
        if d != z_dim:
            x_dim = d
            break
    if x_dim is None:
        raise ValueError(f"Cannot identify x dim in {da.dims}")

    return time_dim, z_dim, x_dim


def get_lon_1d(ds: xr.Dataset, j_idx: int, j_dim: str, x_dim: str) -> np.ndarray:
    """Extract 1D longitude array at the section latitude."""
    for name in ["longitude", "lon", "nav_lon"]:
        if name in ds.coords:
            vals = ds[name].values
            if vals.ndim == 1 and vals.size > 10:
                return vals
            elif vals.ndim == 2:
                if j_dim in ds[name].dims:
                    return ds[name].isel({j_dim: j_idx}).values
    for name in ds.coords:
        if "lon" in name.lower():
            vals = ds[name].values
            if vals.ndim == 1 and vals.size > 10:
                return vals
            elif vals.ndim == 2 and j_dim in ds[name].dims:
                return ds[name].isel({j_dim: j_idx}).values
    raise ValueError(f"Cannot find longitude in {list(ds.coords)}")


def get_depth(ds: xr.Dataset) -> np.ndarray:
    """Extract 1D depth array, converting cm→m if needed."""
    for name in ["lev", "depth", "olevel", "deptht"]:
        if name in ds.coords or name in ds.dims:
            if name in ds.coords:
                vals = ds[name].values.astype(float)
                units = ds[name].attrs.get("units", "m")
                if units in ("centimeters", "cm"):
                    vals = vals / 100.0
                return vals
            else:
                return np.arange(ds.sizes[name], dtype=float)
    raise ValueError(f"Cannot find depth in {list(ds.coords)}")


def compute_fovs_section_cloud(
    vo_section: xr.DataArray,
    so_section: xr.DataArray,
    e1t: np.ndarray,
    e3t: np.ndarray,
    atlantic_mask: np.ndarray,
    x_dim: str,
    z_dim: str,
) -> np.ndarray:
    """Compute F_ovS per timestep with NaN-aware masking.

    This is the per-timestep loop approach (matching compute_cmip6_fovs_timeseries.py)
    to correctly handle NaN-based land masks that vary with depth.

    Parameters
    ----------
    vo_section, so_section : xr.DataArray
        Section data with dims (time, z, x), already loaded into memory.
    e1t : 1D array, shape (n_atlantic,)
    e3t : 1D array, shape (nz,)
    atlantic_mask : 1D bool array
    x_dim, z_dim : dimension names

    Returns
    -------
    1D array of F_ovS values [Sv], shape (n_times,)
    """
    # Apply Atlantic mask (convert bool to integer indices for isel)
    atl_idx = np.where(atlantic_mask)[0]
    vo_atl = vo_section.isel({x_dim: atl_idx}).values  # (time, z, x_atl)
    so_atl = so_section.isel({x_dim: atl_idx}).values

    n_times = vo_atl.shape[0]
    nz = vo_atl.shape[1]
    fovs = np.full(n_times, np.nan)

    for t in range(n_times):
        total = 0.0
        for k in range(nz):
            ocean = ~np.isnan(so_atl[t, k, :])
            if ocean.sum() == 0:
                continue
            v_k = np.where(ocean, np.nan_to_num(vo_atl[t, k, :], nan=0.0), 0.0)
            v_int = (v_k * e1t).sum()
            e1t_ocean = np.where(ocean, e1t, 0.0)
            s_k = np.nan_to_num(so_atl[t, k, :], nan=0.0)
            s_mean = (s_k * e1t_ocean).sum() / e1t_ocean.sum()
            total += v_int * (s_mean - S0) * e3t[k]
        fovs[t] = -(1.0 / S0) * total / 1e6

    return fovs


def find_zstore(
    cat_df,
    source_id: str,
    experiment_id: str,
    variable_id: str,
) -> str | None:
    """Find the best zarr store URL for a given model/experiment/variable."""
    mask = (
        (cat_df["source_id"] == source_id)
        & (cat_df["experiment_id"] == experiment_id)
        & (cat_df["variable_id"] == variable_id)
        & (cat_df["table_id"] == "Omon")
    )
    df = cat_df[mask]
    if len(df) == 0:
        return None

    # Prefer gn grid
    for grid in ["gn", "gr"]:
        subset = df[df["grid_label"] == grid]
        if len(subset) > 0:
            df = subset
            break

    # Prefer r1i1p1f1, then r1i1p1f2
    for member in ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3"]:
        subset = df[df["member_id"] == member]
        if len(subset) > 0:
            df = subset
            break

    return df.iloc[0]["zstore"]


def compute_model_experiment(
    cat_df,
    model: str,
    experiment: str,
    results_dir: Path,
    force: bool = False,
) -> bool:
    """Compute F_ovS for one model/experiment from cloud.

    Returns True on success, False on failure.
    """
    outfile = results_dir / f"fovs_{model}_{experiment}.nc"
    if outfile.exists() and not force:
        print(f"  Already exists: {outfile.name} (use --force to recompute)")
        return True

    # Find zarr stores
    vo_url = find_zstore(cat_df, model, experiment, "vo")
    so_url = find_zstore(cat_df, model, experiment, "so")

    if vo_url is None or so_url is None:
        missing = []
        if vo_url is None:
            missing.append("vo")
        if so_url is None:
            missing.append("so")
        print(f"  NOT FOUND: {', '.join(missing)}")
        return False

    grid_label = cat_df[
        (cat_df["source_id"] == model)
        & (cat_df["experiment_id"] == experiment)
        & (cat_df["variable_id"] == "vo")
        & (cat_df["table_id"] == "Omon")
    ]
    # Get grid info for display
    for grid in ["gn", "gr"]:
        subset = grid_label[grid_label["grid_label"] == grid]
        if len(subset) > 0:
            grid_label = subset
            break
    for member in ["r1i1p1f1", "r1i1p1f2", "r1i1p1f3"]:
        subset = grid_label[grid_label["member_id"] == member]
        if len(subset) > 0:
            grid_label = subset
            break
    grid_str = grid_label.iloc[0]["grid_label"]
    member_str = grid_label.iloc[0]["member_id"]
    print(f"  Opening: {model}/{experiment} ({grid_str}, {member_str})")

    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        vo_ds = xr.open_zarr(vo_url, consolidated=True, decode_times=time_coder)
        so_ds = xr.open_zarr(so_url, consolidated=True, decode_times=time_coder)
    except Exception as e:
        print(f"  FAILED to open zarr: {e}")
        return False

    # Find section latitude
    try:
        j_idx, actual_lat, j_dim, lat_name = find_nearest_j(vo_ds)
    except ValueError as e:
        print(f"  FAILED: {e}")
        return False

    curv = "curvilinear" if vo_ds[lat_name].values.ndim == 2 else "regular"
    print(f"  Nearest lat: {actual_lat:.2f}° (target: {SAMBA_LAT}, {curv})")

    # Extract section lazily (removes j_dim)
    vo_da = vo_ds["vo"]
    vo_section = vo_da.isel({j_dim: j_idx})
    so_section = so_ds["so"].isel({j_dim: j_idx})

    # Identify dimensions from the SLICED section (3D: time, z, x)
    try:
        time_dim, z_dim, x_dim = _identify_dims(vo_section)
    except ValueError as e:
        print(f"  FAILED: {e}")
        return False
    n_times = vo_section.sizes[time_dim]
    print(f"  Section: {dict(vo_section.sizes)} ({n_times} timesteps)")

    # Grid metrics
    try:
        lon = get_lon_1d(vo_ds, j_idx, j_dim, x_dim)
        depth = get_depth(vo_ds)
    except ValueError as e:
        print(f"  FAILED: {e}")
        return False

    # Wrap longitude
    lon_wrapped = np.where(lon > 180, lon - 360, lon)

    # Atlantic mask
    atlantic_mask = (lon_wrapped >= ATLANTIC_LON_MIN) & (lon_wrapped <= ATLANTIC_LON_MAX)
    n_atl = atlantic_mask.sum()
    if n_atl < 5:
        print(f"  FAILED: only {n_atl} Atlantic points")
        return False
    print(f"  Atlantic points: {n_atl}/{len(lon)}")

    # e1t from longitude differences
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    dlon = (dlon + 180) % 360 - 180  # minimal angular difference
    cos_lat = np.cos(np.deg2rad(actual_lat))
    e1t = np.abs(dlon) * 111000.0 * cos_lat
    e1t = np.clip(e1t, 1.0, None)
    e1t_atl = e1t[atlantic_mask]

    # e3t from depth
    e3t = np.diff(depth, prepend=0.0)

    # Compute in batches to limit memory
    t0 = time.time()
    all_fovs = []
    all_times = []
    n_batches = (n_times + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, n_times, BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        end = min(i + BATCH_SIZE, n_times)
        try:
            vo_batch = vo_section.isel({time_dim: slice(i, end)}).load()
            so_batch = so_section.isel({time_dim: slice(i, end)}).load()
            batch_times = vo_batch[time_dim].values

            fovs_batch = compute_fovs_section_cloud(
                vo_batch, so_batch, e1t_atl, e3t, atlantic_mask, x_dim, z_dim
            )
            all_fovs.append(fovs_batch)
            all_times.append(batch_times)

            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n_times - end) / rate if rate > 0 else 0
            valid = np.isfinite(fovs_batch).sum()
            print(
                f"    Batch {batch_num}/{n_batches}: t[{i}:{end}], "
                f"{valid}/{len(fovs_batch)} valid ({elapsed:.0f}s, ~{eta:.0f}s rem)"
            )
        except Exception as e:
            print(f"    Batch {batch_num}/{n_batches}: FAILED ({e})")
            if all_fovs:
                print(f"    Continuing with {len(all_fovs)} batches")
            break

    if not all_fovs:
        print("  FAILED: no data computed")
        return False

    fovs_values = np.concatenate(all_fovs)
    times = np.concatenate(all_times)
    valid = np.isfinite(fovs_values).sum()
    elapsed = time.time() - t0

    fovs_da = xr.DataArray(
        fovs_values,
        dims=("time",),
        coords={"time": times},
        name="F_ovS",
        attrs={
            "units": "Sv",
            "long_name": "Overturning freshwater transport at 34.5S",
            "source_id": model,
            "experiment_id": experiment,
            "section_latitude": actual_lat,
        },
    )
    ds = fovs_da.to_dataset(name="F_ovS")
    ds.to_netcdf(outfile)
    mean_val = float(np.nanmean(fovs_values))
    print(
        f"  Saved: {outfile.name} — {valid}/{len(fovs_values)} valid, "
        f"mean={mean_val:.4f} Sv ({elapsed:.0f}s)"
    )
    return True


def concatenate_hist_ssp(
    model: str, results_dir: Path
) -> None:
    """Concatenate historical + SSP experiments into continuous series."""
    hist_file = results_dir / f"fovs_{model}_historical.nc"
    if not hist_file.exists():
        return

    hist = xr.open_dataset(hist_file)["F_ovS"]

    for ssp in ["ssp245", "ssp585"]:
        ssp_file = results_dir / f"fovs_{model}_{ssp}.nc"
        if not ssp_file.exists():
            continue
        proj = xr.open_dataset(ssp_file)["F_ovS"]
        combined = xr.concat([hist, proj], dim="time")
        # Remove duplicate times
        _, idx = np.unique(combined.time.values, return_index=True)
        combined = combined.isel(time=sorted(idx))
        outfile = results_dir / f"fovs_{model}_hist_{ssp}.nc"
        combined.to_dataset(name="F_ovS").to_netcdf(outfile)
        print(f"  Concatenated: {outfile.name} ({len(combined)} months)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CMIP6 F_ovS directly from Pangeo cloud zarr stores."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to compute. Default: all target models. Use 'all' for every model on Pangeo.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS,
    )
    parser.add_argument("--results-dir", default="data/results/cmip6")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output file exists.",
    )
    parser.add_argument(
        "--skip-concat",
        action="store_true",
        help="Skip historical+SSP concatenation step.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Opening Pangeo CMIP6 catalog...")
    try:
        cat = intake.open_esm_datastore(PANGEO_CATALOG_URL)
    except Exception as e:
        print(f"FAILED to open catalog: {e}")
        sys.exit(1)
    cat_df = cat.df
    print(f"Catalog: {len(cat_df)} entries")

    # Determine model list
    if args.models is None:
        models = TARGET_MODELS
    elif args.models == ["all"]:
        # Find all models with both vo and so on gn grid
        omon = cat_df[cat_df["table_id"] == "Omon"]
        has_vo = set(omon[omon["variable_id"] == "vo"]["source_id"])
        has_so = set(omon[omon["variable_id"] == "so"]["source_id"])
        gn = set(omon[omon["grid_label"] == "gn"]["source_id"])
        models = sorted(has_vo & has_so & gn)
        print(f"Found {len(models)} models with vo+so on gn grid")
    else:
        models = args.models

    print(f"Models: {models}")
    print(f"Experiments: {args.experiments}")
    print()

    successes = 0
    failures = 0
    skipped = 0
    t_start = time.time()

    for model in models:
        print(f"=== {model} ===")

        for experiment in args.experiments:
            print(f"  [{experiment}]")
            result = compute_model_experiment(
                cat_df, model, experiment, results_dir, force=args.force
            )
            if result:
                outfile = results_dir / f"fovs_{model}_{experiment}.nc"
                if outfile.exists():
                    successes += 1
                else:
                    skipped += 1
            else:
                failures += 1

        # Concatenate historical + SSP
        if not args.skip_concat:
            concatenate_hist_ssp(model, results_dir)

        print()

    total_time = time.time() - t_start
    print(
        f"Done in {total_time/60:.1f} min: "
        f"{successes} computed, {failures} failed"
    )


if __name__ == "__main__":
    main()
