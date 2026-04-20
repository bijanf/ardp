#!/usr/bin/env python3
"""Compute F_ovS time series from downloaded CMIP6 34.5S sections.

Reads the latitude slices produced by download_cmip6_fovs_sections.py and
computes the de Vries & Weber (2005) overturning freshwater transport for
each monthly timestep.

For each model, concatenates historical + ssp245 and historical + ssp585
into continuous 1850-2100 time series.

Output: data/results/cmip6/cmip6_fovs_timeseries.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0
from ardp.physics.fovs import compute_fovs_from_section


def _get_lon_and_depth(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1D longitude and depth arrays from a section dataset."""
    # Find longitude
    lon = None
    for name in ["longitude", "lon", "nav_lon", "x"]:
        if name in ds.coords or name in ds:
            vals = ds[name].values
            if vals.ndim == 1 and vals.size > 10:
                lon = vals
                break
    if lon is None:
        # Try any coord with 'lon' in name
        for name in ds.coords:
            if "lon" in name.lower():
                vals = ds[name].values
                if vals.ndim == 1 and vals.size > 10:
                    lon = vals
                    break
    if lon is None:
        raise ValueError(f"Cannot find longitude in {list(ds.coords)}")

    # Find depth
    depth = None
    for name in ["lev", "depth", "olevel", "deptht"]:
        if name in ds.coords or name in ds.dims:
            vals = ds[name].values if name in ds.coords else np.arange(ds.sizes[name])
            if vals.ndim == 1:
                depth = vals.astype(float)
                # Convert centimeters to meters if needed (e.g. CESM2 POP)
                units = ds[name].attrs.get("units", "m") if name in ds.coords else "m"
                if units in ("centimeters", "cm"):
                    depth = depth / 100.0
                break
    if depth is None:
        # Check for density coordinates (e.g., NorESM2 uses 'rho')
        for name in ["rho", "sigma", "rhopoto"]:
            if name in ds.coords or name in ds.dims:
                raise ValueError(
                    f"Vertical coordinate '{name}' is in density space — "
                    f"cannot compute e3t (depth thickness). Skipping."
                )
        raise ValueError(f"Cannot find depth in {list(ds.coords)}")

    return lon, depth


def compute_fovs_section(
    vo_ds: xr.Dataset,
    so_ds: xr.Dataset,
) -> xr.DataArray:
    """Compute F_ovS time series from vo and so section datasets.

    Both datasets should be 34.5S latitude slices with dims (time, depth/lev, x/lon).
    """
    var_vo = "vo"
    var_so = "so"

    lon, depth = _get_lon_and_depth(vo_ds)

    # Wrap longitudes to [-180, 180] if needed
    lon_wrapped = np.where(lon > 180, lon - 360, lon)

    # Atlantic mask
    atlantic_mask = (lon_wrapped >= ATLANTIC_LON_MIN) & (lon_wrapped <= ATLANTIC_LON_MAX)
    if atlantic_mask.sum() < 5:
        raise ValueError(f"Only {atlantic_mask.sum()} Atlantic points found")

    # Grid metrics
    actual_lat = float(vo_ds.attrs.get("section_latitude", -34.5))
    cos_lat = np.cos(np.deg2rad(actual_lat))

    # e1t: zonal grid spacing from longitude differences
    # Use raw (unwrapped) lon for dlon to avoid dateline/seam artifacts
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    # Handle wrap-around: angular differences should be small (~1 degree)
    # Map to [-180, 180] range to get minimal angular difference
    dlon = (dlon + 180) % 360 - 180
    e1t = np.abs(dlon) * 111000.0 * cos_lat
    e1t = np.clip(e1t, 1.0, None)
    e1t_atl = e1t[atlantic_mask]

    # e3t: vertical cell thickness from depth
    e3t = np.diff(depth, prepend=0.0)

    # Find spatial dimension names for vo
    vo_da = vo_ds[var_vo]
    so_da = so_ds[var_so]

    # Get the spatial (x) dimension — the one that's not time and not depth
    time_dim = "time"
    depth_dims = {"lev", "depth", "olevel", "deptht"}
    x_dim_vo = None
    for d in vo_da.dims:
        if d != time_dim and d not in depth_dims:
            x_dim_vo = d
            break
    z_dim_vo = None
    for d in vo_da.dims:
        if d in depth_dims:
            z_dim_vo = d
            break

    if x_dim_vo is None or z_dim_vo is None:
        raise ValueError(f"Cannot identify dims in vo: {vo_da.dims}")

    n_times = vo_da.sizes[time_dim]
    fovs_values = np.full(n_times, np.nan)

    # Process each timestep
    for t in range(n_times):
        try:
            v_full = vo_da.isel({time_dim: t}).values  # (depth, x)
            s_full = so_da.isel({time_dim: t}).values  # (depth, x)

            # Apply Atlantic mask
            v_section = v_full[:, atlantic_mask]
            s_section = s_full[:, atlantic_mask]

            fovs_values[t] = compute_fovs_from_section(
                v_section, s_section, e1t_atl, e3t, s0=S0,
            )

        except Exception:
            continue

    times = vo_da[time_dim].values
    return xr.DataArray(
        fovs_values,
        dims=("time",),
        coords={"time": times},
        name="F_ovS",
        attrs={"units": "Sv", "long_name": "Overturning freshwater transport at 34.5S"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CMIP6 F_ovS time series from downloaded sections."
    )
    parser.add_argument("--sections-dir", default="data/cmip6_sections")
    parser.add_argument("--results-dir", default="data/results/cmip6")
    args = parser.parse_args()

    sections_dir = Path(args.sections_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find all available model/experiment pairs
    vo_files = sorted(sections_dir.glob("*_vo.nc"))
    models_exps = {}
    for f in vo_files:
        parts = f.stem.rsplit("_", 1)  # e.g., "CESM2_historical_vo" -> "CESM2_historical"
        model_exp = parts[0]
        # Check if matching so file exists
        so_file = f.parent / f"{model_exp}_so.nc"
        if so_file.exists():
            # Parse model and experiment
            # model name may contain hyphens, experiment is last segment
            for exp in ["historical", "ssp245", "ssp585", "ssp585ext", "piControl"]:
                if model_exp.endswith(f"_{exp}"):
                    model = model_exp[: -len(f"_{exp}")]
                    if model not in models_exps:
                        models_exps[model] = []
                    models_exps[model].append(exp)
                    break

    if not models_exps:
        print("No section files found. Run download_cmip6_fovs_sections.py first.")
        return

    print(f"Found sections for {len(models_exps)} models:")
    for model, exps in sorted(models_exps.items()):
        print(f"  {model}: {exps}")
    print()

    all_results = {}

    for model, exps in sorted(models_exps.items()):
        print(f"=== {model} ===")

        for exp in exps:
            vo_file = sections_dir / f"{model}_{exp}_vo.nc"
            so_file = sections_dir / f"{model}_{exp}_so.nc"

            print(f"  Computing {exp}...")
            try:
                vo_ds = xr.open_dataset(vo_file)
                so_ds = xr.open_dataset(so_file)
                fovs = compute_fovs_section(vo_ds, so_ds)
                valid = np.isfinite(fovs.values).sum()
                print(f"    {valid}/{len(fovs)} valid timesteps")
                print(f"    Mean: {float(fovs.mean()):.4f} Sv, range: [{float(fovs.min()):.3f}, {float(fovs.max()):.3f}]")

                key = f"{model}_{exp}"
                all_results[key] = fovs
                vo_ds.close()
                so_ds.close()
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

        # Try to concatenate historical + SSP scenarios
        for ssp in ["ssp245", "ssp585"]:
            hist_key = f"{model}_historical"
            ssp_key = f"{model}_{ssp}"
            concat_key = f"{model}_hist_{ssp}"

            if hist_key in all_results and ssp_key in all_results:
                hist = all_results[hist_key]
                proj = all_results[ssp_key]
                # Concatenate, dropping any overlap
                combined = xr.concat([hist, proj], dim="time")
                # Remove duplicate times
                _, idx = np.unique(combined.time.values, return_index=True)
                combined = combined.isel(time=sorted(idx))
                all_results[concat_key] = combined
                print(f"  Concatenated {hist_key} + {ssp_key}: {len(combined)} months")

        print()

    # Save all time series
    if not all_results:
        print("No results to save.")
        return

    datasets = {}
    for key, fovs in all_results.items():
        # Store as individual NetCDF files (time axes differ)
        outfile = results_dir / f"fovs_{key}.nc"
        ds = fovs.to_dataset(name="F_ovS")
        ds.attrs["key"] = key
        ds.to_netcdf(outfile)
        datasets[key] = fovs
        print(f"Saved: {outfile.name}")

    # Also save a summary with annual means (easier to plot)
    print("\n=== Annual mean summary ===")
    summary_file = results_dir / "cmip6_fovs_annual_summary.nc"

    # Build a summary with common time axis (annual)
    annual_data = {}
    for key, fovs in all_results.items():
        try:
            annual = fovs.groupby("time.year").mean()
            annual_data[key] = annual
            years = annual.year.values
            mean_val = float(annual.mean())
            print(f"  {key}: {years[0]}-{years[-1]}, mean={mean_val:.4f} Sv")
        except Exception as e:
            print(f"  {key}: annual mean failed ({e})")

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
