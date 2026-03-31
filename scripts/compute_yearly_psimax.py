#!/usr/bin/env python3
"""Compute yearly Ψ_max from monthly data for ORAS5 and CMIP6 models.

For each year, computes annual-mean zonally-integrated velocity,
then streamfunction, then extracts Ψ_max (0-60°N, 500-4000m).

Output: data/results/yearly_psimax_{product}.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.models import models_sorted_by_fovs
from ardp.spatial.regions import atlantic_lon_bounds


def psi_max_from_vzonal(v_zonal_annual, depth, lat):
    """Compute Ψ_max from annual-mean v_zonal(z, y)."""
    dz = np.diff(depth, prepend=0.0)
    v = np.where(np.isfinite(v_zonal_annual), v_zonal_annual, 0.0)
    transport = v * dz[:, np.newaxis]
    psi = np.cumsum(transport, axis=0) / 1e6  # Sv

    lat_mask = (lat >= 0) & (lat <= 60)
    depth_mask = (depth >= 500) & (depth <= 4000)
    return np.nanmax(psi[np.ix_(depth_mask, lat_mask)])


def compute_oras5_yearly(data_dir):
    """Compute yearly Ψ_max from ORAS5 monthly vomecrty files."""
    files = sorted(data_dir.glob("vomecrty_control_monthly_highres_3D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No vomecrty files in {data_dir}")

    # Parse years
    file_years = {}
    for f in files:
        for p in f.stem.split("_"):
            if len(p) == 6 and p.isdigit():
                file_years[f] = int(p[:4])
                break

    # Read grid
    ds0 = xr.open_dataset(files[0])
    nav_lon = ds0["nav_lon"].values
    nav_lat = ds0["nav_lat"].values
    depth = ds0["depthv"].values
    ny, nx = nav_lon.shape
    nz = len(depth)
    ds0.close()

    lat_1d = np.nanmean(nav_lat, axis=1)

    # Grid metrics + Atlantic mask
    dlon = np.diff(nav_lon, axis=1)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dlon = np.concatenate([dlon, dlon[:, -1:]], axis=1)
    cos_lat = np.cos(np.deg2rad(nav_lat))
    dx = np.abs(dlon) * 111000.0 * cos_lat
    dx = np.clip(dx, 1.0, None)

    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        if lat_1d[j] < -55 or lat_1d[j] > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(lat_1d[j])
        atl_mask[j, :] = (nav_lon[j, :] >= lon_min) & (nav_lon[j, :] <= lon_max)

    weight = (dx * atl_mask)  # (ny, nx)

    # Group files by year
    years_set = sorted(set(file_years.values()))
    print(f"  ORAS5: {years_set[0]}–{years_set[-1]}, {len(years_set)} years")

    all_years = []
    all_psimax = []

    for yr in years_set:
        yr_files = sorted(f for f, y in file_years.items() if y == yr)
        v_zonal_sum = np.zeros((nz, ny), dtype=np.float64)
        count = 0
        for f in yr_files:
            ds = xr.open_dataset(f)
            v = ds["vomecrty"].values[0]
            ds.close()
            v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)
            v_zonal_sum += np.nansum(v * weight[np.newaxis, :, :], axis=2)
            count += 1

        v_zonal_mean = v_zonal_sum / count
        pm = psi_max_from_vzonal(v_zonal_mean, depth, lat_1d)
        all_years.append(yr)
        all_psimax.append(pm)

        if yr % 10 == 0 or yr == years_set[-1]:
            print(f"    {yr}: Ψ_max = {pm:.1f} Sv", flush=True)

    return np.array(all_years), np.array(all_psimax)


def compute_cmip6_yearly(data_dir, model):
    """Compute yearly Ψ_max from CMIP6 v_zonal files (hist + ssp585)."""
    hist_file = data_dir / f"{model}_historical_vo_zonal.nc"
    ssp_file = data_dir / f"{model}_ssp585_vo_zonal.nc"

    datasets = []
    for f in [hist_file, ssp_file]:
        if f.exists():
            datasets.append(xr.open_dataset(f))

    if not datasets:
        return None, None

    combined = xr.concat([ds["v_zonal"] for ds in datasets], dim="time")
    _, idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=sorted(idx))

    depth = datasets[0]["depth"].values
    lat = datasets[0]["lat"].values

    # Extract years
    times = combined.time.values
    try:
        years = np.array([t.year for t in times])
    except AttributeError:
        import pandas as pd
        years = pd.DatetimeIndex(times).year.values

    v_data = combined.values  # (time, nz, ny)

    years_set = sorted(set(years))
    all_years = []
    all_psimax = []

    for yr in years_set:
        mask = years == yr
        v_annual = np.nanmean(v_data[mask], axis=0)
        v_annual = np.where(np.isfinite(v_annual), v_annual, 0.0)

        dz = np.diff(depth, prepend=0.0)
        transport = v_annual * dz[:, np.newaxis]
        psi = np.cumsum(transport, axis=0) / 1e6

        lat_mask = (lat >= 0) & (lat <= 60)
        depth_mask = (depth >= 500) & (depth <= 4000)
        pm = np.nanmax(psi[np.ix_(depth_mask, lat_mask)])

        all_years.append(yr)
        all_psimax.append(pm)

    return np.array(all_years), np.array(all_psimax)


CMIP6_MODELS = models_sorted_by_fovs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True, choices=["oras5", "cmip6"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.product == "oras5":
        print("Computing yearly Ψ_max from ORAS5...")
        years, psimax = compute_oras5_yearly(Path("data/oras5"))
        outfile = args.output_dir / "yearly_psimax_oras5.npz"
        np.savez_compressed(outfile, years=years, psimax=psimax)
        print(f"  Saved: {outfile}")

    elif args.product == "cmip6":
        data_dir = Path("data/cmip6_fullfield")
        results = {}
        for model, fovs in CMIP6_MODELS:
            print(f"  {model}...", end=" ", flush=True)
            yrs, pm = compute_cmip6_yearly(data_dir, model)
            if yrs is not None:
                results[model] = {"years": yrs, "psimax": pm, "fovs": fovs}
                print(f"{len(yrs)} years, {pm[0]:.1f}→{pm[-1]:.1f} Sv")
            else:
                print("SKIP")

        outfile = args.output_dir / "yearly_psimax_cmip6.npz"
        save_dict = {}
        for model, d in results.items():
            save_dict[f"{model}_years"] = d["years"]
            save_dict[f"{model}_psimax"] = d["psimax"]
            save_dict[f"{model}_fovs"] = np.array([d["fovs"]])
        save_dict["models"] = np.array(list(results.keys()))
        np.savez_compressed(outfile, **save_dict)
        print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
