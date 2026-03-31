#!/usr/bin/env python3
"""Compute yearly AMOC strength at 26.5°N from monthly data.

AMOC(26.5N) = max_z Ψ(z) at latitude nearest 26.5°N, depth 500–4000m.
This is directly comparable to RAPID array observations.

Output: data/results/yearly_amoc26n_{product}.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import RAPID_LAT
from ardp.spatial.regions import atlantic_lon_bounds


def amoc_at_26n(v_zonal_annual, depth, lat):
    """Compute AMOC at 26.5°N from annual-mean v_zonal(z, y).

    Returns max of streamfunction over depth 500–4000m at 26.5°N.
    """
    j = np.argmin(np.abs(lat - RAPID_LAT))
    dz = np.diff(depth, prepend=0.0)
    v = np.where(np.isfinite(v_zonal_annual), v_zonal_annual, 0.0)
    transport = v * dz[:, np.newaxis]
    psi = np.cumsum(transport, axis=0) / 1e6  # Sv

    depth_mask = (depth >= 500) & (depth <= 4000)
    return float(np.nanmax(psi[depth_mask, j]))


def compute_oras5(data_dir):
    """Yearly AMOC at 26.5°N from ORAS5."""
    files = sorted(data_dir.glob("vomecrty_control_monthly_highres_3D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No vomecrty files in {data_dir}")

    file_years = {}
    for f in files:
        for p in f.stem.split("_"):
            if len(p) == 6 and p.isdigit():
                file_years[f] = int(p[:4])
                break

    ds0 = xr.open_dataset(files[0])
    nav_lon = ds0["nav_lon"].values
    nav_lat = ds0["nav_lat"].values
    depth = ds0["depthv"].values
    ny, nx = nav_lon.shape
    nz = len(depth)
    ds0.close()

    lat_1d = np.nanmean(nav_lat, axis=1)

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

    weight = dx * atl_mask

    years_set = sorted(set(file_years.values()))
    print(f"  ORAS5: {years_set[0]}–{years_set[-1]}, {len(years_set)} years")

    all_years, all_amoc = [], []
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
        amoc = amoc_at_26n(v_zonal_mean, depth, lat_1d)
        all_years.append(yr)
        all_amoc.append(amoc)
        if yr % 10 == 0 or yr == years_set[-1]:
            print(f"    {yr}: AMOC(26.5N) = {amoc:.1f} Sv", flush=True)

    return np.array(all_years), np.array(all_amoc)


def compute_glorys12(data_dir):
    """Yearly AMOC at 26.5°N from GLORYS12."""
    import os
    files = sorted([f for f in os.listdir(data_dir)
                    if f.startswith("glorys12_") and f.endswith(".nc")])
    if not files:
        raise FileNotFoundError(f"No glorys12 files in {data_dir}")

    ds0 = xr.open_dataset(f"{data_dir}/{files[0]}")[["vo"]]
    lat = ds0["latitude"].values
    depth = ds0["depth"].values
    lon = ds0["longitude"].values
    ds0.close()

    dlon = abs(np.mean(np.diff(lon)))
    dx_1d = dlon * 111000.0 * np.cos(np.deg2rad(lat))
    ny, nx = len(lat), len(lon)

    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        if lat[j] < -55 or lat[j] > 70:
            continue
        lo, hi = atlantic_lon_bounds(float(lat[j]))
        atl_mask[j, :] = (lon >= lo) & (lon <= hi)

    all_years, all_amoc = [], []
    for f in files:
        yr = int(f.split("_")[1].split(".")[0])
        ds = xr.open_dataset(f"{data_dir}/{f}")[["vo"]]
        vo = ds["vo"].values
        ds.close()

        v_zonal_sum = np.zeros((len(depth), ny), dtype=np.float64)
        count = 0
        for m in range(vo.shape[0]):
            v = vo[m]
            v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)
            v_masked = v * atl_mask[np.newaxis, :, :]
            v_zonal = np.nansum(v_masked, axis=2) * dx_1d[np.newaxis, :]
            v_zonal_sum += v_zonal
            count += 1
        del vo

        v_zonal_mean = v_zonal_sum / count
        amoc = amoc_at_26n(v_zonal_mean, depth, lat)
        all_years.append(yr)
        all_amoc.append(amoc)
        if yr % 5 == 0:
            print(f"    {yr}: AMOC(26.5N) = {amoc:.1f} Sv", flush=True)

    return np.array(all_years), np.array(all_amoc)


def compute_cmip6(data_dir):
    """Yearly AMOC at 26.5°N from all CMIP6 models."""
    from ardp.models import models_sorted_by_fovs
    models = models_sorted_by_fovs()

    results = {}
    for model, fovs in models:
        hist_file = data_dir / f"{model}_historical_vo_zonal.nc"
        ssp_file = data_dir / f"{model}_ssp585_vo_zonal.nc"

        datasets = []
        for f in [hist_file, ssp_file]:
            if f.exists():
                datasets.append(xr.open_dataset(f))
        if not datasets:
            print(f"  {model}: SKIP")
            continue

        combined = xr.concat([ds["v_zonal"] for ds in datasets], dim="time")
        _, idx = np.unique(combined.time.values, return_index=True)
        combined = combined.isel(time=sorted(idx))

        depth = datasets[0]["depth"].values
        lat = datasets[0]["lat"].values

        times = combined.time.values
        try:
            years = np.array([t.year for t in times])
        except AttributeError:
            import pandas as pd
            years = pd.DatetimeIndex(times).year.values

        v_data = combined.values
        dz = np.diff(depth, prepend=0.0)
        j26 = np.argmin(np.abs(lat - RAPID_LAT))
        depth_mask = (depth >= 500) & (depth <= 4000)

        years_set = sorted(set(years))
        yr_out, amoc_out = [], []
        for yr in years_set:
            mask = years == yr
            v_annual = np.nanmean(v_data[mask], axis=0)
            v_annual = np.where(np.isfinite(v_annual), v_annual, 0.0)
            transport = v_annual * dz[:, np.newaxis]
            psi = np.cumsum(transport, axis=0) / 1e6
            amoc = float(np.nanmax(psi[depth_mask, j26]))
            yr_out.append(yr)
            amoc_out.append(amoc)

        results[model] = {
            "years": np.array(yr_out),
            "amoc": np.array(amoc_out),
            "fovs": fovs,
        }
        print(f"  {model:25s} {len(yr_out)} years, AMOC(26.5N): {amoc_out[0]:.1f}→{amoc_out[-1]:.1f} Sv")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True, choices=["oras5", "glorys12", "cmip6"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.product == "oras5":
        print("Computing yearly AMOC(26.5N) from ORAS5...")
        years, amoc = compute_oras5(Path("data/oras5"))
        np.savez_compressed(args.output_dir / "yearly_amoc26n_oras5.npz",
                            years=years, amoc=amoc)
        print(f"  Saved ({len(years)} years)")

    elif args.product == "glorys12":
        print("Computing yearly AMOC(26.5N) from GLORYS12...")
        years, amoc = compute_glorys12(Path("data/glorys12"))
        np.savez_compressed(args.output_dir / "yearly_amoc26n_glorys12.npz",
                            years=years, amoc=amoc)
        print(f"  Saved ({len(years)} years)")

    elif args.product == "cmip6":
        print("Computing yearly AMOC(26.5N) from CMIP6...")
        results = compute_cmip6(Path("data/cmip6_fullfield"))
        save_dict = {"models": np.array(list(results.keys()))}
        for model, d in results.items():
            save_dict[f"{model}_years"] = d["years"]
            save_dict[f"{model}_amoc"] = d["amoc"]
            save_dict[f"{model}_fovs"] = np.array([d["fovs"]])
        np.savez_compressed(args.output_dir / "yearly_amoc26n_cmip6.npz", **save_dict)
        print(f"  Saved ({len(results)} models)")


if __name__ == "__main__":
    main()
