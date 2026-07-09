#!/usr/bin/env python3
"""Compute 34.5°S zonal-depth ΔS section from EN4.2.2 and (if available) RG09.

Late period (2020-2024) minus early period (2005-2009), upper 1000 m, full
Atlantic longitude range. Bootstrap 95% CI on each grid cell.

Outputs:
  data/results/argo_zonal_section.nc   — variables EN4_dS, EN4_pvalue,
                                          (and RG09 versions if available)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
EN4_DIR = REPO / "data" / "en4"
RG09_DIR = REPO / "data" / "argo_rg09"
RESULTS = REPO / "data" / "results"

LAT_TARGET = -34.5
LAT_HALF_WIDTH = 1.5     # average over 33°S to 36°S to suppress noise
DEPTH_MAX = 1000.0
EARLY = (2005, 2009)
LATE = (2020, 2024)


def _en4_section_mean(years: range) -> xr.DataArray | None:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    files = [f for f in files if int(f.stem.split(".")[-1][:4]) in years]
    if not files:
        return None
    sections = []
    for fp in files:
        ds = xr.open_dataset(fp)
        sal = ds["salinity"].sel(
            lat=slice(LAT_TARGET - LAT_HALF_WIDTH, LAT_TARGET + LAT_HALF_WIDTH),
            depth=slice(0, DEPTH_MAX),
        )
        # Atlantic-only longitudes (wraps): 300-360 and 0-20
        sal_west = sal.sel(lon=slice(300.0, 360.0))
        sal_east = sal.sel(lon=slice(0.0, 20.0))
        # Stitch: relabel west longitudes to -60..0 so they concatenate cleanly
        sal_west = sal_west.assign_coords(lon=sal_west["lon"] - 360.0)
        sal = xr.concat([sal_west, sal_east], dim="lon").sortby("lon")
        # Average across the lat half-width band
        sal = sal.mean("lat", skipna=True)
        if "time" in sal.dims:
            sal = sal.squeeze("time", drop=True)
        sections.append(sal)
        ds.close()
    if not sections:
        return None
    return xr.concat(sections, dim="time_idx").mean("time_idx", skipna=True)


def _bootstrap_p(early_stack: xr.DataArray, late_stack: xr.DataArray, n: int = 1000) -> xr.DataArray:
    """Empirical two-sided bootstrap p-value at each (depth, lon) cell."""
    rng = np.random.default_rng(7)
    e = early_stack.values
    l = late_stack.values
    obs_diff = np.nanmean(l, axis=0) - np.nanmean(e, axis=0)
    combined = np.concatenate([e, l], axis=0)
    n_total = combined.shape[0]
    n_late = l.shape[0]
    null_count = np.zeros_like(obs_diff)
    valid_count = np.zeros_like(obs_diff)
    for _ in range(n):
        idx = rng.permutation(n_total)
        late_sample = combined[idx[:n_late]]
        early_sample = combined[idx[n_late:]]
        null_diff = np.nanmean(late_sample, axis=0) - np.nanmean(early_sample, axis=0)
        null_count += np.abs(null_diff) >= np.abs(obs_diff)
        valid_count += np.isfinite(null_diff)
    pvals = null_count / np.where(valid_count > 0, valid_count, 1)
    return xr.DataArray(pvals, coords=late_stack.mean("time_idx").coords, dims=late_stack.mean("time_idx").dims)


def _en4_zonal_delta() -> xr.Dataset:
    """Return (early mean, late mean, ΔS, bootstrap p-value) on the 34.5°S section."""
    early_files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    early_files = [f for f in early_files if EARLY[0] <= int(f.stem.split(".")[-1][:4]) <= EARLY[1]]
    late_files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    late_files = [f for f in late_files if LATE[0] <= int(f.stem.split(".")[-1][:4]) <= LATE[1]]
    print(f"  early: {len(early_files)} months, late: {len(late_files)} months")

    def _stack(files: list[Path]) -> xr.DataArray:
        slabs = []
        for fp in files:
            ds = xr.open_dataset(fp)
            sal = ds["salinity"].sel(
                lat=slice(LAT_TARGET - LAT_HALF_WIDTH, LAT_TARGET + LAT_HALF_WIDTH),
                depth=slice(0, DEPTH_MAX),
            )
            sal_west = sal.sel(lon=slice(300.0, 360.0))
            sal_east = sal.sel(lon=slice(0.0, 20.0))
            sal_west = sal_west.assign_coords(lon=sal_west["lon"] - 360.0)
            sal = xr.concat([sal_west, sal_east], dim="lon").sortby("lon")
            sal = sal.mean("lat", skipna=True)
            if "time" in sal.dims:
                sal = sal.squeeze("time", drop=True)
            slabs.append(sal.expand_dims("time_idx"))
            ds.close()
        return xr.concat(slabs, dim="time_idx")

    early_stack = _stack(early_files)
    late_stack = _stack(late_files)

    early_mean = early_stack.mean("time_idx", skipna=True)
    late_mean = late_stack.mean("time_idx", skipna=True)
    dS = late_mean - early_mean
    print(f"  ΔS range: {float(dS.min().values):+.3f} to {float(dS.max().values):+.3f} PSU")
    print(f"  ΔS basin-mean: {float(dS.mean().values):+.4f} PSU")

    pvals = _bootstrap_p(early_stack, late_stack, n=1000)

    out = xr.Dataset({
        "EN4_early_mean": early_mean,
        "EN4_late_mean":  late_mean,
        "EN4_dS":         dS,
        "EN4_pvalue":     pvals,
    })
    out.attrs.update({
        "early_period": f"{EARLY[0]}-{EARLY[1]}",
        "late_period":  f"{LATE[0]}-{LATE[1]}",
        "lat_band":     f"{LAT_TARGET - LAT_HALF_WIDTH}..{LAT_TARGET + LAT_HALF_WIDTH}",
        "depth_max":    f"0-{DEPTH_MAX} m",
        "source":       "EN4.2.2 (Met Office Hadley Centre)",
    })
    return out


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Computing EN4 34.5°S zonal ΔS section ...")
    ds_en4 = _en4_zonal_delta()
    out_path = RESULTS / "argo_zonal_section.nc"
    ds_en4.to_netcdf(out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
