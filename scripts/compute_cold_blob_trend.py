#!/usr/bin/env python3
"""Compute the per-grid-cell HadISST SST linear trend over the subpolar
North Atlantic, 1993-2024.  Output is a NetCDF file consumed by the
Cold-Blob panel of Figure 1.

Data source: HadISST monthly SST, Rayner et al. (2003), public Met Office
Hadley Centre product.  The file is staged at
/home/bijanf/Documents/NEW_Theory/data/external/hadisst/HadISST_sst.nc;
this script reads from there directly (no copy required).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
HADISST = Path(
    "/home/bijanf/Documents/NEW_Theory/data/external/hadisst/HadISST_sst.nc"
)
OUT = REPO / "data" / "results" / "cold_blob_trend_hadisst.nc"

LON_W, LON_E = -180.0, 180.0
LAT_S, LAT_N = -90.0, 90.0
YEAR_0, YEAR_1 = 1993, 2024  # legacy trend window
# Climatology-difference window: modern minus historical.  These windows
# match Caesar et al. (2018) / Rahmstorf et al. (2015) figure conventions
# and produce the canonical absolute-cooling Cold Blob signal.
MOD_Y0, MOD_Y1 = 2014, 2024
HIST_Y0, HIST_Y1 = 1900, 1960


def _trend_per_cell(years: np.ndarray, values: np.ndarray):
    """Return (slope per year, two-sided p-value) along the time axis.
    Cells with any NaN return NaN.
    """
    n_t, n_y, n_x = values.shape
    slope = np.full((n_y, n_x), np.nan, dtype=np.float64)
    pval = np.full((n_y, n_x), np.nan, dtype=np.float64)
    for j in range(n_y):
        for i in range(n_x):
            v = values[:, j, i]
            if not np.all(np.isfinite(v)):
                continue
            res = stats.linregress(years, v)
            slope[j, i] = res.slope
            pval[j, i] = res.pvalue
    return slope, pval


def main() -> int:
    print(f"Reading {HADISST}")
    ds = xr.open_dataset(HADISST)
    sst = ds["sst"]
    # HadISST land/sea mask uses -1000 sentinel; turn into NaN
    sst = sst.where(sst > -100)

    # Subset to user-specified region (default global)
    sst = sst.sel(
        latitude=slice(LAT_N, LAT_S),  # HadISST has lat decreasing
        longitude=slice(LON_W, LON_E),
    )
    # Annual means (full record)
    ann_full = sst.groupby("time.year").mean("time", skipna=True)

    # Legacy 1993-2024 linear trend per cell
    ann_trend = ann_full.sel(year=slice(YEAR_0, YEAR_1))
    years = ann_trend["year"].values.astype(np.float64)
    slope, pval = _trend_per_cell(years, ann_trend.values)
    slope_per_dec = slope * 10.0

    # Climatology-difference map: modern (1995-2024) minus historical (1900-1930)
    mod_mean = ann_full.sel(year=slice(MOD_Y0, MOD_Y1)).mean(
        "year", skipna=True).values
    hist_mean = ann_full.sel(year=slice(HIST_Y0, HIST_Y1)).mean(
        "year", skipna=True).values
    delta_sst = mod_mean - hist_mean

    lat = ann_full["latitude"].values
    lon = ann_full["longitude"].values

    out_ds = xr.Dataset(
        data_vars={
            "trend": (("latitude", "longitude"), slope_per_dec,
                       {"units": "degC per decade",
                        "description": (f"OLS linear trend per grid cell, "
                                        f"{YEAR_0}-{YEAR_1}")}),
            "pvalue": (("latitude", "longitude"), pval,
                        {"description": "Two-sided OLS p-value"}),
            "delta_sst": (("latitude", "longitude"), delta_sst,
                          {"units": "degC",
                           "description": (f"SST difference: {MOD_Y0}-{MOD_Y1} "
                                           f"mean minus {HIST_Y0}-{HIST_Y1} "
                                           f"mean (climatology change)")}),
        },
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "source": str(HADISST),
            "data_product": "HadISST (Rayner et al. 2003)",
            "period_start": YEAR_0,
            "period_end": YEAR_1,
            "modern_window": f"{MOD_Y0}-{MOD_Y1}",
            "historical_window": f"{HIST_Y0}-{HIST_Y1}",
        },
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(OUT)
    finite_d = np.isfinite(delta_sst)
    print(f"Wrote {OUT}")
    print(f"  Grid: {len(lat)} lat x {len(lon)} lon")
    print(f"  Ocean cells: {finite_d.sum()} of {finite_d.size}")
    print(f"  delta_sst ({MOD_Y0}-{MOD_Y1} minus {HIST_Y0}-{HIST_Y1}):")
    print(f"    mean {np.nanmean(delta_sst):+.3f} degC, "
          f"min {np.nanmin(delta_sst):+.3f}, max {np.nanmax(delta_sst):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
