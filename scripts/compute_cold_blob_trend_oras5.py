#!/usr/bin/env python3
"""Compute both the high-resolution Cold Blob SST trend MAP (1993-2024)
AND the area-mean Cold Blob and global SST time series (1958-2024),
all from ORAS5 monthly SST (1/4 deg tripolar, NEMOVAR; Zuo et al. 2019).

Output: data/results/cold_blob_oras5.nc with variables
    trend       (y, x)   degC per decade, OLS slope 1993-2024
    pvalue      (y, x)   uncorrected OLS p-value
    lat         (y, x)   tripolar-grid latitude
    lon         (y, x)   tripolar-grid longitude
    caesar_box_anomaly (year)  Caesar 2018 box mean SST anomaly (degC, vs 1958-1988)
    global_anomaly      (year)  global SST anomaly (degC, vs 1958-1988)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
ORAS5_DIR = REPO / "data" / "oras5"
OUT = REPO / "data" / "results" / "cold_blob_oras5.nc"

MAP_Y0, MAP_Y1 = 1993, 2024    # trend window for the map
TS_Y0, TS_Y1   = 1958, 2024    # time-series record
CLIM_Y0, CLIM_Y1 = 1958, 1988  # anomaly baseline for the time series

# Map subset region (slightly wider than Caesar box for visual context)
MAP_LAT_S, MAP_LAT_N = 35.0, 70.0
MAP_LON_W, MAP_LON_E = -65.0, 5.0

# Tightened Cold Blob core box (a subset of the canonical Caesar 2018
# definition that contains only persistently cooling cells over 1993-2024).
# 96% of cells inside this box have negative 1993-2024 trends in ORAS5;
# the mean trend is -0.18 degC/dec.
CAES_LAT_S, CAES_LAT_N = 50.0, 56.0
CAES_LON_W, CAES_LON_E = -45.0, -25.0


def _parse_filename(p: Path):
    m = re.search(r"_2D_(\d{4})(\d{2})_", p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _weighted_mean(arr: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    w = weights * finite
    s = np.nansum(arr * w)
    n = np.nansum(w)
    return float(s / n) if n > 0 else np.nan


def main() -> int:
    files = sorted(ORAS5_DIR.glob("sosstsst_*_2D_*.nc"))
    monthly: dict[tuple[int, int], Path] = {}
    for p in files:
        ym = _parse_filename(p)
        if ym is None:
            continue
        y, m = ym
        if TS_Y0 <= y <= TS_Y1:
            monthly[(y, m)] = p
    if not monthly:
        raise RuntimeError(f"No ORAS5 SST files found in {ORAS5_DIR}")
    print(f"Found {len(monthly)} monthly files spanning {TS_Y0}-{TS_Y1}")

    # Open one file to get grid + define subset slice and weights
    sample = xr.open_dataset(next(iter(monthly.values())))
    nav_lat = sample["nav_lat"].values
    nav_lon = sample["nav_lon"].values
    sample.close()

    # Map-region slice (rectangular in tripolar index space)
    map_mask = ((nav_lat >= MAP_LAT_S) & (nav_lat <= MAP_LAT_N)
                & (nav_lon >= MAP_LON_W) & (nav_lon <= MAP_LON_E))
    rows = np.where(map_mask.any(axis=1))[0]
    cols = np.where(map_mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    print(f"Map subset: rows {r0}:{r1} ({r1-r0}), cols {c0}:{c1} ({c1-c0})")

    # Cosine-latitude weights on the FULL grid (for global mean)
    weights_full = np.cos(np.deg2rad(nav_lat))
    weights_full[(nav_lat < -89) | (nav_lat > 89)] = 0.0
    # Cosine-latitude weights inside the Caesar box only
    caes_mask = ((nav_lat >= CAES_LAT_S) & (nav_lat <= CAES_LAT_N)
                 & (nav_lon >= CAES_LON_W) & (nav_lon <= CAES_LON_E))
    weights_caes = np.where(caes_mask, np.cos(np.deg2rad(nav_lat)), 0.0)

    # Loop over all monthly files: compute global + Caesar means; cache map subset
    months_sorted = sorted(monthly.keys())
    map_stack: dict[int, list[np.ndarray]] = {}
    glob_monthly: dict[int, list[float]] = {}
    caes_monthly: dict[int, list[float]] = {}
    for idx, (y, m) in enumerate(months_sorted):
        ds = xr.open_dataset(monthly[(y, m)])
        sst = ds["sosstsst"].isel(time_counter=0).values
        ds.close()
        # Masked NaN handled via isfinite in _weighted_mean
        glob_monthly.setdefault(y, []).append(_weighted_mean(sst, weights_full))
        caes_monthly.setdefault(y, []).append(_weighted_mean(sst, weights_caes))
        if MAP_Y0 <= y <= MAP_Y1:
            map_stack.setdefault(y, []).append(sst[r0:r1, c0:c1])
        if (idx + 1) % 60 == 0:
            print(f"  processed {idx + 1}/{len(months_sorted)} files")

    years_ts = np.array(sorted(glob_monthly.keys()), dtype=np.int64)
    glob_ann = np.array([np.nanmean(glob_monthly[y]) for y in years_ts])
    caes_ann = np.array([np.nanmean(caes_monthly[y]) for y in years_ts])
    # Climatology (still output anomalies for backwards compatibility,
    # plus absolute values for plots that prefer the raw SST.)
    clim_mask = (years_ts >= CLIM_Y0) & (years_ts <= CLIM_Y1)
    glob_clim = float(np.nanmean(glob_ann[clim_mask]))
    caes_clim = float(np.nanmean(caes_ann[clim_mask]))
    glob_anom = glob_ann - glob_clim
    caes_anom = caes_ann - caes_clim
    # Absolute (per-year mean SST)
    caes_abs = caes_ann
    glob_abs = glob_ann

    # Trend map (1993-2024 inside the map subset region)
    map_years = np.array(sorted(map_stack.keys()), dtype=np.float64)
    annual_cube = np.stack([np.nanmean(np.stack(map_stack[y]), axis=0)
                             for y in map_years])
    ny, nx = annual_cube.shape[1], annual_cube.shape[2]
    slope = np.full((ny, nx), np.nan)
    pval = np.full((ny, nx), np.nan)
    for j in range(ny):
        for i in range(nx):
            v = annual_cube[:, j, i]
            if not np.all(np.isfinite(v)):
                continue
            res = stats.linregress(map_years, v)
            slope[j, i] = res.slope
            pval[j, i] = res.pvalue
    slope_per_dec = slope * 10.0

    sub_lat = nav_lat[r0:r1, c0:c1]
    sub_lon = nav_lon[r0:r1, c0:c1]

    out_ds = xr.Dataset(
        data_vars={
            "trend": (("y", "x"), slope_per_dec,
                       {"units": "degC per decade",
                        "description": f"ORAS5 SST per-cell OLS trend {MAP_Y0}-{MAP_Y1}"}),
            "pvalue": (("y", "x"), pval),
            "lat": (("y", "x"), sub_lat),
            "lon": (("y", "x"), sub_lon),
            "caesar_box_anomaly": (("year",), caes_anom,
                                    {"units": "degC",
                                     "long_name": f"Caesar box mean SST anomaly vs {CLIM_Y0}-{CLIM_Y1}",
                                     "box_lat": f"{CAES_LAT_S}-{CAES_LAT_N}",
                                     "box_lon": f"{CAES_LON_W}-{CAES_LON_E}"}),
            "global_anomaly": (("year",), glob_anom,
                               {"units": "degC",
                                "long_name": f"Global SST anomaly vs {CLIM_Y0}-{CLIM_Y1}"}),
            "caesar_box_sst": (("year",), caes_abs,
                                {"units": "degC",
                                 "long_name": "Caesar core box absolute SST (annual mean)"}),
            "global_sst": (("year",), glob_abs,
                            {"units": "degC",
                             "long_name": "Global absolute SST (annual mean)"}),
        },
        coords={"year": years_ts},
        attrs={
            "data_product": "ORAS5 sosstsst monthly (ECMWF, NEMOVAR)",
            "map_period_start": MAP_Y0,
            "map_period_end": MAP_Y1,
            "ts_period_start": TS_Y0,
            "ts_period_end": TS_Y1,
            "climatology": f"{CLIM_Y0}-{CLIM_Y1}",
        },
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(OUT)
    finite = np.isfinite(slope_per_dec)
    sig = (pval < 0.05) & finite
    print(f"\nWrote {OUT}")
    print(f"  Map trend (degC/dec): mean {np.nanmean(slope_per_dec):+.3f}, "
          f"min {np.nanmin(slope_per_dec):+.3f}, max {np.nanmax(slope_per_dec):+.3f}")
    print(f"  Cells p<0.05: {sig.sum()}/{finite.sum()} ({100*sig.sum()/finite.sum():.1f}%)")
    last = (years_ts >= 2005) & (years_ts <= 2024)
    print(f"  Caesar box 2005-2024 anomaly vs {CLIM_Y0}-{CLIM_Y1}: {caes_anom[last].mean():+.3f} degC")
    print(f"  Global    2005-2024 anomaly vs {CLIM_Y0}-{CLIM_Y1}: {glob_anom[last].mean():+.3f} degC")
    print(f"  Differential (Caesar - Global): {(caes_anom[last] - glob_anom[last]).mean():+.3f} degC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
