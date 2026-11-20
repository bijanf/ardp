#!/usr/bin/env python3
"""Upper-300 m box-mean salinity trends of ORAS5 and GLORYS12V1 (Table 1 rows).

Applies the procedure used for the Argo-based rows of Table 1
(scripts/compute_argo_sa_trend.py): box 29.5S to 39.5S, 60W to 20E, 0 to 300 m;
monthly fields averaged to annual means; basin mean taken as a uniform mean over
longitude, a cos(latitude)-weighted mean over latitude and a uniform mean over
the model levels above 300 m. A layer-thickness-weighted depth mean is computed
alongside. Trends are ordinary least squares with the Santer et al. (2000)
effective-sample-size 95% interval.

Periods: 1993-2025 (satellite-altimetry era) and 2005-2024 (Argo era, same period
as the Argo-based rows).

Run from the repository root:
    python scripts/compute_reanalysis_box_salinity_trend.py --out data/results
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

LAT_MIN, LAT_MAX = -39.5, -29.5
LON_MIN, LON_MAX = -60.0, 20.0
DEPTH_MAX = 300.0
PERIODS = {"1993-2025": (1993, 2025), "2005-2024": (2005, 2024)}


def santer_trend(years: np.ndarray, values: np.ndarray, alpha: float = 0.05) -> dict:
    yrs = years.astype(float) - years.mean()
    n = len(values)
    slope, intercept, _, p_raw, _ = stats.linregress(yrs, values)
    resid = values - (slope * yrs + intercept)
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1] if n > 2 else 0.0
    n_eff = n * (1.0 - r1) / (1.0 + r1) if r1 < 1 else max(1, n // 4)
    n_eff = max(2.1, min(n, n_eff))
    se = np.sqrt(np.sum(resid**2) / (n_eff - 2)) / np.sqrt(np.sum(yrs**2))
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=int(n_eff) - 2)
    return {
        "trend_psu_per_decade": float(10 * slope),
        "ci95_half_psu_per_decade": float(10 * t_crit * se),
        "r1": float(r1),
        "n_eff": float(n_eff),
        "n_years": int(n),
        "p_raw": float(p_raw),
    }


def thickness(depth: np.ndarray) -> np.ndarray:
    edges = np.concatenate([[0.0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1] + 0.5 * (depth[-1] - depth[-2])]])
    return np.diff(edges)


def box_means(sal: np.ndarray, lat_rows: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """sal (depth, lat, lon) with NaN on land; returns (level-mean, thickness-weighted)."""
    w = np.cos(np.deg2rad(lat_rows))
    lonmean = np.nanmean(sal, axis=2)  # (depth, lat)
    valid = np.isfinite(lonmean)
    band = np.nansum(lonmean * w[None, :], axis=1) / np.sum(np.where(valid, w[None, :], 0.0), axis=1)
    dz = thickness(depth)
    ok = np.isfinite(band)
    return float(np.mean(band[ok])), float(np.sum(band[ok] * dz[ok]) / np.sum(dz[ok]))


def oras5_annual(root: Path, years: range) -> dict[int, tuple[float, float]]:
    files = {}
    for f in sorted(root.glob("vosaline_control_monthly_highres_3D_*.nc")):
        ym = f.name.split("_")[-3]
        files.setdefault(int(ym[:4]), []).append(f)
    g = xr.open_dataset(files[years[0]][0])
    lat = g["nav_lat"].values
    lon = g["nav_lon"].values
    lon = np.where(lon > 180, lon - 360, lon)
    depth = g["deptht"].values
    g.close()
    rows = np.where(np.any((lat >= LAT_MIN) & (lat <= LAT_MAX), axis=1))[0]
    cols = np.where(np.any((lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX), axis=0))[0]
    ys, xs = slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)
    lat_s, lon_s = lat[ys, xs], lon[ys, xs]
    inbox = (lat_s >= LAT_MIN) & (lat_s <= LAT_MAX) & (lon_s >= LON_MIN) & (lon_s <= LON_MAX)
    kz = np.where(depth <= DEPTH_MAX)[0]
    lat_rows = np.nanmean(np.where(inbox, lat_s, np.nan), axis=1)
    keep_rows = np.isfinite(lat_rows)
    out = {}
    for y in years:
        if y not in files or len(files[y]) < 12:
            continue
        acc = []
        for f in files[y]:
            with xr.open_dataset(f) as ds:
                s = ds["vosaline"].isel(time_counter=0, deptht=kz, y=ys, x=xs).values.astype(float)
            s[s <= 0] = np.nan
            s[:, ~inbox] = np.nan
            acc.append(s)
        s_ann = np.nanmean(np.stack(acc), axis=0)[:, keep_rows, :]
        out[y] = box_means(s_ann, lat_rows[keep_rows], depth[kz])
    return out


def glorys_annual(root: Path, years: range) -> dict[int, tuple[float, float]]:
    out = {}
    for y in years:
        f = root / f"glorys12_{y}.nc"
        if not f.exists():
            continue
        with xr.open_dataset(f) as ds:
            sub = ds["so"].sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX),
                               depth=slice(0, DEPTH_MAX))
            if sub.sizes["time"] < 12:
                continue
            s_ann = sub.mean("time", skipna=True).values.astype(float)
            lat_rows = sub["latitude"].values
            depth = sub["depth"].values
        s_ann[s_ann <= 0] = np.nan
        out[y] = box_means(s_ann, lat_rows, depth)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data/results")
    a = p.parse_args()
    data = Path(a.data)
    years = range(1993, 2026)
    series = {
        "ORAS5": oras5_annual(data / "oras5", years),
        "GLORYS12V1": glorys_annual(data / "glorys12", years),
    }
    res = {"box": {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX], "depth_max_m": DEPTH_MAX}}
    for prod, ser in series.items():
        res[prod] = {"annual_level_mean": {str(k): v[0] for k, v in ser.items()},
                     "annual_thickness_weighted": {str(k): v[1] for k, v in ser.items()}}
        for name, (y0, y1) in PERIODS.items():
            yrs = np.array([y for y in sorted(ser) if y0 <= y <= y1])
            if len(yrs) < 5:
                continue
            res[prod][name] = {
                "level_mean": santer_trend(yrs, np.array([ser[y][0] for y in yrs])),
                "thickness_weighted": santer_trend(yrs, np.array([ser[y][1] for y in yrs])),
                "years": [int(yrs[0]), int(yrs[-1])],
            }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reanalysis_box_salinity_trends.json").write_text(json.dumps(res, indent=1))
    for prod in series:
        for name in PERIODS:
            if name in res[prod]:
                r = res[prod][name]
                print(prod, name, r["years"],
                      "level %.3f +- %.3f" % (r["level_mean"]["trend_psu_per_decade"], r["level_mean"]["ci95_half_psu_per_decade"]),
                      "dz %.3f +- %.3f" % (r["thickness_weighted"]["trend_psu_per_decade"], r["thickness_weighted"]["ci95_half_psu_per_decade"]))


if __name__ == "__main__":
    main()
