#!/usr/bin/env python3
"""Compute upper-300 m Atlantic-basin salinity trend from EN4 and RG09.

Defines the South Atlantic mask consistent with project convention:
  latitude:    34.5°S ± 5°  (29.5°S to 39.5°S)
  longitude:   Atlantic basin only — excludes Mediterranean, Baltic,
               Hudson Bay, Gulf of Mexico. For 34.5°S±5° this is roughly
               −60° to +20° E.
  depth:       0-300 m
  time:        2005-01 to 2024-12

For each product, computes:
  - annual basin-mean salinity (volume-weighted)
  - linear OLS trend with Santer 2008 N_eff AR(1) correction for 95% CI
  - bootstrap uncertainty (1000 resamples over years)

Outputs:
  data/results/argo_trends.json   — both products + summary block
  data/results/argo_basin_mean.csv — annual time series for plotting
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
EN4_DIR = REPO / "data" / "en4"
RG09_DIR = REPO / "data" / "argo_rg09"
RESULTS = REPO / "data" / "results"

LAT_MIN, LAT_MAX = -39.5, -29.5
LON_MIN, LON_MAX = -60.0, 20.0
DEPTH_MAX = 300.0
YEAR_START, YEAR_END = 2005, 2024


def _santer_neff_ci(years: np.ndarray, values: np.ndarray, alpha: float = 0.05) -> dict:
    """OLS linear trend with Santer 2008 effective-sample-size CI under AR(1)."""
    yrs = years.astype(float) - years.mean()
    n = len(values)
    slope, intercept, r, p_raw, _ = stats.linregress(yrs, values)
    resid = values - (slope * yrs + intercept)
    # AR(1) coefficient of residuals
    if n > 2:
        r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    else:
        r1 = 0.0
    n_eff = n * (1.0 - r1) / (1.0 + r1) if r1 < 1 else max(1, n // 4)
    n_eff = max(2.1, min(n, n_eff))
    se = np.sqrt(np.sum(resid**2) / (n_eff - 2)) / np.sqrt(np.sum(yrs**2))
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=int(n_eff) - 2)
    return {
        "slope_per_year": float(slope),
        "intercept": float(intercept),
        "p_raw": float(p_raw),
        "r1": float(r1),
        "n_eff": float(n_eff),
        "ci95_half": float(t_crit * se),
        "se": float(se),
    }


def _bootstrap_trend(years: np.ndarray, values: np.ndarray, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    slopes = []
    idx = np.arange(len(values))
    for _ in range(n):
        bidx = rng.choice(idx, size=len(idx), replace=True)
        y = years[bidx].astype(float) - years[bidx].mean()
        v = values[bidx]
        s, _, _, _, _ = stats.linregress(y, v)
        slopes.append(s)
    slopes = np.asarray(slopes)
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


# -------------------------------------------------------------------------
# EN4 processing
# -------------------------------------------------------------------------
def _load_en4_basin_annual() -> pd.DataFrame:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    files = [f for f in files if YEAR_START <= int(f.stem.split(".")[-1][:4]) <= YEAR_END]
    if not files:
        return pd.DataFrame()
    print(f"  EN4: {len(files)} monthly files")
    annual: dict[int, list[float]] = {}
    for fp in files:
        year = int(fp.stem.split(".")[-1][:4])
        ds = xr.open_dataset(fp)
        # Subset Atlantic basin: lat 34.5°S±5°, lon 300-360 OR 0-20 (wrap)
        lat_sel = ds.sel(lat=slice(LAT_MIN, LAT_MAX), depth=slice(0, DEPTH_MAX))
        sal_var = "salinity" if "salinity" in lat_sel else "salinity_observed"
        # Two longitude bands to glue (EN4 uses 0-360°E convention)
        sal_west = lat_sel[sal_var].sel(lon=slice(300.0, 360.0))   # -60..0
        sal_east = lat_sel[sal_var].sel(lon=slice(0.0, LON_MAX))   # 0..20
        sal = xr.concat([sal_west, sal_east], dim="lon")
        if "time" in sal.dims:
            sal = sal.squeeze("time", drop=True)
        # Volume-weighted basin mean: cos(lat) over lat dim, uniform over depth/lon
        lat_w = np.cos(np.deg2rad(sal["lat"]))
        # Mean over lon (uniform) then lat (cos-weighted) then depth (uniform)
        sal_lonmean = sal.mean("lon", skipna=True)             # (depth, lat)
        sal_latband = (sal_lonmean * lat_w).sum("lat", skipna=True) / lat_w.sum()
        sal_basin = float(sal_latband.mean("depth", skipna=True).values)
        annual.setdefault(year, []).append(sal_basin)
        ds.close()
    rows = [(y, np.mean(vs)) for y, vs in sorted(annual.items())]
    return pd.DataFrame(rows, columns=["year", "salinity_psu"])


# -------------------------------------------------------------------------
# RG09 processing
# -------------------------------------------------------------------------
def _load_rg09_basin_annual() -> pd.DataFrame:
    """Build RG09 annual basin-mean salinity by combining the climatology file
    (which carries TIME=180 monthly anomalies for 2004-2018) with the
    monthly-extension files (2019 onward)."""
    annual: dict[int, list[float]] = {}
    files = sorted(RG09_DIR.glob("RG_ArgoClim_*.nc"))
    if not files:
        return pd.DataFrame()
    print(f"  RG09: {len(files)} files")
    clim_path = RG09_DIR / "RG_ArgoClim_Salinity_2019.nc"
    if not clim_path.exists():
        return pd.DataFrame()

    ds_clim = xr.open_dataset(clim_path, decode_times=False)
    sal_mean = ds_clim["ARGO_SALINITY_MEAN"]
    sal_anom_clim = ds_clim["ARGO_SALINITY_ANOMALY"]  # 2004-01 to 2018-12

    # Basin selection
    pressure_mask = ds_clim["PRESSURE"] <= DEPTH_MAX
    lat_mask = (ds_clim["LATITUDE"] >= LAT_MIN) & (ds_clim["LATITUDE"] <= LAT_MAX)
    lon = ds_clim["LONGITUDE"].values
    # RG09 longitudes are 20.5..380.5°E (wraps at 380); Atlantic = 300-360 + 0-20
    lon_atl_mask = ((lon >= 300) & (lon <= 360)) | (lon <= LON_MAX) | (lon >= 380)
    lon_idx = np.where(lon_atl_mask)[0]
    lat_idx = np.where(lat_mask.values)[0]

    sal_mean_sub = sal_mean.isel(PRESSURE=pressure_mask, LATITUDE=lat_idx, LONGITUDE=lon_idx)
    sal_anom_sub = sal_anom_clim.isel(PRESSURE=pressure_mask, LATITUDE=lat_idx, LONGITUDE=lon_idx)

    lat_w = np.cos(np.deg2rad(sal_mean_sub["LATITUDE"]))
    # Basin-mean climatological salinity (scalar)
    mean_lonmean = sal_mean_sub.mean("LONGITUDE", skipna=True)
    mean_latband = (mean_lonmean * lat_w).sum("LATITUDE", skipna=True) / lat_w.sum()
    clim_basin_mean = float(mean_latband.mean("PRESSURE", skipna=True).values)

    # 2004-2018 monthly anomaly time series (basin-mean)
    # Time axis: 'months since 2004-01-01' integer index 0..179
    anom_basin = (sal_anom_sub.mean("LONGITUDE", skipna=True) * lat_w).sum("LATITUDE", skipna=True) / lat_w.sum()
    anom_basin = anom_basin.mean("PRESSURE", skipna=True).values  # shape (180,)
    months_since_2004 = np.arange(180)
    for i, m in enumerate(months_since_2004):
        year = 2004 + m // 12
        if year < YEAR_START or year > YEAR_END:
            continue
        absolute = clim_basin_mean + float(anom_basin[i])
        annual.setdefault(year, []).append(absolute)
    ds_clim.close()

    # Monthly extension files (2019 onward)
    for fp in files:
        if "Salinity_2019" in fp.name:
            continue
        try:
            yymm = fp.stem.split("_")[-2]
            year = int(yymm[:4])
        except (ValueError, IndexError):
            continue
        if year < YEAR_START or year > YEAR_END:
            continue
        ds = xr.open_dataset(fp, decode_times=False)
        anom_var = "ARGO_SALINITY_ANOMALY"
        if anom_var not in ds:
            ds.close()
            continue
        anom = ds[anom_var].isel(PRESSURE=pressure_mask, LATITUDE=lat_idx, LONGITUDE=lon_idx)
        if "TIME" in anom.dims:
            anom = anom.squeeze("TIME", drop=True)
        anom_latlon = (anom.mean("LONGITUDE", skipna=True) * lat_w).sum("LATITUDE", skipna=True) / lat_w.sum()
        anom_mean = float(anom_latlon.mean("PRESSURE", skipna=True).values)
        annual.setdefault(year, []).append(clim_basin_mean + anom_mean)
        ds.close()

    rows = [(y, np.mean(vs)) for y, vs in sorted(annual.items())]
    return pd.DataFrame(rows, columns=["year", "salinity_psu"])


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading EN4.2.2 annual basin-mean salinity ...")
    df_en4 = _load_en4_basin_annual()
    print(f"  EN4 years: {df_en4['year'].tolist() if not df_en4.empty else '(none)'}")

    print("\nLoading RG09 annual basin-mean salinity ...")
    df_rg09 = _load_rg09_basin_annual()
    print(f"  RG09 years: {df_rg09['year'].tolist() if not df_rg09.empty else '(none)'}")

    summary: dict[str, dict] = {}

    for label, df in [("EN4.2.2", df_en4), ("RG09", df_rg09)]:
        if df.empty:
            print(f"  {label}: no data — skipping")
            continue
        yrs = df["year"].values
        vals = df["salinity_psu"].values
        trend = _santer_neff_ci(yrs, vals)
        b_lo, b_hi = _bootstrap_trend(yrs, vals)
        summary[label] = {
            "n_years": int(len(yrs)),
            "year_range": [int(yrs.min()), int(yrs.max())],
            "slope_psu_per_dec": float(trend["slope_per_year"]) * 10.0,
            "ci95_half_psu_per_dec": float(trend["ci95_half"]) * 10.0,
            "bootstrap_ci95_psu_per_dec": [b_lo * 10.0, b_hi * 10.0],
            "n_eff": trend["n_eff"],
            "r1": trend["r1"],
            "p_raw": trend["p_raw"],
        }
        print(f"\n{label}: trend = {summary[label]['slope_psu_per_dec']:+.4f} ± "
              f"{summary[label]['ci95_half_psu_per_dec']:.4f} PSU/decade (Santer)")
        print(f"  bootstrap 95% CI: [{b_lo*10:+.4f}, {b_hi*10:+.4f}] PSU/decade")

    # Add Volkov 2024 SAMBA reference for plotting
    summary["SAMBA-Volkov2024"] = {
        "slope_psu_per_dec": 0.050,
        "ci95_half_psu_per_dec": 0.020,  # approximate from Volkov paper
        "source": "Volkov et al. 2024, SAMBA repeat hydrography 2009-2023",
    }

    out_json = RESULTS / "argo_trends.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    out_csv = RESULTS / "argo_basin_mean.csv"
    combined = pd.DataFrame()
    if not df_en4.empty:
        df_en4["product"] = "EN4.2.2"
        combined = pd.concat([combined, df_en4])
    if not df_rg09.empty:
        df_rg09["product"] = "RG09"
        combined = pd.concat([combined, df_rg09])
    if not combined.empty:
        combined.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
