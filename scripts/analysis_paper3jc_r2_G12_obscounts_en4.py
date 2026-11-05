#!/usr/bin/env python3
"""Observation counts and observation weights by year and depth, and the EN4 test on core-Argo years.

Part (a), EN4 observation weights along the section. The EN4.2.2 objective
analysis carries a salinity observation-weight field (salinity_observation_weights),
the total weight given to the observation increments at each grid point. The
repository holds no profile-level EN4 input, so this field is the only
observing-density measure available. It is read on the 34.0 S row (the EN4 row
used by the paper; the 1-degree grid has no 34.5 S row) over the Atlantic segment
of ``analysis_paper3v2_en4.atlantic_segment``. For each year 2005 to 2024 and each
depth band the script reports:

* the area-weighted mean observation weight over the cells of the band with a
  finite salinity and a finite weight (cell area = zonal width times level
  thickness), averaged over the months of the year;
* the number of section cells with a finite salinity and with a positive
  observation weight, averaged over the months (a grid-cell coverage count, not a
  profile count);
* the trend per decade of the annual mean weight, with the interval, n_eff and
  p-value of ``paper3r1_stats.trend_test`` over 2005 to 2024 and 2008 to 2024.

Depth bands: 0 to 500 m, 500 to 1000 m, 1000 to 2000 m, 2000 to 3000 m and
3000 m to the bottom of the EN4 column.

Part (b), EN4 fixed-circulation test on core-Argo years. The salinity of EN4 is
averaged to the limbs with the velocity frozen at the record mean of ORAS5 or
GLORYS12V1 (``analysis_paper3v2_en4_depth_test.limbs``), for the full column and
for the upper 2000 m, over 2005 to 2024 (the paper's window) and over 2008 to 2024
(the years from which the Argo-era coverage is taken as complete in the paper's
floor argument). The annual values of S_north, S_south, dS and the implied F_ovS
are tested with ``paper3r1_stats.trend_test``. The 2005 to 2024 full-column dS
trends are checked against ``PAPER_3_v2/analysis/jc_n01_en4_window.json``.

Part (c), the 26.5 N drop of the late 1990s. The ORAS5 and GLORYS12V1 annual
overturning at 26.5 N (``data/results/yearly_amoc26n_*.npz``) are summarised by
block means and by the trend over 1993 to 2004. No ORAS5 observation counts exist
in the repository, and the EN4 record starts in 2005, so the drop cannot be set
against counts held in the repository; the block means locate the drop in years.

Seeds: none needed, since the trend tests are analytic (no resampling).
Replicate counts: none.
Units: salinity in PSU; trends per decade; observation weight dimensionless;
overturning in Sv.

Inputs:  data/en4/EN.4.2.2.f.analysis.g10.*.nc (2005 to 2024),
         data/results/paper3r1_section_{oras5,glorys12}.nc (record-mean V_bc),
         data/results/yearly_amoc26n_{oras5,glorys12}.npz,
         PAPER_3_v2/analysis/jc_n01_en4_window.json (validation only).
Outputs: PAPER_3_v2/analysis/jc_r2_G12_obscounts_counts.json (parts a and c),
         PAPER_3_v2/analysis/jc_r2_G12_obscounts_fixedcirc.json (part b).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper3r1_stats as ps  # noqa: E402
from analysis_paper3v2_en4 import EN4_DIR, RESULTS, TARGET_LAT, atlantic_segment, frozen_velocity  # noqa: E402
from analysis_paper3v2_en4_depth_test import ARGO_FLOOR, limbs  # noqa: E402

OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
OUT_COUNTS = OUT_DIR / "jc_r2_G12_obscounts_counts.json"
OUT_FIXED = OUT_DIR / "jc_r2_G12_obscounts_fixedcirc.json"
STORED_N01 = OUT_DIR / "jc_n01_en4_window.json"

YEARS = np.arange(2005, 2025)
WINDOWS = {"2005-2024": (2005, 2024), "2008-2024": (2008, 2024)}
BANDS = {
    "0-500 m": (0.0, 500.0),
    "500-1000 m": (500.0, 1000.0),
    "1000-2000 m": (1000.0, 2000.0),
    "2000-3000 m": (2000.0, 3000.0),
    "3000 m to bottom": (3000.0, np.inf),
}
LAYERS = {"0-1000 m": (0.0, 1000.0), "0-2000 m": (0.0, 2000.0), "below 2000 m": (2000.0, np.inf)}
CIRCS = {"ORAS5": "paper3r1_section_oras5.nc", "GLORYS12V1": "paper3r1_section_glorys12.nc"}
VARIANTS = {"full_column": None, "upper_2000m": ARGO_FLOOR}


def _trend(years, y):
    r = ps.trend_test(np.asarray(years, dtype=float), np.asarray(y, dtype=float))
    return {
        "trend_per_decade": float(r["value"] * 10.0),
        "ci95_per_decade": [float(r["ci"][0] * 10.0), float(r["ci"][1] * 10.0)],
        "n": int(len(years)),
        "n_eff": float(r["n_eff"]),
        "phi_corrected": float(r["phi_corrected"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
        "p": float(r["p"]),
    }


# ── EN4 monthly section fields ────────────────────────────────────────


def load_month(path: Path):
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        depth = ds["depth"].values.astype(float)
        bnds = ds["depth_bnds"].values
        e3t = np.abs(bnds[:, 1] - bnds[:, 0]).astype(float)
        j = int(np.abs(lat - TARGET_LAT).argmin())
        sal = ds["salinity"].isel(time=0, lat=j).values.astype(float)
        wgt = ds["salinity_observation_weights"].isel(time=0, lat=j).values.astype(float)
        time = pd.Timestamp(ds["time"].values[0])
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    ocean = np.isfinite(sal[0])
    if not ocean.any():
        return None
    sel, e1t = atlantic_segment(lon180, np.full(lon.shape, float(lat[j])), ocean)
    s = sal[:, sel]
    w = wgt[:, sel]
    wt = np.where(np.isfinite(s), e1t, 0.0)
    wsum = wt.sum(axis=1)
    s_bar = np.where(wsum > 0, (np.nan_to_num(s) * wt).sum(axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    return {
        "time": time,
        "S": s,
        "W": w,
        "S_bar": s_bar,
        "e1t": e1t,
        "e3t": e3t,
        "depth": depth,
        "lat": float(lat[j]),
        "n_sec_points": int(len(sel)),
    }


def load_all():
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = [r for r in ex.map(load_month, files, chunksize=8) if r is not None]
    rows.sort(key=lambda r: r["time"])
    return rows


# ── part a: observation weights and cell counts by year and band ─────


def _cell_stats(r, m):
    """Area-weighted mean observation weight and cell counts for one month and one level mask."""
    S = r["S"][m]
    W = r["W"][m]
    area = r["e1t"][None, :] * r["e3t"][m][:, None]
    fin_S = np.isfinite(S)
    fin_W = np.isfinite(W) & fin_S
    asum = float((area * fin_W).sum())
    mean_w = float((area * np.where(fin_W, W, 0.0)).sum()) / asum if asum > 0 else np.nan
    return mean_w, int(fin_S.sum()), int((fin_W & (W > 0)).sum())


def part_a(rows):
    depth = rows[0]["depth"]
    times = pd.DatetimeIndex([r["time"] for r in rows])
    months_year = np.array(times.year)
    res = {"years": [int(y) for y in YEARS], "months_per_year": [], "bands": {}}
    res["n_section_points_per_year"] = []
    for y in YEARS:
        idx = np.where(months_year == y)[0]
        res["months_per_year"].append(int(idx.size))
        res["n_section_points_per_year"].append(float(np.mean([rows[i]["n_sec_points"] for i in idx])))
    for b, (lo, hi) in BANDS.items():
        m = (depth >= lo) & (depth < hi)
        mw_y, nfin_y, npos_y = [], [], []
        for y in YEARS:
            idx = np.where(months_year == y)[0]
            stats_m = [_cell_stats(rows[i], m) for i in idx]
            mw_y.append(float(np.nanmean([s[0] for s in stats_m])))
            nfin_y.append(float(np.mean([s[1] for s in stats_m])))
            npos_y.append(float(np.mean([s[2] for s in stats_m])))
        mw_y = np.array(mw_y)
        res["bands"][b] = {
            "per_year": {
                "mean_obs_weight": mw_y.tolist(),
                "cells_finite_S": nfin_y,
                "cells_positive_W": npos_y,
                "fraction_positive_W": [p / f if f > 0 else None for p, f in zip(npos_y, nfin_y)],
            },
            "trend_obs_weight": {
                wname: _trend(YEARS[(YEARS >= y0) & (YEARS <= y1)], mw_y[(YEARS >= y0) & (YEARS <= y1)])
                for wname, (y0, y1) in WINDOWS.items()
            },
            "mean_2005_2007": float(np.mean(mw_y[:3])),
            "mean_2022_2024": float(np.mean(mw_y[-3:])),
        }
    res["layer_summary"] = {}
    for lab, (lo, hi) in LAYERS.items():
        m = (depth >= lo) & (depth < hi)
        vals = []
        for y in YEARS:
            idx = np.where(months_year == y)[0]
            vals.append(float(np.nanmean([_cell_stats(rows[i], m)[0] for i in idx])))
        vals = np.array(vals)
        res["layer_summary"][lab] = {
            "per_year_mean_obs_weight": vals.tolist(),
            "trend_2005_2024": _trend(YEARS.astype(float), vals),
            "trend_2008_2024": _trend(YEARS[3:].astype(float), vals[3:]),
        }
    return res


# ── part b: fixed-circulation EN4 tests on the two windows ───────────


def part_b(rows):
    depth = rows[0]["depth"]
    e3t = rows[0]["e3t"]
    times = pd.DatetimeIndex([r["time"] for r in rows])
    months_year = np.array(times.year)
    out = {"circulations": {}, "validation_against_jc_n01": {}}
    for cname, fname in CIRCS.items():
        v = frozen_velocity(fname, depth, e3t)
        out["circulations"][cname] = {}
        for vname, zmax in VARIANTS.items():
            monthly = [limbs(v, r["S_bar"], e3t, zmax=zmax, depth=depth) for r in rows]
            ann = {}
            for field in ("S_north", "S_south", "dS", "F_ov_Sv", "T_Sv"):
                vals = np.array([m[field] if m is not None else np.nan for m in monthly], dtype=float)
                ann[field] = np.array(
                    [np.nanmean(vals[months_year == y]) if np.isfinite(vals[months_year == y]).any() else np.nan
                     for y in YEARS]
                )
            block = {}
            for wname, (y0, y1) in WINDOWS.items():
                sel = (YEARS >= y0) & (YEARS <= y1)
                yy = YEARS[sel]
                block[wname] = {
                    "window": [y0, y1],
                    "n_years": int(sel.sum()),
                    "n_months_used": int(sum(1 for mo, m in zip(months_year, monthly)
                                             if y0 <= mo <= y1 and m is not None)),
                    "S_north": _trend(yy, ann["S_north"][sel]),
                    "S_south": _trend(yy, ann["S_south"][sel]),
                    "dS": _trend(yy, ann["dS"][sel]),
                    "F_ov_Sv": _trend(yy, ann["F_ov_Sv"][sel]),
                    "level_means": {
                        "S_north": float(np.nanmean(ann["S_north"][sel])),
                        "S_south": float(np.nanmean(ann["S_south"][sel])),
                        "dS": float(np.nanmean(ann["dS"][sel])),
                        "F_ov_Sv": float(np.nanmean(ann["F_ov_Sv"][sel])),
                        "T_Sv": float(np.nanmean(ann["T_Sv"][sel])),
                    },
                }
            out["circulations"][cname][vname] = {
                "zmax_m": None if zmax is None else float(zmax),
                "annual_dS": ann["dS"].tolist(),
                "annual_S_north": ann["S_north"].tolist(),
                "windows": block,
            }
    if STORED_N01.exists():
        n01 = json.loads(STORED_N01.read_text())
        for cname in CIRCS:
            key = "oras5" if cname == "ORAS5" else "glorys12"
            try:
                ref = n01["series"][f"{key}_full_column"]["dS"]["full_2005_2024"]["slope_per_decade"]
                got = out["circulations"][cname]["full_column"]["windows"]["2005-2024"]["dS"]["trend_per_decade"]
                out["validation_against_jc_n01"][cname] = {
                    "recomputed_dS_trend_2005_2024_per_decade": got,
                    "stored_jc_n01_dS_trend_2005_2024_per_decade": ref,
                    "difference": got - ref,
                }
            except KeyError:
                out["validation_against_jc_n01"][cname] = "key not found"
    return out


# ── part c: 26.5 N overturning drop in the late 1990s ─────────────────


def part_c():
    out = {}
    for prod in ("oras5", "glorys12"):
        d = np.load(RESULTS / f"yearly_amoc26n_{prod}.npz")
        yrs, amoc = d["years"].astype(float), d["amoc"].astype(float)
        blocks = {"1993-1996": (1993, 1996), "1997-2001": (1997, 2001),
                  "2002-2006": (2002, 2006), "2007-2011": (2007, 2011)}
        bm = {}
        for lab, (a, b) in blocks.items():
            m = (yrs >= a) & (yrs <= b)
            bm[lab] = {"mean_Sv": float(amoc[m].mean()) if m.any() else None, "n_years": int(m.sum())}
        d1 = None
        if bm["1993-1996"]["mean_Sv"] is not None and bm["2002-2006"]["mean_Sv"] is not None:
            d1 = bm["2002-2006"]["mean_Sv"] - bm["1993-1996"]["mean_Sv"]
        m_pre = yrs <= 2004
        out[prod] = {
            "years": [int(yrs[0]), int(yrs[-1])],
            "block_means": bm,
            "difference_2002_2006_minus_1993_1996_Sv": d1,
            "trend_1993_2004_Sv_per_decade": _trend(yrs[m_pre], amoc[m_pre]),
            "annual_Sv": {str(int(y)): float(a) for y, a in zip(yrs, amoc)},
        }
    return out


def main():
    rows = load_all()
    print(f"loaded {len(rows)} EN4 months", flush=True)
    meta_common = {
        "date": "2026-11-05",
        "group": "G12_obscounts",
        "action_ids": ["A102"],
        "cluster": "C51",
        "seeds": "not used (analytic trend tests, no resampling)",
        "replicate_counts": "none",
        "section_row_lat": TARGET_LAT,
        "n_en4_months": len(rows),
        "trend_test": "paper3r1_stats.trend_test (bias-corrected lag-one, N_eff floor)",
        "units": ("salinity PSU; trends per decade; observation weight dimensionless; "
                  "cell counts are grid-cell counts, not profile counts; overturning Sv"),
        "limitations": {
            "profile_counts": "not in repository: data/en4 holds gridded monthly objective analyses only",
            "oras5_counts": "not in repository",
            "1990s_counts": "EN4 files start in 2005; no counts exist in the repository for the late 1990s",
        },
    }
    a = part_a(rows)
    c = part_c()
    counts = {
        "meta": dict(meta_common,
                     script="scripts/analysis_paper3jc_r2_G12_obscounts_en4.py",
                     inputs=["data/en4/EN.4.2.2.f.analysis.g10.*.nc (34.0 S row)",
                             "data/results/yearly_amoc26n_oras5.npz, yearly_amoc26n_glorys12.npz"],
                     output=OUT_COUNTS.name),
        "part_a_en4_observation_weights_by_year_and_depth": a,
        "part_c_26N_overturning_drop_1990s": c,
    }
    OUT_COUNTS.write_text(json.dumps(counts, indent=1, default=float))
    print(f"wrote {OUT_COUNTS}", flush=True)
    b = part_b(rows)
    fixed = {
        "meta": dict(meta_common,
                     script="scripts/analysis_paper3jc_r2_G12_obscounts_en4.py",
                     inputs=["data/en4/EN.4.2.2.f.analysis.g10.*.nc (34.0 S row)",
                             "data/results/paper3r1_section_{oras5,glorys12}.nc (record-mean V_bc)",
                             "PAPER_3_v2/analysis/jc_n01_en4_window.json (validation only)"],
                     output=OUT_FIXED.name),
        "part_b_en4_fixed_circulation_windows": b,
    }
    OUT_FIXED.write_text(json.dumps(fixed, indent=1, default=float))
    print(f"wrote {OUT_FIXED}", flush=True)


if __name__ == "__main__":
    main()
