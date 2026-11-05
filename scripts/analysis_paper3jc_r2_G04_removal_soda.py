#!/usr/bin/env python3
"""Four-snapshot sampling test for the SODA F_ovS sign (action A33, cluster C09).

SODA 3.15.2 enters the paper only through annual F_ovS values built from four
five-day snapshots per year (January, April, July, October;
scripts/compute_soda_fovs.py). This script tests whether that four-snapshot
sampling changes the sign of the record-mean F_ovS, by sampling the three
monthly products at the same four months and comparing with their full monthly
record.

For ORAS5, GLORYS12V1 and ECCO-V4r4 (corrected F_ovS, uniform removal, from the
monthly section files) it reports

  - the record-mean F_ovS over all months of the product's own record;
  - the record-mean F_ovS over the four sampled months (1, 4, 7, 10) of each
    year, averaged over years, over the same record;
  - the same two numbers restricted to 1980 to 2022, the SODA window, so that
    the sampling and the window are separated;
  - the paper's trend test (trend_stats, annual means) on the four-month
    annual values, for reference.

For SODA, it reads the stored annual values of the current kernel
(data/results/checkpoints/fovs_soda.json, barotropic subtraction, commit
c962ff3) and of the pre-fix backup (fovs_soda.json.preBT, commit 10bf794
kernel), and reports their record means, counts of negative years and the
interval of the mean from the paper's test on the annual values.

Inputs: data/results/paper3r1_section_{oras5,glorys12,ecco}.nc,
data/results/checkpoints/fovs_soda.json, data/results/checkpoints/fovs_soda.json.preBT.
Output: PAPER_3_v2/analysis/jc_r2_G04_removal_soda.json.
Seeds: none (deterministic). Replicates: none.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_attribution import trend_stats  # noqa: E402
from paper3r1_stats import mean_test  # noqa: E402

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G04_removal_soda.json"
PRODUCTS = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}
SNAP_MONTHS = (1, 4, 7, 10)
SODA_WINDOW = (1980, 2022)


def sampled_record(years_m: np.ndarray, months_m: np.ndarray, values: np.ndarray, window=None) -> dict:
    keep = np.ones(len(values), dtype=bool)
    if window is not None:
        keep = (years_m >= window[0]) & (years_m <= window[1])
    full_mean = float(values[keep].mean())
    snap = keep & np.isin(months_m, SNAP_MONTHS)
    ys = np.unique(years_m[snap])
    annual_snap = np.array([values[snap & (years_m == y)].mean() for y in ys])
    m = mean_test(annual_snap)
    tr = trend_stats(ys, annual_snap) if len(ys) >= 15 else None
    return {
        "window": [int(ys[0]), int(ys[-1])] if len(ys) else None,
        "n_months_full": int(keep.sum()),
        "n_years_sampled": int(len(ys)),
        "n_snapshots_per_year": len(SNAP_MONTHS),
        "record_mean_full_monthly_Sv": full_mean,
        "record_mean_four_month_Sv": float(annual_snap.mean()),
        "difference_four_minus_full_Sv": float(annual_snap.mean() - full_mean),
        "mean_test_four_month": {
            "ci_Sv": m["ci"],
            "n_eff": m["n_eff"],
            "n_eff_at_floor": m["n_eff_at_floor"],
        },
        "sign_four_month_mean": "negative" if annual_snap.mean() < 0 else "positive",
        "sign_full_monthly_mean": "negative" if full_mean < 0 else "positive",
        "sign_unchanged": bool(np.sign(annual_snap.mean()) == np.sign(full_mean)),
        "annual_negative_years_four_month": int((annual_snap < 0).sum()),
        "trend_four_month_paper_test": (
            {
                "trend_mSv_per_yr": tr["trend"],
                "ci_mSv_per_yr": tr["ci"],
                "p_paper_test": tr["p_santer"],
                "n_eff": tr["n_eff"],
            }
            if tr is not None
            else None
        ),
    }


def soda_block() -> dict:
    out = {}
    for tag, fname in (
        ("current_kernel_commit_c962ff3", "fovs_soda.json"),
        ("pre_fix_backup_commit_10bf794_kernel", "fovs_soda.json.preBT"),
    ):
        with (RESULTS / "checkpoints" / fname).open() as fh:
            d = json.load(fh)
        yrs = np.array(sorted(int(k) for k in d))
        vals = np.array([d[str(y)] for y in yrs], dtype=float)
        m = mean_test(vals)
        out[tag] = {
            "file": f"data/results/checkpoints/{fname}",
            "window": [int(yrs[0]), int(yrs[-1])],
            "n_years": int(len(yrs)),
            "record_mean_Sv": float(vals.mean()),
            "mean_ci_Sv": m["ci"],
            "n_eff": m["n_eff"],
            "n_negative_years": int((vals < 0).sum()),
        }
    return out


def main() -> None:
    res = {}
    for key, label in PRODUCTS.items():
        with xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc") as ds:
            times = ds["time"].values
            f_ov = ds["F_ov"].values.astype(float)
        yrs = np.array([int(str(t)[:4]) for t in times])
        mos = np.array([int(str(t)[5:7]) for t in times])
        res[key] = {
            "product": label,
            "own_record": sampled_record(yrs, mos, f_ov),
            "window_1980_2022": sampled_record(yrs, mos, f_ov, window=SODA_WINDOW),
        }
    out = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G04_removal_soda.py",
            "action": "A33 (cluster C09, optional)",
            "snapshot_months": list(SNAP_MONTHS),
            "inputs": [
                "data/results/paper3r1_section_{oras5,glorys12,ecco}.nc (monthly F_ov, corrected)",
                "data/results/checkpoints/fovs_soda.json",
                "data/results/checkpoints/fovs_soda.json.preBT",
            ],
            "kernel_history": {
                "10bf794": "SODA v3 catalogue-driven downloads (SODA script)",
                "c962ff3": "2026-04-20, barotropic subtraction in ardp/physics/fovs.py; message says SODA re-run follows",
            },
            "seeds": None,
            "replicates": None,
            "sign_convention": "northward-positive, Sv; F < 0 is a northward salt-flux deficit",
            "units": {"F": "Sv", "trend": "mSv per yr"},
            "tests": "paper3r1_stats.mean_test for intervals; analysis_paper3v2_attribution.trend_stats for trends",
        },
        "products_four_month_sampling": res,
        "soda_stored_annual_values": soda_block(),
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for key in PRODUCTS:
        r = res[key]
        print(
            key,
            round(r["own_record"]["record_mean_full_monthly_Sv"], 5),
            round(r["own_record"]["record_mean_four_month_Sv"], 5),
            round(r["window_1980_2022"]["record_mean_full_monthly_Sv"], 5),
            round(r["window_1980_2022"]["record_mean_four_month_Sv"], 5),
        )
    print(out["soda_stored_annual_values"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
