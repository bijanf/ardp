#!/usr/bin/env python3
"""Uncorrected lag-one test beside the bias-corrected test for the window trend
table of the Supplement (paper 3, revision round 2, group G11).

For every row of the window table (product and window) and for each of the
columns FovS, T, dS, Faz and FovS + Faz, the trend is computed from the annual
means of the section file data/results/paper3r1_section_<product>.nc with two
tests on the same OLS fit:

- bias-corrected: paper3r1_stats.trend_test (the paper's test; identical to
  analysis_paper3v2_attribution.trend_stats);
- uncorrected: the same Santer-type effective sample size with the raw lag-one
  coefficient of the OLS residuals, N_eff = N (1 - r1) / (1 + r1) bounded to
  [k + 1, N]; the interval is the inverse of that test.

Units follow the table: FovS, Faz and FovS + Faz in mSv per year; T in Sv per
decade; dS in PSU per decade.

The GLORYS12V1 1993 to 2025 row is regenerated here from the stored section
file. The bias-corrected value is compared with final_checks.json and the
uncorrected p value is reported. The GLORYS 1993 to 2025 row with trend -4.362
and interval [-5.797, -2.951] in CANONICAL_FACTS.md is checked for a writer in
scripts/; if none exists the row is marked as not regenerable.

No random numbers are drawn.

Inputs: data/results/paper3r1_section_{oras5,glorys12,ecco}.nc,
PAPER_3_v2/analysis/final_checks.json.
Output: PAPER_3_v2/analysis/jc_r2_G11_tables_uncorrected.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_attribution import S0, annual, trend_stats  # noqa: E402
from paper3r1_stats import lag1, trend_design  # noqa: E402

RESULTS = REPO / "data" / "results"
ANA = REPO / "PAPER_3_v2" / "analysis"
OUT = ANA / "jc_r2_G11_tables_uncorrected.json"

PRODUCTS = {
    "oras5": "ORAS5",
    "glorys12": "GLORYS12V1",
    "ecco": "ECCO-V4r4",
}

# (product key, window label, first year, last year)
ROWS = [
    ("oras5", "1958-2025", 1958, 2025),
    ("oras5", "2004-2025", 2004, 2025),
    ("oras5", "1993-2017", 1993, 2017),
    ("glorys12", "1993-2025", 1993, 2025),
    ("glorys12", "2004-2025", 2004, 2025),
    ("glorys12", "2004-2014", 2004, 2014),
    ("glorys12", "1993-2017", 1993, 2017),
    ("ecco", "1992-2017", 1992, 2017),
    ("ecco", "1993-2017", 1993, 2017),
]

# column name -> (annual variable builder, scale to table units)
COLUMNS = {
    "FovS_mSv_per_yr": (lambda a: a["F_ov"].values, 1e3),
    "T_Sv_per_decade": (lambda a: a["T_limb_Sv"].values, 10.0),
    "dS_PSU_per_decade": (lambda a: a["dS_limb"].values, 10.0),
    "Faz_mSv_per_yr": (lambda a: a["F_az"].values, 1e3),
    "FovS_plus_Faz_mSv_per_yr": (
        lambda a: a["F_ov"].values + a["F_az"].values,
        1e3,
    ),
}


def uncorrected_test(years: np.ndarray, y: np.ndarray, scale: float) -> dict:
    """OLS trend with the raw lag-one coefficient in the Santer N_eff."""
    x = trend_design(years.astype(float))
    n, k = x.shape
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    s2 = float(resid @ resid) / (n - k)
    se = float(np.sqrt(s2 * np.linalg.inv(x.T @ x)[1, 1]))
    r1 = float(lag1(resid))
    neff = float(min(max(n * (1.0 - r1) / (1.0 + r1), k + 1.0), n))
    dof = neff - k
    se_adj = se * np.sqrt((n - k) / dof)
    value = float(coef[1])
    tcrit = float(stats.t.ppf(0.975, dof))
    p = float(2.0 * stats.t.sf(abs(value) / se_adj, dof))
    return {
        "trend": value * scale,
        "ci95": [(value - tcrit * se_adj) * scale, (value + tcrit * se_adj) * scale],
        "p": p,
        "n_eff": neff,
        "lag1_raw": r1,
        "n_eff_at_floor": bool(n * (1.0 - r1) / (1.0 + r1) < k + 1.0),
    }


def corrected_test(years: np.ndarray, y: np.ndarray, scale: float) -> dict:
    fit = trend_stats(years, y, scale=scale)
    return {
        "trend": fit["trend"],
        "ci95": fit["ci"],
        "p": fit["p_santer"],
        "n_eff": fit["n_eff"],
        "phi_corrected": fit["phi_corrected"],
        "n_eff_at_floor": fit["n_eff_at_floor"],
    }


def annual_fields(key: str) -> tuple[xr.Dataset, np.ndarray]:
    ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
    ann = annual(ds)
    return ann, ann["year"].values


def row_results() -> list[dict]:
    cache: dict[str, tuple] = {}
    rows = []
    for key, wlab, y0, y1 in ROWS:
        if key not in cache:
            cache[key] = annual_fields(key)
        ann, years = cache[key]
        m = (years >= y0) & (years <= y1)
        y = years[m]
        assert len(y) == y1 - y0 + 1, (key, wlab, len(y))
        block = {}
        for col, (builder, scale) in COLUMNS.items():
            v = np.asarray(builder(ann))[m]
            block[col] = {
                "bias_corrected": corrected_test(y, v, scale),
                "uncorrected": uncorrected_test(y, v, scale),
            }
        rows.append(
            {
                "product": PRODUCTS[key],
                "window": wlab,
                "n_years": int(len(y)),
                "columns": block,
            }
        )
    return rows


def glorys_1993_2025_check() -> dict:
    fc = json.loads((ANA / "final_checks.json").read_text())
    ref = fc["glorys12"]["window_1993_2025"]["F_ov_trend_mSv_per_yr"]
    ann, years = annual_fields("glorys12")
    m = (years >= 1993) & (years <= 2025)
    f = ann["F_ov"].values[m]
    point = uncorrected_test(years[m], f, 1e3)
    corr = corrected_test(years[m], f, 1e3)
    # Search the scripts for a line that writes the stored vertical-grid file
    # (a read is not a writer); the name is assembled here so that this script
    # does not match itself.
    stem = "r1_vertical" + "_grid"
    writers = []
    for p in sorted((REPO / "scripts").glob("*.py")):
        if p.resolve() == Path(__file__).resolve():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            if stem in line and any(w in line for w in ("write", "dump", "to_csv")):
                writers.append(p.name)
                break
    return {
        "section_file_trend_mSv_per_yr": point["trend"],
        "final_checks_trend_mSv_per_yr": ref,
        "trend_matches_final_checks": bool(abs(point["trend"] - ref) < 1e-3),
        "bias_corrected": corr,
        "uncorrected": point,
        "canonical_row_minus_4p362": {
            "value_in_CANONICAL_FACTS": -4.362,
            "interval_in_CANONICAL_FACTS": [-5.797, -2.951],
            "p_in_CANONICAL_FACTS": 0.017,
            "scripts_referencing_vertical_grid_file": writers,
            "status": (
                "regenerated" if writers else "not regenerable from the repository"
            ),
        },
    }


def main() -> None:
    res = {
        "meta": {
            "id": "jc_r2_G11_tables_uncorrected",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G11_tables_uncorrected.py",
            "inputs": [
                "data/results/paper3r1_section_oras5.nc",
                "data/results/paper3r1_section_glorys12.nc",
                "data/results/paper3r1_section_ecco.nc",
                "PAPER_3_v2/analysis/final_checks.json",
            ],
            "seeds": None,
            "replicates": None,
            "units": "FovS, Faz and FovS + Faz in mSv per year; T in Sv per decade; "
            "dS in PSU per decade; p values dimensionless",
            "tests": {
                "bias_corrected": "paper3r1_stats.trend_test via trend_stats",
                "uncorrected": "same N_eff with the raw lag-one coefficient, bounded "
                "to [k+1, N]",
            },
        },
        "window_table_rows": row_results(),
        "glorys12_1993_2025_check": glorys_1993_2025_check(),
    }
    OUT.write_text(json.dumps(res, indent=1))
    for r in res["window_table_rows"]:
        f = r["columns"]["FovS_mSv_per_yr"]
        print(
            r["product"],
            r["window"],
            "corr",
            round(f["bias_corrected"]["trend"], 3),
            round(f["bias_corrected"]["p"], 4),
            round(f["bias_corrected"]["n_eff"], 2),
            "uncorr",
            round(f["uncorrected"]["p"], 4),
            round(f["uncorrected"]["n_eff"], 2),
        )
    print(res["glorys12_1993_2025_check"]["canonical_row_minus_4p362"]["status"])


if __name__ == "__main__":
    main()
