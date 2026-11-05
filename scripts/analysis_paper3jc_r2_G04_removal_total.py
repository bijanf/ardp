#!/usr/bin/env python3
"""Uncorrected section total under the net-transport removal (action A101,
cluster C50), the mass-budget check of the uniform removal, and the optional
step test of the azonal and total transports (action A43, cluster C13).

A101. The uncorrected section total is F_ovS computed from the uncorrected
overturning, F_raw = -(1/S0) sum_k q_k (S_k - S0), plus the azonal term F_az.
Because the removal shifts every velocity by the same v_c, F_az does not
change, so the uncorrected total is F_raw + F_az. For each product this
script gives the record mean, the trend with the paper's test (OLS with the
corrected N_eff, annual means) and its interval over the product's own
record and over the common 1993 to 2017 window, and the difference from the
corrected total F_ovS + F_az. It also reports the total under the
proportional-to-|V| removal (monthly estimator) to show the sensitivity to
the removal scheme.

Mass budget. With V_net the net section transport removed (Sv), A the wet
area of the section and S-bar the area-weighted mean salinity of the section,
the removal changes F_ovS by

    F_ovS - F_raw = (V_net / S0) (S-bar - S0),

because removing v_c = V_net / A from every cell removes V_net carried at the
section-mean salinity. The script checks this identity month by month
(residual reported) and gives the size of the correction.

A43 (optional). Step test of paper3r1_stats.step_test with a step in 2002
applied to the annual F_az and the annual total F_ovS + F_az for ORAS5. The
trend and the step are fitted together, with the corrected N_eff. Trends are
reported in mSv per yr and steps in mSv.

Inputs: data/results/paper3r1_section_<product>.nc (monthly, F_ov, F_ov_raw,
F_az, V_net_Sv, A_xy, S_bar, e3t), data/results/paper3r1_section_oras5.nc.
Output: PAPER_3_v2/analysis/jc_r2_G04_removal_total.json.
Seeds: none (deterministic; the paper's tests draw no random numbers).
Replicates: none.
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

from analysis_paper3r1_cells import abs_v_removal  # noqa: E402
from analysis_paper3v2_attribution import COMMON_WINDOW, S0, annual, trend_stats  # noqa: E402
from paper3r1_stats import step_test  # noqa: E402

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G04_removal_total.json"
PRODUCTS = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}


def windows(years: np.ndarray) -> dict:
    w = {"own": (int(years[0]), int(years[-1]))}
    if years[0] <= COMMON_WINDOW[0] and years[-1] >= COMMON_WINDOW[1]:
        w["common_1993_2017"] = COMMON_WINDOW
    return w


def trend_block(years: np.ndarray, values: np.ndarray) -> dict:
    t = trend_stats(years, values)
    return {
        "trend_mSv_per_yr": t["trend"],
        "ci_mSv_per_yr": t["ci"],
        "p_paper_test": t["p_santer"],
        "n_eff": t["n_eff"],
        "n_eff_at_floor": t["n_eff_at_floor"],
        "window": t["window"],
        "n_years": t["n"],
    }


def product_block(key: str, label: str) -> dict:
    with xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc") as ds:
        ds.load()
        alt = abs_v_removal(ds)
        alt.load()

    f_ov, f_raw, f_az = ds["F_ov"].values, ds["F_ov_raw"].values, ds["F_az"].values
    unc_total = f_raw + f_az
    cor_total = f_ov + f_az
    alt_total = alt["F_ov"].values + f_az

    # Mass budget: F_ov - F_raw = V_net (S-bar_area - S0) / S0 each month.
    a = ds["A_xy"].values * ds["e3t"].values[None, :]
    sbar = ds["S_bar"].values
    s_area = (a * sbar).sum(axis=1) / a.sum(axis=1)
    v_net = ds["V_net_Sv"].values
    predicted = v_net * (s_area - S0) / S0
    residual = (f_ov - f_raw) - predicted

    years_m = ds["time"].dt.year.values
    years = np.unique(years_m)
    ann = {
        "F_ov": np.array([f_ov[years_m == y].mean() for y in years]),
        "F_raw": np.array([f_raw[years_m == y].mean() for y in years]),
        "F_az": np.array([f_az[years_m == y].mean() for y in years]),
        "unc_total": np.array([unc_total[years_m == y].mean() for y in years]),
        "cor_total": np.array([cor_total[years_m == y].mean() for y in years]),
        "alt_total": np.array([alt_total[years_m == y].mean() for y in years]),
    }
    block = {
        "product": label,
        "record": [int(years[0]), int(years[-1])],
        "means_Sv": {
            "F_ov_corrected": float(f_ov.mean()),
            "F_ov_uncorrected": float(f_raw.mean()),
            "F_az": float(f_az.mean()),
            "total_uncorrected_F_raw_plus_F_az": float(unc_total.mean()),
            "total_corrected_F_ov_plus_F_az": float(cor_total.mean()),
            "total_difference_uncorrected_minus_corrected": float((unc_total - cor_total).mean()),
            "total_proportional_to_absV_F_ov_plus_F_az": float(alt_total.mean()),
        },
        "mass_budget": {
            "identity": "F_ov - F_raw = V_net (S-bar_area - S0) / S0 (monthly)",
            "mean_correction_Sv": float((f_ov - f_raw).mean()),
            "mean_predicted_correction_Sv": float(predicted.mean()),
            "max_abs_residual_Sv": float(np.max(np.abs(residual))),
            "mean_V_net_Sv": float(v_net.mean()),
            "mean_S_area_PSU": float(s_area.mean()),
        },
        "windows": {},
    }
    for wname, (y0, y1) in windows(years).items():
        m = (years >= y0) & (years <= y1)
        y = years[m]
        wb = {
            "uncorrected_total": trend_block(y, ann["unc_total"][m]),
            "corrected_total": trend_block(y, ann["cor_total"][m]),
            "uncorrected_F_ov": trend_block(y, ann["F_raw"][m]),
            "corrected_F_ov": trend_block(y, ann["F_ov"][m]),
            "total_proportional_to_absV": trend_block(y, ann["alt_total"][m]),
            "mean_uncorrected_total_Sv": float(ann["unc_total"][m].mean()),
            "mean_corrected_total_Sv": float(ann["cor_total"][m].mean()),
            "mean_total_difference_Sv": float((ann["unc_total"][m] - ann["cor_total"][m]).mean()),
        }
        wb["sign_of_uncorrected_total_changes"] = bool(
            np.sign(wb["mean_uncorrected_total_Sv"]) != np.sign(wb["mean_corrected_total_Sv"])
        )
        wb["resolved_status_changes"] = bool(
            (wb["uncorrected_total"]["p_paper_test"] < 0.05)
            != (wb["corrected_total"]["p_paper_test"] < 0.05)
        )
        block["windows"][wname] = wb
    return block, years, ann, f_az


def step_block(years: np.ndarray, f_az: np.ndarray, f_tot: np.ndarray, step_year: int) -> dict:
    out = {}
    for name, series in (("F_az", f_az), ("F_ovS_plus_F_az", f_tot)):
        r = step_test(years.astype(float), series, step_year)
        tr, st = r["trend"], r["step"]
        out[name] = {
            "step_year": step_year,
            "trend_mSv_per_yr": tr["value"] * 1e3,
            "trend_ci_mSv_per_yr": [c * 1e3 for c in tr["ci"]],
            "trend_p": tr["p"],
            "trend_n_eff": tr["n_eff"],
            "step_mSv": st["value"] * 1e3,
            "step_ci_mSv": [c * 1e3 for c in st["ci"]],
            "step_p": st["p"],
            "step_n_eff": st["n_eff"],
            "n_years": tr["n"],
            "n_eff_at_floor": bool(tr["n_eff_at_floor"]),
        }
    return out


def main() -> None:
    products = {}
    annual_store = {}
    for key, label in PRODUCTS.items():
        block, years, ann, f_az = product_block(key, label)
        products[key] = block
        annual_store[key] = (years, ann, f_az)

    # A43: ORAS5 step test with a 2002 step, annual values.
    years, ann, f_az = annual_store["oras5"]
    step = step_block(years, ann["F_az"], ann["cor_total"], 2002)

    out = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G04_removal_total.py",
            "actions": ["A101 (cluster C50)", "A43 (cluster C13, optional)"],
            "inputs": [
                "data/results/paper3r1_section_{oras5,glorys12,ecco}.nc",
            ],
            "seeds": None,
            "replicates": None,
            "sign_convention": "northward-positive freshwater transport, Sv; F < 0 is a northward salt-flux deficit",
            "units": {"F": "Sv", "trend": "mSv per yr", "step": "mSv"},
            "trend_test": "analysis_paper3v2_attribution.trend_stats (paper test); A43 uses paper3r1_stats.step_test (regression with corrected N_eff)",
            "aggregation": "annual means of monthly values",
        },
        "A101_uncorrected_total": products,
        "A43_step_test_ORAS5_2002": {
            "record": [int(years[0]), int(years[-1])],
            "results": step,
        },
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for key in PRODUCTS:
        b = products[key]
        print(key, b["means_Sv"], b["mass_budget"]["max_abs_residual_Sv"])
    print(json.dumps(step, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
