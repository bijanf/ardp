#!/usr/bin/env python3
"""Trends of the overturning freshwater transport, the northward-limb transport
and the limb salinity contrast over the Argo era, 2004 to 2025.

For ORAS5 and GLORYS12V1 the annual series of F_ov (overturning freshwater
transport at 34.5 S, mSv per year), T_limb (northward-limb transport, Sv) and
dS_limb (limb salinity contrast, PSU) are regressed on time over 2004 to 2025.
Each trend is tested in two ways:

(a) bias-corrected lag-one autocorrelation and effective sample size of
    paper3r1_stats.trend_test (the test used in the paper);
(b) uncorrected lag-one autocorrelation of the OLS residuals, with the same
    effective sample size formula and standard-error inflation. This is the
    comparison form that the revision reports next to the corrected test.

Trends are reported per year for F_ov (mSv/yr) and per decade for T_limb and
dS_limb (Sv/decade and PSU/decade). Intervals are 95 % intervals of the
corrected test. The script also records a verdict on the sentence "Over
2004--2025 no reanalysis resolves a decline", which is judged on the
bias-corrected test of F_ov. ECCO-V4r4 ends in 2017 and has no 2004--2025
window, so it is not part of the verdict.

Output: PAPER_3_v2/analysis/jc_n23_argo_era.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_attribution import annual  # noqa: E402
from paper3r1_stats import lag1, regression_test, trend_design  # noqa: E402

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_n23_argo_era.json"

YEAR_START = 2004
YEAR_END = 2025
ALPHA = 0.05

PRODUCTS = {
    "ORAS5": "paper3r1_section_oras5.nc",
    "GLORYS12V1": "paper3r1_section_glorys12.nc",
}

# variable key, label, scale to the reported unit, reported unit
VARIABLES = {
    "F_ov": ("F_ovS", 1e3, "mSv/yr"),
    "T_limb": ("T_limb (northward-limb transport)", 10.0, "Sv/decade"),
    "dS_limb": ("dS_limb (limb salinity contrast)", 10.0, "PSU/decade"),
}
FILE_VARS = {"F_ov": "F_ov", "T_limb": "T_limb_Sv", "dS_limb": "dS_limb"}


def load_annual(fname: str) -> tuple[np.ndarray, dict]:
    with xr.open_dataset(RESULTS / fname) as ds:
        a = annual(ds)
        years = a["year"].values.astype(int)
        series = {k: a[v].values.astype(float) for k, v in FILE_VARS.items()}
    keep = (years >= YEAR_START) & (years <= YEAR_END)
    return years[keep].astype(float), {k: v[keep] for k, v in series.items()}


def uncorrected_test(years: np.ndarray, y: np.ndarray) -> dict:
    """Trend test with the raw lag-one autocorrelation of the OLS residuals."""
    x = trend_design(years)
    n, k = x.shape
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    s2 = float(resid @ resid) / (n - k)
    se = float(np.sqrt(s2 * np.linalg.inv(x.T @ x)[1, 1]))
    r1 = float(lag1(resid))
    neff_raw = n * (1.0 - r1) / (1.0 + r1)
    neff = float(min(max(neff_raw, k + 1.0), n))
    dof = neff - k
    se_adj = se * np.sqrt((n - k) / dof)
    value = float(coef[1])
    tcrit = float(stats.t.ppf(1.0 - ALPHA / 2.0, dof))
    p = float(2.0 * stats.t.sf(abs(value) / se_adj, dof)) if se_adj > 0 else 0.0
    return {
        "value": value,
        "ci": [value - tcrit * se_adj, value + tcrit * se_adj],
        "p": p,
        "lag1": r1,
        "n_eff": neff,
    }


def scaled(r: dict, scale: float) -> dict:
    return {
        "value": r["value"] * scale,
        "ci": [float(c * scale) for c in r["ci"]],
        "p": float(r["p"]),
        "lag1": float(r["lag1"]),
        "n_eff": float(r["n_eff"]),
    }


def main() -> None:
    out: dict = {
        "window": [YEAR_START, YEAR_END],
        "alpha": ALPHA,
        "units": {k: v[2] for k, v in VARIABLES.items()},
        "products": {},
    }
    for product, fname in PRODUCTS.items():
        years, series = load_annual(fname)
        block: dict = {"file": f"data/results/{fname}", "n": int(len(years))}
        for key, (label, scale, unit) in VARIABLES.items():
            y = series[key]
            corrected = regression_test(trend_design(years), y, 1)
            raw = uncorrected_test(years, y)
            block[key] = {
                "label": label,
                "unit": unit,
                "corrected": {
                    **scaled(
                        {
                            "value": corrected["value"],
                            "ci": corrected["ci"],
                            "p": corrected["p"],
                            "lag1": corrected["phi_corrected"],
                            "n_eff": corrected["n_eff"],
                        },
                        scale,
                    ),
                },
                "uncorrected_lag1": scaled(raw, scale),
            }
        out["products"][product] = block

    f_corr = {p: out["products"][p]["F_ov"]["corrected"]["p"] for p in PRODUCTS}
    out["verdict"] = {
        "sentence": "Over 2004--2025 no reanalysis resolves a decline",
        "basis": "bias-corrected test of F_ov, ORAS5 and GLORYS12V1",
        "p_corrected_F_ov": f_corr,
        "holds": bool(all(p >= ALPHA for p in f_corr.values())),
        "note": "ECCO-V4r4 ends in 2017 and has no 2004--2025 window.",
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["verdict"], indent=2))
    for product in PRODUCTS:
        for key in VARIABLES:
            c = out["products"][product][key]
            print(
                product,
                key,
                "corr",
                round(c["corrected"]["value"], 4),
                [round(v, 4) for v in c["corrected"]["ci"]],
                round(c["corrected"]["p"], 4),
                "uncorr p",
                round(c["uncorrected_lag1"]["p"], 4),
            )


if __name__ == "__main__":
    main()
