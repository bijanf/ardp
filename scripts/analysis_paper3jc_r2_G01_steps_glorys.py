#!/usr/bin/env python3
"""Interval for the GLORYS12V1 2004 step share of the fitted change (review round 2, group G01, A89).

Inputs: data/results/paper3r1_section_glorys12.nc (annual means over 1993 to 2025
via analysis_paper3v2_attribution.annual).

Model for each variable y in {F_ovS (Sv), dS_limb (PSU)}:
    y(t) = a + b (t - mean t) + c I(t >= 2004),   t = 1993 ... 2025.

Step share of the fitted change over the record:
    share_y = c_y / (c_y + b_y * span),   span = 2025 - 1993 = 32 years,
which is the definition of step_fraction_of_fitted_change in
PAPER_3_v2/analysis/r1_step_adjusted.json (F_ovS) extended to dS.

Point values come from OLS with regression_test (bias-corrected N_eff, analytic
intervals). Intervals come from the paired AR(1) sieve bootstrap of
paper3r1_stats.sieve_coefficients with N_BOOT draws and seed SEED: F and dS are
resampled jointly, and the trend and step coefficients are drawn from two calls
with the same seed, which gives identical resampling indices and so paired draws.

Outputs: PAPER_3_v2/analysis/jc_r2_G01_steps_glorys.json
Run from the worktree root: python3 scripts/analysis_paper3jc_r2_G01_steps_glorys.py
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

from analysis_paper3v2_attribution import annual  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    N_BOOT,
    regression_test,
    sieve_coefficients,
    step_design,
)

SEED = 20261104
STEP_YEAR = 2004
RESULTS = REPO / "data" / "results" / "paper3r1_section_glorys12.nc"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G01_steps_glorys.json"


def fraction_summary(step: np.ndarray, trend: np.ndarray, span: float) -> dict:
    """Step share step / (step + trend * span) from paired draws, with percentiles."""
    frac = step / (step + trend * span)
    lo, hi = np.percentile(frac, [2.5, 97.5])
    return {
        "ci_sieve_2.5_97.5": [float(lo), float(hi)],
        "median_of_draws": float(np.median(frac)),
        "prob_outside_0_1": float(np.mean((frac < 0) | (frac > 1))),
        "frac_draws_finite": float(np.mean(np.isfinite(frac))),
    }


def main() -> None:
    with xr.open_dataset(RESULTS) as ds:
        a = annual(ds)
        x = a["year"].values.astype(float)
        f = a["F_ov"].values.astype(float)
        d = a["dS_limb"].values.astype(float)

    design = step_design(x, STEP_YEAR)
    span = float(x[-1] - x[0])

    fits = {}
    for name, y, unit_trend, scale_trend in (
        ("F_ovS", f, "mSv/yr", 1e3),
        ("dS_limb", d, "PSU/decade", 10.0),
    ):
        rt = regression_test(design, y, 1)
        rs = regression_test(design, y, 2)
        fits[name] = {
            "trend": {
                "value": rt["value"] * scale_trend,
                "ci_analytic": [c * scale_trend for c in rt["ci"]],
                "unit": unit_trend,
                "n_eff": rt["n_eff"],
                "n_eff_at_floor": rt["n_eff_at_floor"],
                "p": rt["p"],
            },
            "step": {
                "value": rs["value"],
                "ci_analytic": list(rs["ci"]),
                "unit": "Sv" if name == "F_ovS" else "PSU",
                "n_eff": rs["n_eff"],
                "n_eff_at_floor": rs["n_eff_at_floor"],
                "p": rs["p"],
            },
            "trend_change_over_record": rt["value"] * span,
            "step_share_point": rs["value"]
            / (rs["value"] + rt["value"] * span),
        }

    ys = np.column_stack([f, d])
    draws_trend = sieve_coefficients(design, ys, 1, np.random.default_rng(SEED), N_BOOT)
    draws_step = sieve_coefficients(design, ys, 2, np.random.default_rng(SEED), N_BOOT)

    shares = {}
    for i, name in enumerate(("F_ovS", "dS_limb")):
        s = fraction_summary(draws_step[:, i], draws_trend[:, i], span)
        s["point"] = fits[name]["step_share_point"]
        s["step_draws_ci_Sv_or_PSU"] = [
            float(v) for v in np.percentile(draws_step[:, i], [2.5, 97.5])
        ]
        s["trend_draws_ci_per_yr_native"] = [
            float(v) for v in np.percentile(draws_trend[:, i], [2.5, 97.5])
        ]
        shares[name] = s
        print(
            f"{name}: step share point {s['point']:.3f}, "
            f"sieve [{s['ci_sieve_2.5_97.5'][0]:.3f}, {s['ci_sieve_2.5_97.5'][1]:.3f}], "
            f"median {s['median_of_draws']:.3f}"
        )

    out = {
        "meta": {
            "id": "jc_r2_G01_steps_glorys",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G01_steps_glorys.py",
            "action_ids": ["A89"],
            "cluster_ids": ["C38"],
            "inputs": [
                "data/results/paper3r1_section_glorys12.nc",
                "scripts/analysis_paper3v2_attribution.py (annual)",
                "scripts/paper3r1_stats.py (regression_test, sieve_coefficients, step_design)",
            ],
            "specification": "y = a + b (t - mean t) + c I(t >= 2004), GLORYS12V1",
            "window": [int(x[0]), int(x[-1])],
            "years_in_record": int(len(x)),
            "steps": [STEP_YEAR],
            "span_years": span,
            "seed": SEED,
            "n_boot_sieve": int(N_BOOT),
            "units": {
                "F_ovS trend": "mSv/yr",
                "F_ovS step": "Sv",
                "dS_limb trend": "PSU/decade",
                "dS_limb step": "PSU",
                "shares": "dimensionless fraction",
                "n_eff": "effective sample size (bias-corrected lag-one)",
            },
            "share_definition": "step / (step + trend * span)",
            "interval_definition": "sieve bootstrap 2.5 and 97.5 percentiles; analytic t intervals as ci_analytic",
            "p_values": "present in JSON only, from regression_test with corrected N_eff",
        },
        "fits_OLS": fits,
        "step_share_sieve": shares,
        "reproduction_note": (
            "F_ovS step_share_point should equal step_fraction_of_fitted_change = 0.5003 "
            "in PAPER_3_v2/analysis/r1_step_adjusted.json (glorys12)"
        ),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
