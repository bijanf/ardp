#!/usr/bin/env python3
"""Transport shares and stepped residual of the ORAS5 F_ovS trend (review round 2, group G01).

Inputs: data/results/paper3r1_section_oras5.nc (annual means over 1958 to 2025
via analysis_paper3v2_attribution.annual).

Decomposition used throughout, as in analysis_paper3r1_step_adjusted.py:
    F    = F_ovS(t)                          (Sv)
    F_T  = -(1/S0) T(t) <dS>                 transport term (Sv)
    F_S  = -(1/S0) <T> dS(t)                 salinity term (Sv)
    R    = F - F_T - F_S                     stepped residual (Sv)
with S0 = 35 PSU and <.> the record mean. Trends are reported in mSv per year.

Three specifications of the trend design, all on the window 1958 to 2025:
    unstepped         a + b t
    2015_only         a + b t + c I(t >= 2015)
    joint_2004_2015   a + b t + c1 I(t >= 2004) + c2 I(t >= 2015)

Transport share of the F trend: |b_FT| / |b_F|.

Intervals: the AR(1) sieve bootstrap of paper3r1_stats.sieve_coefficients
(corrected coefficients, joint innovation resampling), N_BOOT draws, seed SEED.
Draws of the series passed in one call are paired.

Outputs: PAPER_3_v2/analysis/jc_r2_G01_steps_oras5.json
Run from the worktree root: python3 scripts/analysis_paper3jc_r2_G01_steps_oras5.py
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

from analysis_paper3v2_attribution import S0, annual  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    N_BOOT,
    ratio_summary,
    regression_test,
    sieve_coefficients,
    trend_design,
)

SEED = 20261104
RESULTS = REPO / "data" / "results" / "paper3r1_section_oras5.nc"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G01_steps_oras5.json"


def design_for(spec: str, x: np.ndarray) -> np.ndarray:
    t = x - x.mean()
    ones = np.ones_like(x)
    if spec == "unstepped":
        return trend_design(x)
    if spec == "2015_only":
        return np.column_stack([ones, t, (x >= 2015).astype(float)])
    if spec == "joint_2004_2015":
        return np.column_stack(
            [ones, t, (x >= 2004).astype(float), (x >= 2015).astype(float)]
        )
    raise ValueError(spec)


def test_pack(res: dict, scale: float, unit: str) -> dict:
    """Value, OLS-based analytic interval, n_eff and p (JSON only) of one coefficient."""
    return {
        "value": res["value"] * scale,
        "ci_analytic": [c * scale for c in res["ci"]],
        "n_eff": res["n_eff"],
        "n_eff_at_floor": res["n_eff_at_floor"],
        "p": res["p"],
        "unit": unit,
    }


def share_block(design, f, ft) -> dict:
    """Share |b_FT| / |b_F| with the paired sieve interval."""
    rf = regression_test(design, f, 1)
    rt = regression_test(design, ft, 1)
    point = abs(rt["value"] / rf["value"])
    draws = sieve_coefficients(
        design, np.column_stack([f, ft]), 1, np.random.default_rng(SEED), N_BOOT
    )
    s = ratio_summary(draws[:, 1], draws[:, 0], point)
    return {
        "share_point": point,
        "share_ci_sieve_2.5_97.5": s["ci"],
        "share_percent_point": 100 * point,
        "share_percent_ci": [100 * s["ci"][0], 100 * s["ci"][1]],
        "prob_share_below_0p25": s["prob_below_0p25"],
        "prob_opposite_sign": s["prob_opposite_sign"],
        "F_trend": test_pack(rf, 1e3, "mSv/yr"),
        "FT_trend": test_pack(rt, 1e3, "mSv/yr"),
        "n_boot": int(N_BOOT),
    }


def main() -> None:
    with xr.open_dataset(RESULTS) as ds:
        a = annual(ds)
        x = a["year"].values.astype(float)
        t = a["T_limb_Sv"].values.astype(float)
        d = a["dS_limb"].values.astype(float)
        f = a["F_ov"].values.astype(float)
    f_t = -(1 / S0) * t * d.mean()
    f_s = -(1 / S0) * t.mean() * d
    resid = f - f_t - f_s

    out = {
        "meta": {
            "id": "jc_r2_G01_steps_oras5",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G01_steps_oras5.py",
            "action_ids": ["A11", "A65"],
            "cluster_ids": ["C02", "C23"],
            "inputs": [
                "data/results/paper3r1_section_oras5.nc",
                "scripts/analysis_paper3v2_attribution.py (S0, annual)",
                "scripts/paper3r1_stats.py (regression_test, sieve_coefficients, ratio_summary)",
            ],
            "record": [int(x[0]), int(x[-1])],
            "years_in_record": int(len(x)),
            "seed": SEED,
            "n_boot_sieve": int(N_BOOT),
            "units": {
                "F, F_T, F_S, R": "Sv internally; trends reported in mSv per year",
                "shares": "dimensionless fraction, and percent",
                "steps": "Sv",
                "n_eff": "effective sample size (bias-corrected lag-one)",
            },
            "interval_definition": (
                "sieve bootstrap 2.5 and 97.5 percentiles (share_ci_sieve_2.5_97.5, "
                "ci_sieve_2.5_97.5); analytic t interval from regression_test as ci_analytic"
            ),
            "p_values": "present in JSON only, from regression_test with corrected N_eff",
            "S0_PSU": S0,
        },
        "A11_transport_share": {},
        "A65_stepped_residual": {},
    }

    for spec in ("unstepped", "2015_only", "joint_2004_2015"):
        design = design_for(spec, x)
        blk = share_block(design, f, f_t)
        blk["window"] = [int(x[0]), int(x[-1])]
        blk["steps"] = {
            "unstepped": [],
            "2015_only": [2015],
            "joint_2004_2015": [2004, 2015],
        }[spec]
        if spec == "2015_only":
            blk["F_step_2015_Sv"] = test_pack(regression_test(design, f, 2), 1.0, "Sv")
        if spec == "joint_2004_2015":
            blk["F_step_2004_Sv"] = test_pack(regression_test(design, f, 2), 1.0, "Sv")
            blk["F_step_2015_Sv"] = test_pack(regression_test(design, f, 3), 1.0, "Sv")
            blk["T_term_step_2004_Sv"] = test_pack(
                regression_test(design, f_t, 2), 1.0, "Sv"
            )
            blk["T_term_step_2015_Sv"] = test_pack(
                regression_test(design, f_t, 3), 1.0, "Sv"
            )
        out["A11_transport_share"][spec] = blk
        print(
            f"A11 {spec}: share {blk['share_percent_point']:.2f}% "
            f"[{blk['share_percent_ci'][0]:.2f}, {blk['share_percent_ci'][1]:.2f}]"
            f" n_eff F {blk['F_trend']['n_eff']:.2f}, FT {blk['FT_trend']['n_eff']:.2f}"
        )

    design15 = design_for("2015_only", x)
    draws = sieve_coefficients(
        design15,
        np.column_stack([f, f_t, f_s]),
        1,
        np.random.default_rng(SEED),
        N_BOOT,
    )
    rd = draws[:, 0] - draws[:, 1] - draws[:, 2]
    r_res = regression_test(design15, resid, 1)
    r_f = regression_test(design15, f, 1)
    r_t = regression_test(design15, f_t, 1)
    r_s = regression_test(design15, f_s, 1)
    pt = r_f["value"] - r_t["value"] - r_s["value"]
    assert abs(pt - r_res["value"]) < 1e-12, (pt, r_res["value"])
    ratio_res_t = ratio_summary(rd, draws[:, 1], abs(pt / r_t["value"]))
    # The ratio is |R| / |F_T| in both the point and the paired draws.
    lo, hi = np.percentile(rd, [2.5, 97.5])
    out["A65_stepped_residual"] = {
        "specification": "2015_only: a + b t + c I(t >= 2015), window 1958-2025",
        "window": [int(x[0]), int(x[-1])],
        "steps": [2015],
        "residual_trend": {
            "point_mSv_per_yr": pt * 1e3,
            "ci_sieve_2.5_97.5_mSv_per_yr": [lo * 1e3, hi * 1e3],
            "ci_analytic_on_R_mSv_per_yr": [c * 1e3 for c in r_res["ci"]],
            "n_eff_on_R": r_res["n_eff"],
            "p_on_R": r_res["p"],
        },
        "component_trends": {
            "F": test_pack(r_f, 1e3, "mSv/yr"),
            "transport_term_F_T": test_pack(r_t, 1e3, "mSv/yr"),
            "salinity_term_F_S": test_pack(r_s, 1e3, "mSv/yr"),
        },
        "residual_over_transport_term": {
            "definition": "|R| / |F_T| (absolute values)",
            "point": float(abs(pt / r_t["value"])),
            "ci_sieve_2.5_97.5": ratio_res_t["ci"],
        },
        "n_boot": int(N_BOOT),
        "paired": "R, F_T and F_S trend draws come from one sieve call on the same innovation indices",
    }
    print(
        f"A65 residual {pt*1e3:+.4f} mSv/yr, sieve [{lo*1e3:+.4f}, {hi*1e3:+.4f}], "
        f"analytic on R [{r_res['ci'][0]*1e3:+.4f}, {r_res['ci'][1]*1e3:+.4f}], "
        f"n_eff(R) {r_res['n_eff']:.2f}"
    )
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
