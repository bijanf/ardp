#!/usr/bin/env python3
"""Transport share of the F_ovS trend after fitting an observing-system step.

Each of F_ovS and its two counterfactuals, F_T(t) = -T(t) <dS> / S0 and
F_S(t) = -<T> dS(t) / S0, is fitted as a + b t + c I(t >= t_step), so that the
level shift and the trend are estimated jointly. The step years are the 2004
Argo transition for GLORYS12V1 and the 2015 join of the consolidated and
operational streams for ORAS5. The tests use the corrected effective sample
size of ``paper3r1_stats.py`` for the step design; the share |b_T| / |b_F| uses
paired draws of the joint AR(1) sieve bootstrap and is quoted only where the
step-adjusted trend of F_ovS is resolved.

Because the split between a step and a trend depends on the year and on the
shape of the transition, the script also scans single-step models over a
range of years and fits a linear ramp over three years in place of the step.

Writes ``PAPER_3_v2/analysis/r1_step_adjusted.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_attribution import S0, annual  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    ratio_summary,
    regression_test,
    sieve_coefficients,
    step_design,
)

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_step_adjusted.json"
STEPS = {"glorys12": 2004, "oras5": 2015}
SCAN = {"glorys12": range(1998, 2013), "oras5": range(2008, 2021)}
RNG = np.random.default_rng(20261105)


def ramp_design(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """Linear transition from 0 at `start` to 1 at `end` in place of a step."""
    r = np.clip((x - start) / (end - start), 0.0, 1.0)
    return np.column_stack([np.ones_like(x), x - x.mean(), r])


def step_summary(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> dict:
    tr = regression_test(design, y, 1)
    st = regression_test(design, y, 2)
    resid = y - design @ np.array(tr["coef"])
    return {
        "trend_mSv_per_yr": tr["value"] * 1e3,
        "trend_ci_mSv_per_yr": [c * 1e3 for c in tr["ci"]],
        "p_trend": tr["p"],
        "step_Sv": st["value"],
        "step_ci_Sv": st["ci"],
        "p_step": st["p"],
        "n_eff": tr["n_eff"],
        "rss": float(resid @ resid),
    }


def main() -> None:
    out = {}
    for key, step in STEPS.items():
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
        ann = annual(ds)
        x = ann["year"].values.astype(float)
        t = ann["T_limb_Sv"].values
        d = ann["dS_limb"].values
        f = ann["F_ov"].values
        f_t = -(1 / S0) * t * d.mean()
        f_s = -(1 / S0) * t.mean() * d
        design = step_design(x, step)
        res = {
            name: step_summary(x, ser, design)
            for name, ser in (("F", f), ("T_term", f_t), ("S_term", f_s))
        }
        draws = sieve_coefficients(design, np.column_stack([f, f_t]), 1, RNG)
        share = ratio_summary(
            draws[:, 1],
            draws[:, 0],
            abs(res["T_term"]["trend_mSv_per_yr"] / res["F"]["trend_mSv_per_yr"]),
        )
        span = x[-1] - x[0]
        trend_change = res["F"]["trend_mSv_per_yr"] * 1e-3 * span
        out[key] = {
            "step_year": step,
            "record": [int(x[0]), int(x[-1])],
            "fits": res,
            "trend_F_mSv_per_yr": res["F"]["trend_mSv_per_yr"],
            "step_F_Sv": res["F"]["step_Sv"],
            "trend_T_term_mSv_per_yr": res["T_term"]["trend_mSv_per_yr"],
            "trend_S_term_mSv_per_yr": res["S_term"]["trend_mSv_per_yr"],
            "F_trend_resolved": bool(res["F"]["p_trend"] < 0.05),
            "share_T_point": share["point"],
            "share_T_ci": share["ci"],
            "change_from_trend_over_record_Sv": float(trend_change),
            "step_fraction_of_fitted_change": float(
                res["F"]["step_Sv"] / (res["F"]["step_Sv"] + trend_change)
            ),
            "step_year_scan": [
                {"step_year": int(sy), **step_summary(x, f, step_design(x, sy))}
                for sy in SCAN[key]
            ],
            "ramp_fit": {
                "start": step - 1,
                "end": step + 1,
                **step_summary(x, f, ramp_design(x, step - 1, step + 1)),
            },
        }
        scan = out[key]["step_year_scan"]
        best = min(scan, key=lambda r: r["rss"])
        print(
            f"{key}: trend {res['F']['trend_mSv_per_yr']:+.2f} "
            f"(p={res['F']['p_trend']:.3f}), step {res['F']['step_Sv']:+.3f} Sv "
            f"(p={res['F']['p_step']:.3f}); step fraction "
            f"{out[key]['step_fraction_of_fitted_change']:.2f}; share "
            f"{share['point']:.3f} [{share['ci'][0]:.3f}, {share['ci'][1]:.3f}]; "
            f"best step year {best['step_year']}"
        )
        for r in scan:
            print(
                f"   step {r['step_year']}: {r['step_Sv']:+.3f} Sv, trend "
                f"{r['trend_mSv_per_yr']:+.2f}, rss {r['rss']:.4f}"
            )
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
