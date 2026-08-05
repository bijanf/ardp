#!/usr/bin/env python3
"""Splice-step regressions, clean windows, and fractional rates.

Three things referees asked for that the main attribution does not provide.

1. **The ORAS5 splice.** The Copernicus distribution joins the consolidated
   reanalysis (to 2014) to the operational stream (2015 onwards), and the record
   mean steps across the join. Any trend fitted across the join absorbs part of
   that step. Here every ORAS5 trend is refitted as
   ``F(t) = a + b t + c I(t >= 2015)`` so the step and the trend are separated,
   and the step is estimated once, by regression, rather than by three different
   informal routes.

2. **The clean window.** 2004 to 2014 is the only window that is both inside
   the Argo era and inside the homogeneous consolidated stream. It is short, so
   its minimum detectable trend is large, but it is the one window with neither
   confound.

3. **Fractional rates.** The transport share is a ratio of trends and inherits
   the pathologies of a ratio. The same comparison expressed as fractional rates
   of change, ``trend(X)/mean(X)`` per decade, is dimensionless, needs no
   denominator that might approach zero, and states the result directly: F_ovS
   changes by tens of percent of its own magnitude per decade while the
   transport that carries it changes by a few percent.

Writes ``PAPER_3_v2/analysis/stepfit.json``.
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

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"

SPLICE_YEAR = 2015
SECTIONS = {
    "oras5": ("ORAS5", "paper3v2_section_oras5.nc"),
    "glorys12": ("GLORYS12V1", "paper3v2_section_glorys12.nc"),
    "ecco": ("ECCO-V4r4", "paper3v2_section_ecco.nc"),
}
WINDOWS = {
    "own record": None,
    "common": (1993, 2017),
    "argo": (2004, None),
    "clean 2004-2014": (2004, 2014),
}


def step_fit(years: np.ndarray, v: np.ndarray, step_year: int) -> dict | None:
    """OLS with a level shift, with the residual autocorrelation accounted for.

    The standard errors are inflated by the same Santer factor used elsewhere,
    computed from the lag-1 autocorrelation of the residuals of this fit.
    """
    x = years.astype(float)
    d = (years >= step_year).astype(float)
    if d.sum() < 3 or (1 - d).sum() < 3:
        return None
    design = np.column_stack([np.ones_like(x), x - x.mean(), d])
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    resid = v - design @ beta
    n = len(x)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    r1 = min(max(r1, -0.99), 0.99)
    neff = max(n * (1 - r1) / (1 + r1), 3.0)
    dof = max(neff - design.shape[1], 1.0)
    s2 = float(resid @ resid) / (n - design.shape[1])
    cov = s2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov)) * np.sqrt((n - design.shape[1]) / dof)
    t_slope = beta[1] / se[1]
    t_step = beta[2] / se[2]
    return {
        "trend_mSv_per_yr": float(beta[1] * 1e3),
        "trend_se": float(se[1] * 1e3),
        "trend_p": float(2 * stats.t.sf(abs(t_slope), df=dof)),
        "step_Sv": float(beta[2]),
        "step_se": float(se[2]),
        "step_p": float(2 * stats.t.sf(abs(t_step), df=dof)),
        "n": int(n),
        "n_eff": float(neff),
        "dof": float(dof),
    }


def main() -> None:
    out: dict = {"splice_year": SPLICE_YEAR}

    for key, (label, fname) in SECTIONS.items():
        with xr.open_dataset(RESULTS / fname) as ds:
            a = ds.groupby("time.year").mean()
        years = a["year"].values.astype(int)
        fov = a["F_ov"].values
        t_limb = a["T_limb_Sv"].values
        s_up = a["S_north"].values

        rec: dict = {"label": label, "record": [int(years[0]), int(years[-1])]}

        # Fractional rates: trend divided by the mean, per decade.
        rec["fractional_rates"] = {}
        for wname, span in WINDOWS.items():
            if span is None:
                m = np.ones(len(years), bool)
            else:
                y0 = span[0]
                y1 = span[1] if span[1] is not None else int(years[-1])
                m = (years >= y0) & (years <= y1)
            if m.sum() < 10:
                continue
            entry = {}
            for vname, series in (
                ("F_ov", fov),
                ("T_limb", t_limb),
                ("S_north", s_up),
            ):
                fit = ols_santer(years[m].astype(float), series[m])
                mean = float(series[m].mean())
                entry[vname] = {
                    "trend_per_decade": fit["slope"] * 10,
                    "mean": mean,
                    "fractional_pct_per_decade": 100.0 * fit["slope"] * 10 / abs(mean),
                    "p_santer": fit["p_santer"],
                    "n_eff": fit["n_eff"],
                }
            entry["ratio_fractional_T_over_F"] = abs(
                entry["T_limb"]["fractional_pct_per_decade"]
                / entry["F_ov"]["fractional_pct_per_decade"]
            )
            rec["fractional_rates"][wname] = entry

        # The splice affects ORAS5 only; the other products have one stream.
        if key == "oras5":
            rec["step_fits"] = {}
            for wname, span in (
                ("own record", (int(years[0]), int(years[-1]))),
                ("argo", (2004, int(years[-1]))),
            ):
                m = (years >= span[0]) & (years <= span[1])
                fits = {}
                for vname, series in (
                    ("F_ov", fov),
                    ("T_limb", t_limb),
                    ("S_north", s_up),
                ):
                    r = step_fit(years[m], series[m], SPLICE_YEAR)
                    if r is not None:
                        fits[vname] = r
                rec["step_fits"][wname] = fits
            # The step estimated three ways, so the paper can quote one number.
            cons = fov[years <= 2014]
            oper = fov[years >= 2015]
            rec["step_estimates"] = {
                "difference_of_stream_means_Sv": float(oper.mean() - cons.mean()),
                "difference_of_adjacent_decades_Sv": float(
                    oper.mean() - fov[(years >= 2005) & (years <= 2014)].mean()
                ),
                "regression_full_record_Sv": rec["step_fits"]["own record"]["F_ov"][
                    "step_Sv"
                ],
                "regression_argo_Sv": rec["step_fits"]["argo"]["F_ov"]["step_Sv"],
            }
        out[key] = rec

    for key, rec in out.items():
        if key == "splice_year":
            continue
        print(f"\n{'=' * 72}\n{rec['label']}\n{'=' * 72}")
        for wname, e in rec["fractional_rates"].items():
            print(
                f"  {wname:15s} F_ov "
                f"{e['F_ov']['fractional_pct_per_decade']:+7.1f}%/dec"
                f"  T {e['T_limb']['fractional_pct_per_decade']:+6.2f}%/dec"
                f"  S_north {e['S_north']['fractional_pct_per_decade']:+6.3f}%/dec"
                f"   ratio T/F {e['ratio_fractional_T_over_F']:.4f}"
            )
        if "step_fits" in rec:
            for wname, fits in rec["step_fits"].items():
                f = fits.get("F_ov")
                if f:
                    print(
                        f"\n  step fit, {wname}: trend {f['trend_mSv_per_yr']:+.3f} "
                        f"+/- {f['trend_se']:.3f} mSv/yr (p={f['trend_p']:.3f}), "
                        f"step {f['step_Sv']:+.4f} +/- {f['step_se']:.4f} Sv "
                        f"(p={f['step_p']:.4f}), n={f['n']}, dof={f['dof']:.1f}"
                    )
            print("\n  step estimated three ways:")
            for k, v in rec["step_estimates"].items():
                print(f"     {k:38s} {v:+.4f} Sv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "stepfit.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
