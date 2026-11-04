"""The early ORAS5 record and the timing of the F_ovS decline (paper 3, J Climate).

Input is the annual means of the ORAS5 section fields written by
compute_paper3r1_section_quantities.py (data/results/paper3r1_section_oras5.nc),
with the limb factorisation F_ovS = -(1/S0) T dS and S0 = 35 PSU.

Every trend, step and interval uses the bias-corrected effective sample size of
paper3r1_stats.py. An interval is the inverse of the test, and a trend or step
is resolved when its 95 per cent interval excludes zero.

1. Trends of F_ovS (mSv per year), of the northward-limb transport T (Sv per
   decade) and of the limb contrast dS (PSU per decade) over fixed windows, and
   the transport and contrast terms of F_ovS over each window, with the other
   factor held at its window mean.
2. A hinge fit at 1979 with continuous segments (one slope before and one after
   the hinge), with the slope change tested directly, and a level-shift variant
   (one common trend plus a step at 1979).
3. A break scan of F_ovS over break years 1962 to 1996 with the sup-type
   statistics of Andrews (1993): the sup-F for a change of intercept and slope,
   and the sup of |t| for a level shift with a common trend. Critical values and
   p-values are simulated from AR(1) noise with the design of the scan, the
   record length of the record and the bias-corrected lag-1 coefficient of the
   full-record fit, with a sensitivity grid over the coefficient.
4. A start-by-end matrix of the F_ovS trend for starts 1958 to 2011 and ends
   2014, 2017, 2020 and 2025. As in the start-year scan of
   analysis_paper3v2_attribution.py, a window needs at least 15 years.
5. A two-segment scan of F_ovS over candidate years 1962 to 1996 with a hinge
   at the candidate year (continuous slope change) plus a jump at it. Its
   sup-F against the linear trend has a critical value and p-value simulated
   from AR(1) noise of the record length with the bias-corrected lag-1
   coefficient of the full-record trend fit (the paper's convention), with
   10,000 simulations. Stored under hinge_jump_scan_1962_1996.

Cross-checks read the existing attribution output without changing it.
Writes PAPER_3_v2/analysis/jc_n15_oras5_early.json.
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
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import (  # noqa: E402
    ar1_noise,
    corrected_phi,
    lag1,
    regression_test,
    step_design,
    step_test,
    trend_design,
    trend_test,
)

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_n15_oras5_early.json"

S0 = 35.0
HINGE_YEAR = 1979
BREAK_YEARS = tuple(range(1962, 1997))
SUP_SIMS = 4000
SEED = 20261104
MIN_LEN = 15
START_YEARS = tuple(range(1958, 2012))
END_YEARS = (2014, 2017, 2020, 2025)
WINDOWS = {
    "1958-2025": (1958, 2025),
    "1958-1978": (1958, 1978),
    "1979-2025": (1979, 2025),
    "1958-2014": (1958, 2014),
    "1979-2014": (1979, 2014),
}


# ── helpers ───────────────────────────────────────────────────────────


def test_dict(r: dict, scale: float) -> dict:
    """Corrected test with its value and interval scaled to the reported unit."""
    ci = [float(r["ci"][0] * scale), float(r["ci"][1] * scale)]
    return {
        "value": float(r["value"] * scale),
        "ci95": ci,
        "p": float(r["p"]),
        "n": int(r["n"]),
        "n_eff": float(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
        "lag1": float(r["lag1"]),
        "phi_corrected": float(r["phi_corrected"]),
        "resolved": bool(ci[0] * ci[1] > 0),
    }


def load_annual() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Annual means of F_ovS (Sv), T (Sv) and dS (PSU) of the ORAS5 section."""
    ds = xr.open_dataset(RESULTS / "paper3r1_section_oras5.nc")
    a = ds[["F_ov", "T_limb_Sv", "dS_limb"]].groupby("time.year").mean()
    years = a["year"].values.astype(int)
    fov = a["F_ov"].values.astype(float)
    t = a["T_limb_Sv"].values.astype(float)
    d = a["dS_limb"].values.astype(float)
    lat = float(ds.attrs["section_latitude"])
    ds.close()
    return years, fov, t, d, lat


def window_summary(years: np.ndarray, fov: np.ndarray, t: np.ndarray, d: np.ndarray) -> dict:
    """Trends of F_ovS and its terms over one window."""
    f_mS = fov * 1e3
    tbar, dbar = float(t.mean()), float(d.mean())
    term_t = -(1.0 / S0) * t * dbar * 1e3  # transport term, contrast at window mean
    term_s = -(1.0 / S0) * tbar * d * 1e3  # contrast term, transport at window mean
    out = {
        "window": [int(years[0]), int(years[-1])],
        "n": int(len(years)),
        "mean_F_ovS_mSv": float(f_mS.mean()),
        "mean_T_limb_Sv": tbar,
        "mean_dS_limb_PSU": dbar,
        "F_ovS_mSv_per_yr": test_dict(trend_test(years, f_mS), 1.0),
        "T_limb_Sv_per_decade": test_dict(trend_test(years, t), 10.0),
        "dS_limb_PSU_per_decade": test_dict(trend_test(years, d), 10.0),
        "transport_term_mSv_per_yr": test_dict(trend_test(years, term_t), 1.0),
        "contrast_term_mSv_per_yr": test_dict(trend_test(years, term_s), 1.0),
    }
    out["cross_term_residual_mSv_per_yr"] = float(
        out["F_ovS_mSv_per_yr"]["value"]
        - out["transport_term_mSv_per_yr"]["value"]
        - out["contrast_term_mSv_per_yr"]["value"]
    )
    return out


def hinge_fit(years: np.ndarray, f_mS: np.ndarray) -> dict:
    """Continuous two-segment fit with the hinge at HINGE_YEAR."""
    x = years.astype(float)
    segs = np.column_stack(
        [
            np.ones_like(x),
            np.minimum(x, HINGE_YEAR) - HINGE_YEAR,
            np.maximum(x, HINGE_YEAR) - HINGE_YEAR,
        ]
    )
    early = regression_test(segs, f_mS, 1)
    late = regression_test(segs, f_mS, 2)
    change_design = np.column_stack(
        [np.ones_like(x), x - HINGE_YEAR, np.maximum(x - HINGE_YEAR, 0.0)]
    )
    change = regression_test(change_design, f_mS, 2)
    return {
        "hinge_year": HINGE_YEAR,
        "slope_before_hinge_mSv_per_yr": test_dict(early, 1.0),
        "slope_after_hinge_mSv_per_yr": test_dict(late, 1.0),
        "slope_change_mSv_per_yr": test_dict(change, 1.0),
        "value_at_hinge_mSv": float(early["coef"][0]),
    }


def level_shift_fit(years: np.ndarray, f_mS: np.ndarray) -> dict:
    """Common trend with a level shift at HINGE_YEAR."""
    r = step_test(years, f_mS, HINGE_YEAR)
    return {
        "step_year": HINGE_YEAR,
        "common_trend_mSv_per_yr": test_dict(r["trend"], 1.0),
        "level_shift_mSv": test_dict(r["step"], 1.0),
    }


def _rss(y: np.ndarray, x: np.ndarray, pinv_x: np.ndarray) -> np.ndarray:
    """Residual sum of squares of each row of y (m, n) on design x (n, k)."""
    coef = y @ pinv_x.T
    resid = y - coef @ x.T
    return (resid * resid).sum(axis=1)


def sup_statistics(y: np.ndarray, years: np.ndarray) -> dict:
    """Sup-F (intercept and slope change) and sup |t| (level shift) over the
    break years, for each row of y (m, n), with the break index of each maximum."""
    t = years.astype(float)
    n = len(t)
    y = np.atleast_2d(y)
    base = trend_design(years)
    rss0 = _rss(y, base, np.linalg.pinv(base))
    f_all = np.empty((y.shape[0], len(BREAK_YEARS)))
    t_all = np.empty_like(f_all)
    for j, b in enumerate(BREAK_YEARS):
        dummy = (t >= b).astype(float)
        xf = np.column_stack([base, dummy, dummy * (t - t.mean())])
        rss1 = _rss(y, xf, np.linalg.pinv(xf))
        f_all[:, j] = ((rss0 - rss1) / 2.0) / (rss1 / (n - 4))
        xt = step_design(years, b)
        pt = np.linalg.pinv(xt)
        coef = y @ pt.T
        resid = y - coef @ xt.T
        s2 = (resid * resid).sum(axis=1) / (n - 3)
        var = np.linalg.inv(xt.T @ xt)[2, 2]
        t_all[:, j] = coef[:, 2] / np.sqrt(s2 * var)
    abs_t = np.abs(t_all)
    return {
        "sup_F": f_all.max(axis=1),
        "argmax_F": f_all.argmax(axis=1),
        "sup_abs_t": abs_t.max(axis=1),
        "argmax_t": abs_t.argmax(axis=1),
        "F_by_break": f_all,
        "t_by_break": t_all,
    }


def regime_1979(years: np.ndarray, f_mS: np.ndarray) -> tuple[dict, float]:
    """Two separate lines joined by a jump at HINGE_YEAR (the interaction form).

    Columns are 1, t - 1979, the post-1979 dummy and the dummy times (t - 1979).
    The dummy coefficient is the jump at 1979 (late line minus early line at
    1979); the coefficient on the interaction is the change of slope. Also
    returns the bias-corrected lag-1 coefficient of the residuals of this fit
    with the trend design of the null, a sensitivity value for the break scan."""
    x = years.astype(float)
    dummy = (x >= HINGE_YEAR).astype(float)
    design = np.column_stack(
        [np.ones_like(x), x - HINGE_YEAR, dummy, dummy * (x - HINGE_YEAR)]
    )
    out = {
        "early_slope_mSv_per_yr": test_dict(regression_test(design, f_mS, 1), 1.0),
        "jump_at_1979_mSv": test_dict(regression_test(design, f_mS, 2), 1.0),
        "slope_change_mSv_per_yr": test_dict(regression_test(design, f_mS, 3), 1.0),
        "early_value_at_1979_mSv": None,
    }
    coef, *_ = np.linalg.lstsq(design, f_mS, rcond=None)
    out["early_value_at_1979_mSv"] = float(coef[0])
    out["late_value_at_1979_mSv"] = float(coef[0] + coef[2])
    out["late_slope_mSv_per_yr"] = float(coef[1] + coef[3])
    resid = f_mS - design @ coef
    phi_regime = float(corrected_phi(lag1(resid), trend_design(years)))
    return out, phi_regime


def two_line_series(years: np.ndarray, y: np.ndarray, scale: float) -> dict:
    """Jump at HINGE_YEAR and slope change of the two-line fit, for one factor.

    The jump and the early and late slopes are in the factor's own unit, with
    the slopes per decade when scale is 10 (and the jump unscaled)."""
    x = years.astype(float)
    dummy = (x >= HINGE_YEAR).astype(float)
    design = np.column_stack(
        [np.ones_like(x), x - HINGE_YEAR, dummy, dummy * (x - HINGE_YEAR)]
    )
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "jump_at_1979": test_dict(regression_test(design, y, 2), 1.0),
        "early_slope_per_yr_scaled": test_dict(regression_test(design, y, 1), scale),
        "slope_change_per_yr_scaled": test_dict(regression_test(design, y, 3), scale),
        "late_slope_per_yr_scaled": float(scale * (coef[1] + coef[3])),
        "scale_note": "slopes scaled by " + str(scale),
    }


def break_scan(
    years: np.ndarray, f_mS: np.ndarray, phi_full: float, phi_regime: float
) -> dict:
    """Observed scan and simulated sup-type critical values under AR(1) noise.

    The null is a linear trend with AR(1) noise of the same length. The
    statistics are invariant to the scale and trend of the noise, so unit
    innovations suffice. The primary coefficient is the one of the full-record
    trend fit, the convention of the paper's tests; the coefficient of the
    residuals of the two-line fit at 1979 is reported as a sensitivity value,
    since a jump in the record inflates the residual autocorrelation of a
    trend-only fit."""
    obs_full = sup_statistics(f_mS, years)
    obs = {
        k: obs_full[k][0]
        for k in ("sup_F", "argmax_F", "sup_abs_t", "argmax_t")
    }
    breaks = list(BREAK_YEARS)
    phi_grid = [
        ("fixed_0.0", 0.0),
        ("fixed_0.3", 0.3),
        ("full_record_fit", phi_full),
        ("two_line_residuals_1979", phi_regime),
        ("fixed_0.7", 0.7),
    ]
    by_phi = []
    for i, (label, phi) in enumerate(phi_grid):
        rng = np.random.default_rng(SEED + i)
        sims = sup_statistics(ar1_noise(SUP_SIMS, len(years), phi, rng), years)
        p_f = float(np.mean(sims["sup_F"] >= obs["sup_F"]))
        p_t = float(np.mean(sims["sup_abs_t"] >= obs["sup_abs_t"]))
        by_phi.append(
            {
                "label": label,
                "phi": float(phi),
                "is_primary": bool(label == "full_record_fit"),
                "sup_F_cv95": float(np.quantile(sims["sup_F"], 0.95)),
                "sup_F_p": p_f,
                "sup_abs_t_cv95": float(np.quantile(sims["sup_abs_t"], 0.95)),
                "sup_abs_t_p": p_t,
            }
        )
    fit_row = next(r for r in by_phi if r["is_primary"])
    b_f = breaks[int(obs["argmax_F"])]
    b_t = breaks[int(obs["argmax_t"])]
    pre_f = years < b_f
    post_f = years >= b_f
    seg_f = {
        "pre_break_trend_mSv_per_yr": test_dict(trend_test(years[pre_f], f_mS[pre_f]), 1.0),
        "post_break_trend_mSv_per_yr": test_dict(trend_test(years[post_f], f_mS[post_f]), 1.0),
    }
    return {
        "break_years": [int(breaks[0]), int(breaks[-1])],
        "n_breaks": len(breaks),
        "phi_full_record_fit": float(phi_full),
        "phi_two_line_residuals_1979": float(phi_regime),
        "n_sims": SUP_SIMS,
        "sup_F": {
            "statistic": float(obs["sup_F"]),
            "argmax_break_year": int(b_f),
            "p_sup": fit_row["sup_F_p"],
            "cv95": fit_row["sup_F_cv95"],
            "resolved": bool(fit_row["sup_F_p"] < 0.05),
            "F_by_break": [float(v) for v in obs_full["F_by_break"][0]],
            "segments_at_argmax": seg_f,
        },
        "sup_abs_t_level_shift": {
            "statistic": float(obs["sup_abs_t"]),
            "argmax_break_year": int(b_t),
            "t_at_argmax": float(obs_full["t_by_break"][0][int(obs["argmax_t"])]),
            "p_sup": fit_row["sup_abs_t_p"],
            "cv95": fit_row["sup_abs_t_cv95"],
            "resolved": bool(fit_row["sup_abs_t_p"] < 0.05),
            "t_by_break": [float(v) for v in obs_full["t_by_break"][0]],
        },
        "sensitivity_to_phi": by_phi,
    }


def start_end_matrix(years: np.ndarray, f_mS: np.ndarray) -> dict:
    """F_ovS trend for every start and end, with the resolved flag."""
    ends = []
    for e in END_YEARS:
        cells = []
        for s in START_YEARS:
            m = (years >= s) & (years <= e)
            n = int(m.sum())
            if n < MIN_LEN:
                cells.append({"start": s, "n": n, "status": "not computed (window shorter than 15 years)"})
                continue
            cell = test_dict(trend_test(years[m], f_mS[m]), 1.0)
            cells.append({"start": s, "status": "computed", **cell})
        done = [c for c in cells if c["status"] == "computed"]
        resolved = [c for c in done if c["resolved"]]
        ends.append(
            {
                "end": e,
                "n_computed": len(done),
                "n_resolved": len(resolved),
                "n_resolved_negative": sum(1 for c in resolved if c["value"] < 0),
                "n_resolved_positive": sum(1 for c in resolved if c["value"] > 0),
                "resolved_starts": [c["start"] for c in resolved],
                "cells": cells,
            }
        )
    return {"unit": "mSv per year", "min_length_years": MIN_LEN, "ends": ends}


HJ_SIMS = 10000
HJ_SEED = SEED + 10


def hinge_jump_design(years: np.ndarray, b: int) -> np.ndarray:
    """Columns 1, t, max(t - b, 0) (continuous hinge at b) and 1[t >= b] (jump at b)."""
    t = years.astype(float)
    return np.column_stack(
        [np.ones_like(t), t, np.maximum(t - b, 0.0), (t >= b).astype(float)]
    )


def hinge_jump_F(y: np.ndarray, years: np.ndarray) -> np.ndarray:
    """F of the hinge-plus-jump model against the linear trend, one column per
    candidate year, for each row of y (m, n). The model adds two parameters."""
    y = np.atleast_2d(y)
    n = len(years)
    base = trend_design(years)
    rss0 = _rss(y, base, np.linalg.pinv(base))
    out = np.empty((y.shape[0], len(BREAK_YEARS)))
    for j, b in enumerate(BREAK_YEARS):
        x = hinge_jump_design(years, b)
        rss1 = _rss(y, x, np.linalg.pinv(x))
        out[:, j] = ((rss0 - rss1) / 2.0) / (rss1 / (n - 4))
    return out


def hinge_jump_scan(years: np.ndarray, f_mS: np.ndarray, phi: float) -> dict:
    """Sup-F of the hinge-plus-jump scan over BREAK_YEARS, with the critical
    value and p-value from AR(1) nulls with the lag-one coefficient phi."""
    obs = hinge_jump_F(f_mS, years)[0]
    rng = np.random.default_rng(HJ_SEED)
    sims = hinge_jump_F(ar1_noise(HJ_SIMS, len(years), phi, rng), years)
    sup_null = sims.max(axis=1)
    sup_obs = float(obs.max())
    j = int(obs.argmax())
    runner = int(np.argsort(obs)[-2])
    p_sup = float(np.mean(sup_null >= sup_obs))
    return {
        "model": "hinge at the candidate year (continuous slope change) plus a jump at it; "
        "columns 1, t, max(t - b, 0), 1[t >= b]",
        "null": "linear trend with AR(1) noise of the record length; lag-one coefficient "
        "is the bias-corrected lag-one of the full-record trend fit",
        "candidate_years": [int(BREAK_YEARS[0]), int(BREAK_YEARS[-1])],
        "n_candidates": len(BREAK_YEARS),
        "phi_bias_corrected_full_record": float(phi),
        "n_sims": HJ_SIMS,
        "seed": HJ_SEED,
        "sup_F": sup_obs,
        "argmax_year": int(BREAK_YEARS[j]),
        "runner_up_year": int(BREAK_YEARS[runner]),
        "runner_up_F": float(obs[runner]),
        "p_sup": p_sup,
        "cv95": float(np.quantile(sup_null, 0.95)),
        "cv90": float(np.quantile(sup_null, 0.90)),
        "null_quantiles": {
            "q50": float(np.quantile(sup_null, 0.50)),
            "q90": float(np.quantile(sup_null, 0.90)),
            "q95": float(np.quantile(sup_null, 0.95)),
            "q99": float(np.quantile(sup_null, 0.99)),
        },
        "resolved_at_5pct": bool(p_sup < 0.05),
        "F_by_year": {str(b): float(v) for b, v in zip(BREAK_YEARS, obs)},
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    years, fov, t, d, lat = load_annual()
    f_mS = fov * 1e3
    attr = json.loads((ANALYSIS / "attribution.json").read_text())["oras5"]

    full = trend_test(years.astype(float), f_mS)
    phi_full = float(full["phi_corrected"])
    regime, phi_regime = regime_1979(years, f_mS)

    checks = {
        "section_latitude": lat,
        "record": [int(years[0]), int(years[-1])],
        "max_abs_diff_F_ov_vs_attribution_Sv": float(
            np.max(np.abs(fov - np.asarray(attr["F_ov_Sv"])))
        ),
        "record_mean_F_ovS_mSv_here": float(f_mS.mean()),
        "record_mean_F_ovS_mSv_attribution": float(attr["mean_F_ov_Sv"] * 1e3),
        "trend_1958_2025_mSv_per_yr_attribution": float(
            attr["trends"]["own record"]["F_ov"]["trend"]
        ),
        "trend_1958_2025_mSv_per_yr_here": float(full["value"] * 1e3),
    }

    windows = {
        name: window_summary(years[(years >= a) & (years <= b)],
                             fov[(years >= a) & (years <= b)],
                             t[(years >= a) & (years <= b)],
                             d[(years >= a) & (years <= b)])
        for name, (a, b) in WINDOWS.items()
    }

    hinge = hinge_fit(years, f_mS)
    level = level_shift_fit(years, f_mS)
    scan = break_scan(years, f_mS, phi_full, phi_regime)
    matrix = start_end_matrix(years, f_mS)
    hj = hinge_jump_scan(years, f_mS, phi_full)
    hj_F = np.array([hj["F_by_year"][str(b)] for b in BREAK_YEARS])
    checks["hinge_jump_max_abs_diff_F_vs_break_scan"] = float(
        np.max(np.abs(hj_F - np.asarray(scan["sup_F"]["F_by_break"])))
    )

    # Start-year scan of the existing attribution output, for the end 2025 row.
    existing = {r["start"]: r["trend_total_mSv_per_yr"] for r in attr["start_year_scan"]}
    end25 = next(e for e in matrix["ends"] if e["end"] == 2025)
    diffs = [
        abs(c["value"] - existing[c["start"]])
        for c in end25["cells"]
        if c["status"] == "computed"
    ]
    checks["start_year_scan_end2025_max_abs_diff_mSv_per_yr"] = float(max(diffs))
    checks["start_year_scan_end2025_starts_compared"] = int(len(diffs))

    result = {
        "id": "N15",
        "product": "ORAS5",
        "units": {
            "F_ovS": "mSv per year (trends), mSv (levels)",
            "T_limb": "Sv per decade",
            "dS_limb": "PSU per decade",
            "intervals": "95 per cent, from the bias-corrected effective sample size",
        },
        "reference_salinity_PSU": S0,
        "method": {
            "trend_test": "paper3r1_stats.trend_test (corrected N_eff)",
            "hinge_and_step": "paper3r1_stats.regression_test and step_test",
            "two_line_jump": "interaction form with a jump at 1979 (regression_test)",
            "break_scan": "Andrews (1993) sup-F and sup-|t| with AR(1) simulated critical values",
            "matrix_min_length_years": MIN_LEN,
        },
        "checks": checks,
        "full_record_trend": test_dict(full, 1.0),
        "windows": windows,
        "hinge_1979": hinge,
        "level_shift_1979": level,
        "two_line_jump_1979": regime,
        "two_line_jump_1979_by_factor": {
            "F_ovS_mSv": two_line_series(years, f_mS, 1.0),
            "T_limb_Sv": two_line_series(years, t, 10.0),
            "dS_limb_PSU": two_line_series(years, d, 10.0),
        },
        "break_scan": scan,
        "hinge_jump_scan_1962_1996": hj,
        "start_end_matrix": matrix,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(result), indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
