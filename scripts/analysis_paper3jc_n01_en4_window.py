#!/usr/bin/env python3
"""Window, break and observing-density tests for the EN4 limb salinity contrast.

The observation-only test (``analysis_paper3v2_en4.py``) holds the circulation
fixed at the record-mean profile of each reanalysis and lets only the EN4
salinity vary. This script asks how the resulting annual northward-limb
salinity S_N and contrast dS = S_N - S_S depend on the choice of window, and
whether a change near 2017 is resolved, for two frozen circulations (ORAS5 and
GLORYS12V1) and two vertical extents (full column and the upper 2000 m).

Tests, all with the bias-corrected effective sample size of
``paper3r1_stats.py`` (the test used throughout the paper):

1. Start-year sensitivity: linear trend over 2005 to 2024 with the start year
   moved from 2005 to 2010, and end-year sensitivity: trend over 2005 to the end
   year, with the end year moved from 2015 to 2024.
2. Window trends over 2005 to 2017 and over 2018 to 2024.
3. Break tests on the years 2012 to 2020:
   a. a step model (trend plus a level shift at year Y), scanned over Y, with a
      sup-type critical value and p value from simulated AR(1) noise with the
      bias-corrected lag-1 coefficient of the trend-only residuals (the noise
      is the null of no break; the t statistic of a column of the design is
      invariant to adding the fitted signal, so the simulated noise can be
      tested directly);
   b. a hinge (continuous change of slope) at 2017, and the same hinge scanned
      over knots 2012 to 2020 with the same sup-type null;
   c. a sub-period model with separate intercepts and slopes for 2005 to 2017
      and 2018 to 2024, with the slope difference as a contrast.
4. Observing-density covariate: the annual transport-weighted EN4 observation
   weight of each limb, regressed together with time on the annual contrast.
   If the time coefficient keeps an interval that excludes zero, the trend is
   not explained by the weight alone.

Monthly limb quantities are computed exactly as in ``analysis_paper3v2_en4.py``
and then averaged to years; the full-column annual contrast is checked against
``PAPER_3_v2/analysis/en4.json``. Trends are reported per decade.

Writes ``PAPER_3_v2/analysis/jc_n01_en4_window.json``.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper3r1_stats as ps  # noqa: E402
from analysis_paper3v2_en4 import EN4_DIR, frozen_velocity  # noqa: E402
from analysis_paper3v2_en4_depth_test import (  # noqa: E402
    ARGO_FLOOR,
    en4_month_full,
    limbs,
)

OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
OUT_FILE = OUT_DIR / "jc_n01_en4_window.json"

SEED = 20261104
N_NULL = 10_000
SCAN_YEARS = tuple(range(2012, 2021))
BREAK_YEAR = 2017
POST_START = 2018
PRODUCTS = (
    ("oras5", "paper3r1_section_oras5.nc"),
    ("glorys12", "paper3r1_section_glorys12.nc"),
)
VARIANTS = (("full_column", None), ("upper_2000m", ARGO_FLOOR))
QUANTITIES = ("dS", "S_north")


# ── vectorised form of paper3r1_stats.regression_test ─────────────────


def contrast_stats(x: np.ndarray, ys: np.ndarray, c, phi_fixed: float | None = None) -> dict:
    """Contrast c' beta for each row of ys (shape (m, n)), with the same
    bias-corrected N_eff, standard error and t interval as
    ``paper3r1_stats.regression_test``. Returns arrays of length m.
    With phi_fixed the lag-1 coefficient is set to that value instead of the
    bias-corrected residual value (sensitivity only)."""
    x = np.asarray(x, dtype=float)
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    c = np.asarray(c, dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = ys @ np.linalg.pinv(x).T
    resid = ys - coef @ x.T
    s2 = (resid * resid).sum(axis=1) / (n - k)
    se = np.sqrt(s2 * float(c @ xtx_inv @ c))
    if phi_fixed is None:
        phi = ps.corrected_phi(ps.lag1(resid), x)
    else:
        phi = np.full(ys.shape[0], float(phi_fixed))
    neff_raw = n * (1.0 - phi) / (1.0 + phi)
    neff = np.clip(neff_raw, k + 1.0, n)
    dof = neff - k
    se_adj = se * np.sqrt((n - k) / dof)
    value = coef @ c
    tcrit = stats.t.ppf(0.975, dof)
    t = value / se_adj
    return {
        "value": value,
        "ci_lo": value - tcrit * se_adj,
        "ci_hi": value + tcrit * se_adj,
        "t": t,
        "p": 2.0 * stats.t.sf(np.abs(t), dof),
        "n_eff": neff,
        "phi": phi,
        "n_eff_at_floor": neff_raw < k + 1.0,
    }


def one(d: dict, i: int = 0) -> dict:
    """Scalar summary of row i of contrast_stats output, in per-decade units."""
    return {
        "value_per_decade": float(d["value"][i]) * 10.0,
        "ci_per_decade": [float(d["ci_lo"][i]) * 10.0, float(d["ci_hi"][i]) * 10.0],
        "t": float(d["t"][i]),
        "p": float(d["p"][i]),
        "n_eff": float(d["n_eff"][i]),
        "phi_corrected": float(d["phi"][i]),
        "n_eff_at_floor": bool(d["n_eff_at_floor"][i]),
    }


def one_level(d: dict, i: int = 0) -> dict:
    """Scalar summary of a level shift (PSU, no rescaling) for row i."""
    return {
        "level_shift_PSU": float(d["value"][i]),
        "ci_PSU": [float(d["ci_lo"][i]), float(d["ci_hi"][i])],
        "t": float(d["t"][i]),
        "p": float(d["p"][i]),
        "n_eff": float(d["n_eff"][i]),
        "phi_corrected": float(d["phi"][i]),
        "n_eff_at_floor": bool(d["n_eff_at_floor"][i]),
    }


def check_against_paper(x, y, j, c) -> float:
    """Largest absolute difference between the vectorised test and
    paper3r1_stats.regression_test on one series (value, p and interval)."""
    ref = ps.regression_test(x, y, j)
    new = contrast_stats(x, np.asarray(y, float)[None, :], np.eye(x.shape[1])[j])
    return float(
        max(
            abs(ref["value"] - new["value"][0]),
            abs(ref["p"] - new["p"][0]),
            abs(ref["ci"][0] - new["ci_lo"][0]),
            abs(ref["ci"][1] - new["ci_hi"][0]),
        )
    )


# ── monthly EN4 limb quantities ───────────────────────────────────────


def load_months() -> list[dict]:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for r in ex.map(en4_month_full, files):
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["time"])
    return rows


def annual_limbs(rows: list[dict], product_file: str, zmax, uniq, months) -> dict:
    """Monthly limb quantities (as in the observation-only test) averaged to years."""
    depth = rows[0]["depth"]
    e3t = rows[0]["e3t"]
    v = frozen_velocity(product_file, depth, e3t)
    monthly = []
    for r in rows:
        monthly.append(
            limbs(
                v,
                r["S_bar"],
                e3t,
                extra={"obsw": r["W_bar"], "unc": r["U_bar"]},
                zmax=zmax,
                depth=depth,
            )
        )
    ann: dict[str, np.ndarray] = {}
    for field in (
        "S_north",
        "S_south",
        "dS",
        "T_Sv",
        "F_ov_Sv",
        "obsw_north",
        "obsw_south",
        "unc_north",
        "unc_south",
    ):
        vals = np.array(
            [m.get(field, np.nan) if m is not None else np.nan for m in monthly],
            dtype=float,
        )
        ann[field] = np.array(
            [np.nanmean(vals[months == y]) if np.isfinite(vals[months == y]).any()
             else np.nan for y in uniq]
        )
    ann["n_valid_months"] = np.array(
        [sum(1 for m, mo in zip(monthly, months) if mo == y and m is not None)
         for y in uniq]
    )
    return ann


# ── window, break and covariate tests ─────────────────────────────────


def window_trend(years, y, lo, hi) -> dict:
    m = (years >= lo) & (years <= hi)
    r = ps.trend_test(years[m], y[m])
    return {
        "window": [int(lo), int(hi)],
        "n": int(m.sum()),
        "slope_per_decade": r["value"] * 10.0,
        "ci_per_decade": [r["ci"][0] * 10.0, r["ci"][1] * 10.0],
        "p": r["p"],
        "n_eff": r["n_eff"],
        "phi_corrected": r["phi_corrected"],
        "n_eff_at_floor": r["n_eff_at_floor"],
    }


def null_noise(years, y) -> np.ndarray:
    """AR(1) noise under the null of a trend with no break, with the lag-1
    coefficient of the trend-only residuals after bias correction."""
    x = ps.trend_design(years)
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    phi0 = float(ps.corrected_phi(float(ps.lag1(resid)), x))
    rng = np.random.default_rng(SEED)
    return ps.ar1_noise(N_NULL, len(years), phi0, rng), phi0


def sup_scan(years, y, eps, designs: dict, contrast, phi_fixed=None, level=False) -> dict:
    """Scan designs (keyed by knot) on |t| of a contrast; sup-type null from eps."""
    per_year = {}
    t_obs = []
    t_null = []
    for knot, x in designs.items():
        obs = contrast_stats(x, y, contrast, phi_fixed)
        nul = contrast_stats(x, eps, contrast, phi_fixed)["t"]
        t_obs.append(abs(float(obs["t"][0])))
        t_null.append(np.abs(nul))
        per_year[str(knot)] = one_level(obs) if level else one(obs)
    sup_null = np.max(np.column_stack(t_null), axis=1)
    sup_obs = max(t_obs)
    best = list(designs)[int(np.argmax(t_obs))]
    return {
        "knots": [int(k) for k in designs],
        "per_knot": per_year,
        "sup_abs_t_obs": sup_obs,
        "best_knot": int(best),
        "crit95_sup": float(np.quantile(sup_null, 0.95)),
        "crit99_sup": float(np.quantile(sup_null, 0.99)),
        "p_sup": float(np.mean(sup_null >= sup_obs)),
        "n_null": int(sup_null.size),
    }


def break_tests(years, y, eps, phi0) -> dict:
    t_c = years - years.mean()
    t_col = t_c
    step_designs = {yr: ps.step_design(years, yr) for yr in SCAN_YEARS}
    step_block = sup_scan(years, y, eps, step_designs, [0.0, 0.0, 1.0], level=True)
    # Trend in the step model at each candidate year, and the level shift again
    # from the same model, for the table.
    step_trend = {}
    for yr, x in step_designs.items():
        step_trend[str(yr)] = one(contrast_stats(x, y, [0.0, 1.0, 0.0]))
    step_block["trend_in_step_model"] = step_trend
    step_block["phi0_null"] = phi0
    # Cross-check against the paper's step_test on the best candidate year.
    ref = ps.step_test(years, y, BREAK_YEAR)["step"]
    new = contrast_stats(ps.step_design(years, BREAK_YEAR), y, [0.0, 0.0, 1.0])
    step_block["check_step_2017_vs_step_test"] = float(
        abs(ref["value"] - new["value"][0])
    )

    # Hinge at 2017: slope changes from 2017 onward.
    hinge_x = {
        BREAK_YEAR: np.column_stack(
            [np.ones_like(years), t_col, np.maximum(0.0, years - BREAK_YEAR)]
        )
    }
    hinge_fixed = {
        "change_of_slope": one(contrast_stats(hinge_x[BREAK_YEAR], y, [0, 0, 1.0])),
        "slope_pre_2017_within_model": one(
            contrast_stats(hinge_x[BREAK_YEAR], y, [0, 1.0, 0])
        ),
        "slope_post_2017_within_model": one(
            contrast_stats(hinge_x[BREAK_YEAR], y, [0, 1.0, 1.0])
        ),
    }
    hinge_scan_designs = {
        k: np.column_stack([np.ones_like(years), t_col, np.maximum(0.0, years - k)])
        for k in SCAN_YEARS
    }
    hinge_scan = sup_scan(years, y, eps, hinge_scan_designs, [0, 0, 1.0])

    # Sensitivity: the same scans with the lag-1 coefficient held at the
    # trend-only value for every knot, instead of re-estimated per knot.
    fixed_phi = {
        "phi_fixed": phi0,
        "step_scan_2012_2020": {
            k: v for k, v in sup_scan(
                years, y, eps, step_designs, [0.0, 0.0, 1.0], phi_fixed=phi0, level=True
            ).items() if k != "per_knot"
        },
        "hinge_scan_2012_2020": {
            k: v for k, v in sup_scan(
                years, y, eps, hinge_scan_designs, [0, 0, 1.0], phi_fixed=phi0
            ).items() if k != "per_knot"
        },
    }

    # Sub-period model: separate intercept and slope before and after 2017.
    pre = (years <= BREAK_YEAR).astype(float)
    post = (years >= POST_START).astype(float)
    split_x = np.column_stack([pre, t_col * pre, post, t_col * post])
    split = {
        "slope_2005_2017": one(contrast_stats(split_x, y, [0, 1.0, 0, 0])),
        "slope_2018_2024": one(contrast_stats(split_x, y, [0, 0, 0, 1.0])),
        "difference_post_minus_pre": one(contrast_stats(split_x, y, [0, -1.0, 0, 1.0])),
    }
    return {
        "step_scan_2012_2020": step_block,
        "hinge_2017": hinge_fixed,
        "hinge_scan_2012_2020": hinge_scan,
        "subperiod_split_2017": split,
        "fixed_phi_sensitivity": fixed_phi,
    }


def covariate_tests(years, y, w_n, w_s) -> dict:
    """Time coefficient with the observation weight of each limb as covariate."""
    t_c = years - years.mean()
    wn_c = w_n - w_n.mean()
    ws_c = w_s - w_s.mean()
    x1 = np.column_stack([np.ones_like(years), t_c, wn_c])
    x2 = np.column_stack([np.ones_like(years), t_c, wn_c, ws_c])
    m0 = ps.trend_test(years, y)
    m1 = ps.regression_test(x1, y, 1)
    m1w = ps.regression_test(x1, y, 2)
    m2 = ps.regression_test(x2, y, 1)
    m2n = ps.regression_test(x2, y, 2)
    m2s = ps.regression_test(x2, y, 3)
    weight_trend = ps.trend_test(years, w_n)
    return {
        "time_only": {
            "slope_per_decade": m0["value"] * 10,
            "ci_per_decade": [m0["ci"][0] * 10, m0["ci"][1] * 10],
            "p": m0["p"],
        },
        "time_plus_north_weight": {
            "slope_per_decade": m1["value"] * 10,
            "ci_per_decade": [m1["ci"][0] * 10, m1["ci"][1] * 10],
            "p": m1["p"],
            "n_eff": m1["n_eff"],
            "coef_north_weight": m1w["value"],
            "ci_north_weight": m1w["ci"],
            "p_north_weight": m1w["p"],
        },
        "time_plus_both_weights": {
            "slope_per_decade": m2["value"] * 10,
            "ci_per_decade": [m2["ci"][0] * 10, m2["ci"][1] * 10],
            "p": m2["p"],
            "n_eff": m2["n_eff"],
            "coef_north_weight": m2n["value"],
            "p_north_weight": m2n["p"],
            "coef_south_weight": m2s["value"],
            "p_south_weight": m2s["p"],
        },
        "north_weight_trend_per_decade": {
            "slope_per_decade": weight_trend["value"] * 10,
            "ci_per_decade": [weight_trend["ci"][0] * 10, weight_trend["ci"][1] * 10],
            "p": weight_trend["p"],
        },
        "corr_year_north_weight": float(np.corrcoef(years, w_n)[0, 1]),
        "north_weight_values": [float(v) for v in w_n],
        "south_weight_values": [float(v) for v in w_s],
    }


def analyse(years, y, w_n, w_s) -> dict:
    eps, phi0 = null_noise(years, y)
    out = {
        "full_2005_2024": window_trend(years, y, 2005, 2024),
        "start_year_2005_to_2010_end_2024": {
            str(s): window_trend(years, y, s, 2024) for s in range(2005, 2011)
        },
        "end_year_2015_to_2024_start_2005": {
            str(e): window_trend(years, y, 2005, e) for e in range(2015, 2025)
        },
        "window_2005_2017": window_trend(years, y, 2005, BREAK_YEAR),
        "window_2018_2024": window_trend(years, y, POST_START, 2024),
    }
    out.update(break_tests(years, y, eps, phi0))
    out["covariate"] = covariate_tests(years, y, w_n, w_s)
    return out


def main() -> None:
    rows = load_months()
    times = pd.DatetimeIndex([r["time"] for r in rows])
    months = np.array([t.year for t in times])
    uniq = np.unique(months)
    years = uniq.astype(float)

    # Checks on the vectorised test against the paper's test.
    x_trend = ps.trend_design(years)

    result: dict = {
        "id": "n01",
        "question": (
            "EN4 limb salinity contrast dS and northward-limb salinity S_N with the "
            "circulation frozen at each reanalysis record-mean section: start and end "
            "year sensitivity, break tests near 2017 and observing-density covariate."
        ),
        "record": [int(uniq[0]), int(uniq[-1])],
        "n_months": int(len(rows)),
        "units": "PSU per decade for trends; PSU for levels",
        "test": (
            "paper3r1_stats: OLS on annual values, lag-1 coefficient bias-corrected "
            "by inversion of its AR(1) simulated mean, N_eff = N(1-phi)/(1+phi), "
            "t interval with N_eff - k degrees of freedom (95 per cent)."
        ),
        "break_null": {
            "description": (
                "AR(1) noise with the bias-corrected lag-1 coefficient of the "
                "trend-only residuals; sup over the scanned knots of |t|; "
                "critical value and p from N_NULL simulations."
            ),
            "n_null": N_NULL,
            "scan_years": list(SCAN_YEARS),
            "seed": SEED,
        },
        "series": {},
    }

    annual: dict = {}
    for pkey, pfile in PRODUCTS:
        for vkey, zmax in VARIANTS:
            annual[(pkey, vkey)] = annual_limbs(rows, pfile, zmax, uniq, months)

    # Reconciliation with the stored full-column annual series (en4.json).
    with open(OUT_DIR / "en4.json") as fh:
        en4_json = json.load(fh)
    recon = {}
    for pkey, _ in PRODUCTS:
        ref_dS = np.array(en4_json[pkey]["dS"]["series"])
        ref_S = np.array(en4_json[pkey]["S_north"]["series"])
        ann = annual[(pkey, "full_column")]
        recon[pkey] = {
            "max_abs_diff_dS_vs_en4_json": float(np.max(np.abs(ann["dS"] - ref_dS))),
            "max_abs_diff_S_north_vs_en4_json": float(
                np.max(np.abs(ann["S_north"] - ref_S))
            ),
        }
    result["reconciliation_with_en4_json"] = recon

    # Vectorised test against the paper's test on one annual series.
    result["vectorised_check_max_abs_diff"] = check_against_paper(
        x_trend, annual[("oras5", "full_column")]["dS"], 1, None
    )

    for pkey, _ in PRODUCTS:
        for vkey, _ in VARIANTS:
            ann = annual[(pkey, vkey)]
            key = f"{pkey}_{vkey}"
            result["series"][key] = {
                "annual_years": [int(v) for v in uniq],
                "n_valid_months_per_year": [int(v) for v in ann["n_valid_months"]],
                "obs_weight_north_annual": [float(v) for v in ann["obsw_north"]],
                "obs_weight_south_annual": [float(v) for v in ann["obsw_south"]],
            }
            for q in QUANTITIES:
                y = ann[q]
                if not np.all(np.isfinite(y)):
                    raise RuntimeError(f"non-finite annual values in {key} {q}")
                result["series"][key][q] = analyse(
                    years, y, ann["obsw_north"], ann["obsw_south"]
                )
                result["series"][key][q]["annual_values"] = [float(v) for v in y]

    # Compact summary for the manuscript placeholders.
    summ = {}
    for pkey, _ in PRODUCTS:
        for vkey, _ in VARIANTS:
            key = f"{pkey}_{vkey}"
            s = result["series"][key]
            summ[key] = {
                "dS_full_2005_2024": s["dS"]["full_2005_2024"]["slope_per_decade"],
                "dS_window_2018_2024": s["dS"]["window_2018_2024"],
                "dS_window_2005_2017": s["dS"]["window_2005_2017"],
                "S_north_window_2005_2017": s["S_north"]["window_2005_2017"],
                "S_north_window_2018_2024": s["S_north"]["window_2018_2024"],
                "dS_step_scan_p_sup": s["dS"]["step_scan_2012_2020"]["p_sup"],
                "dS_hinge_change": s["dS"]["hinge_2017"]["change_of_slope"],
                "dS_split_difference": s["dS"]["subperiod_split_2017"][
                    "difference_post_minus_pre"
                ],
                "dS_covariate_time_plus_north_weight": s["dS"]["covariate"][
                    "time_plus_north_weight"
                ],
            }
    result["summary"] = summ

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=1, default=_default))
    print(f"wrote {OUT_FILE}")


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not serialisable: {type(o)}")


if __name__ == "__main__":
    main()
