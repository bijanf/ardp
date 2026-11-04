#!/usr/bin/env python3
"""Observing-system steps in the annual overturning freshwater series.

The overturning freshwater transport at 34.5 S is F = -(1/S0) T dS, with T the
northward-limb transport and dS the limb salinity contrast (S0 = 35 PSU). This
script tests whether the trends of F, T, dS and the northward-limb salinity S_N
depend on a level shift at an observing-system change. Four parts.

(a) Step-year scans. For every candidate break year s the annual series are
    fitted as a + b t + c I(t >= s). The step c is tested with the bias-corrected
    effective sample size of paper3r1_stats.py. Because the scan picks the best
    s from the data, the scan-level test uses the supremum over the candidate
    years of |t_c(s)|. Its null distribution is simulated from AR(1) noise whose
    coefficient is the bias-corrected lag-one autocorrelation of the trend-only
    residuals of the same series. Each simulated series is passed through the
    same estimator as the data, so the small-sample floor of the effective sample
    size enters the null as well. A sensitivity version keeps only break years
    with at least MIN_SIDE years on each side.
(b) ORAS5 joint model with steps at 2004 and 2015 over 1958 to 2025.
(c) GLORYS12V1 fit with a step at 2004 over 1993 to 2025, with Holm's adjustment
    across F, T and dS (family A) and across F and its two counterfactual terms
    (family B).
(d) Reconciliation of the two ORAS5 trends over 2004 to 2025 (unstepped and
    stepped at 2015). With beta the slope of I(t >= 2015) on t, the unstepped
    slope equals the stepped slope plus the step times beta, which is exact in
    ordinary least squares. The script checks this identity numerically.

Every number in the output is written to PAPER_3_v2/analysis/jc_n03_steps.json.
Trends are per year for F (mSv/yr) and per decade for T, dS and S_N (Sv/decade
and PSU/decade). Steps are in the native units of each series (Sv or PSU).
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

from analysis_paper3v2_attribution import S0, annual  # noqa: E402
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
OUT = ANALYSIS / "jc_n03_steps.json"

N_NULL = 10_000
SEED = 20261104
MIN_SIDE = 5
ALPHA = 0.05

# Series keys, trend scale (to the reported unit), trend unit, step unit.
SCALE = {"F": 1e3, "T": 10.0, "dS": 10.0, "SN": 10.0, "F_T": 1e3, "F_S": 1e3}
TREND_UNIT = {
    "F": "mSv/yr",
    "T": "Sv/decade",
    "dS": "PSU/decade",
    "SN": "PSU/decade",
    "F_T": "mSv/yr",
    "F_S": "mSv/yr",
}
STEP_UNIT = {"F": "Sv", "T": "Sv", "dS": "PSU", "SN": "PSU", "F_T": "Sv", "F_S": "Sv"}
LABEL = {
    "F": "F_ovS",
    "T": "T_limb (northward-limb transport)",
    "dS": "dS_limb (limb salinity contrast)",
    "SN": "S_north (northward-limb salinity)",
    "F_T": "F_T = -(1/S0) T(t) <dS>",
    "F_S": "F_S = -(1/S0) <T> dS(t)",
}
VARIABLES = ("F", "T", "dS", "SN")

PRODUCTS = {
    "oras5": {
        "label": "ORAS5",
        "file": "paper3r1_section_oras5.nc",
        "families": {
            "1995-2015": list(range(1995, 2016)),
            "1975-1985": list(range(1975, 1986)),
            "1975-1985 and 1995-2015": list(range(1975, 1986))
            + list(range(1995, 2016)),
        },
    },
    "glorys12": {
        "label": "GLORYS12V1",
        "file": "paper3r1_section_glorys12.nc",
        "families": {"1995-2015": list(range(1995, 2016))},
    },
}


# ── data ──────────────────────────────────────────────────────────────


def load_series(fname: str) -> tuple[np.ndarray, dict]:
    """Annual means of the section file and the two counterfactual terms."""
    with xr.open_dataset(RESULTS / fname) as ds:
        a = annual(ds)
        years = a["year"].values.astype(float)
        t = a["T_limb_Sv"].values.astype(float)
        d = a["dS_limb"].values.astype(float)
        series = {
            "F": a["F_ov"].values.astype(float),
            "T": t,
            "dS": d,
            "SN": a["S_north"].values.astype(float),
            "F_T": -(1 / S0) * t * d.mean(),
            "F_S": -(1 / S0) * t.mean() * d,
        }
    return years, series


# ── vectorised version of the step test (checked against regression_test) ──


def fit_rows(x: np.ndarray, rows: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """t value of coefficient j and its dof, for each row of rows (m, n).

    The same estimator as paper3r1_stats.regression_test, applied to many
    series on one design.
    """
    n, k = x.shape
    coef = rows @ np.linalg.pinv(x).T
    resid = rows - coef @ x.T
    s2 = np.sum(resid * resid, axis=-1) / (n - k)
    se = np.sqrt(s2 * np.linalg.inv(x.T @ x)[j, j])
    phi = corrected_phi(lag1(resid), x)
    neff = np.clip(n * (1.0 - phi) / (1.0 + phi), k + 1.0, n)
    dof = neff - k
    se_adj = se * np.sqrt((n - k) / dof)
    return coef[..., j] / se_adj, dof


def null_block(
    years: np.ndarray, y: np.ndarray, designs: list, offset: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Simulated null of |t| for the step coefficient on every design.

    The null is AR(1) noise with the bias-corrected lag-one coefficient of the
    trend-only residuals. The t value of the step coefficient is unchanged by a
    linear trend, so pure noise is the exact null for these designs.
    """
    rng = np.random.default_rng(SEED + offset)
    x0 = trend_design(years)
    c0 = np.linalg.lstsq(x0, y, rcond=None)[0]
    phi0 = float(corrected_phi(lag1(y - x0 @ c0), x0))
    eps = ar1_noise(N_NULL, len(y), phi0, rng)
    abs_t = np.empty((N_NULL, len(designs)))
    size = np.empty(len(designs))
    for i, x in enumerate(designs):
        t, dof = fit_rows(x, eps, 2)
        abs_t[:, i] = np.abs(t)
        size[i] = float(np.mean(2.0 * stats.t.sf(np.abs(t), dof) < ALPHA))
    return phi0, abs_t, size


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in the order given."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def pack_trend(r: dict, scale: float) -> dict:
    return {
        "value": r["value"] * scale,
        "ci": [c * scale for c in r["ci"]],
        "p": r["p"],
        "n": r["n"],
        "n_eff": r["n_eff"],
        "n_eff_at_floor": r["n_eff_at_floor"],
        "phi_corrected": r["phi_corrected"],
    }


def pack_step(r: dict) -> dict:
    return {
        "value": r["value"],
        "ci": r["ci"],
        "p": r["p"],
        "n_eff": r["n_eff"],
        "n_eff_at_floor": r["n_eff_at_floor"],
    }


# ── (a) step-year scans with sup-type critical values ─────────────────


def family_summary(
    obs: list[dict], t_obs: np.ndarray, abs_t: np.ndarray, idx: list[int]
) -> dict:
    """Supremum test over one family of candidate break years."""

    def block(sel: list[int]) -> dict:
        sup_obs = float(t_obs[sel].max())
        sup_null = abs_t[:, sel].max(axis=1)
        best = sel[int(np.argmax(t_obs[sel]))]
        return {
            "break_years": [obs[i]["step_year"] for i in sel],
            "sup_abs_t_obs": sup_obs,
            "best_break_year": obs[best]["step_year"],
            "best_break_step": obs[best]["step"],
            "sup_crit95": float(np.quantile(sup_null, 1 - ALPHA)),
            "p_sup": float((1 + np.sum(sup_null >= sup_obs)) / (N_NULL + 1)),
            "resolved_at_5pct": bool(
                (1 + np.sum(sup_null >= sup_obs)) / (N_NULL + 1) < ALPHA
            ),
        }

    out = block(idx)
    restricted = [
        i for i in idx if min(obs[i]["n_pre"], obs[i]["n_post"]) >= MIN_SIDE
    ]
    out["restricted"] = (
        {"min_side_years": MIN_SIDE, **block(restricted)} if restricted else None
    )
    return out


def scan_product(years: np.ndarray, series: dict, families: dict, key_idx: int) -> dict:
    cands = sorted({s for fam in families.values() for s in fam})
    designs = [step_design(years, s) for s in cands]
    out: dict = {"candidate_years": cands, "variables": {}}
    for v_i, v in enumerate(VARIABLES):
        y = series[v]
        obs = []
        for s in cands:
            r = step_test(years, y, s)
            tr, st = r["trend"], r["step"]
            obs.append(
                {
                    "step_year": s,
                    "n_pre": int(np.sum(years < s)),
                    "n_post": int(np.sum(years >= s)),
                    "trend": tr["value"] * SCALE[v],
                    "trend_ci": [c * SCALE[v] for c in tr["ci"]],
                    "trend_p": tr["p"],
                    "step": st["value"],
                    "step_ci": st["ci"],
                    "step_p": st["p"],
                    "t_step": st["value"] / st["stderr_adj"],
                    "n_eff": st["n_eff"],
                    "n_eff_at_floor": st["n_eff_at_floor"],
                    "phi_corrected": st["phi_corrected"],
                }
            )
        phi0, abs_t, size = null_block(years, y, designs, 100 * key_idx + v_i)
        t_obs = np.array([abs(r["t_step"]) for r in obs])
        t_check = np.array(
            [abs(fit_rows(x, y[None, :], 2)[0][0]) for x in designs]
        )
        for i, r in enumerate(obs):
            r["size_5pct_under_null"] = float(size[i])
        out["variables"][v] = {
            "label": LABEL[v],
            "trend_unit": TREND_UNIT[v],
            "step_unit": STEP_UNIT[v],
            "phi_null": phi0,
            "n_null": N_NULL,
            "check_max_abs_t_difference": float(np.max(np.abs(t_check - t_obs))),
            "families": {
                fname: family_summary(
                    obs, t_obs, abs_t, [cands.index(s) for s in fam]
                )
                for fname, fam in families.items()
            },
            "scan": obs,
        }
    return out


# ── (b) ORAS5 joint model with steps at 2004 and 2015 ─────────────────


def joint_2004_2015(years: np.ndarray, series: dict) -> dict:
    x = np.column_stack(
        [
            np.ones_like(years),
            years - years.mean(),
            (years >= 2004).astype(float),
            (years >= 2015).astype(float),
        ]
    )
    out = {}
    for v in VARIABLES:
        y = series[v]
        tr = regression_test(x, y, 1)
        s04 = regression_test(x, y, 2)
        s15 = regression_test(x, y, 3)
        base = trend_test(years, y)
        out[v] = {
            "label": LABEL[v],
            "trend_unit": TREND_UNIT[v],
            "step_unit": STEP_UNIT[v],
            "unstepped_trend_1958_2025": pack_trend(base, SCALE[v]),
            "joint_trend": pack_trend(tr, SCALE[v]),
            "joint_step_2004": pack_step(s04),
            "joint_step_2015": pack_step(s15),
        }
    return out


# ── (c) GLORYS12V1 step at 2004 with Holm adjustment ──────────────────


def glorys_2004(years: np.ndarray, series: dict) -> dict:
    fits = {}
    for name in ("F", "T", "dS", "F_T", "F_S"):
        r = step_test(years, series[name], 2004)
        fits[name] = {
            "label": LABEL[name],
            "trend_unit": TREND_UNIT[name],
            "step_unit": STEP_UNIT[name],
            "trend": pack_trend(r["trend"], SCALE[name]),
            "step": pack_step(r["step"]),
        }
    fam_a = ["F", "T", "dS"]
    fam_b = ["F", "F_T", "F_S"]
    for fam, tag in ((fam_a, "holm_family_A"), (fam_b, "holm_family_B")):
        adj = holm([fits[n]["step"]["p"] for n in fam])
        for n, p_adj in zip(fam, adj):
            fits[n].setdefault("holm", {})[tag] = p_adj
    return {
        "step_year": 2004,
        "record": [int(years[0]), int(years[-1])],
        "family_A": fam_a,
        "family_B": fam_b,
        "fits": fits,
    }


# ── (d) reconciliation of the two ORAS5 Argo-era trends ───────────────


def argo_reconciliation(years: np.ndarray, series: dict) -> dict:
    m = (years >= 2004) & (years <= 2025)
    yr = years[m]
    ind = (yr >= 2015).astype(float)
    beta = float(np.sum((yr - yr.mean()) * (ind - ind.mean())) / np.sum((yr - yr.mean()) ** 2))
    out: dict = {
        "window": [2004, 2025],
        "n": int(m.sum()),
        "beta_indicator_slope_per_year": beta,
        "variables": {},
    }
    for v in VARIABLES:
        y = series[v][m]
        unst = trend_test(yr, y)
        stp = step_test(yr, y, 2015)
        b_u = unst["value"]
        b_s = stp["trend"]["value"]
        c_s = stp["step"]["value"]
        out["variables"][v] = {
            "label": LABEL[v],
            "trend_unit": TREND_UNIT[v],
            "step_unit": STEP_UNIT[v],
            "unstepped": pack_trend(unst, SCALE[v]),
            "stepped_2015": {
                "trend": pack_trend(stp["trend"], SCALE[v]),
                "step_2015": pack_step(stp["step"]),
            },
            "difference_unstepped_minus_stepped": (b_u - b_s) * SCALE[v],
            "omitted_step_term_step_times_beta": c_s * beta * SCALE[v],
            "identity_residual": (b_u - b_s - c_s * beta) * SCALE[v],
        }
    sub = {}
    for name, (y0, y1) in {"2004-2014": (2004, 2014), "2015-2025": (2015, 2025)}.items():
        mm = (years >= y0) & (years <= y1)
        sub[name] = {
            v: pack_trend(trend_test(years[mm], series[v][mm]), SCALE[v])
            for v in ("F", "T", "dS")
        }
    out["unstepped_subwindows"] = sub
    out["note_step_2004"] = (
        "Over 2004 to 2025 the indicator I(t >= 2004) is identically one, so a "
        "2004 step is absorbed by the intercept and is not identified in this window."
    )
    return out


# ── reproduction of stored values ─────────────────────────────────────


def reproduction(
    years_o: np.ndarray, s_o: dict, years_g: np.ndarray, s_g: dict
) -> list[dict]:
    r1 = json.loads((ANALYSIS / "r1_step_adjusted.json").read_text())
    sf = json.loads((ANALYSIS / "stepfit.json").read_text())
    cw = json.loads((ANALYSIS / "r1_common_window.json").read_text())

    m = (years_o >= 2004) & (years_o <= 2025)
    mine = {
        "GLORYS12V1 F step at 2004, 1993-2025 (Sv)": (
            step_test(years_g, s_g["F"], 2004)["step"]["value"],
            r1["glorys12"]["fits"]["F"]["step_Sv"],
        ),
        "GLORYS12V1 F trend at 2004 step, 1993-2025 (mSv/yr)": (
            step_test(years_g, s_g["F"], 2004)["trend"]["value"] * 1e3,
            r1["glorys12"]["fits"]["F"]["trend_mSv_per_yr"],
        ),
        "ORAS5 F step at 2015, 1958-2025 (Sv)": (
            step_test(years_o, s_o["F"], 2015)["step"]["value"],
            sf["oras5"]["step_fits"]["own record"]["F_ov"]["step_Sv"],
        ),
        "ORAS5 F trend with 2015 step, 1958-2025 (mSv/yr)": (
            step_test(years_o, s_o["F"], 2015)["trend"]["value"] * 1e3,
            sf["oras5"]["step_fits"]["own record"]["F_ov"]["trend_mSv_per_yr"],
        ),
        "ORAS5 F trend with 2015 step, 2004-2025 (mSv/yr)": (
            step_test(years_o[m], s_o["F"][m], 2015)["trend"]["value"] * 1e3,
            sf["oras5"]["step_fits"]["argo"]["F_ov"]["trend_mSv_per_yr"],
        ),
        "ORAS5 F trend unstepped, 2004-2025 (mSv/yr)": (
            trend_test(years_o[m], s_o["F"][m])["value"] * 1e3,
            cw["oras5"]["argo"]["trend_F_ov"]["trend"],
        ),
    }
    rows = []
    for name, (a, b) in mine.items():
        rows.append(
            {"quantity": name, "this_script": float(a), "stored": float(b), "difference": float(a - b)}
        )
    return rows


# ── output ────────────────────────────────────────────────────────────


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    data = {k: load_series(p["file"]) for k, p in PRODUCTS.items()}
    years_o, s_o = data["oras5"]
    years_g, s_g = data["glorys12"]

    scans = {}
    for i, (k, p) in enumerate(PRODUCTS.items()):
        years, series = data[k]
        scans[k] = {
            "label": p["label"],
            "record": [int(years[0]), int(years[-1])],
            **scan_product(years, series, p["families"], i),
        }

    out = {
        "id": "n03",
        "description": (
            "Step-year scans with supremum critical values, the ORAS5 joint model "
            "with steps at 2004 and 2015, the GLORYS12V1 2004 fit with Holm "
            "adjustment, and the reconciliation of the ORAS5 Argo-era trends."
        ),
        "conventions": {
            "S0_PSU": S0,
            "alpha": ALPHA,
            "n_null": N_NULL,
            "seed": SEED,
            "min_side_years_restricted": MIN_SIDE,
            "trend_units": TREND_UNIT,
            "step_units": STEP_UNIT,
            "ci_level": 0.95,
            "pvalue_note": (
                "sup p values count the simulated null maxima at or above the observed "
                "maximum, with the +1 correction."
            ),
        },
        "a_step_scans": scans,
        "b_oras5_joint_2004_2015": {
            "record": [int(years_o[0]), int(years_o[-1])],
            "variables": joint_2004_2015(years_o, s_o),
        },
        "c_glorys12_2004_holm": glorys_2004(years_g, s_g),
        "d_oras5_argo_reconciliation": argo_reconciliation(years_o, s_o),
        "reproduction_checks": reproduction(years_o, s_o, years_g, s_g),
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(out), indent=1))

    for k, sc in scans.items():
        for v in VARIABLES:
            vv = sc["variables"][v]
            for fname, fam in vv["families"].items():
                print(
                    f"{sc['label']:10s} {v:3s} {fname:24s} best {fam['best_break_year']} "
                    f"|t| {fam['sup_abs_t_obs']:.2f} crit95 {fam['sup_crit95']:.2f} "
                    f"p_sup {fam['p_sup']:.3f}"
                )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
