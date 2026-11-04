"""Cross-model relation between the 1950 to 1980 mean F_ovS and the forced change
of the overturning, at 26.5 N and at 34.5 S (CMIP6, historical and SSP5-8.5).

For each model of the 25-model set the following are formed from the stored
annual series:

* F_ovS, the 1950 to 1980 mean at 34.5 S (data/results/paper3r1_cmip6_<model>.nc);
* the change of the overturning at 26.5 N, from 1850 to 1900 to 2081 to 2100
  (data/results/yearly_amoc26n_cmip6.npz), and the change of the 34.5 S
  streamfunction maximum, from 1850 to 1900 to 2081 to 2100 (the nc files);
* the 1850 to 1900 base-state strength at both latitudes.

Each window mean carries a sampling error from interannual variability: the
standard error of the mean about a linear trend, with the lag-1 corrected
effective sample size of paper3r1_stats. These errors give the weights of the
errors-in-variables regressions (York, 2004; Deming with one error ratio; reduced
major axis, which uses no error information).

Statistics per set of models:

* Pearson, Spearman and Kendall correlations of F_ovS with the change;
* OLS, reduced major axis, Deming and York slopes;
* partial correlation with the 1850 to 1900 base-state strength (Pearson and
  Spearman on ranks), with Fisher intervals of n - 4 degrees of freedom;
* the same quantities for the fractional change (change divided by the base
  state);
* a family-cluster bootstrap over the genealogy grouping of
  analysis_paper3r1_cmip6.py (families are resampled with replacement and all
  members of a drawn family enter the sample), and leave-one-model-out and
  leave-one-family-out ranges of the partial correlation.

Three model sets are used: the 26.5 N set (models with a 1850 to 1900 and a
2081 to 2100 overturning value at 26.5 N), the 34.5 S set (models with a
1850 to 1900 and a 2081 to 2100 streamfunction value at 34.5 S), and their
intersection. The difference of the partial correlations at the two latitudes
is bootstrapped with the same cluster draws on the intersection.

Cross-model correlations are plain statistics over models, with n - 2 degrees
of freedom; the autocorrelation correction of paper3r1_stats applies to the
time series and is used here only for the window-mean sampling errors.

Writes ``PAPER_3_v2/analysis/jc_n11_cmip6_relation.json``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import fitted_mean_test, trend_design  # noqa: E402

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_n11_cmip6_relation.json"
N_BOOT = 10_000
SEED = 20261104
UNIT = 0.1  # slopes are also reported per 0.1 Sv of F_ovS

# Genealogy grouping, as in analysis_paper3r1_cmip6.py (main()).
FAMILIES = {
    "CESM2": ["CESM2", "CESM2-WACCM", "TaiESM1"],
    "MPI": ["MPI-ESM1-2-LR", "MPI-ESM1-2-HR"],
    "ACCESS": ["ACCESS-CM2", "ACCESS-ESM1-5"],
    "FGOALS": ["FGOALS-f3-L", "FGOALS-g3"],
    "CMCC": ["CMCC-CM2-SR5", "CMCC-ESM2"],
    "UKMO": ["UKESM1-0-LL", "HadGEM3-GC31-LL"],
}


# ── window means and their sampling errors ────────────────────────────


def window_mean(series: pd.Series, y0: int, y1: int) -> dict | None:
    """Mean over [y0, y1] with its standard error about a linear trend.

    The interval of fitted_mean_test (trend counted as signal, lag-1 corrected
    effective sample size) gives the standard error by inversion.
    """
    sub = series.loc[y0:y1].dropna()
    n_full = y1 - y0 + 1
    if len(sub) < 0.8 * n_full:
        return None
    yrs = sub.index.to_numpy(dtype=float)
    y = sub.to_numpy(dtype=float)
    x = trend_design(yrs)
    res = fitted_mean_test(x, y)
    k = res["design_columns"]
    tcrit = stats.t.ppf(0.975, res["n_eff"] - k)
    se = 0.5 * (res["ci"][1] - res["ci"][0]) / tcrit
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sd_int = float(np.sqrt(resid @ resid / (len(y) - 2)))
    return {
        "mean": float(res["mean"]),
        "se": float(se),
        "n_eff": float(res["n_eff"]),
        "n_years": int(len(y)),
        "sd_interannual": sd_int,
        "trend_per_year": float(coef[1]),
    }


def load_models(models: list[str]) -> pd.DataFrame:
    amoc = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    m26 = set(str(m) for m in amoc["models"])
    rows = []
    for m in models:
        row: dict = {"model": m}
        nc = xr.open_dataset(RESULTS / f"paper3r1_cmip6_{m}.nc")
        f = nc["F_ov"].to_series()
        f.index = f.index.astype(int)
        psi = nc["psi34"].to_series()
        psi.index = psi.index.astype(int)
        nc.close()
        wf = window_mean(f, 1950, 1980)
        if wf:
            row.update({"F_mean": wf["mean"], "F_se": wf["se"],
                        "F_sd_interannual": wf["sd_interannual"]})
        b34 = window_mean(psi, 1850, 1900)
        l34 = window_mean(psi, 2081, 2100)
        if b34 and l34:
            row.update({
                "psi34_base": b34["mean"],
                "psi34_base_se": b34["se"],
                "psi34_base_sd_interannual": b34["sd_interannual"],
                "dpsi34": l34["mean"] - b34["mean"],
                "dpsi34_se": math.hypot(b34["se"], l34["se"]),
                "dpsi34_sd_interannual": math.hypot(
                    b34["sd_interannual"], l34["sd_interannual"]
                ) / math.sqrt(2.0),
            })
        if m in m26:
            yrs = amoc[f"{m}_years"].astype(int)
            a = pd.Series(amoc[f"{m}_amoc"].astype(float), index=yrs)
            a = a[~a.index.duplicated()]
            ba = window_mean(a, 1850, 1900)
            la = window_mean(a, 2081, 2100)
            if ba and la:
                row.update({
                    "amoc26_base": ba["mean"],
                    "amoc26_base_se": ba["se"],
                    "amoc26_base_sd_interannual": ba["sd_interannual"],
                    "damoc26": la["mean"] - ba["mean"],
                    "damoc26_se": math.hypot(ba["se"], la["se"]),
                    "damoc26_sd_interannual": math.hypot(
                        ba["sd_interannual"], la["sd_interannual"]
                    ) / math.sqrt(2.0),
                })
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ── estimators ────────────────────────────────────────────────────────


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    a = np.column_stack([np.ones_like(z), z])
    rx = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]
    ry = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_test(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    n = len(x)
    r = partial_r(x, y, z)
    dof = n - 3
    t = r * np.sqrt(dof / (1.0 - r * r))
    zr, se = np.arctanh(r), 1.0 / np.sqrt(n - 4)
    return {
        "r": r,
        "n": int(n),
        "p": float(2 * stats.t.sf(abs(t), dof)),
        "fisher_ci95": [float(np.tanh(zr - 1.96 * se)), float(np.tanh(zr + 1.96 * se))],
        "smallest_detectable_abs_r": float(
            stats.t.ppf(0.975, dof) / np.sqrt(dof + stats.t.ppf(0.975, dof) ** 2)
        ),
    }


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    sxy = np.sum((x - x.mean()) * (y - y.mean()))
    return float(sxy / np.sum((x - x.mean()) ** 2))


def rma_slope(x: np.ndarray, y: np.ndarray) -> float:
    sxx = np.sum((x - x.mean()) ** 2)
    syy = np.sum((y - y.mean()) ** 2)
    sxy = np.sum((x - x.mean()) * (y - y.mean()))
    return float(np.sign(sxy) * np.sqrt(syy / sxx))


def deming_slope(x: np.ndarray, y: np.ndarray, delta: float) -> float:
    """Deming slope for error variance ratio delta = var(error of y) / var(error of x)."""
    sxx = np.sum((x - x.mean()) ** 2)
    syy = np.sum((y - y.mean()) ** 2)
    sxy = np.sum((x - x.mean()) * (y - y.mean()))
    d = syy - delta * sxx
    return float((d + np.sqrt(d * d + 4.0 * delta * sxy * sxy)) / (2.0 * sxy))


def york_fit(x, y, sx, sy, tol: float = 1e-12, maxit: int = 500):
    """York (2004) iterative fit with independent per-point errors sx, sy.

    Returns slope, intercept, the analytic standard error of the slope
    sigma_b = (sum W_i u_i^2)^(-1/2) with u_i = x_i - xbar_W (measurement
    errors only), and the scatter-scaled version of that error,
    sigma_b * sqrt(max(1, chi2 / (n - 2))), with chi2 the York misfit.
    """
    wx, wy = 1.0 / sx**2, 1.0 / sy**2
    b = ols_slope(x, y)
    for _ in range(maxit):
        w = wx * wy / (wx + b * b * wy)
        xbar = np.sum(w * x) / np.sum(w)
        ybar = np.sum(w * y) / np.sum(w)
        u, v = x - xbar, y - ybar
        beta = w * (u / wy + b * v / wx)
        b_new = float(np.sum(w * beta * v) / np.sum(w * beta * u))
        done = abs(b_new - b) <= tol * max(1.0, abs(b))
        b = b_new
        if done:
            break
    w = wx * wy / (wx + b * b * wy)
    xbar = np.sum(w * x) / np.sum(w)
    ybar = np.sum(w * y) / np.sum(w)
    u = x - xbar
    a = ybar - b * xbar
    se = float(1.0 / np.sqrt(np.sum(w * u * u)))
    chi2 = float(np.sum(wx * wy * (y - b * x - a) ** 2 / (wx + b * b * wy)))
    se_scaled = se * float(np.sqrt(max(1.0, chi2 / (len(x) - 2))))
    return float(b), float(a), se, se_scaled, chi2


def ranks(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a)


KEYS = [
    "r",
    "rho",
    "slope_ols",
    "slope_rma",
    "slope_deming_pooled",
    "slope_deming_interannual",
    "slope_york",
    "partial_r",
    "partial_rho",
    "r_frac",
    "partial_r_frac",
]


def estimates(x, y, b, sx, sy, yfrac, delta_pool, delta_alt) -> np.ndarray:
    """Vector of point estimates in the order of KEYS."""
    return np.array([
        pearson(x, y),
        float(stats.spearmanr(x, y)[0]),
        ols_slope(x, y),
        rma_slope(x, y),
        deming_slope(x, y, delta_pool),
        deming_slope(x, y, delta_alt),
        york_fit(x, y, sx, sy)[0],
        partial_r(x, y, b),
        partial_r(ranks(x), ranks(y), ranks(b)),
        pearson(x, yfrac),
        partial_r(x, yfrac, b),
    ])


# ── sets and clusters ─────────────────────────────────────────────────


def clusters_of(members: list[str]) -> list[list[str]]:
    out, used = [], set()
    for fam in FAMILIES.values():
        present = [m for m in fam if m in members]
        if present:
            out.append(present)
            used.update(present)
    out += [[m] for m in members if m not in used]
    return out


def set_arrays(tab: pd.DataFrame, latitude: str, members: list[str]) -> dict:
    t = tab.loc[members]
    if latitude == "26.5N":
        y = t["damoc26"].to_numpy(float)
        sy = t["damoc26_se"].to_numpy(float)
        base = t["amoc26_base"].to_numpy(float)
        var_y_win = t["damoc26_sd_interannual"].to_numpy(float) ** 2
    else:
        y = t["dpsi34"].to_numpy(float)
        sy = t["dpsi34_se"].to_numpy(float)
        base = t["psi34_base"].to_numpy(float)
        var_y_win = t["dpsi34_sd_interannual"].to_numpy(float) ** 2
    x = t["F_mean"].to_numpy(float)
    sx = t["F_se"].to_numpy(float)
    var_x_win = t["F_sd_interannual"].to_numpy(float) ** 2
    # Error ratio delta = var(error of y) / var(error of x), pooled over the set,
    # from the sampling errors of the window means (primary), and from the
    # interannual variances of the annual values (sensitivity).
    delta_pool = float(np.mean(sy**2) / np.mean(sx**2))
    delta_alt = float(np.mean(var_y_win) / np.mean(var_x_win))
    return {
        "members": members,
        "x": x, "y": y, "b": base, "sx": sx, "sy": sy,
        "yfrac": y / base,
        "delta_pool": delta_pool,
        "delta_alt": delta_alt,
        "delta_median_model": float(np.median(sy**2 / sx**2)),
        "var_x_win": var_x_win, "var_y_win": var_y_win,
    }


def point_block(a: dict) -> dict:
    x, y, b = a["x"], a["y"], a["b"]
    n = len(x)
    vec = estimates(x, y, b, a["sx"], a["sy"], a["yfrac"], a["delta_pool"], a["delta_alt"])
    r_p, p_p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    tau, p_tau = stats.kendalltau(x, y)
    rf, pf = stats.pearsonr(x, a["yfrac"])
    york_b, york_a, york_se, york_se_scaled, york_chi2 = york_fit(x, y, a["sx"], a["sy"])
    part = partial_test(x, y, b)
    part_frac = partial_test(x, a["yfrac"], b)
    part_rho = partial_r(ranks(x), ranks(y), ranks(b))
    loo = [partial_r(np.delete(x, i), np.delete(y, i), np.delete(b, i)) for i in range(n)]
    i_min, i_max = int(np.argmin(loo)), int(np.argmax(loo))
    fam_drop = []
    for cl in clusters_of(a["members"]):
        keep = np.array([m not in cl for m in a["members"]])
        if keep.sum() < 6:
            continue
        fam_drop.append({
            "dropped": cl,
            "partial_r": partial_r(x[keep], y[keep], b[keep]),
        })
    fam_vals = [d["partial_r"] for d in fam_drop]
    return {
        "n": int(n),
        "n_clusters": len(clusters_of(a["members"])),
        "members": a["members"],
        "delta_pooled": a["delta_pool"],
        "delta_interannual_sensitivity": a["delta_alt"],
        "delta_median_model": a["delta_median_model"],
        "reliability_ratio_x": float(1.0 - np.mean(a["sx"] ** 2) / np.var(x, ddof=1)),
        "raw": {
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "kendall_tau": float(tau), "kendall_p": float(p_tau),
        },
        "slopes": {
            "ols": ols_slope(x, y),
            "rma": rma_slope(x, y),
            "deming_pooled_delta": deming_slope(x, y, a["delta_pool"]),
            "deming_interannual_delta": deming_slope(x, y, a["delta_alt"]),
            "york": york_b,
            "york_intercept": york_a,
            "york_analytic_se_measurement_only": york_se,
            "york_analytic_se_scatter_scaled": york_se_scaled,
            "york_chi2": york_chi2,
            "york_slope_per_0p1_Sv_F": york_b * UNIT,
            "ols_per_0p1_Sv_F": ols_slope(x, y) * UNIT,
        },
        "partial_base": {
            "absolute": part,
            "absolute_spearman_r": part_rho,
            "fractional": part_frac,
            "leave_one_model_out_range": [float(min(loo)), float(max(loo))],
            "leave_one_model_out_min_without": a["members"][i_min],
            "leave_one_model_out_max_without": a["members"][i_max],
            "leave_one_family_out_range": [float(min(fam_vals)), float(max(fam_vals))] if fam_vals else None,
            "leave_one_family_out_detail": fam_drop,
        },
        "fractional_change": {
            "pearson_r": float(rf), "pearson_p": float(pf),
        },
        "base_state_correlations": {
            "F_with_base": {"r": pearson(x, b), "p": float(stats.pearsonr(x, b)[1])},
            "base_with_change": {"r": pearson(b, y), "p": float(stats.pearsonr(b, y)[1])},
        },
        "point_vector": dict(zip(KEYS, [float(v) for v in vec])),
    }


def boot_draws(a: dict, rng: np.random.Generator, n_boot: int, cl: list[list[int]] | None = None):
    members = a["members"]
    if cl is None:
        cl_names = clusters_of(members)
        cl = [[members.index(m) for m in c] for c in cl_names]
    K = len(cl)
    out = np.full((n_boot, len(KEYS)), np.nan)
    draws = []
    for b_i in range(n_boot):
        pick = rng.integers(0, K, K)
        idx = np.array([j for k in pick for j in cl[k]], dtype=int)
        draws.append(idx)
        if len(idx) < 6:
            continue
        try:
            out[b_i] = estimates(
                a["x"][idx], a["y"][idx], a["b"][idx], a["sx"][idx], a["sy"][idx],
                a["yfrac"][idx], a["delta_pool"], a["delta_alt"],
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return out, draws, cl


def summarise_boot(out: np.ndarray, point: dict) -> dict:
    res = {}
    for j, k in enumerate(KEYS):
        col = out[:, j]
        col = col[np.isfinite(col)]
        lo, hi = np.percentile(col, [2.5, 97.5])
        pt = point["point_vector"][k]
        res[k] = {
            "point": pt,
            "ci95": [float(lo), float(hi)],
            "n_valid": int(col.size),
            "share_same_sign_as_point": float(np.mean(np.sign(col) == np.sign(pt))),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
        if k.startswith("slope"):
            res[k]["ci95_per_0p1_Sv_F"] = [float(lo) * UNIT, float(hi) * UNIT]
    return res


def tabulate_checks(tab: pd.DataFrame, ref: dict) -> dict:
    """Window means recomputed from the annual series against r1_cmip6.json."""
    per = {r["model"]: r for r in ref["per_model"]}
    pairs = [
        ("F_mean", "F_1950-1980", None),
        ("amoc26_base", "amoc26_1850-1900", None),
        ("damoc26", "damoc26", None),
        ("psi34_base", "psi34_1850-1900", None),
        ("dpsi34", "dpsi34", None),
    ]
    out = {}
    for mine, theirs, _ in pairs:
        diffs = []
        for m in tab.index:
            if mine not in tab.columns or pd.isna(tab.loc[m].get(mine, np.nan)):
                continue
            v = per[m].get(theirs)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            diffs.append(abs(float(tab.loc[m, mine]) - float(v)))
        out[mine] = {"n_compared": len(diffs), "max_abs_diff": float(max(diffs)) if diffs else None}
    return out


def main() -> None:
    ref = json.loads((ANALYSIS / "r1_cmip6.json").read_text())
    models = [r["model"] for r in ref["per_model"]]
    tab = load_models(models)

    set26 = tab.dropna(subset=["F_mean", "damoc26", "amoc26_base"]).index.tolist()
    set34 = tab.dropna(subset=["F_mean", "dpsi34", "psi34_base"]).index.tolist()
    common = [m for m in set26 if m in set34]
    sets = {
        "26.5N_21": set_arrays(tab, "26.5N", set26),
        "34.5S_19": set_arrays(tab, "34.5S", set34),
        "common_17_26.5N": set_arrays(tab, "26.5N", common),
        "common_17_34.5S": set_arrays(tab, "34.5S", common),
    }

    rng = np.random.default_rng(SEED)
    blocks, boots = {}, {}
    for name, a in sets.items():
        blocks[name] = point_block(a)
        out, _, _ = boot_draws(a, rng, N_BOOT)
        boots[name] = summarise_boot(out, blocks[name])
        blocks[name]["bootstrap_family_cluster"] = boots[name]

    # Paired difference of the 26.5 N and 34.5 S statistics on the common models,
    # with the same family-cluster draws for both latitudes.
    a26, a34 = sets["common_17_26.5N"], sets["common_17_34.5S"]
    cl = [[a26["members"].index(m) for m in c] for c in clusters_of(a26["members"])]
    rng2 = np.random.default_rng(SEED + 1)
    d_out = np.full((N_BOOT, len(KEYS)), np.nan)
    for b_i in range(N_BOOT):
        pick = rng2.integers(0, len(cl), len(cl))
        idx = np.array([j for k in pick for j in cl[k]], dtype=int)
        try:
            v26 = estimates(a26["x"][idx], a26["y"][idx], a26["b"][idx], a26["sx"][idx],
                            a26["sy"][idx], a26["yfrac"][idx], a26["delta_pool"], a26["delta_alt"])
            v34 = estimates(a34["x"][idx], a34["y"][idx], a34["b"][idx], a34["sx"][idx],
                            a34["sy"][idx], a34["yfrac"][idx], a34["delta_pool"], a34["delta_alt"])
            d_out[b_i] = v26 - v34
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    paired = {}
    for j, k in enumerate(KEYS):
        col = d_out[:, j]
        col = col[np.isfinite(col)]
        lo, hi = np.percentile(col, [2.5, 97.5])
        pt = blocks["common_17_26.5N"]["point_vector"][k] - blocks["common_17_34.5S"]["point_vector"][k]
        paired[k] = {
            "difference_26N_minus_34S": float(pt),
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_valid": int(col.size),
        }

    # Statements of the paper, evaluated from the blocks above.
    s26 = blocks["26.5N_21"]
    s34 = blocks["34.5S_19"]
    sc26 = blocks["common_17_26.5N"]
    sc34 = blocks["common_17_34.5S"]
    def claim(statement, value, ci, survives):
        return {"statement": statement, "value": value, "ci95": ci,
                "survives_at_95": bool(survives)}

    b26, b34 = boots["26.5N_21"], boots["34.5S_19"]
    bc26, bc34 = boots["common_17_26.5N"], boots["common_17_34.5S"]
    claims = {
        "P1_partial_26N_positive": claim(
            "Given the 1850 to 1900 base state, a more positive F_ovS goes with less "
            "weakening at 26.5 N (21 models; family-cluster bootstrap)",
            s26["partial_base"]["absolute"]["r"], b26["partial_r"]["ci95"],
            b26["partial_r"]["ci_excludes_zero"] and s26["partial_base"]["absolute"]["r"] > 0),
        "P2_partial_26N_fractional_positive": claim(
            "Same, with the fractional change of the overturning at 26.5 N",
            s26["partial_base"]["fractional"]["r"], b26["partial_r_frac"]["ci95"],
            b26["partial_r_frac"]["ci_excludes_zero"] and s26["partial_base"]["fractional"]["r"] > 0),
        "P3_raw_26N_21_models": claim(
            "Raw correlation of F_ovS with the 26.5 N change, 21 models",
            s26["raw"]["pearson_r"], b26["r"]["ci95"],
            b26["r"]["ci_excludes_zero"] and s26["raw"]["pearson_r"] > 0),
        "P4_raw_26N_common_17_models_negative": claim(
            "Raw correlation of F_ovS with the 26.5 N change on the 17 models with both "
            "latitudes (sign opposite to the partial)",
            sc26["raw"]["pearson_r"], bc26["r"]["ci95"],
            bc26["r"]["ci_excludes_zero"] and sc26["raw"]["pearson_r"] < 0),
        "P5_raw_34S_19_models_negative": claim(
            "Raw correlation of F_ovS with the 34.5 S change, 19 models (the paper's "
            "\"absent at 34.5 S\" statement concerns the partial only)",
            s34["raw"]["pearson_r"], b34["r"]["ci95"],
            b34["r"]["ci_excludes_zero"] and s34["raw"]["pearson_r"] < 0),
        "P6_partial_34S_not_distinguishable_from_zero": claim(
            "Partial correlation at 34.5 S given the base state is not distinguishable from zero",
            s34["partial_base"]["absolute"]["r"], b34["partial_r"]["ci95"],
            not b34["partial_r"]["ci_excludes_zero"]),
        "P7_partial_26N_minus_34S_contrast": claim(
            "The 26.5 N partial exceeds the 34.5 S partial (\"absent at 34.5 S\"), paired on "
            "the 17 common models",
            paired["partial_r"]["difference_26N_minus_34S"], paired["partial_r"]["ci95"],
            paired["partial_r"]["ci_excludes_zero"]
            and paired["partial_r"]["difference_26N_minus_34S"] > 0),
        "P8_partial_26N_stable_leave_one_model_out": claim(
            "Partial at 26.5 N keeps its sign when any one model is dropped (21 models)",
            s26["partial_base"]["leave_one_model_out_range"],
            s26["partial_base"]["leave_one_model_out_range"],
            min(s26["partial_base"]["leave_one_model_out_range"]) > 0),
        "P9_partial_26N_on_common_17_models": claim(
            "Partial at 26.5 N on the 17 models with both latitudes",
            sc26["partial_base"]["absolute"]["r"], bc26["partial_r"]["ci95"],
            bc26["partial_r"]["ci_excludes_zero"] and sc26["partial_base"]["absolute"]["r"] > 0),
        "P10_errors_in_variables_negligible": claim(
            "Errors in F_ovS (interannual sampling only) do not change the slope: reliability "
            "ratio of F_ovS and York versus OLS slope, 21 models, per 0.1 Sv of F_ovS",
            {"reliability_ratio_x": s26["reliability_ratio_x"],
             "ols": s26["slopes"]["ols_per_0p1_Sv_F"],
             "york": s26["slopes"]["york_slope_per_0p1_Sv_F"]},
            None,
            abs(s26["slopes"]["york_slope_per_0p1_Sv_F"] - s26["slopes"]["ols_per_0p1_Sv_F"])
            < 0.1 * abs(s26["slopes"]["ols_per_0p1_Sv_F"])),
    }

    # The 26.5 N cluster bootstrap for the partial is on all 21 models, so the
    # claim checks above use the same draws as the bootstrap summaries.
    out = {
        "description": (
            "CMIP6 cross-model relation between the 1950 to 1980 mean F_ovS at 34.5 S and "
            "the forced change of the overturning at 26.5 N (1850 to 1900 to 2081 to 2100, "
            "ssp585) and of the streamfunction maximum at 34.5 S; errors-in-variables "
            "regressions, rank and partial correlations, family-cluster bootstrap."
        ),
        "units": {"F_ovS": "Sv", "overturning change": "Sv", "slopes": "Sv per Sv"},
        "n_boot": N_BOOT,
        "seed": {"bootstrap": SEED, "paired": SEED + 1},
        "model_sets": {
            "26.5N_21": set26,
            "34.5S_19": set34,
            "common_17": common,
            "excluded_from_26.5N": [m for m in models if m not in set26],
            "excluded_from_34.5S": [m for m in models if m not in set34],
        },
        "family_clusters": {
            "26.5N_21": len(clusters_of(set26)),
            "34.5S_19": len(clusters_of(set34)),
            "common_17": len(clusters_of(common)),
        },
        "error_model": (
            "Sampling error of each window mean from interannual variability: standard error "
            "about a linear trend with the lag-1 corrected effective sample size "
            "(paper3r1_stats.fitted_mean_test). Change errors add the two window errors in "
            "quadrature. Deming delta = mean(error variance of y) / mean(error variance of F) "
            "over the set; the interannual sensitivity uses the ratio of the mean interannual "
            "variances instead."
        ),
        "consistency_with_r1_cmip6_json": tabulate_checks(tab, ref),
        "per_model": tab.reset_index().to_dict(orient="records"),
        "sets": blocks,
        "paired_difference_common_17": paired,
        "claims": claims,
    }
    OUT.write_text(json.dumps(_clean(out), indent=1))
    print(f"wrote {OUT}")


def _clean(o):
    """Recursively convert numpy scalars and non-finite floats for JSON."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if o is None or isinstance(o, str):
        return o
    return str(o)


if __name__ == "__main__":
    main()
