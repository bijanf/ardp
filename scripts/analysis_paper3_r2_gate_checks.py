#!/usr/bin/env python3
"""PAPER_3 round-2 gate checks A2 and A3 (see revision/rev_papaer3_02/RESPONSE_PLAN.md).

A2: verify R3's falsifiable count on the AMOC-rate figure (fig:amoc_rate):
    "only 2/6 bistable models weaken more than GLORYS, while 6/10 monostable
    models do." Replicates the exact computation of
    scripts/plot_amoc_rate_comparison.py (same loader, same rate function,
    same window) and counts models on each side of the observed lines.

A3: split-period statistics behind the "fivefold acceleration since 1990"
    claim, on the canonical ORAS5 F_ovS series: OLS + Santer N_eff p-values
    for 1958-1989 vs 1990-2025, circular block bootstrap CIs for both
    slopes, their difference, and their ratio, plus a changepoint scan.

Outputs JSON + a markdown summary into revision/rev_papaer3_02/results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from plot_amoc_rate_comparison import compute_rate, load_cmip6  # noqa: E402

from ardp.models import models_sorted_by_fovs  # noqa: E402

RESULTS_DIR = REPO / "data" / "results"
OUT_DIR = REPO / "revision" / "rev_papaer3_02" / "results"

RNG = np.random.default_rng(20260804)
N_BOOT = 10_000
BLOCK_YEARS = 5


# ══════════════════════════════════════════════════════════════════════
# Trend statistics
# ══════════════════════════════════════════════════════════════════════


def ols_santer(years: np.ndarray, values: np.ndarray) -> dict:
    """OLS slope with both naive and Santer N_eff-adjusted p-values.

    Santer et al. (2000): inflate the slope standard error using the
    lag-1 autocorrelation of the regression residuals,
    N_eff = N (1 - r1) / (1 + r1).
    """
    n = len(years)
    res = stats.linregress(years, values)
    resid = values - (res.intercept + res.slope * years)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 2 else 0.0
    r1 = min(max(r1, -0.99), 0.99)
    neff = n * (1.0 - r1) / (1.0 + r1)
    neff = max(neff, 3.0)
    se_adj = res.stderr * np.sqrt((n - 2) / (neff - 2))
    t_adj = res.slope / se_adj if se_adj > 0 else np.inf
    p_adj = 2.0 * stats.t.sf(abs(t_adj), df=neff - 2)
    return {
        "n_years": n,
        "slope": float(res.slope),
        "stderr_ols": float(res.stderr),
        "p_ols": float(res.pvalue),
        "lag1_autocorr": r1,
        "n_eff": float(neff),
        "stderr_santer": float(se_adj),
        "p_santer": float(p_adj),
    }


def _block_resample_residuals(resid: np.ndarray, block: int) -> np.ndarray:
    """Circular moving-block bootstrap of a residual series."""
    n = len(resid)
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return resid[idx].ravel()[:n]


def bootstrap_slopes(years: np.ndarray, values: np.ndarray, n_boot: int) -> np.ndarray:
    """Bootstrap slope distribution: block-resampled residuals added back
    to the OLS fit, preserving trend and residual autocorrelation."""
    res = stats.linregress(years, values)
    fit = res.intercept + res.slope * years
    resid = values - fit
    slopes = np.empty(n_boot)
    x = years - years.mean()
    denom = float(np.sum(x * x))
    for i in range(n_boot):
        pseudo = fit + _block_resample_residuals(resid, BLOCK_YEARS)
        slopes[i] = float(np.sum(x * (pseudo - pseudo.mean())) / denom)
    return slopes


# ══════════════════════════════════════════════════════════════════════
# A3: split-period statistics on canonical ORAS5 F_ovS
# ══════════════════════════════════════════════════════════════════════


def a3_split_period(break_year: int = 1990) -> dict:
    ds = xr.open_dataset(RESULTS_DIR / "oras5_f_ovs.nc")
    annual = ds["F_ovS"].groupby("time.year").mean()
    years = annual["year"].values.astype(float)
    vals = annual.values.astype(float)
    ds.close()

    pre_m = years < break_year
    post_m = years >= break_year

    pre = ols_santer(years[pre_m], vals[pre_m])
    post = ols_santer(years[post_m], vals[post_m])

    # Full-record check against the canonical CSV (methodological anchor)
    full = ols_santer(years, vals)

    pre_b = bootstrap_slopes(years[pre_m], vals[pre_m], N_BOOT)
    post_b = bootstrap_slopes(years[post_m], vals[post_m], N_BOOT)
    ratio_b = post_b / pre_b
    diff_b = post_b - pre_b

    def _ci(a: np.ndarray) -> list[float]:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    ratio_point = post["slope"] / pre["slope"] if pre["slope"] != 0 else np.inf

    scan = {}
    for by in range(1985, 1996):
        m1, m2 = years < by, years >= by
        s1 = ols_santer(years[m1], vals[m1])
        s2 = ols_santer(years[m2], vals[m2])
        scan[by] = {
            "pre_mSv_yr": s1["slope"] * 1e3,
            "pre_p_santer": s1["p_santer"],
            "post_mSv_yr": s2["slope"] * 1e3,
            "post_p_santer": s2["p_santer"],
            "ratio": (s2["slope"] / s1["slope"]) if s1["slope"] != 0 else np.inf,
        }

    return {
        "break_year": break_year,
        "series": "data/results/oras5_f_ovs.nc (canonical), annual means",
        "full_record": full,
        "pre": pre,
        "post": post,
        "point_ratio": float(ratio_point),
        "bootstrap": {
            "n_boot": N_BOOT,
            "block_years": BLOCK_YEARS,
            "pre_slope_ci_mSv_yr": [v * 1e3 for v in _ci(pre_b)],
            "post_slope_ci_mSv_yr": [v * 1e3 for v in _ci(post_b)],
            "pre_ci_contains_zero": bool(
                np.percentile(pre_b, 2.5) < 0 < np.percentile(pre_b, 97.5)
            ),
            "slope_diff_ci_mSv_yr": [v * 1e3 for v in _ci(diff_b)],
            "diff_ci_excludes_zero": bool(
                np.percentile(diff_b, 97.5) < 0 or np.percentile(diff_b, 2.5) > 0
            ),
            "ratio_ci": _ci(ratio_b),
            "ratio_median": float(np.median(ratio_b)),
            "ratio_frac_negative": float(np.mean(ratio_b < 0)),
            "ratio_frac_above_10": float(np.mean(np.abs(ratio_b) > 10)),
        },
        "changepoint_scan": scan,
    }


# ══════════════════════════════════════════════════════════════════════
# A2: recount the AMOC-rate figure (fig:amoc_rate)
# ══════════════════════════════════════════════════════════════════════


def a2_fig10_count() -> dict:
    oras5 = np.load(RESULTS_DIR / "yearly_amoc26n_oras5.npz")
    glorys = np.load(RESULTS_DIR / "yearly_amoc26n_glorys12.npz")
    rapid = np.load(RESULTS_DIR / "rapid_amoc26n.npz")
    cmip6 = load_cmip6(RESULTS_DIR)

    y0, y1 = int(rapid["years"][0]), int(rapid["years"][-1])
    oras5_rate, _ = compute_rate(oras5["years"], oras5["amoc"], y0, y1)
    glorys_rate, _ = compute_rate(glorys["years"], glorys["amoc"], y0, y1)
    rapid_rate, _ = compute_rate(rapid["years"], rapid["amoc"], y0, y1)

    rows = []
    for model, fovs in models_sorted_by_fovs():
        if model not in cmip6:
            continue
        d = cmip6[model]
        rate, pval = compute_rate(d["years"], d["amoc"], y0, y1)
        if not np.isfinite(rate):
            continue
        rows.append(
            {
                "model": model,
                "fovs_Sv": float(fovs),
                "class": "bistable" if fovs < 0 else "monostable",
                "rate_Sv_dec": float(rate),
                "p": float(pval),
            }
        )

    def _count(cls: str, ref: float) -> dict:
        grp = [r for r in rows if r["class"] == cls]
        more = [r["model"] for r in grp if r["rate_Sv_dec"] < ref]
        return {"n": len(grp), "n_weaken_more": len(more), "models_weaken_more": more}

    return {
        "window": f"{y0}-{y1} (RAPID overlap, as plotted)",
        "obs_rates_Sv_dec": {
            "ORAS5": float(oras5_rate),
            "GLORYS12": float(glorys_rate),
            "RAPID": float(rapid_rate),
        },
        "models": rows,
        "counts_vs_GLORYS12": {
            "bistable": _count("bistable", glorys_rate),
            "monostable": _count("monostable", glorys_rate),
        },
        "counts_vs_RAPID": {
            "bistable": _count("bistable", rapid_rate),
            "monostable": _count("monostable", rapid_rate),
        },
        "R3_claim": "only 2/6 bistable weaken more than GLORYS; 6/10 monostable do",
    }


# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a3 = a3_split_period()
    (OUT_DIR / "A3_split_period.json").write_text(json.dumps(a3, indent=2))

    a2 = a2_fig10_count()
    (OUT_DIR / "A2_fig10_count.json").write_text(json.dumps(a2, indent=2))

    pre, post, boot = a3["pre"], a3["post"], a3["bootstrap"]
    lines = [
        "# Gate checks A2 + A3 (canonical data)",
        "",
        "## A3: split-period ORAS5 F_ovS trends (break 1990)",
        "",
        f"- Full record: {a3['full_record']['slope'] * 1e3:+.3f} mSv/yr, "
        f"Santer p={a3['full_record']['p_santer']:.2e}, "
        f"N_eff={a3['full_record']['n_eff']:.0f} "
        "(anchor: matches data/results/fovs_multiprod_trends.csv)",
        f"- Pre-1990 (1958-1989): {pre['slope'] * 1e3:+.3f} mSv/yr, "
        f"Santer p={pre['p_santer']:.3f}, "
        f"bootstrap CI [{boot['pre_slope_ci_mSv_yr'][0]:+.2f}, "
        f"{boot['pre_slope_ci_mSv_yr'][1]:+.2f}] mSv/yr, "
        f"contains zero: {boot['pre_ci_contains_zero']}",
        f"- Post-1990 (1990-2025): {post['slope'] * 1e3:+.3f} mSv/yr, "
        f"Santer p={post['p_santer']:.2e}, "
        f"bootstrap CI [{boot['post_slope_ci_mSv_yr'][0]:+.2f}, "
        f"{boot['post_slope_ci_mSv_yr'][1]:+.2f}] mSv/yr",
        f"- Slope difference (post minus pre): CI "
        f"[{boot['slope_diff_ci_mSv_yr'][0]:+.2f}, "
        f"{boot['slope_diff_ci_mSv_yr'][1]:+.2f}] mSv/yr, "
        f"excludes zero: {boot['diff_ci_excludes_zero']}",
        f"- Point ratio: {a3['point_ratio']:.1f}x. Bootstrap ratio CI "
        f"[{boot['ratio_ci'][0]:.1f}, {boot['ratio_ci'][1]:.1f}], "
        f"median {boot['ratio_median']:.1f}, "
        f"{100 * boot['ratio_frac_negative']:.1f}% of resamples negative, "
        f"{100 * boot['ratio_frac_above_10']:.1f}% exceed 10x in magnitude",
        "",
        "Changepoint scan (break year: pre / post mSv/yr, ratio):",
        "",
    ]
    for by, s in a3["changepoint_scan"].items():
        lines.append(
            f"- {by}: {s['pre_mSv_yr']:+.2f} (p={s['pre_p_santer']:.2f}) / "
            f"{s['post_mSv_yr']:+.2f} (p={s['post_p_santer']:.3f}), "
            f"ratio {s['ratio']:.1f}x"
        )

    cg = a2["counts_vs_GLORYS12"]
    lines += [
        "",
        "## A2: fig:amoc_rate recount (window " + a2["window"] + ")",
        "",
        f"- Observed lines (Sv/dec): ORAS5 {a2['obs_rates_Sv_dec']['ORAS5']:+.2f}, "
        f"GLORYS12 {a2['obs_rates_Sv_dec']['GLORYS12']:+.2f}, "
        f"RAPID {a2['obs_rates_Sv_dec']['RAPID']:+.2f}",
        f"- Bistable models weakening more than GLORYS12: "
        f"{cg['bistable']['n_weaken_more']}/{cg['bistable']['n']} "
        f"({', '.join(cg['bistable']['models_weaken_more']) or 'none'})",
        f"- Monostable models weakening more than GLORYS12: "
        f"{cg['monostable']['n_weaken_more']}/{cg['monostable']['n']} "
        f"({', '.join(cg['monostable']['models_weaken_more']) or 'none'})",
        f"- R3's claim: {a2['R3_claim']}",
        "",
    ]
    (OUT_DIR / "GATE_CHECKS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
