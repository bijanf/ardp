#!/usr/bin/env python3
"""Attribution of the F_ovS change, rebuilt on an exact two-factor factorisation.

The earlier version of this analysis factorised F_ovS as -Psi DvS / S0 with DvS
defined as -S0 F_ovS / Psi. That is not a decomposition: the second factor is
the first divided out, so it absorbs every change in the vertical structure of
the velocity as well as every change in salinity, it closes to machine
precision by construction, and the resulting "overturning share" is fixed by
the algebraic identity

    share = (F_early / dF) * (dPsi / Psi_early).

This script discards that construction and uses the exact limb factorisation
computed in ``compute_paper3v2_section_profiles.py``,

    F_ovS = -(1/S0) * T * dS ,

with T the northward-limb transport of the barotropic-corrected velocity and
dS = S_north - S_south the transport-weighted salinity contrast between the two
limbs. Both are measured directly from the section profile, neither is defined
in terms of the other, and T is linear in the velocity field, so it carries
none of the upward sampling bias of a streamfunction maximum.

The epoch decomposition is taken in the symmetric (Shapley) form

    dF = -(1/S0) * ( <dS> * dT  +  <T> * ddS ) ,

with <T> and <dS> the means of the two epochs. This is exact, has no cross
term to discard, and does not depend on which epoch is chosen as the base
point. The asymmetric early-base and late-base conventions are also reported,
because the difference between them is large and the previous version quoted
only the convention that gave the smallest number.

Everything reported here carries an uncertainty. Trends use the Santer
effective sample size with N_eff stated, means use a block bootstrap whose
block length is varied rather than fixed at an arbitrary value, and shares use
a paired block bootstrap.

Writes ``PAPER_3_v2/analysis/attribution.json``.
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

S0 = 35.0
RNG = np.random.default_rng(20260805)
N_BOOT = 10_000
BLOCK_LENGTHS = (2, 3, 5, 10, 15, 20)
PRIMARY_BLOCK = 5

SECTIONS = {
    "oras5": ("ORAS5", "paper3v2_section_oras5.nc"),
    "glorys12": ("GLORYS12V1", "paper3v2_section_glorys12.nc"),
    "ecco": ("ECCO-V4r4", "paper3v2_section_ecco.nc"),
}

COMMON_WINDOW = (1993, 2017)
ARGO_START = 2004


# ── helpers ───────────────────────────────────────────────────────────


def block_idx(n: int, block: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n, size=n_blocks)
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def mean_ci(v: np.ndarray, block: int) -> list[float]:
    boot = np.array([v[block_idx(len(v), block)].mean() for _ in range(N_BOOT)])
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def trend_stats(years: np.ndarray, v: np.ndarray, scale: float = 1e3) -> dict:
    """OLS trend with Santer N_eff, its bootstrap CI, and the detection floor."""
    x = years.astype(float)
    fit = ols_santer(x, v)
    coef = np.polyfit(x, v, 1)
    resid = v - np.polyval(coef, x)
    fitted = np.polyval(coef, x)
    boot = np.array(
        [
            np.polyfit(x, fitted + resid[block_idx(len(x), PRIMARY_BLOCK)], 1)[0]
            for _ in range(N_BOOT)
        ]
    )
    # Smallest trend that this record could have resolved at the 5% level.
    t_crit = stats.t.ppf(0.975, df=max(fit["n_eff"] - 2, 1))
    return {
        "n": int(len(x)),
        "window": [int(years[0]), int(years[-1])],
        "trend": fit["slope"] * scale,
        "p_santer": fit["p_santer"],
        "p_ols": fit["p_ols"],
        "n_eff": fit["n_eff"],
        "lag1": fit["lag1_autocorr"],
        "significant": bool(fit["p_santer"] < 0.05),
        "ci": [
            float(np.percentile(boot, 2.5) * scale),
            float(np.percentile(boot, 97.5) * scale),
        ],
        "min_detectable_trend": float(t_crit * fit["stderr_santer"] * scale),
    }


def annual(ds: xr.Dataset) -> xr.Dataset:
    return ds.groupby("time.year").mean()


# ── the two-factor decomposition ──────────────────────────────────────


def decompose(t1: float, t2: float, d1: float, d2: float) -> dict:
    """Exact two-factor split of the change in F = -(1/S0) T D.

    The symmetric form uses the epoch-pair means as coefficients and has no
    cross term; the two asymmetric forms are the usual early-base and late-base
    expansions, whose difference is exactly the cross term.
    """
    f1, f2 = -(1 / S0) * t1 * d1, -(1 / S0) * t2 * d2
    dt, dd = t2 - t1, d2 - d1
    total = f2 - f1
    sym_t = -(1 / S0) * 0.5 * (d1 + d2) * dt
    sym_d = -(1 / S0) * 0.5 * (t1 + t2) * dd
    early_t = -(1 / S0) * d1 * dt
    late_t = -(1 / S0) * d2 * dt
    cross = -(1 / S0) * dt * dd
    return {
        "T_early": t1,
        "T_late": t2,
        "dT": dt,
        "dT_pct": 100.0 * dt / t1,
        "dS_early": d1,
        "dS_late": d2,
        "ddS": dd,
        "ddS_pct": 100.0 * dd / d1,
        "F_early_mSv": f1 * 1e3,
        "F_late_mSv": f2 * 1e3,
        "total_mSv": total * 1e3,
        "term_T_sym_mSv": sym_t * 1e3,
        "term_S_sym_mSv": sym_d * 1e3,
        "term_T_earlybase_mSv": early_t * 1e3,
        "term_T_latebase_mSv": late_t * 1e3,
        "cross_mSv": cross * 1e3,
        "closure_sym_mSv": (sym_t + sym_d - total) * 1e3,
        "share_T_sym": abs(sym_t / total) if total else np.nan,
        "share_T_earlybase": abs(early_t / total) if total else np.nan,
        "share_T_latebase": abs(late_t / total) if total else np.nan,
    }


def continuous(years: np.ndarray, t: np.ndarray, d: np.ndarray, f: np.ndarray) -> dict:
    """Trend attribution using every year, by freezing one factor at its mean."""
    t_bar, d_bar = float(t.mean()), float(d.mean())
    f_t = -(1 / S0) * t * d_bar
    f_d = -(1 / S0) * t_bar * d

    x = years.astype(float)
    parts = {}
    fits = {}
    for name, series in (("total", f), ("T", f_t), ("S", f_d)):
        coef = np.polyfit(x, series, 1)
        parts[name] = (np.polyval(coef, x), series - np.polyval(coef, x))
        fits[name] = ols_santer(x, series)

    boot = {k: np.empty(N_BOOT) for k in parts}
    for i in range(N_BOOT):
        idx = block_idx(len(x), PRIMARY_BLOCK)
        for k, (fitted, resid) in parts.items():
            boot[k][i] = np.polyfit(x, fitted + resid[idx], 1)[0]

    ratio = np.abs(boot["T"]) / np.abs(boot["total"])
    out = {
        "n": int(len(x)),
        "window": [int(years[0]), int(years[-1])],
        "T_mean_Sv": t_bar,
        "dS_mean_PSU": d_bar,
        "residual_mSv_per_yr": (
            fits["total"]["slope"] - fits["T"]["slope"] - fits["S"]["slope"]
        )
        * 1e3,
        "share_T_point": float(abs(fits["T"]["slope"] / fits["total"]["slope"])),
        "share_T_ci": [
            float(np.percentile(ratio, 2.5)),
            float(np.percentile(ratio, 97.5)),
        ],
        "prob_share_T_below_0p25": float(np.mean(ratio < 0.25)),
        "prob_T_term_opposes_total": float(
            np.mean(np.sign(boot["T"]) != np.sign(boot["total"]))
        ),
    }
    for k in ("total", "T", "S"):
        out[f"trend_{k}_mSv_per_yr"] = fits[k]["slope"] * 1e3
        out[f"p_{k}"] = fits[k]["p_santer"]
        out[f"n_eff_{k}"] = fits[k]["n_eff"]
        out[f"ci_{k}_mSv_per_yr"] = [
            float(np.percentile(boot[k], 2.5) * 1e3),
            float(np.percentile(boot[k], 97.5) * 1e3),
        ]
    return out


def census(
    years: np.ndarray, t: np.ndarray, d: np.ndarray, lengths=(10, 13, 15)
) -> dict:
    """Every non-overlapping early/late epoch pair the record supports."""
    shares_sym, shares_early, shares_late, totals, terms_t = [], [], [], [], []
    n = len(years)
    for length in lengths:
        for i in range(0, n - 2 * length + 1):
            for j in range(i + length, n - length + 1):
                e, lt = slice(i, i + length), slice(j, j + length)
                r = decompose(
                    float(t[e].mean()),
                    float(t[lt].mean()),
                    float(d[e].mean()),
                    float(d[lt].mean()),
                )
                totals.append(r["total_mSv"])
                terms_t.append(r["term_T_sym_mSv"])
                shares_sym.append(r["share_T_sym"])
                shares_early.append(r["share_T_earlybase"])
                shares_late.append(r["share_T_latebase"])
    totals = np.array(totals)
    terms_t = np.array(terms_t)
    keep = np.abs(totals) > 5.0
    ss = np.array(shares_sym)[keep]
    return {
        "n_pairs": int(len(totals)),
        "n_pairs_usable": int(keep.sum()),
        "epoch_lengths": list(lengths),
        "frac_total_negative": float(np.mean(totals < 0)),
        "frac_T_term_opposes_total": float(
            np.mean(np.sign(terms_t[keep]) != np.sign(totals[keep]))
        ),
        "share_T_sym_median": float(np.median(ss)),
        "share_T_sym_p90": float(np.percentile(ss, 90)),
        "share_T_sym_p95": float(np.percentile(ss, 95)),
        "share_T_sym_max": float(ss.max()),
        "frac_share_below_0p25": float(np.mean(ss < 0.25)),
        "share_T_earlybase_median": float(np.median(np.array(shares_early)[keep])),
        "share_T_latebase_median": float(np.median(np.array(shares_late)[keep])),
    }


def epoch_bootstrap(
    years: np.ndarray, t: np.ndarray, d: np.ndarray, early: tuple, late: tuple
) -> dict:
    me = (years >= early[0]) & (years <= early[1])
    ml = (years >= late[0]) & (years <= late[1])
    point = decompose(
        float(t[me].mean()),
        float(t[ml].mean()),
        float(d[me].mean()),
        float(d[ml].mean()),
    )
    bt, bs, btot = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
    te, de, tl, dl = t[me], d[me], t[ml], d[ml]
    for i in range(N_BOOT):
        ie, il = block_idx(len(te), PRIMARY_BLOCK), block_idx(len(tl), PRIMARY_BLOCK)
        r = decompose(
            float(te[ie].mean()),
            float(tl[il].mean()),
            float(de[ie].mean()),
            float(dl[il].mean()),
        )
        bt[i], bs[i], btot[i] = r["term_T_sym_mSv"], r["term_S_sym_mSv"], r["total_mSv"]
    ratio = np.abs(bt) / np.abs(btot)
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    point.update(
        {
            "term_T_ci_mSv": pct(bt),
            "term_S_ci_mSv": pct(bs),
            "total_ci_mSv": pct(btot),
            "share_T_ci": pct(ratio),
            "prob_share_below_0p25": float(np.mean(ratio < 0.25)),
            "prob_S_term_larger": float(np.mean(np.abs(bs) > np.abs(bt))),
        }
    )
    return point


# ── per-product driver ────────────────────────────────────────────────


def run_product(key: str) -> dict:
    label, fname = SECTIONS[key]
    ds = xr.open_dataset(RESULTS / fname)
    a = annual(ds)
    years = a["year"].values.astype(int)
    depth = ds["depth"].values
    e3t = np.diff(depth, prepend=0.0)

    fov = a["F_ov"].values
    t_limb = a["T_limb_Sv"].values
    ds_limb = a["dS_limb"].values

    # F_ov formed from annual-mean fields, the route ECCO and SODA were
    # originally computed by. The gap is the discarded within-year covariance.
    fov_from_annual = (
        -(1 / S0)
        * np.nansum(a["V_bc"].values * (a["S_bar"].values - S0) * e3t, axis=1)
        / 1e6
    )

    out: dict = {
        "label": label,
        "section_latitude": float(ds.attrs["section_latitude"]),
        "n_atlantic_points": int(ds.attrs["n_atlantic_points"]),
        "record": [int(years[0]), int(years[-1])],
        "n_years": int(len(years)),
        "years": years.tolist(),
        "F_ov_Sv": fov.tolist(),
        "T_limb_Sv": t_limb.tolist(),
        "dS_limb_PSU": ds_limb.tolist(),
        "mean_F_ov_Sv": float(fov.mean()),
        "mean_F_ov_uncorrected_Sv": float(a["F_ov_raw"].values.mean()),
        "barotropic_correction_Sv": float((a["F_ov"] - a["F_ov_raw"]).values.mean()),
        "mean_V_net_Sv": float(a["V_net_Sv"].values.mean()),
        "mean_F_az_Sv": float(a["F_az"].values.mean()),
        "mean_F_ov_from_annual_fields_Sv": float(fov_from_annual.mean()),
        "annual_field_bias_Sv": float(fov_from_annual.mean() - fov.mean()),
        "mean_T_limb_Sv": float(t_limb.mean()),
        "mean_dS_limb_PSU": float(ds_limb.mean()),
        "mean_S_north": float(a["S_north"].values.mean()),
        "mean_S_south": float(a["S_south"].values.mean()),
        "mean_psi_max_surface_Sv": float(a["psi_max_surface_Sv"].values.mean()),
        "mean_psi_max_250m_Sv": float(a["psi_max_250m_Sv"].values.mean()),
        "n_years_F_ov_negative": int((fov < 0).sum()),
        "lag1_F_ov": float(np.corrcoef(fov[:-1], fov[1:])[0, 1]),
    }

    # Record mean, and how its interval depends on the block length.
    out["mean_ci_by_block"] = {str(b): mean_ci(fov, b) for b in BLOCK_LENGTHS}
    out["mean_ci_excludes_zero_by_block"] = {
        str(b): bool(ci[0] * ci[1] > 0) for b, ci in out["mean_ci_by_block"].items()
    }
    # A mean over a trending record is not a well-defined population quantity;
    # the most recent decade is, so report it too.
    recent = years >= years[-1] - 9
    out["recent_decade"] = {
        "window": [int(years[recent][0]), int(years[-1])],
        "mean_F_ov_Sv": float(fov[recent].mean()),
        "ci": mean_ci(fov[recent], 3),
    }

    windows = {"own record": (int(years[0]), int(years[-1])), "common": COMMON_WINDOW}
    if years[-1] - ARGO_START >= 14:
        windows["argo"] = (ARGO_START, int(years[-1]))
    out["trends"] = {}
    for name, (y0, y1) in windows.items():
        m = (years >= y0) & (years <= y1)
        if m.sum() < 10:
            continue
        out["trends"][name] = {
            "F_ov": trend_stats(years[m], fov[m]),
            "T_limb": trend_stats(years[m], t_limb[m], scale=10.0),
            "dS_limb": trend_stats(years[m], ds_limb[m], scale=10.0),
            "S_north": trend_stats(years[m], a["S_north"].values[m], scale=10.0),
            "S_south": trend_stats(years[m], a["S_south"].values[m], scale=10.0),
            "F_az": trend_stats(years[m], a["F_az"].values[m]),
        }

    out["continuous"] = {
        name: continuous(
            years[(years >= y0) & (years <= y1)],
            t_limb[(years >= y0) & (years <= y1)],
            ds_limb[(years >= y0) & (years <= y1)],
            fov[(years >= y0) & (years <= y1)],
        )
        for name, (y0, y1) in windows.items()
        if ((years >= y0) & (years <= y1)).sum() >= 15
    }
    out["census"] = census(years, t_limb, ds_limb)

    # The epoch pair used by the previous version, kept only so that the old
    # and new attributions can be compared directly.
    if years[0] <= 1993 and years[-1] >= 2020:
        late_end = min(2025, int(years[-1]))
        out["epoch_pair"] = epoch_bootstrap(
            years, t_limb, ds_limb, (1993, 2005), (2013, late_end)
        )

    # Attribution under three definitions of the overturning strength, to show
    # the conclusion does not depend on which scalar is called "the overturning".
    out["psi_definition_sensitivity"] = {}
    for name, series in (
        ("exchange transport T", t_limb),
        ("streamfunction max, surface start", a["psi_max_surface_Sv"].values),
        ("streamfunction max, 250 m start", a["psi_max_250m_Sv"].values),
    ):
        d_eff = -S0 * fov / series  # the contrast implied by that choice
        c = continuous(years, series, d_eff, fov)
        out["psi_definition_sensitivity"][name] = {
            "mean_Sv": float(series.mean()),
            "trend_per_decade": trend_stats(years, series, scale=10.0)["trend"],
            "p_santer": trend_stats(years, series, scale=10.0)["p_santer"],
            "share_T_point": c["share_T_point"],
            "share_T_ci": c["share_T_ci"],
        }

    # ORAS5 is a splice: the consolidated reanalysis to 2014, the operational
    # stream thereafter. Attribution on the homogeneous part only.
    if key == "oras5":
        m = years <= 2014
        out["consolidated_only"] = {
            "window": [int(years[0]), 2014],
            "trend_F_ov": trend_stats(years[m], fov[m]),
            "continuous": continuous(years[m], t_limb[m], ds_limb[m], fov[m]),
            "mean_F_ov_Sv": float(fov[m].mean()),
        }
        m2 = years >= 2015
        out["operational_only"] = {
            "window": [2015, int(years[-1])],
            "n": int(m2.sum()),
            "mean_F_ov_Sv": float(fov[m2].mean()),
        }
        out["splice_step_Sv"] = float(fov[m2].mean() - fov[(years >= 2005) & m].mean())
    return out


def report(res: dict) -> None:
    for r in res.values():
        print(
            f"\n{'=' * 78}\n{r['label']}  {r['record'][0]}-{r['record'][1]}  "
            f"({r['n_atlantic_points']} ocean points at "
            f"{r['section_latitude']:.2f})\n{'=' * 78}"
        )
        print(
            f"  F_ov mean {r['mean_F_ov_Sv']:+.4f} Sv (uncorrected "
            f"{r['mean_F_ov_uncorrected_Sv']:+.4f}, barotropic correction "
            f"{r['barotropic_correction_Sv']:+.4f}, net section transport "
            f"{r['mean_V_net_Sv']:+.2f} Sv)"
        )
        print(
            f"F_ov from annual fields {r['mean_F_ov_from_annual_fields_Sv']:+.4f} Sv, "
            f"bias {r['annual_field_bias_Sv']:+.4f} Sv"
        )
        print(
            f"  F_az {r['mean_F_az_Sv']:+.4f} Sv   T {r['mean_T_limb_Sv']:.2f} Sv   "
            f"dS {r['mean_dS_limb_PSU']:+.4f} PSU  "
            f"(S_N {r['mean_S_north']:.3f}, S_S {r['mean_S_south']:.3f})"
        )
        print(
            f"  Psi_max surface {r['mean_psi_max_surface_Sv']:.2f} Sv, "
            f"250 m start {r['mean_psi_max_250m_Sv']:.2f} Sv"
        )
        print(
            f"F_ov < 0 in {r['n_years_F_ov_negative']}/{r['n_years']} years, lag-1 "
            f"{r['lag1_F_ov']:.3f}"
        )
        print("  mean CI by block length:")
        for b, ci in r["mean_ci_by_block"].items():
            flag = (
                "excludes 0" if r["mean_ci_excludes_zero_by_block"][b] else "INCLUDES 0"
            )
            print(f"     L={b:>2}: [{ci[0]:+.4f}, {ci[1]:+.4f}]  {flag}")
        rd = r["recent_decade"]
        print(
            f"  most recent decade {rd['window'][0]}-{rd['window'][1]}: "
            f"{rd['mean_F_ov_Sv']:+.4f} [{rd['ci'][0]:+.4f}, {rd['ci'][1]:+.4f}] Sv"
        )
        for name, tr in r["trends"].items():
            f = tr["F_ov"]
            print(
                f"  trend {name:12s} {f['window'][0]}-{f['window'][1]}: "
                f"{f['trend']:+7.3f} [{f['ci'][0]:+.3f},{f['ci'][1]:+.3f}] mSv/yr "
                f"p={f['p_santer']:.4f} N_eff={f['n_eff']:.1f} "
                f"(detectable at {f['min_detectable_trend']:.2f})"
            )
        for name, c in r["continuous"].items():
            print(
                f"\n  -- continuous attribution, {name} "
                f"({c['window'][0]}-{c['window'][1]}, n={c['n']})"
            )
            for k, lab in (
                ("total", "dF_ov/dt"),
                ("T", "transport"),
                ("S", "salinity "),
            ):
                print(
                    f"     {lab:10s} {c[f'trend_{k}_mSv_per_yr']:+7.3f} "
                    f"[{c[f'ci_{k}_mSv_per_yr'][0]:+.3f},"
                    f"{c[f'ci_{k}_mSv_per_yr'][1]:+.3f}] mSv/yr "
                    f"p={c[f'p_{k}']:.3f} N_eff={c[f'n_eff_{k}']:.1f}"
                )
            print(
                f"     share_T {c['share_T_point']:.3f} "
                f"[{c['share_T_ci'][0]:.3f},{c['share_T_ci'][1]:.3f}]  "
                f"P(<0.25)={c['prob_share_T_below_0p25']:.3f}  residual "
                f"{c['residual_mSv_per_yr']:+.3f}"
            )
        e = r["census"]
        print(
            f"\n  -- census: {e['n_pairs_usable']}/{e['n_pairs']} usable pairs; "
            f"total negative in {100 * e['frac_total_negative']:.1f}%; "
            f"transport opposes total in "
            f"{100 * e['frac_T_term_opposes_total']:.1f}%"
        )
        print(
            f"     share_T (symmetric): median {e['share_T_sym_median']:.3f}, "
            f"p95 {e['share_T_sym_p95']:.3f}, max {e['share_T_sym_max']:.3f}, "
            f"below 0.25 in {100 * e['frac_share_below_0p25']:.1f}%"
        )
        print(
            f"     median share under early base {e['share_T_earlybase_median']:.3f}, "
            f"late base {e['share_T_latebase_median']:.3f}"
        )
        if "epoch_pair" in r:
            p = r["epoch_pair"]
            print(
                f"\n  -- epoch pair 1993-2005 vs 2013-{p['T_late'] and ''}: "
                f"T {p['T_early']:.2f} to {p['T_late']:.2f} Sv ({p['dT_pct']:+.1f}%), "
                f"dS {p['dS_early']:+.4f} to {p['dS_late']:+.4f} PSU "
                f"({p['ddS_pct']:+.1f}%)"
            )
            print(
                f"     transport term {p['term_T_sym_mSv']:+6.2f} "
                f"[{p['term_T_ci_mSv'][0]:+.2f},{p['term_T_ci_mSv'][1]:+.2f}]  "
                f"salinity term {p['term_S_sym_mSv']:+7.2f} "
                f"[{p['term_S_ci_mSv'][0]:+.2f},{p['term_S_ci_mSv'][1]:+.2f}]  "
                f"total {p['total_mSv']:+7.2f}"
            )
            print(
                f"     share_T sym {p['share_T_sym']:.3f} "
                f"[{p['share_T_ci'][0]:.3f},{p['share_T_ci'][1]:.3f}], "
                f"early base {p['share_T_earlybase']:.3f}, late base "
                f"{p['share_T_latebase']:.3f}, closure {p['closure_sym_mSv']:+.1e}"
            )
        for name, s in r["psi_definition_sensitivity"].items():
            print(
                f"  Psi definition '{name}': mean {s['mean_Sv']:.2f} Sv, trend "
                f"{s['trend_per_decade']:+.3f}/dec (p={s['p_santer']:.3f}), share "
                f"{s['share_T_point']:.3f} "
                f"[{s['share_T_ci'][0]:.3f},{s['share_T_ci'][1]:.3f}]"
            )
        if "consolidated_only" in r:
            c = r["consolidated_only"]
            print(
                f"\n  -- consolidated stream only ({c['window'][0]}-{c['window'][1]}): "
                f"mean {c['mean_F_ov_Sv']:+.4f} Sv, trend "
                f"{c['trend_F_ov']['trend']:+.3f} mSv/yr "
                f"p={c['trend_F_ov']['p_santer']:.4f}, "
                f"share_T {c['continuous']['share_T_point']:.3f} "
                f"[{c['continuous']['share_T_ci'][0]:.3f},{c['continuous']['share_T_ci'][1]:.3f}]"
            )
            print(
                f"     operational stream {r['operational_only']['window'][0]}-"
                f"{r['operational_only']['window'][1]} mean "
                f"{r['operational_only']['mean_F_ov_Sv']:+.4f} Sv; "
                f"step across the splice {r['splice_step_Sv']:+.4f} Sv"
            )


def main() -> None:
    res = {}
    for key in SECTIONS:
        if (RESULTS / SECTIONS[key][1]).exists():
            res[key] = run_product(key)
        else:
            print(f"skipping {key}: section file not built yet")
    report(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "attribution.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
