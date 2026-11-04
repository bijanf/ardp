#!/usr/bin/env python3
"""MPI-ESM1-2-LR initial-condition ensemble at 34.5 S.

Reads the annual parts written by ``analysis_paper3jc_n20_extract.py`` (1950-2014
historical and 2015-2100 ssp585 for each member extracted from the Pangeo archive),
writes the per-member annual series to ``data/results/paper3jc_smile_mpi_lr/<member>.nc``,
and writes the summary ``PAPER_3_v2/analysis/jc_n20_smile.json`` with four parts:

(a) internal variability of the 1958-2025 trends of F_ovS, T and dS, of the
    transport and contrast terms of the F_ovS trend, and of the 2005-2024 dS
    trend; forced (ensemble-mean) trend, internal spread, and the ORAS5 and EN4
    trends in standardised form; the ratio of the internal spread of the F_ovS
    trend to the mean test standard error, and the ORAS5 interval with its
    half-width scaled by that ratio;
(b) size of the paper's trend test (``trend_test`` of ``paper3r1_stats``) applied
    to each member's departure from the ensemble mean, for record lengths 68, 33,
    26 and 20 years, with the rejection rate against the nominal 5 percent;
(c) forced change 1993-2014 to 2081-2100 of F_ovS, T and dS, split exactly into
    transport, contrast and product terms, against the CMIP6 single-member changes;
(d) internal spread of the 26.5 N AMOC change (1850-1900 to 2081-2100) across the
    50 members of ``smile_amoc26n_mpi_lr.npz`` and of the 1950-1980 mean F_ovS
    across the 34.5 S members, as error variances.

Conventions: F_ovS = -(1/S0) T dS with S0 = 35 PSU; annual means of monthly values,
as in the r1 pipeline; trends are ordinary least squares with the bias-corrected
effective sample size of ``paper3r1_stats.trend_test``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import trend_test  # noqa: E402

S0 = 35.0
ENSEMBLE = REPO / "data" / "results" / "paper3jc_smile_mpi_lr"
PARTS = ENSEMBLE / "parts"
ORAS5_SECTION = REPO / "data" / "results" / "paper3r1_section_oras5.nc"
ORAS5_TRENDS = REPO / "PAPER_3_v2" / "analysis" / "jc_n15_oras5_early.json"
CMIP6_REF = REPO / "PAPER_3_v2" / "analysis" / "r1_cmip6.json"
AMOC_NPZ = REPO / "data" / "results" / "smile_amoc26n_mpi_lr.npz"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n20_smile.json"

TREND_WINDOW = (1958, 2025)
LATE_WINDOW = (2005, 2024)
ALIGNED_LENGTHS = (68, 33, 26, 20)
SPECIFIED_WINDOWS = {68: (1958, 2025), 33: (1993, 2025), 26: (1992, 2017), 20: (2005, 2024)}
CHANGE_WINDOWS = {"early": (1993, 2014), "late": (2081, 2100)}
ORAS5_REPORTED = {"F_ovS_mSv_per_yr": -1.42, "contrast_mSv_per_yr": -1.53,
                  "transport_mSv_per_yr": 0.11}
EN4_DS_2005_2024_PSU_PER_DECADE = (0.024, 0.038)
ALPHA = 0.05


# ── loading ───────────────────────────────────────────────────────────


def members_available() -> list[str]:
    names = {p.name.split("_")[0] for p in PARTS.glob("*_historical.nc")}
    names &= {p.name.split("_")[0] for p in PARTS.glob("*_ssp585.nc")}
    return sorted(names, key=lambda s: int(s[1:].split("i")[0]))


def load_member(member: str) -> pd.DataFrame:
    """Annual 1950-2100 series of one member (historical then ssp585)."""
    frames = []
    for experiment in ("historical", "ssp585"):
        with xr.open_dataset(PARTS / f"{member}_{experiment}.nc") as ds:
            frames.append(ds.to_dataframe())
    ann = pd.concat(frames)
    ann = ann[~ann.index.duplicated(keep="first")].sort_index()
    ann.index = ann.index.astype(int)
    ann.index.name = "year"
    if list(ann.index) != list(range(1950, 2101)):
        raise ValueError(f"{member}: years are not 1950-2100")
    return ann


# ── statistics helpers ────────────────────────────────────────────────


def fit(years, y, per_decade: bool) -> dict:
    """trend_test in units per year, or per decade when per_decade is set."""
    r = trend_test(np.asarray(years, float), np.asarray(y, float))
    k = 10.0 if per_decade else 1.0
    return {
        "value": k * r["value"],
        "se": k * r["stderr_adj"],
        "ci95": [k * r["ci"][0], k * r["ci"][1]],
        "p": r["p"],
        "p_ols": r["p_ols"],
        "n": r["n"],
        "n_eff": r["n_eff"],
        "phi_corrected": r["phi_corrected"],
    }


def ensemble(values: dict) -> dict | None:
    """Ensemble mean, internal spread (sd of member minus mean) and range."""
    if not values:
        return None
    names = list(values)
    arr = np.array([values[m] for m in names], float)
    n = arr.size
    mean = float(arr.mean())
    dev = arr - mean
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    out = {
        "n": int(n),
        "members": names,
        "mean": mean,
        "sd_internal": sd,
        "dev_min": float(dev.min()),
        "dev_max": float(dev.max()),
        "range": float(arr.max() - arr.min()),
        "n_positive": int((arr > 0).sum()),
        "n_negative": int((arr < 0).sum()),
    }
    if n > 2:
        lo = sd * np.sqrt((n - 1) / stats.chi2.ppf(0.975, n - 1))
        hi = sd * np.sqrt((n - 1) / stats.chi2.ppf(0.025, n - 1))
        out["sd_ci95"] = [float(lo), float(hi)]
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson score interval for a binomial proportion k of n."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(centre - half), float(centre + half)]


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ── (a) trends ─────────────────────────────────────────────────────────


def trend_block(ann: pd.DataFrame, window: tuple[int, int]) -> dict:
    """Trends of F_ovS, T, dS and of the two terms of the F_ovS trend, 1958-2025."""
    sub = ann.loc[window[0]:window[1]]
    years = sub.index.values
    f_mS = sub["F_ov"].values * 1e3
    t = sub["T"].values
    d = sub["dS"].values
    t_bar, d_bar = float(t.mean()), float(d.mean())
    f_bar = float(sub["F_ov"].mean())
    term_t = -(1.0 / S0) * t * d_bar * 1e3
    term_s = -(1.0 / S0) * t_bar * d * 1e3
    out = {
        "window": list(window),
        "means": {"F_ovS_Sv": f_bar, "T_Sv": t_bar, "dS_PSU": d_bar},
        "F_ovS_mSv_per_yr": fit(years, f_mS, False),
        "T_Sv_per_decade": fit(years, t, True),
        "dS_PSU_per_decade": fit(years, d, True),
        "transport_term_mSv_per_yr": fit(years, term_t, False),
        "contrast_term_mSv_per_yr": fit(years, term_s, False),
    }
    out["cross_residual_mSv_per_yr"] = (
        out["F_ovS_mSv_per_yr"]["value"]
        - out["transport_term_mSv_per_yr"]["value"]
        - out["contrast_term_mSv_per_yr"]["value"]
    )
    out["relative_per_decade_pct"] = {
        "F_ovS": 100.0 * 10.0 * (out["F_ovS_mSv_per_yr"]["value"] / 1e3) / abs(f_bar),
        "T": 100.0 * out["T_Sv_per_decade"]["value"] / abs(t_bar),
        "dS": 100.0 * out["dS_PSU_per_decade"]["value"] / abs(d_bar),
    }
    return out


def oras5_block() -> dict:
    """ORAS5 section trends recomputed with the same definitions, for the context."""
    with xr.open_dataset(ORAS5_SECTION) as ds:
        frame = pd.DataFrame(
            {
                "F_ov": ds["F_ov"].values.astype(float),
                "T": ds["T_limb_Sv"].values.astype(float),
                "dS": ds["dS_limb"].values.astype(float),
            },
            index=ds["time"].dt.year.values,
        )
    ann = frame.groupby(level=0).mean()
    ann.index = ann.index.astype(int)
    block = trend_block(ann, TREND_WINDOW)
    late = ann.loc[LATE_WINDOW[0]:LATE_WINDOW[1]]
    block["dS_2005_2024_PSU_per_decade"] = fit(late.index.values, late["dS"].values, True)
    block["reported_in_paper"] = ORAS5_REPORTED
    return block


def part_a(ann_by_member: dict) -> dict:
    blocks = {m: trend_block(a, TREND_WINDOW) for m, a in ann_by_member.items()}
    late = {m: fit(a.loc[LATE_WINDOW[0]:LATE_WINDOW[1]].index.values,
                   a.loc[LATE_WINDOW[0]:LATE_WINDOW[1]]["dS"].values, True)
            for m, a in ann_by_member.items()}
    ens_keys = {
        "F_ovS_mSv_per_yr": {m: b["F_ovS_mSv_per_yr"]["value"] for m, b in blocks.items()},
        "transport_term_mSv_per_yr": {m: b["transport_term_mSv_per_yr"]["value"]
                                      for m, b in blocks.items()},
        "contrast_term_mSv_per_yr": {m: b["contrast_term_mSv_per_yr"]["value"]
                                     for m, b in blocks.items()},
        "cross_residual_mSv_per_yr": {m: b["cross_residual_mSv_per_yr"]
                                      for m, b in blocks.items()},
        "T_Sv_per_decade": {m: b["T_Sv_per_decade"]["value"] for m, b in blocks.items()},
        "dS_PSU_per_decade": {m: b["dS_PSU_per_decade"]["value"] for m, b in blocks.items()},
        "dS_2005_2024_PSU_per_decade": {m: late[m]["value"] for m in late},
        "relative_F_ovS_pct_per_decade": {m: b["relative_per_decade_pct"]["F_ovS"]
                                          for m, b in blocks.items()},
        "relative_dS_pct_per_decade": {m: b["relative_per_decade_pct"]["dS"]
                                       for m, b in blocks.items()},
        "relative_T_pct_per_decade": {m: b["relative_per_decade_pct"]["T"]
                                      for m, b in blocks.items()},
    }
    ensembles = {k: ensemble(v) for k, v in ens_keys.items()}
    ora = oras5_block()
    sd_F = ensembles["F_ovS_mSv_per_yr"]["sd_internal"]
    sd_tr = ensembles["transport_term_mSv_per_yr"]["sd_internal"]
    sd_co = ensembles["contrast_term_mSv_per_yr"]["sd_internal"]
    sd_dS_late = ensembles["dS_2005_2024_PSU_per_decade"]["sd_internal"]
    mean_F = ensembles["F_ovS_mSv_per_yr"]["mean"]
    mean_tr = ensembles["transport_term_mSv_per_yr"]["mean"]
    mean_co = ensembles["contrast_term_mSv_per_yr"]["mean"]
    mean_dS_late = ensembles["dS_2005_2024_PSU_per_decade"]["mean"]
    en4_lo, en4_hi = EN4_DS_2005_2024_PSU_PER_DECADE
    standardised = {
        "note": ("z = value / internal sd (sd of member trends, n = 8); z_dev = (value "
                 "minus ensemble-mean trend) / internal sd. The model F_ovS and dS have "
                 "opposite signs by construction (F_ovS = -(1/S0) T dS with T > 0 in the "
                 "model limb), see means below."),
        "ORAS5_F_ovS_1958_2025": {
            "value": ORAS5_REPORTED["F_ovS_mSv_per_yr"],
            "recomputed": ora["F_ovS_mSv_per_yr"]["value"],
            "z": ORAS5_REPORTED["F_ovS_mSv_per_yr"] / sd_F,
            "z_dev": (ORAS5_REPORTED["F_ovS_mSv_per_yr"] - mean_F) / sd_F,
        },
        "ORAS5_contrast_term_1958_2025": {
            "value": ORAS5_REPORTED["contrast_mSv_per_yr"],
            "recomputed": ora["contrast_term_mSv_per_yr"]["value"],
            "z": ORAS5_REPORTED["contrast_mSv_per_yr"] / sd_co,
            "z_dev": (ORAS5_REPORTED["contrast_mSv_per_yr"] - mean_co) / sd_co,
        },
        "ORAS5_transport_term_1958_2025": {
            "value": ORAS5_REPORTED["transport_mSv_per_yr"],
            "recomputed": ora["transport_term_mSv_per_yr"]["value"],
            "z": ORAS5_REPORTED["transport_mSv_per_yr"] / sd_tr,
            "z_dev": (ORAS5_REPORTED["transport_mSv_per_yr"] - mean_tr) / sd_tr,
        },
        "EN4_dS_contrast_2005_2024": {
            "values_PSU_per_decade": list(EN4_DS_2005_2024_PSU_PER_DECADE),
            "z": [en4_lo / sd_dS_late, en4_hi / sd_dS_late],
            "z_dev": [(en4_lo - mean_dS_late) / sd_dS_late,
                      (en4_hi - mean_dS_late) / sd_dS_late],
        },
    }
    spread = f_ovs_spread(blocks, ensembles["F_ovS_mSv_per_yr"], ann_by_member)
    window_keys = ["F_ovS_mSv_per_yr", "transport_term_mSv_per_yr",
                   "contrast_term_mSv_per_yr", "cross_residual_mSv_per_yr",
                   "T_Sv_per_decade", "dS_PSU_per_decade"]
    rate_keys = ["relative_F_ovS_pct_per_decade", "relative_dS_pct_per_decade",
                 "relative_T_pct_per_decade"]
    def mean_of(key: str) -> float:
        return float(np.mean([b["means"][key] for b in blocks.values()]))

    return {
        "window_1958_2025": {
            "per_member": blocks,
            "ensemble": {k: ensembles[k] for k in window_keys},
        },
        "dS_2005_2024": {
            "per_member": late,
            "ensemble": ensembles["dS_2005_2024_PSU_per_decade"],
        },
        "ensemble_relative_rates": {k: ensembles[k] for k in rate_keys},
        "oras5": ora,
        "standardised": standardised,
        "F_ovS_spread": spread,
        "sign_check": {
            "model_mean_F_ovS_1958_2025_Sv": mean_of("F_ovS_Sv"),
            "model_mean_T_1958_2025_Sv": mean_of("T_Sv"),
            "model_mean_dS_1958_2025_PSU": mean_of("dS_PSU"),
            "oras5_mean_F_ovS_1958_2025_Sv": ora["means"]["F_ovS_Sv"],
            "oras5_mean_T_1958_2025_Sv": ora["means"]["T_Sv"],
            "oras5_mean_dS_1958_2025_PSU": ora["means"]["dS_PSU"],
        },
    }


# ── (a') spread of F_ovS trends against the test, and ORAS5 on that scale ──


def f_ovs_spread(blocks: dict, ens_f: dict, ann_by_member: dict) -> dict:
    """Ratio of the 1958-2025 F_ovS trend spread to the mean test SE, and the ORAS5
    1958-2025 and 1958-2014 trends in internal standard deviations and with their
    intervals widened by that ratio."""
    mean_se = float(np.mean([b["F_ovS_mSv_per_yr"]["se"] for b in blocks.values()]))
    sd = ens_f["sd_internal"]
    sd_lo, sd_hi = ens_f["sd_ci95"]
    ratio = sd / mean_se
    trends_2014 = {
        m: fit(a.loc[1958:2014].index.values, a.loc[1958:2014, "F_ov"].values * 1e3,
               False)["value"]
        for m, a in ann_by_member.items()
    }
    ens_2014 = ensemble(trends_2014)
    oras5 = json.loads(ORAS5_TRENDS.read_text())["windows"]

    def oras5_entry(key: str, mean: float, sd_c: float, sd_u: float) -> dict:
        rec = oras5[key]["F_ovS_mSv_per_yr"]
        value = float(rec["value"])
        lo, hi = rec["ci95"]
        half = 0.5 * (hi - lo)
        return {
            "value": value,
            "ci95": [float(lo), float(hi)],
            "ci95_half_width_scaled_by_ratio": [value - ratio * half, value + ratio * half],
            "z_dev": (value - mean) / sd_c,
            "z_dev_at_sd_upper": (value - mean) / sd_u,
        }

    return {
        "mean_test_se_mSv_per_yr": mean_se,
        "sd_internal_1958_2025_mSv_per_yr": sd,
        "sd_ci95_1958_2025_mSv_per_yr": [sd_lo, sd_hi],
        "ratio_sd_to_mean_test_se": ratio,
        "ratio_range_from_sd_ci95": [sd_lo / mean_se, sd_hi / mean_se],
        "ensemble_1958_2025_mean_mSv_per_yr": ens_f["mean"],
        "ensemble_1958_2014": ens_2014,
        "ORAS5_1958_2025": oras5_entry("1958-2025", ens_f["mean"], sd, sd_hi),
        "ORAS5_1958_2014": oras5_entry("1958-2014", ens_2014["mean"],
                                       ens_2014["sd_internal"], ens_2014["sd_ci95"][1]),
    }


# ── (b) size of the trend test on the departures ───────────────────────


def aligned_windows(length: int) -> list[tuple[int, int]]:
    out = []
    start = 1950
    while start + length - 1 <= 2100:
        out.append((start, start + length - 1))
        start += length
    return out


def rejections(dep: dict, windows: list[tuple[int, int]], var: str) -> dict:
    """Rejection counts of the corrected test and of plain OLS, over members and windows."""
    years_all = np.arange(1950, 2101)
    n_rej = n_ols = n_tot = 0
    p_vals = []
    for m, series in dep.items():
        for (y0, y1) in windows:
            sel = (years_all >= y0) & (years_all <= y1)
            r = trend_test(years_all[sel].astype(float), series[sel])
            n_tot += 1
            n_rej += int(r["p"] < ALPHA)
            n_ols += int(r["p_ols"] < ALPHA)
            p_vals.append(r["p"])
    return {
        "variable": var,
        "n_tests": n_tot,
        "n_reject_corrected": n_rej,
        "rate_corrected": n_rej / n_tot,
        "wilson95_corrected": wilson(n_rej, n_tot),
        "n_reject_ols": n_ols,
        "rate_ols": n_ols / n_tot,
        "median_p_corrected": float(np.median(p_vals)),
    }


def part_b(data: dict) -> dict:
    members = list(data)
    stack = {v: np.vstack([data[m][v] for m in members]) for v in ("F_ov", "T", "dS")}
    scale = {"F_ov": 1e3, "T": 1.0, "dS": 1.0}
    dep_full, dep_loo = {}, {}
    for v in stack:
        mean_all = stack[v].mean(axis=0)
        n = len(members)
        for i, m in enumerate(members):
            dep_full.setdefault(m, {})[v] = (stack[v][i] - mean_all) * scale[v]
            loo_mean = (stack[v].sum(axis=0) - stack[v][i]) / (n - 1)
            dep_loo.setdefault(m, {})[v] = (stack[v][i] - loo_mean) * scale[v]
    out = {"n_members": len(members), "members": members,
           "departure": "member minus ensemble mean of all members (main); leave-one-out "
                        "departure reported as a check",
           "nominal_size": ALPHA, "by_record_length": {}}
    for length in ALIGNED_LENGTHS:
        wins = aligned_windows(length)
        spec = SPECIFIED_WINDOWS[length]
        entry = {"record_length": length, "aligned_windows": wins,
                 "specified_window": list(spec), "variables": {}, "leave_one_out": {},
                 "specified_window_tests": {}}
        for v in ("F_ov", "T", "dS"):
            entry["variables"][v] = rejections(
                {m: dep_full[m][v] for m in members}, wins, v)
            entry["leave_one_out"][v] = rejections(
                {m: dep_loo[m][v] for m in members}, wins, v)
            entry["specified_window_tests"][v] = rejections(
                {m: dep_full[m][v] for m in members}, [spec], v)
        out["by_record_length"][str(length)] = entry
    return out


# ── (c) forced change and its decomposition ────────────────────────────


def decompose(t0, d0, t1, d1, f0, f1) -> dict:
    """Exact split of the change of -(T dS)/S0 (T in Sv, dS in PSU, F in Sv)."""
    dT, dD = t1 - t0, d1 - d0
    transport = -dT * d0 / S0
    contrast = -t0 * dD / S0
    product = -dT * dD / S0
    fhat0, fhat1 = -t0 * d0 / S0, -t1 * d1 / S0
    return {
        "dF_ovS_window_means_Sv": f1 - f0,
        "dF_hat_Sv": fhat1 - fhat0,
        "transport_term_Sv": transport,
        "contrast_term_Sv": contrast,
        "product_term_Sv": product,
        "identity_error_Sv": transport + contrast + product - (fhat1 - fhat0),
        "covariance_residual_Sv": (f1 - f0) - (fhat1 - fhat0),
        "dT_Sv": dT,
        "ddS_PSU": dD,
    }


def window_means(ann: pd.DataFrame, window: tuple[int, int]) -> tuple[float, float, float]:
    sub = ann.loc[window[0]:window[1]]
    return float(sub["T"].mean()), float(sub["dS"].mean()), float(sub["F_ov"].mean())


def part_c(ann_by_member: dict, cmip6: dict) -> dict:
    per_member = {}
    for m, a in ann_by_member.items():
        t0, d0, f0 = window_means(a, CHANGE_WINDOWS["early"])
        t1, d1, f1 = window_means(a, CHANGE_WINDOWS["late"])
        per_member[m] = decompose(t0, d0, t1, d1, f0, f1)
    keys = ["dF_ovS_window_means_Sv", "transport_term_Sv", "contrast_term_Sv",
            "product_term_Sv", "covariance_residual_Sv", "dT_Sv", "ddS_PSU"]
    ensembles = {k: ensemble({m: per_member[m][k] for m in per_member}) for k in keys}
    # CMIP6 single-member forced changes (r1 pipeline, same windows and definitions).
    cmip_rows = {}
    for row in cmip6["per_model"]:
        if "F_1993-2014" not in row or "F_2081-2100" not in row:
            continue
        dec = decompose(row["T_1993-2014"], row["dS_1993-2014"], row["T_2081-2100"],
                        row["dS_2081-2100"], row["F_1993-2014"], row["F_2081-2100"])
        cmip_rows[row["model"]] = {
            "dF_ovS_window_means_Sv": dec["dF_ovS_window_means_Sv"],
            "transport_term_Sv": dec["transport_term_Sv"],
            "contrast_term_Sv": dec["contrast_term_Sv"],
            "product_term_Sv": dec["product_term_Sv"],
            "dT_Sv": dec["dT_Sv"],
            "ddS_PSU": dec["ddS_PSU"],
        }

    def band(key):
        vals = np.array([r[key] for r in cmip_rows.values()], float)
        return {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "p16": float(np.percentile(vals, 16)),
            "p84": float(np.percentile(vals, 84)),
            "sd": float(vals.std(ddof=1)),
        }

    cmip_band = {k: band(k) for k in ["dF_ovS_window_means_Sv", "transport_term_Sv",
                                       "contrast_term_Sv", "dT_Sv", "ddS_PSU"]}
    mpi_r1_cmip = cmip_rows.get("MPI-ESM1-2-LR")
    r1_key = "r1i1p1f1"
    r1_check = None
    if r1_key in per_member and mpi_r1_cmip is not None:
        r1_check = {
            "extract_dF_Sv": per_member[r1_key]["dF_ovS_window_means_Sv"],
            "cmip6_pipeline_dF_Sv": mpi_r1_cmip["dF_ovS_window_means_Sv"],
            "difference_Sv": per_member[r1_key]["dF_ovS_window_means_Sv"]
            - mpi_r1_cmip["dF_ovS_window_means_Sv"],
        }
    return {
        "windows": {k: list(v) for k, v in CHANGE_WINDOWS.items()},
        "per_member": per_member,
        "ensemble": ensembles,
        "cmip6_single_member_changes": cmip_rows,
        "cmip6_noise_band": cmip_band,
        "r1_consistency_with_cmip6_pipeline": r1_check,
    }


# ── (d) error variances ────────────────────────────────────────────────


def part_d(f_1950_1980: dict, cmip6: dict) -> dict:
    npz = np.load(AMOC_NPZ, allow_pickle=True)
    members = [str(m) for m in npz["members"]]
    change, base = {}, {}
    for m in members:
        yrs = npz[f"{m}_years"].astype(float)
        amoc = npz[f"{m}_amoc"].astype(float)
        base[m] = float(amoc[(yrs >= 1850) & (yrs <= 1900)].mean())
        change[m] = float(amoc[(yrs >= 2081) & (yrs <= 2100)].mean() - base[m])
    amoc_ens = ensemble(change)
    amoc_ens["error_variance_single_member"] = amoc_ens["sd_internal"] ** 2
    amoc_ens["variance_of_ensemble_mean"] = amoc_ens["sd_internal"] ** 2 / amoc_ens["n"]
    f_ens = ensemble(f_1950_1980)
    f_ens["error_variance_single_member"] = f_ens["sd_internal"] ** 2
    f_ens["variance_of_ensemble_mean"] = f_ens["sd_internal"] ** 2 / f_ens["n"]
    # Consistency with the CMIP6 26.5 N series for the MPI member used in r1_cmip6.
    mpi_row = next((r for r in cmip6["per_model"] if r["model"] == "MPI-ESM1-2-LR"), None)
    check = None
    if mpi_row is not None and "r1i1p1f1" in base:
        check = {
            "npz_r1_base_1850_1900": base["r1i1p1f1"],
            "r1_cmip6_base_1850_1900": mpi_row["amoc26_1850-1900"],
            "npz_r1_change": change["r1i1p1f1"],
            "r1_cmip6_change": mpi_row["damoc26"],
            "note": ("the two 26.5 N series differ for the same member: the npz is the "
                     "50-member smile file used here; the r1_cmip6 value comes from "
                     "yearly_amoc26n_cmip6.npz (its own definition)"),
        }
    return {
        "amoc26n_change_1850_1900_to_2081_2100_Sv": amoc_ens,
        "amoc26n_base_1850_1900_Sv": {"mean": float(np.mean(list(base.values()))),
                                      "sd": float(np.std(list(base.values()), ddof=1))},
        "F_ovS_1950_1980_Sv_34S": f_ens,
        "cross_check_with_cmip6_amoc26_series": check,
        "per_member_amoc26n_change_Sv": change,
        "per_member_F_ovS_1950_1980_Sv": f_1950_1980,
    }


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    ENSEMBLE.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    members = members_available()
    ann_by_member = {}
    for m in members:
        ann = load_member(m)
        ann_by_member[m] = ann
        ann.to_xarray().to_netcdf(ENSEMBLE / f"{m}.nc")
    cmip6 = json.loads(CMIP6_REF.read_text())
    f_1950_1980 = {m: float(a.loc[1950:1980, "F_ov"].mean()) for m, a in ann_by_member.items()}
    summary = {
        "meta": {
            "model": "MPI-ESM1-2-LR",
            "members": members,
            "n_members": len(members),
            "members_not_extracted": ["r9i1p1f1", "r10i1p1f1"],
            "members_not_extracted_reason": "queue stopped at the 14:50 start cutoff",
            "esgf_members": "skipped (DKRZ OPeNDAP attempts failed)",
            "source": "Pangeo CMIP6 zarr (gn, v20190710); vo, so, thkcello",
            "S0_PSU": S0,
            "trend_test": "paper3r1_stats.trend_test (bias-corrected N_eff)",
        },
        "a_trends": part_a(ann_by_member),
        "b_test_size": part_b({m: {"F_ov": a["F_ov"].values, "T": a["T"].values,
                                   "dS": a["dS"].values} for m, a in ann_by_member.items()}),
        "c_forced_change": part_c(ann_by_member, cmip6),
        "d_error_variances": part_d(f_1950_1980, cmip6),
    }
    OUT.write_text(json.dumps(jsonable(summary), indent=1))
    print(f"wrote {OUT} for {len(members)} members")


if __name__ == "__main__":
    main()
