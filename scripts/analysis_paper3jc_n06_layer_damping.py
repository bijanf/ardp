#!/usr/bin/env python3
"""Layer test and observation-weight damping check for the limb salinity contrast.

Part (a), layer test. The overturning freshwater transport F_ovS and the limb
salinity contrast dS are recomputed over 2005 to 2024 for several cut depths
(1000, 1500, 2000 and 2500 m, and the full column). EN4 is held at the
record-mean velocity of each reanalysis, as in ``analysis_paper3v2_en4_depth_test.py``.
Each reanalysis is evaluated twice: with its own monthly velocity, and with its
record-mean velocity (the same circulation as EN4). At every cut the profile is
truncated and the net transport is removed again inside the retained layer, so
that the limb factorisation stays exact there. The relative change of F_ovS from
the full column to each cut is reported for every family, which shows whether the
change is a property of the cut or of one dataset.

Part (b), damping check. For each EN4 level the annual zonal-mean salinity over
2005 to 2024 is compared with the reanalysis salinity interpolated to the same
depth. Levels are assigned to the northward or southward limb by the sign of the
EN4 velocity frozen to the ORAS5 record mean, and each level carries its mean EN4
salinity observation weight over the window. Within each group of levels the
transport-weighted trend of EN4 salinity is divided by the transport-weighted
trend of the reanalysis. The intervals come from the joint AR(1) sieve bootstrap
of ``paper3r1_stats``, which keeps the cross-correlation between the EN4 and
reanalysis series at each level. If relaxation towards climatology damps the EN4
trend where observations are sparse, the ratio should be smaller in the
low-weight groups than in the high-weight groups.

ECCO-V4r4 is not used: its section record ends in December 2017, so it does not
cover 2005 to 2024.

Writes ``PAPER_3_v2/analysis/jc_n06_layer_damping.json``.
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_en4 import EN4_DIR, RESULTS, S0, frozen_velocity  # noqa: E402
from analysis_paper3v2_en4_depth_test import ARGO_FLOOR, en4_month_full, limbs  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    N_BOOT,
    mean_test,
    sieve_coefficients,
    trend_design,
    trend_test,
)

OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n06_layer_damping.json"
STORED_EN4_DEPTH = REPO / "PAPER_3_v2" / "analysis" / "en4_depth_test.json"

YEARS = np.arange(2005, 2025)
N_YEARS = YEARS.size
CUTS = {"1000": 1000.0, "1500": 1500.0, "2000": 2000.0, "2500": 2500.0, "full": None}
PRODUCTS = {
    "oras5": "paper3r1_section_oras5.nc",
    "glorys12": "paper3r1_section_glorys12.nc",
}
WEIGHT_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9)
LIMB_KEYS = ("T_Sv", "S_north", "S_south", "dS", "F_ov_Sv")
RNG_SEED = 20261104


def _f(x) -> float:
    return float(x)


# ── EN4 along the section, annual means ───────────────────────────────


def en4_annual():
    """Annual zonal-mean EN4 salinity and observation weight per level, 2005 to 2024."""
    files = []
    for f in sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc")):
        yr = int(f.name.split(".")[-2][:4])
        if YEARS[0] <= yr <= YEARS[-1]:
            files.append(f)
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(en4_month_full, f) for f in files]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["time"])
    depth = np.asarray(rows[0]["depth"], float)
    e3t = np.asarray(rows[0]["e3t"], float)
    yrs = pd.DatetimeIndex([r["time"] for r in rows]).year.values
    sbar = np.vstack([r["S_bar"] for r in rows])
    wbar = np.vstack([r["W_bar"] for r in rows])
    s_ann, w_ann, counts = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for y in YEARS:
            m = yrs == y
            counts.append(int(m.sum()))
            s_ann.append(np.nanmean(sbar[m], axis=0))
            w_ann.append(np.nanmean(wbar[m], axis=0))
    return {
        "depth": depth,
        "e3t": e3t,
        "S_ann": np.array(s_ann),
        "W_ann": np.array(w_ann),
        "months_per_year": counts,
        "n_months": len(rows),
    }


# ── reanalysis sections ───────────────────────────────────────────────


def load_reanalysis(fname: str) -> dict:
    with xr.open_dataset(RESULTS / fname) as ds:
        R = {
            "time": pd.DatetimeIndex(ds["time"].values),
            "depth": ds["depth"].values.astype(float),
            "e3t": ds["e3t"].values.astype(float),
            "V_bc": ds["V_bc"].values.astype(float),
            "S_bar": ds["S_bar"].values.astype(float),
            "A_xy": ds["A_xy"].values.astype(float),
            "F_ov": ds["F_ov"].values.astype(float),
        }
    R["V_frozen"] = np.nanmean(R["V_bc"], axis=0)
    yrs = R["time"].year.values
    R["F_ov_stored_annual"] = np.array([R["F_ov"][yrs == y].mean() for y in YEARS])
    return R


def reanalysis_annual_salinity(R: dict, depth_target: np.ndarray) -> np.ndarray:
    """Annual-mean reanalysis salinity at the target depths (NaN where dry or outside)."""
    yrs = R["time"].year.values
    out = np.full((N_YEARS, depth_target.size), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, y in enumerate(YEARS):
            m = yrs == y
            s = np.where(R["A_xy"][m] > 0, R["S_bar"][m], np.nan)
            sm = np.nanmean(s, axis=0)
            ok = np.isfinite(sm)
            out[i] = np.interp(
                depth_target, R["depth"][ok], sm[ok], left=np.nan, right=np.nan
            )
    return out


def reanalysis_limb_series(R: dict, mode: str, zmax) -> dict:
    """Annual means of the monthly limb quantities on the wet levels of each month."""
    yrs = R["time"].year.values
    rec = {k: np.full(N_YEARS, np.nan) for k in LIMB_KEYS}
    for i, y in enumerate(YEARS):
        vals = {k: [] for k in LIMB_KEYS}
        for t in np.where(yrs == y)[0]:
            wet = R["A_xy"][t] > 0
            s = np.where(wet, R["S_bar"][t], np.nan)
            v_raw = R["V_bc"][t] if mode == "own" else R["V_frozen"]
            v = np.where(wet, v_raw, np.nan)
            q = limbs(v, s, R["e3t"], zmax=zmax, depth=R["depth"])
            if q is None:
                continue
            for k in LIMB_KEYS:
                vals[k].append(q[k])
        if len(vals["dS"]) == 12:
            for k in LIMB_KEYS:
                rec[k][i] = float(np.mean(vals[k]))
    return rec


def en4_limb_series(v: np.ndarray, S_ann: np.ndarray, e3t: np.ndarray, depth: np.ndarray, zmax) -> dict:
    rec = {k: np.full(N_YEARS, np.nan) for k in LIMB_KEYS}
    for i in range(N_YEARS):
        q = limbs(v, S_ann[i], e3t, zmax=zmax, depth=depth)
        if q is None:
            continue
        for k in LIMB_KEYS:
            rec[k][i] = q[k]
    return rec


# ── summaries ─────────────────────────────────────────────────────────


def summarise(rec: dict) -> dict:
    """Mean and trend with calibrated intervals for each limb quantity (all 20 years needed)."""
    out = {}
    for k, y in rec.items():
        y = np.asarray(y, float)
        if not np.all(np.isfinite(y)):
            out[k] = None
            continue
        mn = mean_test(y)
        tr = trend_test(YEARS, y)
        tcrit_ols = stats.t.ppf(0.975, N_YEARS - 2)
        out[k] = {
            "mean": _f(mn["value"]),
            "mean_ci": [_f(c) for c in mn["ci"]],
            "trend_per_decade": _f(10.0 * tr["value"]),
            "trend_ci_per_decade": [_f(10.0 * c) for c in tr["ci"]],
            "p": _f(tr["p"]),
            "n_eff": _f(tr["n_eff"]),
            "n_eff_at_floor": bool(tr["n_eff_at_floor"]),
            "trend_ci_per_decade_uncorrected": [
                _f(10.0 * (tr["value"] - tcrit_ols * tr["stderr_ols"])),
                _f(10.0 * (tr["value"] + tcrit_ols * tr["stderr_ols"])),
            ],
            "n": N_YEARS,
        }
    return out


def cut_effect(blocks: dict) -> dict:
    """Relative change of mean F_ovS and trend of dS from the full column to each cut."""
    full = blocks["full"]
    out = {}
    for cut in CUTS:
        b = blocks[cut]
        f_full, f_cut = full["F_ov_Sv"]["mean"], b["F_ov_Sv"]["mean"]
        ds_full, ds_cut = full["dS"]["trend_per_decade"], b["dS"]["trend_per_decade"]
        out[cut] = {
            "F_ov_mean_Sv": f_cut,
            "F_ov_change_vs_full_percent": 100.0 * (f_cut / f_full - 1.0),
            "dS_trend_per_decade": ds_cut,
            "dS_trend_ratio_to_full": ds_cut / ds_full if ds_full != 0 else None,
        }
    return out


# ── part (b): EN4-to-reanalysis trend ratio stratified by observation weight ──


def damping_for(
    S_E: np.ndarray,
    W_mean: np.ndarray,
    S_R: np.ndarray,
    depth: np.ndarray,
    e3t: np.ndarray,
    v_cls: np.ndarray,
    zcut,
    seed: int,
) -> dict:
    lev = (
        np.all(np.isfinite(S_E), axis=0)
        & np.all(np.isfinite(S_R), axis=0)
        & np.isfinite(W_mean)
        & (v_cls != 0)
    )
    if zcut is not None:
        lev &= depth <= zcut
    idx = np.where(lev)[0]
    m = idx.size
    x = trend_design(YEARS)
    ys = np.column_stack([S_E[:, idx], S_R[:, idx]])
    beta_E = np.array([trend_test(YEARS, S_E[:, k])["value"] for k in idx])
    beta_R = np.array([trend_test(YEARS, S_R[:, k])["value"] for k in idx])
    rng = np.random.default_rng(seed)
    draws = sieve_coefficients(x, ys, 1, rng, n_boot=N_BOOT)
    dE_all, dR_all = draws[:, :m], draws[:, m:]
    w_all = np.abs(v_cls[idx] * e3t[idx])
    W = W_mean[idx]
    north = v_cls[idx] > 0
    depth_i = depth[idx]

    def group(sel: np.ndarray) -> dict | None:
        if sel.sum() == 0:
            return None
        wi = w_all[sel] / w_all[sel].sum()
        pE = float(np.sum(wi * beta_E[sel]))
        pR = float(np.sum(wi * beta_R[sel]))
        dE = dE_all[:, sel] @ wi
        dR = dR_all[:, sel] @ wi
        ratio = dE / dR
        ok = np.isfinite(ratio)
        return {
            "n_levels": int(sel.sum()),
            "depth_range_m": [_f(depth_i[sel].min()), _f(depth_i[sel].max())],
            "mean_obs_weight": _f(W[sel].mean()),
            "share_of_abs_transport_in_set": _f(w_all[sel].sum() / w_all.sum()),
            "trend_EN4_per_decade": {
                "point": 10 * pE,
                "ci": [_f(10 * np.percentile(dE, 2.5)), _f(10 * np.percentile(dE, 97.5))],
            },
            "trend_reanalysis_per_decade": {
                "point": 10 * pR,
                "ci": [_f(10 * np.percentile(dR, 2.5)), _f(10 * np.percentile(dR, 97.5))],
            },
            "ratio_EN4_to_reanalysis": {
                "point": pE / pR,
                "ci": [_f(np.percentile(ratio[ok], 2.5)), _f(np.percentile(ratio[ok], 97.5))],
                "prob_ratio_below_1": _f(np.mean(ratio[ok] < 1.0)),
            },
            "_draws_ratio": ratio,
        }

    limb_masks = {"northward": north, "southward": ~north, "both": np.ones(m, bool)}
    bins = {}
    for lname, lmask in limb_masks.items():
        bins[lname] = {}
        for lo, hi in zip(WEIGHT_EDGES[:-1], WEIGHT_EDGES[1:]):
            sel = lmask & (W >= lo) & (W < hi)
            g = group(sel)
            if g is not None:
                bins[lname][f"[{lo:.2f},{min(hi, 1.0):.2f})"] = g

    low = group((W < 0.5))
    high = group((W >= 0.75))
    contrast = None
    if low is not None and high is not None:
        diff = low["_draws_ratio"] - high["_draws_ratio"]
        diff = diff[np.isfinite(diff)]
        contrast = {
            "definition": "ratio for mean observation weight < 0.5 minus ratio for weight >= 0.75 (all limb levels)",
            "point": low["ratio_EN4_to_reanalysis"]["point"] - high["ratio_EN4_to_reanalysis"]["point"],
            "ci": [_f(np.percentile(diff, 2.5)), _f(np.percentile(diff, 97.5))],
            "prob_below_0": _f(np.mean(diff < 0.0)),
        }

    for lname in bins:
        for key in bins[lname]:
            bins[lname][key].pop("_draws_ratio", None)

    levels = []
    for j, k in enumerate(idx):
        levels.append(
            {
                "depth_m": _f(depth[k]),
                "limb": "northward" if v_cls[k] > 0 else "southward",
                "mean_obs_weight": _f(W_mean[k]),
                "transport_weight_Sv": _f(w_all[j] / 1e6),
                "EN4_trend_per_decade": _f(10 * beta_E[j]),
                "reanalysis_trend_per_decade": _f(10 * beta_R[j]),
            }
        )
    corr_wd = float(np.corrcoef(W, depth_i)[0, 1]) if m > 2 else None
    return {
        "n_levels": int(m),
        "corr_obs_weight_depth": corr_wd,
        "bins": bins,
        "contrast_low_minus_high_weight": contrast,
        "levels": levels,
    }


def main() -> None:
    en4 = en4_annual()
    depth, e3t = en4["depth"], en4["e3t"]
    S_E, W_E = en4["S_ann"], en4["W_ann"]
    W_mean = np.nanmean(W_E, axis=0)

    out = {
        "purpose": (
            "Layer test of the EN4 and reanalysis limb contrast over 2005 to 2024, "
            "with the F_ovS change at each depth cut, and a damping check that stratifies "
            "the EN4-to-reanalysis salinity trend ratio by EN4 observation weight."
        ),
        "window_years": [int(YEARS[0]), int(YEARS[-1])],
        "argo_floor_m": ARGO_FLOOR,
        "reference_salinity_PSU": S0,
        "cuts_m": {k: v for k, v in CUTS.items()},
        "en4_months_loaded": en4["n_months"],
        "en4_months_per_year": en4["months_per_year"],
        "units": {
            "trends": "PSU per decade (salinities), Sv per decade (transports) where stated",
            "F_ov": "Sv",
            "intervals": "95 per cent; trends and means from the calibrated test of paper3r1_stats",
        },
        "layer_test": {"en4_frozen": {}, "reanalysis": {}, "cut_effect": {}},
        "damping": {},
    }

    # ── (a) layer test ───────────────────────────────────────────────
    v_frozen_oras5_en4 = None
    for label, fname in PRODUCTS.items():
        v_fr = frozen_velocity(fname, depth, e3t)
        if label == "oras5":
            v_frozen_oras5_en4 = v_fr
        blocks = {}
        for cut, zmax in CUTS.items():
            blocks[cut] = summarise(en4_limb_series(v_fr, S_E, e3t, depth, zmax))
        out["layer_test"]["en4_frozen"][label] = blocks
        out["layer_test"]["cut_effect"][f"en4_frozen_{label}"] = cut_effect(blocks)

        R = load_reanalysis(fname)
        for mode in ("own", "frozen"):
            fam = f"reanalysis_{mode}_{label}"
            blocks = {}
            for cut, zmax in CUTS.items():
                blocks[cut] = summarise(reanalysis_limb_series(R, mode, zmax))
            out["layer_test"]["reanalysis"][fam] = blocks
            out["layer_test"]["cut_effect"][fam] = cut_effect(blocks)
        own_full = reanalysis_limb_series(R, "own", None)["F_ov_Sv"]
        out.setdefault("checks", {})[f"{label}_full_column_F_ov_vs_stored_max_abs_diff_Sv"] = _f(
            np.max(np.abs(own_full - R["F_ov_stored_annual"]))
        )
        out["checks"][f"{label}_stored_F_ov_mean_2005_2024_Sv"] = _f(R["F_ov_stored_annual"].mean())

    # Reconcile with the stored OLS (Santer) values in en4_depth_test.json.
    stored = json.loads(STORED_EN4_DEPTH.read_text())
    out["layer_test"]["stored_en4_depth_test_upper_2000m_dS_per_decade_santer"] = {
        "oras5": stored["oras5"]["upper_2000m"]["dS"]["full"]["trend_per_decade"],
        "glorys12": stored["glorys12"]["upper_2000m"]["dS"]["full"]["trend_per_decade"],
    }
    out["layer_test"]["stored_en4_depth_test_F_ov_mean_upper_2000m_Sv"] = {
        "oras5": stored["oras5"]["upper_2000m"]["F_ov_Sv"]["mean"],
        "glorys12": stored["glorys12"]["upper_2000m"]["F_ov_Sv"]["mean"],
    }

    # ── (b) damping check ────────────────────────────────────────────
    v_cls = np.asarray(v_frozen_oras5_en4, float)
    for i, (label, fname) in enumerate(PRODUCTS.items()):
        R = load_reanalysis(fname)
        S_R = reanalysis_annual_salinity(R, depth)
        res = {}
        for zname, zcut in (("all_levels", None), ("upper_2000m", ARGO_FLOOR)):
            res[zname] = damping_for(
                S_E, W_mean, S_R, depth, e3t, v_cls, zcut, seed=RNG_SEED + 10 * i + (0 if zcut is None else 1)
            )
        out["damping"][label] = res

    out["damping"]["classification"] = (
        "Limb sign from the EN4 velocity frozen to the ORAS5 record mean, applied to all products; "
        "transport weights |V e3| from the same velocity."
    )
    upper = depth <= ARGO_FLOOR
    out["damping"]["weight_range_upper_2000m"] = {
        "min": _f(np.nanmin(W_mean[upper])),
        "max": _f(np.nanmax(W_mean[upper])),
        "n_levels": int(np.sum(upper & np.isfinite(W_mean))),
    }
    out["damping"]["weight_range_below_2000m"] = {
        "min": _f(np.nanmin(W_mean[~upper & np.isfinite(W_mean)])),
        "max": _f(np.nanmax(W_mean[~upper & np.isfinite(W_mean)])),
        "n_levels": int(np.sum(~upper & np.isfinite(W_mean))),
    }
    out["damping"]["upper_2000m_annual_mean_weight"] = {
        str(int(y)): _f(np.nanmean(W_E[i][upper])) for i, y in enumerate(YEARS)
    }
    out["damping"]["en4_mean_weight_by_depth_m"] = {
        str(_f(d)): _f(w) for d, w in zip(depth, W_mean) if np.isfinite(w)
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
