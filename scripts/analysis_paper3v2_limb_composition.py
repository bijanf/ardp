#!/usr/bin/env python3
"""Is the limb salinification a water-mass change, or a repartition of transport?

The limb factorisation defines the northward limb by the sign of the
barotropic-corrected velocity. At 34.5 S that sign is positive in more than one
depth range: the upper cell, and, in some products, a northward abyssal limb of
Antarctic Bottom Water beneath the southward North Atlantic Deep Water. The
transport-weighted limb salinity S_up is then an average over sub-limbs of very
different salinity, and moving transport between them changes S_up with no water
mass changing salinity at all. A referee is right to insist this be excluded
before the salinification is called a water-mass signal.

This script does three things.

1. **Counts the structure.** Number of sign changes of V_bc(z) per year, the
   depths of the zero crossings as time series, and the transport and
   transport-weighted salinity of every northward sub-limb.

2. **Decomposes the change in S_up exactly.** Writing S_up = sum_i w_i S_i with
   weights w_i = T_i / T summing to one,

       dS_up = sum_i ( w_i dS_i )      water-mass term
             + sum_i ( S_i dw_i )      partition term
             + cross term,

   evaluated continuously by holding one set at its record mean, exactly as the
   main attribution does. If the partition term is comparable to the water-mass
   term the salinification claim is not supported.

3. **Repeats the whole attribution on the upper cell alone.** The section is
   truncated at the deep zero crossing, so the abyssal cell is excluded
   entirely, the barotropic correction is re-applied within the truncated layer
   so that the limb factorisation is again exact, and F_ov, T and dS are
   recomputed. This is the quantity the title actually claims.

Writes ``PAPER_3_v2/analysis/limb_composition.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"

S0 = 35.0
RNG = np.random.default_rng(20260805)
N_BOOT = 10_000
BLOCK = 5

SECTIONS = {
    "oras5": ("ORAS5", "paper3v2_section_oras5.nc"),
    "glorys12": ("GLORYS12V1", "paper3v2_section_glorys12.nc"),
    "ecco": ("ECCO-V4r4", "paper3v2_section_ecco.nc"),
}


def block_idx(n: int, block: int) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n, size=nb)
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def runs_of_sign(v: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs of constant sign: list of (start, stop_exclusive, sign)."""
    sign = np.sign(v)
    out = []
    i = 0
    n = len(sign)
    while i < n:
        j = i
        while j + 1 < n and sign[j + 1] == sign[i]:
            j += 1
        if sign[i] != 0:
            out.append((i, j + 1, int(sign[i])))
        i = j + 1
    return out


def sublimbs(vbc: np.ndarray, sbar: np.ndarray, e3t: np.ndarray, depth: np.ndarray):
    """Transport and transport-weighted salinity of every northward sub-limb."""
    trans = vbc * e3t
    out = []
    for a, b, s in runs_of_sign(trans):
        if s <= 0:
            continue
        t = float(trans[a:b].sum())
        if t <= 0:
            continue
        sal = float((trans[a:b] * sbar[a:b]).sum() / t)
        out.append(
            {
                "top_m": float(depth[a]),
                "base_m": float(depth[b - 1]),
                "T_Sv": t / 1e6,
                "S": sal,
            }
        )
    return out


def truncated_factorisation(
    vbc_raw: np.ndarray,
    v_int: np.ndarray,
    a_xy: np.ndarray,
    sbar: np.ndarray,
    e3t: np.ndarray,
    kmax: int,
):
    """Re-do the limb factorisation on the layer above index kmax only.

    The barotropic correction is re-applied within the truncated layer so that
    the depth integral of the corrected velocity vanishes there and the limb
    split is again exact.
    """
    v = v_int[:kmax]
    a = a_xy[:kmax]
    s = sbar[:kmax]
    e = e3t[:kmax]
    v_net = float((v * e).sum())
    a_tot = float((a * e).sum())
    vbar = v_net / a_tot if a_tot > 0 else 0.0
    vbc = v - vbar * a
    trans = vbc * e
    north = trans > 0
    t_n = float(trans[north].sum())
    if t_n <= 0:
        return None
    s_n = float((trans[north] * s[north]).sum() / t_n)
    s_s = float((-trans[~north] * s[~north]).sum() / float(-trans[~north].sum()))
    f_ov = -(1.0 / S0) * float((vbc * (s - S0) * e).sum()) / 1e6
    return {
        "F_ov": f_ov,
        "T_Sv": t_n / 1e6,
        "S_north": s_n,
        "S_south": s_s,
        "dS": s_n - s_s,
    }


def continuous_split(years, comp_a, comp_b, total):
    """Trend of `total` split by freezing each of two contributions at its mean."""
    x = years.astype(float)
    fits = {
        "total": ols_santer(x, total),
        "A": ols_santer(x, comp_a),
        "B": ols_santer(x, comp_b),
    }
    parts = {}
    for k, series in (("total", total), ("A", comp_a), ("B", comp_b)):
        coef = np.polyfit(x, series, 1)
        parts[k] = (np.polyval(coef, x), series - np.polyval(coef, x))
    boot = {k: np.empty(N_BOOT) for k in parts}
    for i in range(N_BOOT):
        idx = block_idx(len(x), BLOCK)
        for k, (fit, res) in parts.items():
            boot[k][i] = np.polyfit(x, fit + res[idx], 1)[0]
    out = {}
    for k in ("total", "A", "B"):
        out[f"trend_{k}"] = fits[k]["slope"] * 10  # per decade
        out[f"p_{k}"] = fits[k]["p_santer"]
        out[f"ci_{k}"] = [
            float(np.percentile(boot[k], 2.5) * 10),
            float(np.percentile(boot[k], 97.5) * 10),
        ]
    out["residual_per_decade"] = (
        fits["total"]["slope"] - fits["A"]["slope"] - fits["B"]["slope"]
    ) * 10
    ratio = np.abs(boot["B"]) / np.abs(boot["total"])
    out["partition_share_point"] = float(
        abs(fits["B"]["slope"] / fits["total"]["slope"])
    )
    out["partition_share_ci"] = [
        float(np.percentile(ratio, 2.5)),
        float(np.percentile(ratio, 97.5)),
    ]
    return out


def run(key: str) -> dict:
    label, fname = SECTIONS[key]
    ds = xr.open_dataset(RESULTS / fname)
    depth = ds["depth"].values
    e3t = ds["e3t"].values if "e3t" in ds else np.diff(depth, prepend=0.0)
    a = ds.groupby("time.year").mean()
    years = a["year"].values.astype(int)
    vbc = a["V_bc"].values
    v_int = a["V_int"].values
    a_xy = a["A_xy"].values
    sbar = a["S_bar"].values

    n_years = len(years)
    n_runs = np.zeros(n_years, int)
    n_north = np.zeros(n_years, int)
    t_up = np.zeros(n_years)
    s_up = np.zeros(n_years)
    t_deep = np.zeros(n_years)
    s_deep = np.zeros(n_years)
    deep_base = np.zeros(n_years)
    first_cross = np.zeros(n_years)

    for i in range(n_years):
        subs = sublimbs(vbc[i], sbar[i], e3t, depth)
        n_north[i] = len(subs)
        n_runs[i] = len(runs_of_sign(vbc[i] * e3t))
        # The upper cell is the shallowest northward sub-limb; everything else
        # northward is deep (abyssal) water.
        subs = sorted(subs, key=lambda r: r["top_m"])
        t_up[i] = subs[0]["T_Sv"]
        s_up[i] = subs[0]["S"]
        first_cross[i] = subs[0]["base_m"]
        rest = subs[1:]
        tt = sum(r["T_Sv"] for r in rest)
        t_deep[i] = tt
        s_deep[i] = sum(r["T_Sv"] * r["S"] for r in rest) / tt if tt > 0 else np.nan
        deep_base[i] = rest[0]["top_m"] if rest else np.nan

    t_tot = t_up + t_deep
    w_up = t_up / t_tot
    w_deep = t_deep / t_tot
    s_north = w_up * s_up + np.where(t_deep > 0, w_deep * np.nan_to_num(s_deep), 0.0)

    # Water-mass term freezes the weights; partition term freezes the salinities.
    wm = w_up.mean() * s_up + w_deep.mean() * np.nan_to_num(
        s_deep, nan=float(np.nanmean(s_deep)) if np.isfinite(s_deep).any() else 0.0
    )
    sd_fill = float(np.nanmean(s_deep)) if np.isfinite(s_deep).any() else 0.0
    part = w_up * s_up.mean() + w_deep * sd_fill

    out: dict = {
        "label": label,
        "record": [int(years[0]), int(years[-1])],
        "n_sign_runs_min": int(n_runs.min()),
        "n_sign_runs_max": int(n_runs.max()),
        "n_sign_runs_modal": int(np.bincount(n_runs).argmax()),
        "n_northward_sublimbs_min": int(n_north.min()),
        "n_northward_sublimbs_max": int(n_north.max()),
        "n_northward_sublimbs_modal": int(np.bincount(n_north).argmax()),
        "years_with_deep_northward_limb": int((t_deep > 0).sum()),
        "upper_limb": {
            "mean_T_Sv": float(t_up.mean()),
            "mean_S": float(s_up.mean()),
            "mean_weight": float(w_up.mean()),
            "base_depth_mean_m": float(first_cross.mean()),
            "trend_T_Sv_per_decade": ols_santer(years.astype(float), t_up)["slope"]
            * 10,
            "p_T": ols_santer(years.astype(float), t_up)["p_santer"],
            "trend_S_per_decade": ols_santer(years.astype(float), s_up)["slope"] * 10,
            "p_S": ols_santer(years.astype(float), s_up)["p_santer"],
        },
        "deep_northward_limb": {
            "mean_T_Sv": float(t_deep.mean()),
            "mean_S": float(np.nanmean(s_deep)) if np.isfinite(s_deep).any() else None,
            "mean_weight": float(w_deep.mean()),
            "top_depth_mean_m": float(np.nanmean(deep_base))
            if np.isfinite(deep_base).any()
            else None,
            "trend_T_Sv_per_decade": ols_santer(years.astype(float), t_deep)["slope"]
            * 10,
            "p_T": ols_santer(years.astype(float), t_deep)["p_santer"],
        },
        "salinity_gap_upper_minus_deep": (
            float(s_up.mean() - np.nanmean(s_deep))
            if np.isfinite(s_deep).any()
            else None
        ),
    }

    if (t_deep > 0).sum() > 0.5 * n_years:
        out["composition"] = continuous_split(years, wm, part, s_north)

    # Attribution on the upper cell alone, abyssal cell excluded.
    trunc = []
    for i in range(n_years):
        subs = sorted(sublimbs(vbc[i], sbar[i], e3t, depth), key=lambda r: r["top_m"])
        rest = subs[1:]
        kmax = int(np.searchsorted(depth, rest[0]["top_m"])) if rest else len(depth)
        r = truncated_factorisation(vbc[i], v_int[i], a_xy[i], sbar[i], e3t, kmax)
        trunc.append(r)
    if all(r is not None for r in trunc):
        f = np.array([r["F_ov"] for r in trunc])
        tt = np.array([r["T_Sv"] for r in trunc])
        dd = np.array([r["dS"] for r in trunc])
        sn = np.array([r["S_north"] for r in trunc])
        x = years.astype(float)
        f_t = -(1 / S0) * tt * dd.mean()
        f_d = -(1 / S0) * tt.mean() * dd
        fit_tot = ols_santer(x, f)
        fit_t = ols_santer(x, f_t)
        fit_d = ols_santer(x, f_d)
        out["upper_cell_only"] = {
            "mean_F_ov_Sv": float(f.mean()),
            "mean_T_Sv": float(tt.mean()),
            "mean_dS_PSU": float(dd.mean()),
            "trend_F_ov_mSv_per_yr": fit_tot["slope"] * 1e3,
            "p_F_ov": fit_tot["p_santer"],
            "trend_S_north_per_decade": ols_santer(x, sn)["slope"] * 10,
            "p_S_north": ols_santer(x, sn)["p_santer"],
            "transport_term_mSv_per_yr": fit_t["slope"] * 1e3,
            "salinity_term_mSv_per_yr": fit_d["slope"] * 1e3,
            "share_T": float(abs(fit_t["slope"] / fit_tot["slope"]))
            if fit_tot["slope"]
            else None,
        }
    return out


def report(res: dict) -> None:
    for r in res.values():
        print(
            f"\n{'=' * 74}\n{r['label']} {r['record'][0]}-{r['record'][1]}\n{'=' * 74}"
        )
        print(
            f"  sign runs in V_bc: {r['n_sign_runs_min']} to "
            f"{r['n_sign_runs_max']} (modal {r['n_sign_runs_modal']}); "
            f"northward sub-limbs {r['n_northward_sublimbs_min']} to "
            f"{r['n_northward_sublimbs_max']} (modal "
            f"{r['n_northward_sublimbs_modal']})"
        )
        u, d = r["upper_limb"], r["deep_northward_limb"]
        print(
            f"  upper northward limb: T {u['mean_T_Sv']:.2f} Sv "
            f"(weight {u['mean_weight']:.3f}), S {u['mean_S']:.4f}, base "
            f"{u['base_depth_mean_m']:.0f} m"
        )
        print(
            f"     trend T {u['trend_T_Sv_per_decade']:+.3f} Sv/dec "
            f"(p={u['p_T']:.3f}); "
            f"trend S {u['trend_S_per_decade']:+.4f} /dec (p={u['p_S']:.3f})"
        )
        print(
            f"  deep northward limb: T {d['mean_T_Sv']:.2f} Sv "
            f"(weight {d['mean_weight']:.3f}), S "
            f"{d['mean_S'] and round(d['mean_S'], 4)}, present in "
            f"{r['years_with_deep_northward_limb']} years"
        )
        gap = r["salinity_gap_upper_minus_deep"]
        gap = "n/a" if gap is None else f"{gap:.4f}"
        print(
            f"     trend T {d['trend_T_Sv_per_decade']:+.3f} Sv/dec "
            f"(p={d['p_T']:.3f}); "
            f"upper minus deep salinity gap {gap}"
        )
        c = r.get("composition")
        if c:
            print(
                f"  dS_north/dt {c['trend_total']:+.4f}/dec: water-mass "
                f"{c['trend_A']:+.4f} [{c['ci_A'][0]:+.4f},{c['ci_A'][1]:+.4f}], "
                f"partition {c['trend_B']:+.4f} "
                f"[{c['ci_B'][0]:+.4f},{c['ci_B'][1]:+.4f}]"
            )
            print(
                f"     partition share {c['partition_share_point']:.3f} "
                f"[{c['partition_share_ci'][0]:.3f},"
                f"{c['partition_share_ci'][1]:.3f}], residual "
                f"{c['residual_per_decade']:+.5f}"
            )
        uc = r.get("upper_cell_only")
        if uc:
            print(
                f"  upper cell only: F_ov {uc['mean_F_ov_Sv']:+.4f} Sv, T "
                f"{uc['mean_T_Sv']:.2f} Sv, dS {uc['mean_dS_PSU']:+.4f} PSU"
            )
            print(
                f"     trend F_ov {uc['trend_F_ov_mSv_per_yr']:+.3f} mSv/yr "
                f"(p={uc['p_F_ov']:.4f}); S_north "
                f"{uc['trend_S_north_per_decade']:+.4f}/dec "
                f"(p={uc['p_S_north']:.4f}); share_T "
                f"{uc['share_T'] and round(uc['share_T'], 3)}"
            )


def main() -> None:
    res = {k: run(k) for k in SECTIONS if (RESULTS / SECTIONS[k][1]).exists()}
    report(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "limb_composition.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
