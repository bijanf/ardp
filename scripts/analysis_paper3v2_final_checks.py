#!/usr/bin/env python3
"""The remaining referee checks, on machinery already built.

1. **Frozen-field test at full vertical resolution.** The two-sub-limb
   composition excludes a coarse upper-to-abyssal redistribution but not a
   redistribution of transport *within* the upper limb. Recomputing the limb
   salinities with the salinity profile frozen at its record mean, and again
   with the velocity profile frozen, separates the two completely at the
   resolution of the archive.

2. **Step fits for GLORYS12V1**, at the Argo onset year the paper itself
   nominates, so that the inhomogeneity test applied to ORAS5 is applied to both
   sequential-assimilation products rather than to one.

3. **Limb composition on a monthly basis**, so that the sub-limb transports sum
   to the same T used everywhere else and the two limb-salinity trends
   reconcile. The previous version evaluated the composition on annual-mean
   profiles, which is a different and slightly smaller T.

4. **An alternative removal of the net section transport**, to show the limb
   factorisation does not depend on how that transport is taken out.

5. **The 1993--2025 window shared by the two eddy-permitting products**, which
   is longer than the three-product window and is where both limb-salinity
   trends are resolvable.

Writes ``PAPER_3_v2/analysis/final_checks.json``.
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
from analysis_paper3v2_limb_composition import sublimbs  # noqa: E402
from analysis_paper3v2_stepfit import step_fit  # noqa: E402
from scipy import stats  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
S0 = 35.0

SECTIONS = {
    "oras5": ("ORAS5", "paper3v2_section_oras5.nc"),
    "glorys12": ("GLORYS12V1", "paper3v2_section_glorys12.nc"),
    "ecco": ("ECCO-V4r4", "paper3v2_section_ecco.nc"),
}


def limbs(v: np.ndarray, s: np.ndarray, e3t: np.ndarray) -> tuple[float, float, float]:
    trans = v * e3t
    north = trans > 0
    t_n = float(trans[north].sum())
    t_s = float(-trans[~north].sum())
    if t_n <= 0 or t_s <= 0:
        return np.nan, np.nan, np.nan
    s_n = float((trans[north] * s[north]).sum() / t_n)
    s_s = float((-trans[~north] * s[~north]).sum() / t_s)
    return t_n / 1e6, s_n, s_s


def main() -> None:
    out: dict = {}
    for key, (label, fname) in SECTIONS.items():
        ds = xr.open_dataset(RESULTS / fname)
        depth = ds["depth"].values
        e3t = ds["e3t"].values if "e3t" in ds else np.diff(depth, prepend=0.0)
        vbc_m = ds["V_bc"].values
        s_m = ds["S_bar"].values
        v_int_m = ds["V_int"].values
        yrs_m = ds["time"].dt.year.values
        years = np.unique(yrs_m)
        x = years.astype(float)
        rec: dict = {"label": label, "record": [int(years[0]), int(years[-1])]}

        # ── 1. frozen-field test, full vertical resolution ────────────
        v_bar = vbc_m.mean(axis=0)
        s_bar = s_m.mean(axis=0)
        both, salonly, velonly = [], [], []
        for i in range(len(yrs_m)):
            both.append(limbs(vbc_m[i], s_m[i], e3t))
            salonly.append(limbs(v_bar, s_m[i], e3t))  # only salinity varies
            velonly.append(limbs(vbc_m[i], s_bar, e3t))  # only velocity varies
        frozen = {}
        for name, arr in (
            ("full", both),
            ("salinity_only", salonly),
            ("velocity_only", velonly),
        ):
            ds_series = np.array([r[1] - r[2] for r in arr])
            sn = np.array([r[1] for r in arr])
            ann_ds = np.array([np.nanmean(ds_series[yrs_m == y]) for y in years])
            ann_sn = np.array([np.nanmean(sn[yrs_m == y]) for y in years])
            fd, fs = ols_santer(x, ann_ds), ols_santer(x, ann_sn)
            # A null here is only meaningful next to what the window could
            # have resolved; several of these have N_eff at the floor of 3.
            t_crit = stats.t.ppf(0.975, df=max(fd["n_eff"] - 2, 1))
            frozen[name] = {
                "dS_trend_per_decade": fd["slope"] * 10,
                "dS_p": fd["p_santer"],
                "dS_n_eff": fd["n_eff"],
                "dS_min_detectable": float(t_crit * fd["stderr_santer"] * 10),
                "dS_lag1": fd["lag1_autocorr"],
                "S_north_trend_per_decade": fs["slope"] * 10,
                "S_north_p": fs["p_santer"],
                "S_north_n_eff": fs["n_eff"],
                "n_eff": fd["n_eff"],
            }
        tot = frozen["full"]["dS_trend_per_decade"]
        frozen["salinity_share_of_dS"] = (
            abs(frozen["salinity_only"]["dS_trend_per_decade"] / tot) if tot else None
        )
        frozen["velocity_share_of_dS"] = (
            abs(frozen["velocity_only"]["dS_trend_per_decade"] / tot) if tot else None
        )
        frozen["share_residual"] = (
            1.0
            - (frozen["salinity_share_of_dS"] or 0)
            - (frozen["velocity_share_of_dS"] or 0)
            if tot
            else None
        )
        rec["frozen_field"] = frozen

        # ── 3. limb composition on a monthly basis ────────────────────
        t_up, s_up, t_dp, s_dp = [], [], [], []
        for i in range(len(yrs_m)):
            subs = sorted(
                sublimbs(vbc_m[i], s_m[i], e3t, depth), key=lambda r: r["top_m"]
            )
            if not subs:
                t_up.append(np.nan)
                s_up.append(np.nan)
                t_dp.append(np.nan)
                s_dp.append(np.nan)
                continue
            t_up.append(subs[0]["T_Sv"])
            s_up.append(subs[0]["S"])
            rest = subs[1:]
            tt = sum(r["T_Sv"] for r in rest)
            t_dp.append(tt)
            s_dp.append(
                sum(r["T_Sv"] * r["S"] for r in rest) / tt if tt > 0 else np.nan
            )
        ann = {}
        for nm, arr in (("t_up", t_up), ("s_up", s_up), ("t_dp", t_dp), ("s_dp", s_dp)):
            v = np.array(arr, dtype=float)
            ann[nm] = np.array([np.nanmean(v[yrs_m == y]) for y in years])
        t_tot = ann["t_up"] + ann["t_dp"]
        w_up, w_dp = ann["t_up"] / t_tot, ann["t_dp"] / t_tot
        s_north = w_up * ann["s_up"] + w_dp * np.nan_to_num(ann["s_dp"])
        wm = w_up.mean() * ann["s_up"] + w_dp.mean() * np.nan_to_num(ann["s_dp"])
        part = w_up * np.nanmean(ann["s_up"]) + w_dp * np.nanmean(ann["s_dp"])
        ft, fw, fp = ols_santer(x, s_north), ols_santer(x, wm), ols_santer(x, part)
        with xr.open_dataset(RESULTS / fname) as d2:
            t_ref = float(d2["T_limb_Sv"].values.mean())
        rec["composition_monthly"] = {
            "sublimb_T_sum_Sv": float(t_tot.mean()),
            "T_limb_Sv": t_ref,
            "closure_gap_Sv": float(t_tot.mean() - t_ref),
            "S_north_trend_per_decade": ft["slope"] * 10,
            "water_mass_trend_per_decade": fw["slope"] * 10,
            "partition_trend_per_decade": fp["slope"] * 10,
            "partition_share": abs(fp["slope"] / ft["slope"]) if ft["slope"] else None,
            "residual_per_decade": (ft["slope"] - fw["slope"] - fp["slope"]) * 10,
            "upper_T_trend_per_decade": ols_santer(x, ann["t_up"])["slope"] * 10,
            "upper_T_p": ols_santer(x, ann["t_up"])["p_santer"],
            "abyssal_T_trend_per_decade": ols_santer(x, ann["t_dp"])["slope"] * 10,
            "abyssal_T_p": ols_santer(x, ann["t_dp"])["p_santer"],
            "upper_S_trend_per_decade": ols_santer(x, ann["s_up"])["slope"] * 10,
            "upper_S_p": ols_santer(x, ann["s_up"])["p_santer"],
            "mean_upper_T_Sv": float(ann["t_up"].mean()),
            "mean_abyssal_T_Sv": float(ann["t_dp"].mean()),
            "abyssal_frac_of_T": float(ann["t_dp"].mean() / t_ref),
        }

        # ── 4. alternative removal of the net section transport ───────
        # Instead of a uniform velocity offset spread over the whole area,
        # remove the net transport in proportion to |V| so it is taken out
        # where the flow actually is.
        alt = []
        for i in range(len(yrs_m)):
            v_net = float((v_int_m[i] * e3t).sum())
            wgt = np.abs(v_int_m[i]) * e3t
            wsum = wgt.sum()
            v_alt = (
                v_int_m[i] - (v_net / wsum) * np.abs(v_int_m[i])
                if wsum > 0
                else v_int_m[i]
            )
            alt.append(limbs(v_alt, s_m[i], e3t))
        ann_t = np.array(
            [np.nanmean(np.array([r[0] for r in alt])[yrs_m == y]) for y in years]
        )
        ann_d = np.array(
            [
                np.nanmean(np.array([r[1] - r[2] for r in alt])[yrs_m == y])
                for y in years
            ]
        )
        f_alt = -(1 / S0) * ann_t * ann_d
        ft_alt, fd_alt = ols_santer(x, ann_t), ols_santer(x, ann_d)
        f_fit = ols_santer(x, f_alt)
        term_t = -(1 / S0) * ols_santer(x, ann_t)["slope"] * ann_d.mean()
        rec["alternative_removal"] = {
            "scheme": "net transport removed in proportion to |V| rather than area",
            "mean_T_Sv": float(ann_t.mean()),
            "mean_dS_PSU": float(ann_d.mean()),
            "mean_F_ov_Sv": float(f_alt.mean()),
            "T_trend_per_decade": ft_alt["slope"] * 10,
            "dS_trend_per_decade": fd_alt["slope"] * 10,
            "F_ov_trend_mSv_per_yr": f_fit["slope"] * 1e3,
            "transport_share": abs(term_t / f_fit["slope"]) if f_fit["slope"] else None,
        }

        # ── 5. the two-product window, and 2. step fits ───────────────
        with xr.open_dataset(RESULTS / fname) as d2:
            aa = d2.groupby("time.year").mean()
        yy = aa["year"].values.astype(int)
        m = (yy >= 1993) & (yy <= 2025)
        if m.sum() >= 15:
            fs = ols_santer(yy[m].astype(float), aa["S_north"].values[m])
            fdd = ols_santer(yy[m].astype(float), aa["dS_limb"].values[m])
            ff = ols_santer(yy[m].astype(float), aa["F_ov"].values[m])
            ftt = ols_santer(yy[m].astype(float), aa["T_limb_Sv"].values[m])
            share = abs(
                (-(1 / S0) * ftt["slope"] * aa["dS_limb"].values[m].mean())
                / ff["slope"]
            )
            rec["window_1993_2025"] = {
                "S_north_trend_per_decade": fs["slope"] * 10,
                "S_north_p": fs["p_santer"],
                "S_north_n_eff": fs["n_eff"],
                "dS_trend_per_decade": fdd["slope"] * 10,
                "dS_p": fdd["p_santer"],
                "F_ov_trend_mSv_per_yr": ff["slope"] * 1e3,
                "F_ov_p": ff["p_santer"],
                "transport_share": share,
            }
        if key == "glorys12":
            rec["step_fits_2004"] = {}
            for vname, series in (
                ("F_ov", aa["F_ov"].values),
                ("dS_limb", aa["dS_limb"].values),
                ("S_north", aa["S_north"].values),
            ):
                r = step_fit(yy, series, 2004)
                if r is not None:
                    rec["step_fits_2004"][vname] = r
        out[key] = rec
        ds.close()

    for r in out.values():
        print(f"\n{'=' * 72}\n{r['label']}\n{'=' * 72}")
        f = r["frozen_field"]
        print("  frozen-field test on dS (full vertical resolution):")
        for nm in ("full", "salinity_only", "velocity_only"):
            d = f[nm]
            print(
                f"     {nm:14s} dS {d['dS_trend_per_decade']:+.4f}/dec "
                f"(p={d['dS_p']:.4f})"
                f"   S_north {d['S_north_trend_per_decade']:+.4f}/dec "
                f"(p={d['S_north_p']:.4f})"
            )
        print(
            f"     salinity share of dS trend {f['salinity_share_of_dS']:.3f}, "
            f"velocity share {f['velocity_share_of_dS']:.3f}"
        )
        c = r["composition_monthly"]
        print(
            f"  composition on monthly basis: sub-limb T sum "
            f"{c['sublimb_T_sum_Sv']:.3f} vs "
            f"T {c['T_limb_Sv']:.3f} Sv (gap {c['closure_gap_Sv']:+.4f})"
        )
        print(
            f"     S_north {c['S_north_trend_per_decade']:+.4f} = water-mass "
            f"{c['water_mass_trend_per_decade']:+.4f} + partition "
            f"{c['partition_trend_per_decade']:+.4f} (share "
            f"{c['partition_share']:.3f}, "
            f"residual {c['residual_per_decade']:+.5f})"
        )
        print(
            f"     upper T {c['upper_T_trend_per_decade']:+.3f}/dec "
            f"p={c['upper_T_p']:.4f} | "
            f"abyssal T {c['abyssal_T_trend_per_decade']:+.3f} "
            f"p={c['abyssal_T_p']:.4f} | "
            f"upper S {c['upper_S_trend_per_decade']:+.4f} p={c['upper_S_p']:.4f} | "
            f"abyssal frac {100 * c['abyssal_frac_of_T']:.1f}%"
        )
        a = r["alternative_removal"]
        print(
            f"  alternative removal: T {a['mean_T_Sv']:.2f} Sv, dS "
            f"{a['mean_dS_PSU']:+.4f}, "
            f"F_ov {a['mean_F_ov_Sv']:+.4f} Sv, trend "
            f"{a['F_ov_trend_mSv_per_yr']:+.3f} mSv/yr, "
            f"share {a['transport_share']:.3f}"
        )
        w = r.get("window_1993_2025")
        if w:
            print(
                f"  1993-2025: S_north {w['S_north_trend_per_decade']:+.4f}/dec "
                f"(p={w['S_north_p']:.4f}, N_eff={w['S_north_n_eff']:.1f}), F_ov "
                f"{w['F_ov_trend_mSv_per_yr']:+.3f} mSv/yr (p={w['F_ov_p']:.4f}), "
                f"share {w['transport_share']:.3f}"
            )
        for vname, s in r.get("step_fits_2004", {}).items():
            print(
                f"  step 2004, {vname}: trend {s['trend_mSv_per_yr']:+.4f}+/-"
                f"{s['trend_se']:.4f} (p={s['trend_p']:.4f}), step {s['step_Sv']:+.5f}"
                f"+/-{s['step_se']:.5f} (p={s['step_p']:.4f})"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "final_checks.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
