#!/usr/bin/env python3
"""Overturning cells defined by the streamfunction, the common-window comparison,
and the sensitivity of the section integrals to the vertical grid.

Cells. With q_k = V_bc,k e3t_k the volume transport of level k (northward
positive) and

    psi(z) = int_z^0 V_bc dz'        (northward transport above depth z),

psi vanishes at the surface and, because int V_bc dz = 0, at the bottom. Its
maximum marks the base of the northward upper limb. The upper (North Atlantic
Deep Water) cell is the layer between the surface and the first depth below that
maximum at which psi returns to zero; the abyssal (Antarctic Bottom Water) cell
is the layer between that depth and the bottom. The level that contains the
zero crossing is split at the crossing, with its salinity taken as uniform
across the level. Each cell then carries zero net volume transport, so

    F_ovS = F_upper + F_abyssal,   F_cell = -(1/S0) T_cell (S_N,cell - S_S,cell),

exactly, with T_cell the northward transport within the cell and S_N,cell and
S_S,cell the transport-weighted salinities of its northward and southward
branches. The southward branch of the abyssal cell is the lower part of the
southward deep water, so neither cell is mass-imbalanced.

Common window. Trends of every factor over 1993 to 2017, the years all three
reanalyses share, with the azonal component and the total.

Grid and removal. Means, trends and the transport share are recomputed for every
variant written by ``compute_paper3r1_section_quantities.py`` (vertical cell
thickness from the spacing of level centres; the deepest wet cell of each
column at half and at one fifth of its thickness) and for the net transport
removed level by level in proportion to |V_int(z)|.

SAMBA. Monthly anomalies of the upper and abyssal cell strengths are compared
with the SAMBA array over the months they share.

Writes ``PAPER_3_v2/analysis/r1_cells.json``,
``PAPER_3_v2/analysis/r1_common_window.json``,
``PAPER_3_v2/analysis/r1_grid_sensitivity.json`` and
``data/results/paper3r1_cells_<product>.nc``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from analysis_paper3v2_attribution import (  # noqa: E402
    ARGO_START,
    COMMON_WINDOW,
    N_BOOT,
    PRIMARY_BLOCK,
    S0,
    SECTIONS,
    annual,
    block_idx,
    continuous,
    trend_stats,
)

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
SAMBA = REPO / "data" / "external" / "samba_moc_monthly.nc"


# ── cells ─────────────────────────────────────────────────────────────


def limb_terms(q: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """Northward transport and transport-weighted branch salinities of a layer."""
    pos, neg = np.clip(q, 0, None), np.clip(-q, 0, None)
    t_n, t_s = pos.sum(), neg.sum()
    if t_n <= 0 or t_s <= 0:
        return 0.0, np.nan, np.nan
    return float(t_n), float((pos * s).sum() / t_n), float((neg * s).sum() / t_s)


def split_cells(q: np.ndarray, s: np.ndarray, e3t: np.ndarray) -> dict:
    """Split one monthly profile into its upper and abyssal cells."""
    nz = len(q)
    psi = np.concatenate([[0.0], np.cumsum(q)])
    zw = np.concatenate([[0.0], np.cumsum(e3t)])
    kmax = int(np.argmax(psi))
    up_q, ab_q = q.copy(), np.zeros(nz)
    z0 = np.nan
    psi_min = 0.0
    cross = [i for i in range(kmax + 1, nz) if psi[i] < 0.0]
    if cross:
        i = cross[0]
        k = i - 1
        frac = psi[k] / (psi[k] - psi[i])
        z0 = zw[k] + frac * e3t[k]
        up_q = np.zeros(nz)
        up_q[:k] = q[:k]
        up_q[k] = -psi[k]
        ab_q[k] = q[k] + psi[k]
        ab_q[k + 1 :] = q[k + 1 :]
        psi_min = float(psi[i:].min())
    t_up, sn_up, ss_up = limb_terms(up_q, s)
    t_ab, sn_ab, ss_ab = limb_terms(ab_q, s)
    f_up = -(1.0 / S0) * float((up_q * (s - S0)).sum())
    f_ab = -(1.0 / S0) * float((ab_q * (s - S0)).sum())
    return {
        "F_up": f_up / 1e6,
        "F_ab": f_ab / 1e6,
        "T_up": t_up / 1e6,
        "Sn_up": sn_up,
        "Ss_up": ss_up,
        "T_ab": t_ab / 1e6,
        "Sn_ab": sn_ab,
        "Ss_ab": ss_ab,
        "psi_max": float(psi[kmax]) / 1e6,
        "psi_min": psi_min / 1e6,
        "z_psimax": float(zw[kmax]),
        "z0": float(z0),
        "net_up": float(up_q.sum()) / 1e6,
        "net_ab": float(ab_q.sum()) / 1e6,
    }


def cell_series(ds: xr.Dataset) -> xr.Dataset:
    e3t = ds["e3t"].values
    vbc = ds["V_bc"].values
    sbar = ds["S_bar"].values
    rows = [split_cells(vbc[t] * e3t, sbar[t], e3t) for t in range(ds.sizes["time"])]
    out = xr.Dataset(
        {k: ("time", np.array([r[k] for r in rows])) for k in rows[0]},
        coords={"time": ds["time"].values},
    )
    out["dS_up"] = out["Sn_up"] - out["Ss_up"]
    out["dS_ab"] = out["Sn_ab"] - out["Ss_ab"]
    out["F_ov"] = ds["F_ov"]
    out["F_az"] = ds["F_az"]
    out["T_limb"] = ds["T_limb_Sv"]
    out["dS_limb"] = ds["dS_limb"]
    # psi from the uncorrected velocity, for comparison with the observing array.
    psi_raw = np.cumsum(ds["V_int"].values * e3t, axis=1) / 1e6
    out["psi_max_raw"] = ("time", psi_raw.max(axis=1))
    return out


def joint_share(years, part, total) -> dict:
    """Trend of one additive part as a fraction of the trend of the total."""
    x = years.astype(float)
    fits = {}
    comps = {}
    for name, ser in (("part", part), ("total", total)):
        coef = np.polyfit(x, ser, 1)
        fitted = np.polyval(coef, x)
        comps[name] = (fitted, ser - fitted)
        fits[name] = ols_santer(x, ser)
    boot = {k: np.empty(N_BOOT) for k in comps}
    for i in range(N_BOOT):
        idx = block_idx(len(x), PRIMARY_BLOCK)
        for k, (fitted, resid) in comps.items():
            boot[k][i] = np.polyfit(x, fitted + resid[idx], 1)[0]
    ratio = boot["part"] / boot["total"]
    return {
        "share_point": float(fits["part"]["slope"] / fits["total"]["slope"]),
        "share_ci": [
            float(np.percentile(ratio, 2.5)),
            float(np.percentile(ratio, 97.5)),
        ],
        "trend_part_mSv_per_yr": fits["part"]["slope"] * 1e3,
        "p_part": fits["part"]["p_santer"],
        "trend_total_mSv_per_yr": fits["total"]["slope"] * 1e3,
        "p_total": fits["total"]["p_santer"],
    }


def windows_for(years: np.ndarray) -> dict:
    w = {"own": (int(years[0]), int(years[-1])), "argo": (ARGO_START, int(years[-1]))}
    if years[0] <= COMMON_WINDOW[0] and years[-1] >= COMMON_WINDOW[1]:
        w["common"] = COMMON_WINDOW
    return w


def run_cells(key: str) -> dict:
    label, fname = SECTIONS[key]
    ds = xr.open_dataset(RESULTS / fname)
    cs = cell_series(ds)
    cs.attrs.update(
        product=label,
        note="cells defined by the zero crossing of psi below its maximum",
    )
    cs.to_netcdf(RESULTS / f"paper3r1_cells_{key}.nc")

    closure = float(np.nanmax(np.abs(cs["F_up"] + cs["F_ab"] - cs["F_ov"])))
    net = float(np.nanmax(np.abs(np.concatenate([cs["net_up"], cs["net_ab"]]))))
    ann = annual(cs)
    years = ann["year"].values
    res: dict = {
        "product": label,
        "n_months": int(cs.sizes["time"]),
        "closure_max_abs_Sv": closure,
        "cell_net_transport_max_abs_Sv": net,
        "months_without_abyssal_cell": int(np.isnan(cs["z0"]).sum()),
        "means": {
            k: float(cs[k].mean())
            for k in (
                "F_ov",
                "F_up",
                "F_ab",
                "T_up",
                "dS_up",
                "Sn_up",
                "Ss_up",
                "T_ab",
                "dS_ab",
                "Sn_ab",
                "Ss_ab",
                "psi_max",
                "psi_min",
                "psi_max_raw",
                "z_psimax",
                "z0",
            )
        },
        "windows": {},
    }
    res["means"]["z0_median"] = float(np.nanmedian(cs["z0"]))
    for wname, (y0, y1) in windows_for(years).items():
        m = (years >= y0) & (years <= y1)
        y = years[m]
        sub = {k: ann[k].values[m] for k in ann.data_vars}
        block = {
            "trend_F_ov": trend_stats(y, sub["F_ov"]),
            "trend_F_up": trend_stats(y, sub["F_up"]),
            "trend_F_ab": trend_stats(y, sub["F_ab"]),
            "trend_T_up_Sv_per_decade": trend_stats(y, sub["T_up"], scale=10.0),
            "trend_T_ab_Sv_per_decade": trend_stats(y, sub["T_ab"], scale=10.0),
            "trend_dS_up_PSU_per_decade": trend_stats(y, sub["dS_up"], scale=10.0),
            "trend_Sn_up_PSU_per_decade": trend_stats(y, sub["Sn_up"], scale=10.0),
            "trend_Ss_up_PSU_per_decade": trend_stats(y, sub["Ss_up"], scale=10.0),
            "trend_dS_ab_PSU_per_decade": trend_stats(y, sub["dS_ab"], scale=10.0),
            "upper_cell_attribution": continuous(
                y, sub["T_up"], sub["dS_up"], sub["F_up"]
            ),
            "upper_cell_share_of_F_ov_trend": joint_share(y, sub["F_up"], sub["F_ov"]),
            "abyssal_cell_share_of_F_ov_trend": joint_share(
                y, sub["F_ab"], sub["F_ov"]
            ),
        }
        res["windows"][wname] = block
    return res


# ── SAMBA ─────────────────────────────────────────────────────────────


def samba_compare(cells: dict[str, xr.Dataset]) -> dict:
    """Monthly anomalies of the cell strengths against the SAMBA array.

    The SAMBA record held here is a series of transport anomalies (its mean is
    zero to within rounding), so it constrains variability, not the mean.
    """
    sb = xr.open_dataset(SAMBA)
    st = pd.DatetimeIndex(sb["time"].values)
    s_key = list(st.strftime("%Y-%m"))
    out = {
        "samba_months": int(len(st)),
        "samba_window": [str(st[0].date()), str(st[-1].date())],
        "samba_upper_mean_Sv": float(sb["upper_cell"].mean()),
        "samba_upper_std_Sv": float(sb["upper_cell"].std()),
        "samba_abyssal_mean_Sv": float(sb["abyssal_cell"].mean()),
        "samba_abyssal_std_Sv": float(sb["abyssal_cell"].std()),
        "samba_upper_attrs": {k: str(v) for k, v in sb["upper_cell"].attrs.items()},
        "samba_abyssal_attrs": {k: str(v) for k, v in sb["abyssal_cell"].attrs.items()},
        "products": {},
    }
    for key, cs in cells.items():
        rt = list(pd.DatetimeIndex(cs["time"].values).strftime("%Y-%m"))
        common = sorted(set(rt) & set(s_key))
        ri = [rt.index(c) for c in common]
        si = [s_key.index(c) for c in common]
        up_r = cs["psi_max"].values[ri]
        ab_r = -cs["psi_min"].values[ri]
        up_s = sb["upper_cell"].values[si]
        ab_s = sb["abyssal_cell"].values[si]
        ok = ~np.isnan(up_s) & ~np.isnan(up_r)
        oka = ~np.isnan(ab_s) & ~np.isnan(ab_r)
        out["products"][key] = {
            "n_common_months": int(ok.sum()),
            "upper_mean_Sv": float(up_r[ok].mean()),
            "upper_std_Sv": float(up_r[ok].std()),
            "upper_anomaly_corr": float(np.corrcoef(up_r[ok], up_s[ok])[0, 1]),
            "abyssal_mean_Sv": float(ab_r[oka].mean()),
            "abyssal_std_Sv": float(ab_r[oka].std()),
            "abyssal_anomaly_corr_with_sign_as_stored": float(
                np.corrcoef(ab_r[oka], ab_s[oka])[0, 1]
            ),
            "T_limb_mean_same_months_Sv": float(cs["T_limb"].values[ri][ok].mean()),
        }
    return out


# ── common window and azonal component ───────────────────────────────


def common_window(key: str) -> dict:
    label, fname = SECTIONS[key]
    ds = xr.open_dataset(RESULTS / fname)
    ann = annual(ds)
    years = ann["year"].values
    res: dict = {"product": label}
    for wname, (y0, y1) in windows_for(years).items():
        m = (years >= y0) & (years <= y1)
        y = years[m]

        def g(v, m=m):
            return ann[v].values[m]

        f_tot = g("F_ov") + g("F_az")
        cont = continuous(y, g("T_limb_Sv"), g("dS_limb"), g("F_ov"))
        res[wname] = {
            "window": [int(y[0]), int(y[-1])],
            "mean_F_ov_Sv": float(g("F_ov").mean()),
            "mean_F_az_Sv": float(g("F_az").mean()),
            "mean_F_total_Sv": float(f_tot.mean()),
            "ratio_abs_F_az_to_F_ov": float(abs(g("F_az").mean() / g("F_ov").mean())),
            "mean_T_Sv": float(g("T_limb_Sv").mean()),
            "mean_dS_PSU": float(g("dS_limb").mean()),
            "trend_F_ov": trend_stats(y, g("F_ov")),
            "trend_F_az": trend_stats(y, g("F_az")),
            "trend_F_total": trend_stats(y, f_tot),
            "trend_T_Sv_per_decade": trend_stats(y, g("T_limb_Sv"), scale=10.0),
            "trend_dS_PSU_per_decade": trend_stats(y, g("dS_limb"), scale=10.0),
            "trend_S_north_PSU_per_decade": trend_stats(y, g("S_north"), scale=10.0),
            "trend_S_south_PSU_per_decade": trend_stats(y, g("S_south"), scale=10.0),
            "trend_psi_max_surface_Sv_per_decade": trend_stats(
                y, g("psi_max_surface_Sv"), scale=10.0
            ),
            "attribution": cont,
            "fractional_rate_F_ov_pct_per_decade": float(
                100
                * 10
                * ols_santer(y.astype(float), g("F_ov"))["slope"]
                / abs(g("F_ov").mean())
            ),
            "fractional_rate_T_pct_per_decade": float(
                100
                * 10
                * ols_santer(y.astype(float), g("T_limb_Sv"))["slope"]
                / abs(g("T_limb_Sv").mean())
            ),
        }
    return res


# ── sensitivity to the grid and to the net-transport removal ─────────


def abs_v_removal(ds: xr.Dataset) -> xr.Dataset:
    """Net transport removed level by level in proportion to |V_int(z)|.

    V_alt(z) = V_int(z) - c |V_int(z)| with c = sum V_int dz / sum |V_int| dz.
    Because |c| < 1, V_alt has the sign of V_int at every level, so the limb
    boundaries are those of the uncorrected velocity.
    """
    e3t = ds["e3t"].values
    vint = ds["V_int"].values
    sbar = ds["S_bar"].values
    c = (vint * e3t).sum(axis=1) / (np.abs(vint) * e3t).sum(axis=1)
    q = (vint - c[:, None] * np.abs(vint)) * e3t
    pos = np.clip(q, 0, None)
    neg = np.clip(-q, 0, None)
    t_n = pos.sum(axis=1)
    sn = (pos * sbar).sum(axis=1) / t_n
    ss = (neg * sbar).sum(axis=1) / neg.sum(axis=1)
    f = -(1.0 / S0) * (q * (sbar - S0)).sum(axis=1) / 1e6
    out = xr.Dataset(
        {
            "F_ov": ("time", f),
            "T_limb_Sv": ("time", t_n / 1e6),
            "dS_limb": ("time", sn - ss),
            "F_az": ds["F_az"],
            "V_net_Sv": ds["V_net_Sv"],
        },
        coords={"time": ds["time"].values},
    )
    out.attrs["max_abs_c"] = float(np.abs(c).max())
    return out


def summarise_variant(ds: xr.Dataset) -> dict:
    ann = annual(ds)
    y = ann["year"].values
    cont = continuous(
        y, ann["T_limb_Sv"].values, ann["dS_limb"].values, ann["F_ov"].values
    )
    return {
        "mean_F_ov_Sv": float(ds["F_ov"].mean()),
        "mean_T_Sv": float(ds["T_limb_Sv"].mean()),
        "mean_dS_PSU": float(ds["dS_limb"].mean()),
        "mean_F_az_Sv": float(ds["F_az"].mean()),
        "mean_V_net_Sv": float(ds["V_net_Sv"].mean()),
        "trend_F_ov_mSv_per_yr": cont["trend_total_mSv_per_yr"],
        "p_F_ov": cont["p_total"],
        "trend_T_term_mSv_per_yr": cont["trend_T_mSv_per_yr"],
        "trend_S_term_mSv_per_yr": cont["trend_S_mSv_per_yr"],
        "share_T": cont["share_T_point"],
        "share_T_ci": cont["share_T_ci"],
    }


def grid_sensitivity(key: str) -> dict:
    label = SECTIONS[key][0]
    out: dict = {"product": label}
    for suffix, name in (
        ("", "baseline: level bounds, uniform removal"),
        ("_centres", "thickness from the spacing of level centres"),
        ("_bottom0p5", "deepest wet cell of each column at half thickness"),
        ("_bottom0p2", "deepest wet cell of each column at one fifth thickness"),
    ):
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}{suffix}.nc")
        out[suffix or "_baseline"] = {"description": name, **summarise_variant(ds)}
        if suffix == "":
            alt = abs_v_removal(ds)
            out["_absV"] = {
                "description": "net transport removed in proportion to |V_int(z)|",
                "max_abs_c": alt.attrs["max_abs_c"],
                **summarise_variant(alt),
            }
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells, cell_ds = {}, {}
    for key in SECTIONS:
        cells[key] = run_cells(key)
        cell_ds[key] = xr.open_dataset(RESULTS / f"paper3r1_cells_{key}.nc")
        m = cells[key]["means"]
        print(
            f"{key:9s} closure {cells[key]['closure_max_abs_Sv']:.1e} Sv  "
            f"T_up {m['T_up']:.2f}  T_ab {m['T_ab']:.2f}  z0 {m['z0_median']:.0f} m  "
            f"F_up {m['F_up']:+.4f}  F_ab {m['F_ab']:+.4f}  F_ov {m['F_ov']:+.4f}",
            flush=True,
        )
    cells["samba"] = samba_compare(cell_ds)
    (OUT_DIR / "r1_cells.json").write_text(json.dumps(_clean(cells), indent=1))

    cw = {key: common_window(key) for key in SECTIONS}
    (OUT_DIR / "r1_common_window.json").write_text(json.dumps(_clean(cw), indent=1))

    vg = {key: grid_sensitivity(key) for key in SECTIONS}
    (OUT_DIR / "r1_grid_sensitivity.json").write_text(json.dumps(_clean(vg), indent=1))
    print("wrote r1_cells.json, r1_common_window.json, r1_grid_sensitivity.json")


if __name__ == "__main__":
    main()
