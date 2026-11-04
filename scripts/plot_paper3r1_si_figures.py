#!/usr/bin/env python3
"""Supplementary figures added in the revision.

FigS_soda              annual F_ovS of SODA 3.15.2 (annual fields) beside the
                       three reanalyses computed from monthly fields.
FigS_trend_sections    longitude-depth sections of the salinity trend over each
                       product's own record.
FigS_samba             monthly anomalies of the upper- and abyssal-cell
                       strengths against the SAMBA array, 2013-2017.
FigS_amoc26_anomalies  annual overturning anomalies at 26.5 N in four reanalyses
                       and RAPID, with trends from the corrected test.
FigS_cmip6_trajectories  decadal running mean of F_ovS in the CMIP6 models
                       (historical and ssp585) with the reanalyses.

Writes PAPER_3_v2/figures_r1/FigS_*.{pdf,png}.
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

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from paper3r1_stats import corr_test  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    C_SODA,
    TWO_COL,
    panel_letter,
    set_style,
)  # noqa: E402

RES = REPO / "data" / "results"
AN = REPO / "PAPER_3_v2" / "analysis"
FIG = REPO / "PAPER_3_v2" / "figures_r1"
PROD = [
    ("oras5", "ORAS5", C_ORAS5),
    ("glorys12", "GLORYS12V1", C_GLORYS),
    ("ecco", "ECCO-V4r4", C_ECCO),
]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print("wrote", name)


def soda():
    fig, ax = plt.subplots(figsize=(TWO_COL * 0.6, 2.2))
    with xr.open_dataset(RES / "soda_f_ovs.nc") as ds:
        ax.plot(
            ds["year"],
            ds["F_ovS"],
            color=C_SODA,
            lw=0.9,
            label="SODA 3.15.2, annual fields",
        )
    for key, label, col in PROD:
        a = (
            xr.open_dataset(RES / f"paper3r1_section_{key}.nc")["F_ov"]
            .groupby("time.year")
            .mean()
        )
        ax.plot(a["year"], a, color=col, lw=0.9, label=f"{label}, monthly fields")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(1958, 2026)
    ax.set_xlabel("Year")
    ax.set_ylabel("$F_{\\mathrm{ovS}}$ (Sv)")
    ax.legend(loc="lower left", fontsize=5.3, ncol=2)
    save(fig, "FigS_soda")


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def trend_sections():
    lev = np.array([-0.15, -0.10, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10, 0.15])
    cmap = plt.get_cmap("BrBG_r", len(lev) + 1)
    norm = colors.BoundaryNorm(lev, cmap.N, extend="both")
    fig, axs = plt.subplots(3, 1, figsize=(TWO_COL, 6.0), sharex=True)
    for ax, (key, label, _), win, letter in zip(
        axs, PROD, ("own", "own", "own"), "abc", strict=False
    ):
        ds = xr.open_dataset(RES / f"paper3r1_salinity_trend_section_{key}_{win}.nc")
        cf = ax.contourf(
            ds["lon"],
            ds["depth"],
            ds["trend"],
            levels=lev,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        p = ds["p"].values
        step = max(1, ds.sizes["lon"] // 70)
        lon2, dep2 = np.meshgrid(ds["lon"].values[::step], ds["depth"].values)
        sig = (p[:, ::step] < 0.05) & np.isfinite(ds["trend"].values[:, ::step])
        ax.plot(lon2[sig], dep2[sig], ".", ms=0.5, color="black", alpha=0.6)
        ax.set_yscale("function", functions=(depth_fwd, depth_inv))
        ax.set_ylim(5800, 0)
        ax.set_yticks([0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000])
        ax.set_facecolor("#4d4d4d")
        ax.set_ylabel("Depth (m)")
        ax.text(
            0.01,
            0.03,
            f"{label}, {ds.attrs['window']}",
            transform=ax.transAxes,
            fontsize=6,
            color="white",
        )
        panel_letter(ax, letter, x=-0.07, y=1.0)
    axs[-1].set_xlabel("Longitude (degrees east)")
    cb = fig.colorbar(cf, ax=axs, pad=0.015, fraction=0.03)
    cb.set_label("Salinity trend (PSU decade$^{-1}$)", fontsize=6)
    save(fig, "FigS_trend_sections")


def amoc26_anomalies():
    """Annual overturning anomalies at 26.5 N in four reanalyses and RAPID,
    relative to the ORAS5 mean of 1958-2009; shorter records are offset to
    ORAS5 over their overlap. Trends use the test of paper3r1_stats.py and
    are written to PAPER_3_v2/analysis/r1_amoc26_trends.json."""
    from paper3r1_stats import trend_test

    series = {}
    for key, label, col in (
        ("oras5", "ORAS5", C_ORAS5),
        ("glorys12", "GLORYS12V1", C_GLORYS),
        ("soda", "SODA 3.15.2", C_SODA),
        ("ecco", "ECCO-V4r4", C_ECCO),
    ):
        z = np.load(RES / f"yearly_amoc26n_{key}.npz")
        series[key] = (label, col, pd.Series(z["amoc"], index=z["years"].astype(int)))
    rp = xr.open_dataset(REPO / "data" / "external" / "rapid_moc_monthly.nc")
    moc = rp["moc_mar_hc10"].to_series()
    moc.index = pd.DatetimeIndex(moc.index)
    cnt = moc.groupby(moc.index.year).count()
    rapid = moc.groupby(moc.index.year).mean()[cnt == 12]
    series["rapid"] = ("RAPID", "black", rapid)

    oras5 = series["oras5"][2]
    ref = float(oras5.loc[1958:2009].mean())
    fig, ax = plt.subplots(figsize=(TWO_COL * 0.7, 2.6))
    trends = {"reference_ORAS5_1958_2009_Sv": ref}
    handles = []
    for key, (label, col, s) in series.items():
        overlap = s.index.intersection(oras5.index)
        offset = (
            0.0
            if key == "oras5"
            else float(oras5.loc[overlap].mean() - s.loc[overlap].mean())
        )
        anom = s + offset - ref
        ax.plot(anom.index, anom.values, color=col, lw=0.5, alpha=0.6)
        (h,) = ax.plot(
            anom.index,
            anom.rolling(5, center=True, min_periods=3).mean().values,
            color=col,
            lw=1.3,
        )
        r = trend_test(s.index.values.astype(float), s.values)
        trends[key] = {
            "window": [int(s.index[0]), int(s.index[-1])],
            "trend_Sv_per_decade": r["value"] * 10,
            "ci_Sv_per_decade": [c * 10 for c in r["ci"]],
            "p": r["p"],
            "n_eff": r["n_eff"],
            "offset_Sv": offset,
        }
        if key == "soda":
            # SODA is shown for comparison only and carries no trend in the legend.
            legend_text = f"{label} {s.index[0]}–{s.index[-1]}"
        else:
            sig = "" if r["p"] >= 0.05 else ", significant"
            legend_text = (
                f"{label} {s.index[0]}–{s.index[-1]}: "
                f"{r['value'] * 10:+.2f} Sv per decade{sig}"
            )
        handles.append((h, legend_text))
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Overturning anomaly at 26.5°N (Sv)")
    ax.legend(
        [h for h, _ in handles], [t for _, t in handles], loc="lower left", fontsize=5.2
    )
    save(fig, "FigS_amoc26_anomalies")
    (AN / "r1_amoc26_trends.json").write_text(json.dumps(trends, indent=1))
    print(json.dumps(trends, indent=1))


def layer_transport(key: str, z_top: float, z_bot: float) -> pd.Series:
    """Monthly transport (Sv) of the velocity after removal of the net section
    transport between two depths, from the section file."""
    ds = xr.open_dataset(RES / f"paper3r1_section_{key}.nc")
    zw = np.concatenate([[0.0], np.cumsum(ds["e3t"].values)])
    w = np.clip(np.minimum(zw[1:], z_bot) - np.maximum(zw[:-1], z_top), 0, None)
    q = np.nansum(ds["V_bc"].values * w, axis=1) / 1e6
    return pd.Series(q, index=pd.DatetimeIndex(ds["time"].values))


def samba():
    """SAMBA cell anomalies and the reanalysis transports over the same layers:
    the surface to 1315 dbar (northward positive) and 3155 to 4295 dbar
    (southward positive), as SAMBA defines them."""
    import gsw

    sb = xr.open_dataset(REPO / "data" / "external" / "samba_moc_monthly.nc")
    t = pd.DatetimeIndex(sb["time"].values)
    d1, d2, d3 = (-float(gsw.z_from_p(p, -34.5)) for p in (1315.0, 3155.0, 4295.0))
    fig, axs = plt.subplots(2, 1, figsize=(TWO_COL * 0.7, 3.4), sharex=True)
    for ax, var, layer, sign, letter, lab in (
        (axs[0], "upper_cell", (0.0, d1), 1.0, "a", "Upper cell"),
        (axs[1], "abyssal_cell", (d2, d3), -1.0, "b", "Abyssal cell"),
    ):
        ax.plot(t, sb[var], color="black", lw=1.0, label="SAMBA")
        for key, label, col in PROD:
            s = sign * layer_transport(key, *layer)
            s = s[(s.index >= t[0]) & (s.index <= t[-1] + pd.Timedelta(days=31))]
            ax.plot(s.index, s - s.mean(), color=col, lw=0.8, label=label)
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_ylabel(f"{lab} anomaly (Sv)")
        panel_letter(ax, letter, x=-0.1, y=1.0)
    import matplotlib.dates as mdates

    axs[0].legend(loc="upper left", ncol=4, fontsize=5.3)
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axs[1].set_xlabel("Year")
    save(fig, "FigS_samba")


def cmip6_trajectories():
    d = json.loads((AN / "r1_cmip6.json").read_text())
    fig, ax = plt.subplots(figsize=(TWO_COL * 0.7, 2.6))
    for r in d["per_model"]:
        f = RES / f"paper3r1_cmip6_{r['model']}.nc"
        if not f.exists():
            continue
        a = xr.open_dataset(f)["F_ov"].to_series()
        a = a[(a.index >= 1850) & (a.index <= 2100)]
        ax.plot(a.index, a.rolling(10, center=True).mean(), color="0.55", lw=0.5)
    for key, label, col in PROD:
        a = (
            xr.open_dataset(RES / f"paper3r1_section_{key}.nc")["F_ov"]
            .groupby("time.year")
            .mean()
        )
        sr = a.to_series()
        ax.plot(
            sr.index, sr.rolling(10, center=True).mean(), color=col, lw=1.4, label=label
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(1850, 2100)
    ax.set_xlabel("Year")
    ax.set_ylabel("$F_{\\mathrm{ovS}}$, 10-year running mean (Sv)")
    ax.legend(
        handles=[plt.Line2D([], [], color="0.55", lw=0.5, label="CMIP6 model")]
        + [plt.Line2D([], [], color=c, lw=1.4, label=lab) for _, lab, c in PROD],
        loc="lower left",
        fontsize=5.3,
        ncol=2,
    )
    save(fig, "FigS_cmip6_trajectories")


def pileup():
    """Pile-up index against F_ovS; writes the correlations used in the caption."""

    fig, axs = plt.subplots(
        1, 2, figsize=(TWO_COL * 0.85, 2.3), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    out = {}
    for key, label, col in PROD[:2]:
        pi = (
            xr.open_dataset(RES / f"pileup_wrapped_{key}.nc")["pileup"]
            .groupby("time.year")
            .mean()
        )
        fo = (
            xr.open_dataset(RES / f"paper3r1_section_{key}.nc")["F_ov"]
            .groupby("time.year")
            .mean()
        )
        common = np.intersect1d(pi["year"].values, fo["year"].values)
        x = pi.sel(year=common).values
        y = fo.sel(year=common).values
        common.astype(float)
        axs[0].plot(pi["year"], pi, color=col, lw=0.9, label=label)
        axs[1].plot(x, y, "o", ms=2.5, color=col, label=label)
        # Correlations of autocorrelated series, with the effective sample size
        # of Bretherton et al. (1999) from the corrected lag-1 coefficients.
        raw = corr_test(x, y, detrend=False)
        det = corr_test(x, y, detrend=True)
        slope = np.polyfit(pi["year"].values.astype(float), pi.values, 1)[0]
        out[key] = {
            "years": [int(common[0]), int(common[-1])],
            "r": raw["r"],
            "p": raw["p"],
            "n_eff": raw["n_eff"],
            "r_detrended": det["r"],
            "p_detrended": det["p"],
            "n_eff_detrended": det["n_eff"],
            "pileup_trend_PSU_per_decade": float(10 * slope),
        }
    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Pile-up index (PSU)")
    axs[0].legend(loc="upper left", fontsize=5.3)
    axs[1].set_xlabel("Pile-up index (PSU)")
    axs[1].set_ylabel("$F_{\\mathrm{ovS}}$ (Sv)")
    panel_letter(axs[0], "a", x=-0.15, y=1.0)
    panel_letter(axs[1], "b", x=-0.25, y=1.0)
    save(fig, "FigS_pileup")
    (AN / "r1_pileup.json").write_text(json.dumps(out, indent=1))
    print(out)


def main() -> None:
    set_style()
    FIG.mkdir(parents=True, exist_ok=True)
    soda()
    trend_sections()
    samba()
    cmip6_trajectories()
    pileup()
    amoc26_anomalies()


if __name__ == "__main__":
    main()
