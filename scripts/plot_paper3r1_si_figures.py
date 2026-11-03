#!/usr/bin/env python3
"""Supplementary figures added in the revision.

FigS_soda              annual F_ovS of SODA 3.15.2 (annual fields) beside the
                       three reanalyses computed from monthly fields.
FigS_trend_sections    longitude-depth sections of the salinity trend over each
                       product's own record.
FigS_samba             monthly anomalies of the upper- and abyssal-cell
                       strengths against the SAMBA array, 2013-2017.
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


def samba():
    sb = xr.open_dataset(REPO / "data" / "external" / "samba_moc_monthly.nc")
    t = pd.DatetimeIndex(sb["time"].values)
    fig, axs = plt.subplots(2, 1, figsize=(TWO_COL * 0.7, 3.4), sharex=True)
    for ax, var, cellvar, sign, letter, lab in (
        (axs[0], "upper_cell", "psi_max", 1.0, "a", "Upper cell"),
        (axs[1], "abyssal_cell", "psi_min", -1.0, "b", "Abyssal cell"),
    ):
        ax.plot(t, sb[var], color="black", lw=1.0, label="SAMBA")
        for key, label, col in PROD:
            c = xr.open_dataset(RES / f"paper3r1_cells_{key}.nc")
            s = (sign * c[cellvar]).sel(time=slice(t[0], t[-1] + pd.Timedelta(days=31)))
            ax.plot(
                pd.DatetimeIndex(s["time"].values),
                s - s.mean(),
                color=col,
                lw=0.8,
                label=label,
            )
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
    from scipy import stats

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
        yrs = common.astype(float)
        axs[0].plot(pi["year"], pi, color=col, lw=0.9, label=label)
        axs[1].plot(x, y, "o", ms=2.5, color=col, label=label)
        r, p = stats.pearsonr(x, y)
        dx = x - np.polyval(np.polyfit(yrs, x, 1), yrs)
        dy = y - np.polyval(np.polyfit(yrs, y, 1), yrs)
        rd, pd_ = stats.pearsonr(dx, dy)
        slope = np.polyfit(pi["year"].values.astype(float), pi.values, 1)[0]
        out[key] = {
            "years": [int(common[0]), int(common[-1])],
            "r": float(r),
            "p": float(p),
            "r_detrended": float(rd),
            "p_detrended": float(pd_),
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


if __name__ == "__main__":
    main()
