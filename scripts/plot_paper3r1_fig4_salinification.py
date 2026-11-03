#!/usr/bin/env python3
"""Figure 4: where and at which depths the northward limb gains salt.

a  Transport-weighted salinity of the northward limb (solid) and southward limb
   (dashed) as anomalies from their record means.
b  Zonal-mean salinity trend against depth over the full water column, drawn
   bold where the Santer-adjusted p is below 0.05; grey bands mark the depth
   ranges of the northward limb.
c  Longitude-depth section of the salinity trend in GLORYS12V1, 2005-2024.
d  The same in the EN4.2.2 objective analysis on its 34 S row.
e  Contributions of four longitude bands and three layers to the trend of the
   northward-limb salinity with the circulation frozen at its record mean,
   2005-2024; diamonds give each band's share of the total if the trend were
   uniform along the section.

Writes PAPER_3_v2/figures_r1/Fig4_salinification.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    TWO_COL,
    panel_letter,
    set_style,
)

RES = REPO / "data" / "results"
AN = REPO / "PAPER_3_v2" / "analysis"
OUT = REPO / "PAPER_3_v2" / "figures_r1" / "Fig4_salinification"
PROD = [
    ("oras5", "ORAS5", C_ORAS5),
    ("glorys12", "GLORYS12V1", C_GLORYS),
    ("ecco", "ECCO-V4r4", C_ECCO),
]
BANDS = [(-54.0, -45.0), (-45.0, -20.0), (-20.0, 0.0), (0.0, 19.4)]
BAND_LABELS = ["WB", "WI", "EI", "CB"]
LEV = np.array([-0.15, -0.10, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10, 0.15])


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def trend_profile(ds: xr.Dataset, y0: int, y1: int):
    a = ds["S_bar"].groupby("time.year").mean()
    yrs = a["year"].values
    m = (yrs >= y0) & (yrs <= y1)
    tr, pv = [], []
    for k in range(a.sizes["depth"]):
        y = a.values[m, k]
        if np.ptp(y) == 0 or not np.isfinite(y).all():
            tr.append(np.nan)
            pv.append(np.nan)
            continue
        f = ols_santer(yrs[m].astype(float), y)
        tr.append(f["slope"] * 10)
        pv.append(f["p_santer"])
    return np.array(tr), np.array(pv)


def section_panel(ax, fname: str, fig, add_cbar: bool):
    ds = xr.open_dataset(RES / fname)
    lon, depth = ds["lon"].values, ds["depth"].values.copy()
    depth[0] = 0.0  # draw the uppermost level up to the surface
    tr = ds["trend"].values
    cmap = plt.get_cmap("BrBG_r", len(LEV) + 1)
    cf = ax.contourf(
        lon,
        depth,
        tr,
        levels=LEV,
        cmap=cmap,
        extend="both",
        norm=colors.BoundaryNorm(LEV, cmap.N, extend="both"),
    )
    ax.contour(
        lon, depth, tr, levels=[-0.03], colors="k", linewidths=0.35, linestyles="--"
    )
    ax.contour(
        lon, depth, tr, levels=[0.03], colors="k", linewidths=0.35, linestyles="-"
    )
    p = ds["p"].values
    step = max(1, len(lon) // 60)
    lon2, dep2 = np.meshgrid(lon[::step], depth)
    sig = (p[:, ::step] < 0.05) & np.isfinite(tr[:, ::step])
    ax.plot(lon2[sig], dep2[sig], ".", ms=0.6, color="black", alpha=0.6)
    for lo in [b[0] for b in BANDS[1:]]:
        ax.axvline(lo, color="0.3", lw=0.4, ls=":")
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(1500, 0)
    ax.set_yticks([0, 100, 300, 600, 1000, 1500])
    ax.set_xlim(-54, 19.4)
    ax.set_facecolor("#4d4d4d")
    ax.set_xlabel("Longitude (degrees east)")
    for (lo, hi), lab in zip(BANDS, BAND_LABELS, strict=False):
        ax.text(
            0.5 * (lo + hi),
            1.01,
            lab,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    if add_cbar:
        cb = fig.colorbar(
            cf, ax=ax, pad=0.02, fraction=0.04, ticks=LEV[::2].tolist() + [0.15]
        )
        cb.set_label("Salinity trend (PSU decade$^{-1}$)", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)


def main() -> None:
    set_style()
    struct = json.loads((AN / "r1_salinification_structure.json").read_text())
    fig = plt.figure(figsize=(TWO_COL, 6.4))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 0.95, 0.9],
        hspace=0.48,
        left=0.07,
        right=0.97,
        top=0.97,
        bottom=0.06,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.3)
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.12)

    # a
    ax = fig.add_subplot(top[0])
    for key, _label, col in PROD:
        ds = (
            xr.open_dataset(RES / f"paper3r1_section_{key}.nc")
            .groupby("time.year")
            .mean()
        )
        yrs = ds["year"].values
        for var, ls in (("S_north", "-"), ("S_south", (0, (2.5, 1.2)))):
            v = ds[var].values
            ax.plot(
                yrs, v - v.mean(), color=col, lw=0.9 if key != "ecco" else 0.6, ls=ls
            )
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(1958, 2026)
    ax.set_xlabel("Year")
    ax.set_ylabel("Limb salinity anomaly (PSU)")
    handles = [plt.Line2D([], [], color=c, lw=0.9, label=lab) for _, lab, c in PROD]
    handles += [
        plt.Line2D([], [], color="0.3", lw=0.9, label="northward limb"),
        plt.Line2D(
            [], [], color="0.3", lw=0.9, ls=(0, (2.5, 1.2)), label="southward limb"
        ),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2)
    panel_letter(ax, "a", x=-0.08)

    # b
    ax = fig.add_subplot(top[1])
    for key, _label, col in PROD:
        ds = xr.open_dataset(RES / f"paper3r1_section_{key}.nc")
        y0, y1 = (1992, 2017) if key == "ecco" else (1993, 2025)
        tr, pv = trend_profile(ds, y0, y1)
        z = ds["depth"].values
        ax.plot(tr, z, color=col, lw=0.6)
        sig = np.where(pv < 0.05, tr, np.nan)
        ax.plot(sig, z, color=col, lw=1.6)
    ax.axhspan(0, 1360, color="0.85", lw=0, zorder=0)
    ax.axhspan(4100, 5800, color="0.85", lw=0, zorder=0)
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(5800, 0)
    ax.set_yticks([0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel("Salinity trend (PSU decade$^{-1}$)")
    ax.set_ylabel("Depth (m)")
    panel_letter(ax, "b", x=-0.3)

    # c, d
    axc = fig.add_subplot(mid[0])
    section_panel(
        axc, "paper3r1_salinity_trend_section_glorys12_2005-2024.nc", fig, False
    )
    axc.set_ylabel("Depth (m)")
    panel_letter(axc, "c", x=-0.12, y=1.06)
    axd = fig.add_subplot(mid[1], sharey=axc)
    section_panel(axd, "paper3r1_salinity_trend_section_en4_2005-2024.nc", fig, True)
    plt.setp(axd.get_yticklabels(), visible=False)
    panel_letter(axd, "d", x=-0.04, y=1.06)

    # e
    ax = fig.add_subplot(outer[2])
    sets = [
        (
            "EN4.2.2",
            struct["en4"]["frozen"]["oras5"],
            struct["en4"]["band_width_fraction"],
            "black",
        ),
        (
            "ORAS5",
            struct["oras5"]["windows"]["2005-2024"],
            struct["oras5"]["windows"]["2005-2024"]["band_width_fraction"],
            C_ORAS5,
        ),
        (
            "GLORYS12V1",
            struct["glorys12"]["windows"]["2005-2024"],
            struct["glorys12"]["windows"]["2005-2024"]["band_width_fraction"],
            C_GLORYS,
        ),
    ]
    layers = ["0-100 m", "100-500 m", "500-1300 m"]
    shades = ["0.15", "0.5", "0.82"]
    bands = list(sets[1][1]["northward_by_band"].keys())
    width = 0.26
    for j, (_name, rec, wfrac, col) in enumerate(sets):
        tab = rec.get("northward_by_band_and_layer")
        total = rec.get(
            "S_north_frozen_trend_sum_of_cells", rec.get("S_north_frozen_trend_direct")
        )
        for i, b in enumerate(bands):
            x = i + (j - 1) * width
            bottom = 0.0
            for ly, sh in zip(layers, shades, strict=False):
                if tab is not None:
                    val = tab[f"{b} | {ly}"]
                else:
                    val = (
                        rec["northward_by_band"][b]
                        * rec["northward_by_layer"][ly]
                        / max(sum(rec["northward_by_layer"][q] for q in layers), 1e-12)
                    )
                ax.bar(
                    x,
                    val,
                    width * 0.9,
                    bottom=bottom if val >= 0 else 0.0,
                    color=sh,
                    edgecolor=col,
                    lw=0.8,
                )
                bottom += max(val, 0.0)
            ax.plot(x, total * wfrac[b], "D", ms=2.6, mfc="white", mec=col, mew=0.8)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(
        [
            "western boundary\n(54 to 45$^\\circ$W)",
            "western interior\n(45 to 20$^\\circ$W)",
            "eastern interior\n(20$^\\circ$W to 0$^\\circ$)",
            "Cape Basin\n(0$^\\circ$ to 19$^\\circ$E)",
        ]
    )
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Contribution to the $S_{\\uparrow}$ trend\n(PSU decade$^{-1}$)")
    handles = [
        Patch(fc=sh, ec="black", lw=0.5, label=ly.replace("-", "–"))
        for ly, sh in zip(layers, shades, strict=False)
    ]
    handles += [Patch(fc="white", ec=c, lw=0.8, label=n) for n, _, _, c in sets]
    handles += [
        plt.Line2D(
            [],
            [],
            marker="D",
            ls="none",
            ms=2.6,
            mfc="white",
            mec="0.3",
            label="uniform along the section",
        )
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=5.3)
    panel_letter(ax, "e", x=-0.05)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=600)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
