#!/usr/bin/env python3
"""Figure 7: CMIP6 at 34.5 S on the reanalysis basis.

a  Mean F_ovS over 1993-2014 in 25 CMIP6 historical simulations (bars) against
   the three reanalyses over the same years (vertical lines).
b  The same models and reanalyses in the plane of the two factors, northward
   limb transport T and limb salinity contrast dS, over contours of
   F_ovS = -T dS / S0.
c  Change of the overturning strength at 34.5 S against that at 26.5 N from
   1850-1900 to 2081-2100 under historical and ssp585 forcing.
d  Mean F_ovS over 1950-1980 against the change of the overturning at 26.5 N,
   coloured by the 1850-1900 overturning strength at 26.5 N.

Writes PAPER_3_v2/figures_r1/Fig7_models.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    TWO_COL,
    panel_letter,
    set_style,
)

AN = REPO / "PAPER_3_v2" / "analysis"
OUT = REPO / "PAPER_3_v2" / "figures_r1" / "Fig7_models"
S0 = 35.0
REAN = [
    ("oras5", "ORAS5", C_ORAS5, "o"),
    ("glorys12", "GLORYS12V1", C_GLORYS, "s"),
    ("ecco", "ECCO-V4r4", C_ECCO, "^"),
]


def main() -> None:
    set_style()
    d = json.loads((AN / "r1_cmip6.json").read_text())
    rows = d["per_model"]
    rw = d["reanalysis_windows"]
    fig = plt.figure(figsize=(TWO_COL, 4.9))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.15, 1.0],
        wspace=0.32,
        hspace=0.42,
        left=0.12,
        right=0.98,
        top=0.97,
        bottom=0.08,
    )

    # a
    ax = fig.add_subplot(gs[0, 0])
    rr = sorted(rows, key=lambda r: r["F_1993-2014"])
    y = np.arange(len(rr))
    f = np.array([r["F_1993-2014"] for r in rr])
    ax.barh(
        y,
        f,
        color=np.where(f > 0, "0.35", "0.75"),
        edgecolor="black",
        lw=0.3,
        height=0.75,
    )
    ax.plot([r["F_1950-1980"] for r in rr], y, "|", color="black", ms=4, mew=0.8)
    for key, label, col, _ in REAN:
        ax.axvline(
            rw[key]["1993-2014"]["F_ov"],
            color=col,
            lw=1.0,
            ls=(0, (3, 1.5)),
            label=label,
        )
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in rr], fontsize=5)
    ax.set_ylim(-0.7, len(rr) - 0.3)
    ax.set_xlabel("Mean $F_{\\mathrm{ovS}}$, 1993–2014 (Sv)")
    handles = [
        plt.Line2D([], [], color=c, lw=1.0, ls=(0, (3, 1.5)), label=lab)
        for _, lab, c, _ in REAN
    ]
    handles.append(
        plt.Line2D(
            [], [], color="black", marker="|", ls="none", ms=4, label="model, 1950–1980"
        )
    )
    ax.legend(handles=handles, loc="lower right", fontsize=5.3)
    panel_letter(ax, "a", x=-0.32)

    # b
    ax = fig.add_subplot(gs[0, 1])
    t_grid = np.linspace(6, 32, 200)
    d_grid = np.linspace(-1.2, 1.0, 200)
    tt, dd = np.meshgrid(t_grid, d_grid)
    ff = -tt * dd / S0
    lev = np.array(
        [-0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    )
    cmap = plt.get_cmap("PuOr", len(lev) - 1)
    cf = ax.contourf(
        tt,
        dd,
        ff,
        levels=lev,
        cmap=cmap,
        norm=colors.BoundaryNorm(lev, cmap.N),
        alpha=0.85,
    )
    cs = ax.contour(tt, dd, ff, levels=lev, colors="0.35", linewidths=0.3)
    ax.clabel(cs, levels=[-0.2, -0.1, 0.1, 0.2], fmt="%.1f", fontsize=4.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.plot(
        [r["T_1993-2014"] for r in rows],
        [r["dS_1993-2014"] for r in rows],
        "o",
        ms=2.6,
        mfc="white",
        mec="black",
        mew=0.5,
        label="CMIP6 models",
    )
    for key, label, col, mk in REAN:
        w = rw[key]["1993-2014"]
        ax.plot(w["T"], w["dS"], mk, ms=5, mfc=col, mec="black", mew=0.5, label=label)
    dsv = np.array(
        [r["dS_1993-2014"] for r in rows] + [rw[k]["1993-2014"]["dS"] for k, *_ in REAN]
    )
    tv = np.array(
        [r["T_1993-2014"] for r in rows] + [rw[k]["1993-2014"]["T"] for k, *_ in REAN]
    )
    ax.set_xlim(max(8, tv.min() - 1.5), min(30, tv.max() + 1.5))
    ax.set_ylim(min(-0.9, dsv.min() - 0.05), max(0.7, dsv.max() + 0.05))
    ax.set_xlabel("Northward limb transport $T$, 1993–2014 (Sv)")
    ax.set_ylabel("Limb salinity contrast $\\Delta S$ (PSU)")
    cb = fig.colorbar(
        cf, ax=ax, pad=0.02, fraction=0.05, ticks=[-0.4, -0.2, 0, 0.2, 0.4]
    )
    cb.set_label("$F_{\\mathrm{ovS}}$ (Sv)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.legend(
        loc="upper right", fontsize=5.3, framealpha=0.85, frameon=True, edgecolor="none"
    )
    panel_letter(ax, "b", x=-0.2)

    # c
    ax = fig.add_subplot(gs[1, 0])
    sub = [
        r
        for r in rows
        if r.get("dpsi34") is not None
        and r.get("damoc26") is not None
        and np.isfinite(r.get("dpsi34", np.nan))
        and np.isfinite(r.get("damoc26", np.nan))
    ]
    x = np.array([r["damoc26"] for r in sub])
    yv = np.array([r["dpsi34"] for r in sub])
    ax.plot(x, yv, "o", ms=3, mfc="0.4", mec="black", mew=0.4)
    lim = [-14, 2]
    ax.plot(lim, lim, color="0.5", lw=0.5, ls=":")
    coef = np.polyfit(x, yv, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(coef, xx), color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.3)
    ax.axvline(0, color="black", lw=0.3)
    ax.set_xlim(*lim)
    ax.set_ylim(-10, 5)
    ax.set_xlabel("Change at 26.5$^\\circ$N (Sv)")
    ax.set_ylabel("Change at 34.5$^\\circ$S (Sv)")
    panel_letter(ax, "c", x=-0.32)

    # d
    ax = fig.add_subplot(gs[1, 1])
    sub = [
        r
        for r in rows
        if r.get("damoc26") is not None and np.isfinite(r.get("damoc26", np.nan))
    ]
    sc = ax.scatter(
        [r["damoc26"] for r in sub],
        [r["F_1950-1980"] for r in sub],
        c=[r["amoc26_1850-1900"] for r in sub],
        cmap="cividis",
        s=12,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("Change at 26.5$^\\circ$N, 1850–1900 to 2081–2100 (Sv)")
    ax.set_ylabel("Mean $F_{\\mathrm{ovS}}$, 1950–1980 (Sv)")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Overturning at 26.5$^\\circ$N,\n1850–1900 (Sv)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    panel_letter(ax, "d", x=-0.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=600)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
