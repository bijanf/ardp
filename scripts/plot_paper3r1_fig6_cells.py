#!/usr/bin/env python3
"""Figure 6: the overturning cells defined by the streamfunction.

a  Time-mean streamfunction of the velocity after removal of the net section
   transport; circles mark its maximum, the base of the northward upper limb,
   and squares the depth at which it returns to zero, the boundary between the
   upper and abyssal cells.
b  Annual freshwater transport of the upper cell (solid) and abyssal cell
   (dashed).
c  Trend of F_ovS over each product's own record and its split between the two
   cells, with 95% block-bootstrap intervals.
d  Trend of the upper-cell freshwater transport split into its transport and
   salinity terms, with 95% intervals.

Writes PAPER_3_v2/figures_r1/Fig6_cells.{pdf,png}.
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
from matplotlib.patches import Patch  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    C_SALINITY,
    C_TOTAL,
    C_TRANSPORT,
    TWO_COL,
    panel_letter,
    set_style,
)

RES = REPO / "data" / "results"
AN = REPO / "PAPER_3_v2" / "analysis"
OUT = REPO / "PAPER_3_v2" / "figures_r1" / "Fig6_cells"
PROD = [
    ("oras5", "ORAS5", C_ORAS5),
    ("glorys12", "GLORYS12V1", C_GLORYS),
    ("ecco", "ECCO-V4r4", C_ECCO),
]


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def mean_psi(key: str):
    ds = xr.open_dataset(RES / f"paper3r1_section_{key}.nc")
    e3t = ds["e3t"].values
    q = ds["V_bc"].mean("time").values * e3t
    psi = np.concatenate([[0.0], np.cumsum(q)]) / 1e6
    zw = np.concatenate([[0.0], np.cumsum(e3t)])
    kmax = int(np.argmax(psi))
    i = next(i for i in range(kmax + 1, len(psi) - 1) if psi[i] < 0)
    z0 = zw[i - 1] + psi[i - 1] / (psi[i - 1] - psi[i]) * (zw[i] - zw[i - 1])
    return psi, zw, zw[kmax], psi[kmax], z0


def main() -> None:
    set_style()
    cells = json.loads((AN / "r1_cells.json").read_text())
    fig = plt.figure(figsize=(TWO_COL, 4.4))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.75, 1.3, 1.0],
        height_ratios=[1, 1],
        wspace=0.45,
        hspace=0.55,
        left=0.07,
        right=0.99,
        top=0.93,
        bottom=0.09,
    )

    # a
    ax = fig.add_subplot(gs[:, 0])
    for key, label, col in PROD:
        psi, zw, zmax, pmax, z0 = mean_psi(key)
        ax.plot(psi, zw, color=col, lw=0.9, label=label)
        ax.plot(pmax, zmax, "o", color=col, ms=3)
        ax.plot(0, z0, "s", color=col, ms=3)
    ax.axvline(0, color="black", lw=0.4)
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(5800, 0)
    ax.set_yticks([0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel("Streamfunction (Sv)")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="lower right")
    panel_letter(ax, "a", x=-0.35, y=1.02)

    # b
    ax = fig.add_subplot(gs[0, 1:])
    for key, _label, col in PROD:
        ds = (
            xr.open_dataset(RES / f"paper3r1_cells_{key}.nc")
            .groupby("time.year")
            .mean()
        )
        ax.plot(ds["year"], ds["F_up"], color=col, lw=0.9)
        ax.plot(ds["year"], ds["F_ab"], color=col, lw=0.8, ls=(0, (2.5, 1.2)))
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(1958, 2026)
    ax.set_xlabel("Year")
    ax.set_ylabel("Freshwater transport (Sv)")
    ax.legend(
        handles=[
            plt.Line2D([], [], color="0.3", lw=0.9, label="upper cell"),
            plt.Line2D(
                [], [], color="0.3", lw=0.8, ls=(0, (2.5, 1.2)), label="abyssal cell"
            ),
        ],
        loc="lower left",
        ncol=2,
    )
    panel_letter(ax, "b", x=-0.08, y=1.02)

    # c
    ax = fig.add_subplot(gs[1, 1])
    width = 0.26
    for j, (var, _name, fc) in enumerate(
        (
            ("trend_F_ov", "full column", C_TOTAL),
            ("trend_F_up", "upper cell", "0.55"),
            ("trend_F_ab", "abyssal cell", "white"),
        )
    ):
        for i, (key, _, _) in enumerate(PROD):
            t = cells[key]["windows"]["own"][var]
            x = i + (j - 1) * width
            ax.bar(x, t["trend"], width * 0.9, color=fc, edgecolor="black", lw=0.5)
            ax.errorbar(
                x,
                t["trend"],
                yerr=[[t["trend"] - t["ci"][0]], [t["ci"][1] - t["trend"]]],
                fmt="none",
                ecolor="0.2",
                elinewidth=0.5,
                capsize=1.0,
            )
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(range(len(PROD)))
    ax.set_xticklabels([p[1] for p in PROD])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Trend (mSv yr$^{-1}$)")
    ax.legend(
        handles=[
            Patch(fc=C_TOTAL, ec="black", lw=0.5, label="full column"),
            Patch(fc="0.55", ec="black", lw=0.5, label="upper cell"),
            Patch(fc="white", ec="black", lw=0.5, label="abyssal cell"),
        ],
        loc="lower left",
        fontsize=5.3,
    )
    panel_letter(ax, "c", x=-0.2, y=1.02)

    # d
    ax = fig.add_subplot(gs[1, 2])
    for j, (tk, ck, _name, fc) in enumerate(
        (
            ("trend_total_mSv_per_yr", "ci_total_mSv_per_yr", "upper cell", "0.55"),
            ("trend_T_mSv_per_yr", "ci_T_mSv_per_yr", "transport term", C_TRANSPORT),
            ("trend_S_mSv_per_yr", "ci_S_mSv_per_yr", "salinity term", C_SALINITY),
        )
    ):
        for i, (key, _, _) in enumerate(PROD):
            a = cells[key]["windows"]["own"]["upper_cell_attribution"]
            x = i + (j - 1) * width
            ax.bar(x, a[tk], width * 0.9, color=fc, edgecolor="black", lw=0.5)
            ax.errorbar(
                x,
                a[tk],
                yerr=[[a[tk] - a[ck][0]], [a[ck][1] - a[tk]]],
                fmt="none",
                ecolor="0.2",
                elinewidth=0.5,
                capsize=1.0,
            )
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(range(len(PROD)))
    ax.set_xticklabels([p[1] for p in PROD])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Upper-cell trend (mSv yr$^{-1}$)")
    ax.legend(
        handles=[
            Patch(fc="0.55", ec="black", lw=0.5, label="upper cell"),
            Patch(fc=C_TRANSPORT, ec="black", lw=0.5, label="transport term"),
            Patch(fc=C_SALINITY, ec="black", lw=0.5, label="salinity term"),
        ],
        loc="lower left",
        fontsize=5.3,
    )
    panel_letter(ax, "d", x=-0.25, y=1.02)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=600)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
