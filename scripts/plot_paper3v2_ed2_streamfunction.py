#!/usr/bin/env python3
"""Extended Data Figure 2: Atlantic overturning streamfunction in both products.

The figure previously carried under this caption showed ORAS5 only, over
2005-2024 only, with a 250 m integration-restart experiment and profiles at
10 N and 26.5 N, none of which the caption described and none of which the
manuscript used. This replaces it with the figure the caption promises: the
time-mean depth-space streamfunction for ORAS5 and GLORYS12V1 with the 34.5 S
section marked, plus the streamfunction profile at that section, which is where
every transport quantity in the paper is evaluated.

Writes PAPER_3_v2/extended_data/ED2_streamfunction.{pdf,png}.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
OUTBASE = REPO / "PAPER_3_v2" / "extended_data" / "ED2_streamfunction"

SECTION_LAT = -34.5
LEVELS = np.arange(-20, 22, 2)

PRODUCTS = [
    ("ORAS5", "moc_streamfunction_oras5_1993_2024.npz", "1993-2024"),
    ("GLORYS12V1", "moc_streamfunction_glorys12_2005_2024.npz", "2005-2024"),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "lines.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.06,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    set_style()
    fig = plt.figure(figsize=(7.09, 4.6))
    # A dedicated middle column holds the colourbar so it cannot overlap the
    # maps or the profile panel.
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[3.0, 0.10, 1.05],
        hspace=0.30,
        wspace=0.30,
        left=0.07,
        right=0.99,
        top=0.96,
        bottom=0.09,
    )

    norm = BoundaryNorm(LEVELS, ncolors=256, clip=False)
    profiles = []
    mesh = None

    for row, (label, fname, period) in enumerate(PRODUCTS):
        d = np.load(RESULTS / fname, allow_pickle=True)
        psi, lat, depth = d["psi"], d["lat"], d["depth"]
        # The ORAS5 tripolar grid folds at the north pole, so its row-mean
        # latitude is not monotonic; sort before drawing.
        order = np.argsort(lat)
        lat, psi = lat[order], psi[:, order]

        ax = fig.add_subplot(gs[row, 0])
        mesh = ax.pcolormesh(lat, depth, psi, cmap="RdBu_r", norm=norm, shading="auto")
        ax.contour(lat, depth, psi, levels=[0.0], colors="k", linewidths=0.6)
        ax.axvline(SECTION_LAT, color="k", lw=0.6, ls="--")
        ax.set_facecolor("#4d4d4d")
        ax.invert_yaxis()
        ax.set_xlim(-35, 70)
        ax.set_ylim(5500, 0)
        ax.set_ylabel("Depth (m)")
        if row == 1:
            ax.set_xlabel("Latitude ($^\\circ$N)")
        else:
            ax.set_xticklabels([])
        panel_letter(ax, "ab"[row])

        j = int(np.abs(lat - SECTION_LAT).argmin())
        profiles.append((label, psi[:, j], depth, period))

    cax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(mesh, cax=cax, ticks=LEVELS[::2])
    cb.set_label("$\\Psi$ (Sv)")
    cb.outline.set_linewidth(0.5)

    ax = fig.add_subplot(gs[:, 2])
    for (label, prof, depth, _), colour in zip(
        profiles, ("#20558a", "#b3541e"), strict=False
    ):
        ax.plot(prof, depth, color=colour, label=label)
    ax.axvline(0.0, color="0.4", lw=0.5)
    ax.invert_yaxis()
    ax.set_ylim(5500, 0)
    ax.set_xlabel("$\\Psi$ at 34.5$^\\circ$S (Sv)")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="lower right")
    panel_letter(ax, "c")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTBASE}.{ext}", bbox_inches="tight")
    print(f"wrote {OUTBASE}.pdf and .png")
    for label, prof, depth, period in profiles:
        k = int(np.nanargmax(prof))
        print(
            f"  {label} ({period}): upper-cell maximum {prof[k]:.2f} Sv at "
            f"{depth[k]:.0f} m"
        )


if __name__ == "__main__":
    main()
