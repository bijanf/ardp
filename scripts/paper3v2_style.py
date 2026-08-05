"""Shared figure style for the PAPER_3_v2 figure set.

Springer Nature artwork rules: sans-serif, 5 to 7 pt text, RGB, vector PDF with
TrueType fonts, single-column 88 mm (3.46 in) or double-column 180 mm (7.09 in).
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

ONE_COL = 3.46
TWO_COL = 7.09

C_ORAS5 = "#20558a"
C_GLORYS = "#b3541e"
C_ECCO = "#3f7d52"
C_SODA = "#8d6ea8"
C_TRANSPORT = "#1B5E8C"
C_SALINITY = "#F0A93B"
C_TOTAL = "#111111"
C_GREY = "#9A9A9A"

PRODUCT_COLOURS = {
    "ORAS5": C_ORAS5,
    "GLORYS12V1": C_GLORYS,
    "ECCO-V4r4": C_ECCO,
    "SODA 3.15.2": C_SODA,
}


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
            "legend.fontsize": 5.5,
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


def panel_letter(ax, letter: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save(fig, outbase) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{outbase}.{ext}", bbox_inches="tight")
    print(f"wrote {outbase}.pdf and .png")
