#!/usr/bin/env python3
"""Figure 2: the sign of the overturning freshwater transport and its azonal companion.

a  Annual F_ovS at 34.5 S in ORAS5, GLORYS12V1 and ECCO-V4r4.
b  Record mean (circles) and mean of the last ten years (diamonds) with 95%
   intervals from the effective sample size (paper3r1_stats.py).
c  Mean F_ovS over 1993-2025 against section latitude; open markers where the
   interval includes zero.
d  Record means of the overturning, azonal and total freshwater transport.
e  Their trends over each product's own record with 95% intervals of the
   trend test; hatched bars where the trend is not significant.

Reads PAPER_3_v2/analysis/{attribution,latitude_sensitivity,r1_common_window}.json.
Writes PAPER_3_v2/figures_r1/Fig2_sign.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    TWO_COL,
    panel_letter,
    set_style,
)

AN = REPO / "PAPER_3_v2" / "analysis"
OUT = REPO / "PAPER_3_v2" / "figures_r1" / "Fig2_sign"
ORDER = [("oras5", C_ORAS5), ("glorys12", C_GLORYS), ("ecco", C_ECCO)]
COMP = [
    ("F_ov", "overturning", "#4d4d4d"),
    ("F_az", "azonal", "#bdbdbd"),
    ("F_total", "total", "white"),
]


def main() -> None:
    set_style()
    d = json.loads((AN / "attribution.json").read_text())
    lat = json.loads((AN / "latitude_sensitivity.json").read_text())
    cw = json.loads((AN / "r1_common_window.json").read_text())

    fig = plt.figure(figsize=(TWO_COL, 4.6))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.45, 1.0, 0.9],
        height_ratios=[1, 1],
        wspace=0.42,
        hspace=0.55,
        left=0.07,
        right=0.995,
        top=0.94,
        bottom=0.09,
    )

    # a
    ax = fig.add_subplot(gs[0, 0])
    for key, colour in ORDER:
        r = d[key]
        ax.plot(r["years"], r["F_ov_Sv"], color=colour, lw=0.9, label=r["label"])
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("$F_{\\mathrm{ovS}}$ (Sv)")
    ax.set_xlim(1958, 2026)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.4,
    )
    panel_letter(ax, "a", x=-0.12)

    # b
    ax = fig.add_subplot(gs[0, 1])
    labels, ypos = [], []
    for i, (key, colour) in enumerate(ORDER):
        r = d[key]
        y = len(ORDER) - i
        ax.plot(r["mean_stats"]["ci"], [y, y], color=colour, lw=1.1)
        ax.plot(r["mean_F_ov_Sv"], y, "o", color=colour, ms=3.0)
        rd = r["recent_decade"]
        ax.plot(rd["mean_F_ov_Sv"], y - 0.34, "D", color=colour, ms=2.6)
        labels.append(r["label"])
        ypos.append(y - 0.17)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(0.35, len(ORDER) + 0.45)
    ax.set_xlabel("$F_{\\mathrm{ovS}}$ (Sv)")
    ax.legend(
        handles=[
            plt.Line2D(
                [], [], color="0.3", marker="o", ms=3.0, lw=1.1, label="record mean"
            ),
            plt.Line2D(
                [],
                [],
                color="0.3",
                marker="D",
                ms=2.6,
                ls="none",
                label="last ten years",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.4,
    )
    panel_letter(ax, "b", x=-0.42)

    # c
    ax = fig.add_subplot(gs[0, 2])
    for prod, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        entries = sorted(lat[prod].values(), key=lambda s: s["section_latitude"])
        lats = np.array([s["section_latitude"] for s in entries])
        means = np.array([s["mean_F_ov_Sv"] for s in entries])
        lo = np.array([s["ci_F_ov_Sv"][0] for s in entries])
        hi = np.array([s["ci_F_ov_Sv"][1] for s in entries])
        crosses = lo * hi <= 0
        ax.plot(means, lats, "-", color=colour, lw=0.9)
        ax.plot(means[~crosses], lats[~crosses], "o", color=colour, ms=2.6)
        ax.plot(
            means[crosses], lats[crosses], "o", ms=2.6, mfc="white", mec=colour, mew=0.7
        )
        ax.errorbar(
            means,
            lats,
            xerr=[means - lo, hi - means],
            fmt="none",
            ecolor=colour,
            elinewidth=0.7,
            capsize=1.2,
        )
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_xlabel("$F_{\\mathrm{ovS}}$ (Sv)")
    ax.set_ylabel("Section latitude ($^\\circ$S)")
    ax.set_yticks([-34.5, -32, -30, -28, -25])
    ax.set_yticklabels(["34.5", "32", "30", "28", "25"])
    panel_letter(ax, "c", x=-0.45)

    # d and e
    width = 0.24
    x = np.arange(len(ORDER))
    axd = fig.add_subplot(gs[1, 0:1])
    axe = fig.add_subplot(gs[1, 1:3])
    for j, (comp, name, fc) in enumerate(COMP):
        means, trends, lo, hi, sig = [], [], [], [], []
        for key, _ in ORDER:
            w = cw[key]["own"]
            means.append(w[f"mean_{comp}_Sv"])
            t = w[f"trend_{comp}"]
            trends.append(t["trend"])
            lo.append(t["trend"] - t["ci"][0])
            hi.append(t["ci"][1] - t["trend"])
            sig.append(t["p_santer"] < 0.05)
        xs = x + (j - 1) * width
        axd.bar(
            xs, means, width * 0.92, color=fc, edgecolor="black", lw=0.5, label=name
        )
        for xi, ti, s in zip(xs, trends, sig, strict=False):
            axe.bar(xi, ti, width * 0.92, color=fc, edgecolor="black", lw=0.5)
            if not s:
                axe.bar(
                    xi,
                    ti,
                    width * 0.92,
                    color="none",
                    edgecolor="0.55",
                    lw=0.0,
                    hatch="/////",
                )
        ylim_e = (-9.0, 12.0)
        t_arr, lo_arr, hi_arr = np.array(trends), np.array(lo), np.array(hi)
        low_end = np.maximum(t_arr - lo_arr, ylim_e[0])
        high_end = np.minimum(t_arr + hi_arr, ylim_e[1])
        axe.errorbar(
            xs,
            trends,
            yerr=[t_arr - low_end, high_end - t_arr],
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=1.2,
        )
        # Intervals beyond the axis are drawn to its edge with an arrow.
        for xi, a, b in zip(xs, t_arr - lo_arr, t_arr + hi_arr, strict=False):
            for edge, start, cond in (
                (ylim_e[0], ylim_e[0] + 1.2, a < ylim_e[0]),
                (ylim_e[1], ylim_e[1] - 1.2, b > ylim_e[1]),
            ):
                if cond:
                    axe.annotate(
                        "",
                        xy=(xi, edge),
                        xytext=(xi, start),
                        arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "black"},
                    )
        axe.set_ylim(*ylim_e)
    for ax in (axd, axe):
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([d[k]["label"] for k, _ in ORDER])
        ax.tick_params(axis="x", length=0)
    axd.set_ylabel("Mean freshwater transport (Sv)")
    axe.set_ylabel("Trend (mSv yr$^{-1}$)")
    from matplotlib.patches import Patch

    axd.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2)
    axe.legend(
        handles=[Patch(fc=fc, ec="black", lw=0.5, label=name) for _, name, fc in COMP]
        + [
            Patch(fc="white", ec="0.55", hatch="/////", lw=0.0, label="not significant")
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        handlelength=1.2,
    )
    panel_letter(axd, "d", x=-0.12)
    panel_letter(axe, "e", x=-0.07)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=600)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
