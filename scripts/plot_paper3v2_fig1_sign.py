#!/usr/bin/env python3
"""Figure 1: the sign of the overturning freshwater transport at 34.5 S.

(a) Annual F_ovS in four reanalyses. ORAS5, GLORYS12V1 and ECCO-V4r4 are
    recomputed here from monthly fields on the contiguous Atlantic section;
    SODA 3.15.2 is carried over from the earlier annual-field pipeline and is
    drawn in grey-violet because it is not computed on the same basis.
(b) Record mean and most recent decade with 95% circular block-bootstrap
    intervals. The pale bar behind each interval spans the intervals obtained
    for block lengths of 2 to 20 years, so the reader can see that the sign
    does not depend on that choice.
(c) Record-mean F_ovS against section latitude in the two eddy-permitting
    products over the matched 1993-2025 window, with the same intervals.

Supersedes plot_paper3v2_fig1_regime.py. Reads
PAPER_3_v2/analysis/{attribution,latitude_sensitivity}.json.
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
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    C_SODA,
    TWO_COL,
    panel_letter,
    save,
    set_style,
)

ANALYSIS = REPO / "PAPER_3_v2" / "analysis" / "attribution.json"
LATFILE = REPO / "PAPER_3_v2" / "analysis" / "latitude_sensitivity.json"
SODA = REPO / "data" / "results" / "soda_f_ovs.nc"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig1_sign"

ORDER = [("oras5", C_ORAS5), ("glorys12", C_GLORYS), ("ecco", C_ECCO)]


def main() -> None:
    set_style()
    d = json.loads(ANALYSIS.read_text())
    lat = json.loads(LATFILE.read_text())

    fig = plt.figure(figsize=(TWO_COL, 2.5))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.45, 1.0, 0.9],
        wspace=0.42,
        left=0.06,
        right=0.995,
        top=0.90,
        bottom=0.17,
    )

    # ── a: time series ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    with xr.open_dataset(SODA) as ds:
        # SODA is archived already annual, indexed by year rather than time.
        ax.plot(
            ds["year"].values,
            ds["F_ovS"].values,
            color=C_SODA,
            lw=0.7,
            alpha=0.7,
            label="SODA 3.15.2",
        )
    for key, colour in ORDER:
        r = d[key]
        ax.plot(r["years"], r["F_ov_Sv"], color=colour, lw=0.9, label=r["label"])
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("$F_{\\mathrm{ov}}^{S}$ (Sv)")
    ax.set_xlim(1958, 2026)
    # The GLORYS12V1 and ECCO traces run through the lower-left quadrant, so
    # the legend goes above the axes rather than on top of them.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.4,
    )
    panel_letter(ax, "a")

    # ── b: means with block-length envelope ───────────────────────────
    ax = fig.add_subplot(gs[1])
    labels, ypos = [], []
    for i, (key, colour) in enumerate(ORDER):
        r = d[key]
        y = len(ORDER) - i
        blocks = np.array(list(r["mean_ci_by_block"].values()))
        ax.plot([blocks.min(), blocks.max()], [y, y], color=colour, lw=3.4, alpha=0.22)
        ci = r["mean_ci_by_block"]["5"]
        ax.plot(ci, [y, y], color=colour, lw=1.1)
        ax.plot(r["mean_F_ov_Sv"], y, "o", color=colour, ms=3.0)
        rd = r["recent_decade"]
        ax.plot(rd["ci"], [y - 0.34, y - 0.34], color=colour, lw=1.1)
        ax.plot(rd["mean_F_ov_Sv"], y - 0.34, "D", color=colour, ms=2.6)
        labels.append(r["label"])
        ypos.append(y - 0.17)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(0.35, len(ORDER) + 0.45)
    ax.set_xlabel("$F_{\\mathrm{ov}}^{S}$ (Sv)")
    handles = [
        plt.Line2D(
            [], [], color="0.3", marker="o", ms=3.0, lw=1.1, label="record mean"
        ),
        plt.Line2D(
            [], [], color="0.3", marker="D", ms=2.6, lw=1.1, label="recent decade"
        ),
    ]
    ax.legend(handles=handles, loc="upper left")
    panel_letter(ax, "b", x=-0.36)

    # ── c: latitude sensitivity ───────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    for prod, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        entries = sorted(lat[prod].values(), key=lambda s: s["section_latitude"])
        lats = np.array([s["section_latitude"] for s in entries])
        means = np.array([s["mean_F_ov_Sv"] for s in entries])
        lo = np.array([s["ci_F_ov_Sv"][0] for s in entries])
        hi = np.array([s["ci_F_ov_Sv"][1] for s in entries])
        # Open markers where the bootstrap interval includes zero, matching the
        # solid/dashed convention used for significance elsewhere.
        crosses = lo * hi <= 0
        ax.plot(means, lats, "-", color=colour, lw=0.9)
        ax.plot(means[~crosses], lats[~crosses], "o", color=colour, ms=2.6)
        ax.plot(
            means[crosses],
            lats[crosses],
            "o",
            ms=2.6,
            mfc="white",
            mec=colour,
            mew=0.7,
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
    ax.set_xlabel("$F_{\\mathrm{ov}}^{S}$ (Sv)")
    ax.set_ylabel("Section latitude ($^\\circ$S)")
    ax.set_yticks([-34.5, -32, -30, -28, -25])
    ax.set_yticklabels(["34.5", "32", "30", "28", "25"])
    panel_letter(ax, "c", x=-0.40)

    save(fig, OUTBASE)


if __name__ == "__main__":
    main()
