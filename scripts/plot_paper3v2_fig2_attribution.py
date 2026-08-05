#!/usr/bin/env python3
"""Figure 2: what changes F_ovS, on an exact factorisation with uncertainty.

(a) The two independently measured factors of F_ovS = -(1/S0) T dS: the
    overturning exchange transport T and the transport-weighted salinity
    contrast between the limbs, each as an anomaly from its record mean so the
    two can share an axis.
(b) Trend of F_ovS split into a transport contribution and a salinity
    contribution, with 95% block-bootstrap intervals, over each product's own
    record and over the Argo era.
(c) The transport share over every admissible early/late epoch pair the record
    supports, so the attribution does not rest on hand-picked epochs.

Reads PAPER_3_v2/analysis/attribution.json.
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
    C_GLORYS,
    C_GREY,
    C_ORAS5,
    C_SALINITY,
    C_TOTAL,
    C_TRANSPORT,
    TWO_COL,
    panel_letter,
    save,
    set_style,
)

ANALYSIS = REPO / "PAPER_3_v2" / "analysis" / "attribution.json"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig2_attribution"

PRODUCTS = [("oras5", "ORAS5", C_ORAS5), ("glorys12", "GLORYS12V1", C_GLORYS)]


def main() -> None:
    set_style()
    d = json.loads(ANALYSIS.read_text())

    fig = plt.figure(figsize=(TWO_COL, 2.55))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.15, 0.95],
        wspace=0.42,
        left=0.06,
        right=0.995,
        top=0.90,
        bottom=0.16,
    )

    # ── a: the two factors ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    for key, label, colour in PRODUCTS:
        r = d[key]
        yrs = np.array(r["years"])
        t = np.array(r["T_limb_Sv"])
        s = np.array(r["dS_limb_PSU"])
        ax.plot(
            yrs,
            100 * (t - t.mean()) / t.mean(),
            color=colour,
            lw=0.9,
            label=f"{label}, transport",
        )
        ax.plot(
            yrs,
            100 * (s - s.mean()) / s.mean(),
            color=colour,
            lw=0.9,
            ls=(0, (2.5, 1.2)),
            label=f"{label}, salinity contrast",
        )
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly from record mean (%)")
    ax.set_xlim(1958, 2026)
    ax.legend(loc="upper left", ncol=1, handlelength=2.0)
    panel_letter(ax, "a")

    # ── b: trend attribution with intervals ───────────────────────────
    ax = fig.add_subplot(gs[1])
    rows = []
    for key, label, _ in PRODUCTS:
        for win, wlabel in (("own record", "full"), ("argo", "2004$-$")):
            c = d[key]["continuous"].get(win)
            if c is not None:
                rows.append((f"{label}\n{wlabel}", c))
    ypos = np.arange(len(rows))[::-1]
    height = 0.19
    series = [
        ("trend_total_mSv_per_yr", "ci_total_mSv_per_yr", C_TOTAL, "total"),
        ("trend_T_mSv_per_yr", "ci_T_mSv_per_yr", C_TRANSPORT, "transport"),
        ("trend_S_mSv_per_yr", "ci_S_mSv_per_yr", C_SALINITY, "salinity"),
        ("residual_mSv_per_yr", None, C_GREY, "residual"),
    ]
    for k, (vkey, ckey, colour, name) in enumerate(series):
        off = (1.5 - k) * height
        vals = np.array([r[1][vkey] for r in rows])
        ax.barh(ypos + off, vals, height=height * 0.88, color=colour, label=name)
        if ckey is None:
            continue
        lo = np.array([r[1][ckey][0] for r in rows])
        hi = np.array([r[1][ckey][1] for r in rows])
        ax.errorbar(
            vals,
            ypos + off,
            xerr=[vals - lo, hi - vals],
            fmt="none",
            ecolor="0.25",
            elinewidth=0.6,
            capsize=1.2,
        )
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Trend in $F_{\\mathrm{ov}}^{S}$ (mSv yr$^{-1}$)")
    ax.legend(loc="lower left", ncol=1)
    panel_letter(ax, "b", x=-0.30)

    # ── c: census over all epoch pairs ────────────────────────────────
    ax = fig.add_subplot(gs[2])
    for i, (key, _label, colour) in enumerate(PRODUCTS):
        c = d[key]["census"]
        stats = [
            c["share_T_sym_median"],
            c["share_T_sym_p90"],
            c["share_T_sym_p95"],
            c["share_T_sym_max"],
        ]
        ax.plot(
            [i] * len(stats),
            stats,
            marker="_",
            ms=11,
            mew=1.1,
            ls="none",
            color=colour,
        )
        ax.plot([i, i], [stats[0], stats[-1]], color=colour, lw=0.8, alpha=0.5)
        ax.annotate(
            f"{100 * c['frac_share_below_0p25']:.0f}%",
            (i, 0.25),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=5.5,
            color=colour,
        )
    ax.axhline(0.25, color="0.4", lw=0.5, ls=(0, (2, 1.5)))
    ax.set_xticks(range(len(PRODUCTS)))
    ax.set_xticklabels([p[1] for p in PRODUCTS])
    ax.set_xlim(-0.5, len(PRODUCTS) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Transport share of the change")
    panel_letter(ax, "c", x=-0.30)

    save(fig, OUTBASE)


if __name__ == "__main__":
    main()
