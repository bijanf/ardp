#!/usr/bin/env python3
"""Figure 3: attribution of the F_ovS trend to transport and the limb contrast.

a  The two factors of F_ovS = -(1/S0) T dS as percentage anomalies from their
   record means.
b  Trend of F_ovS split into a transport term and a salinity term, with 95%
   block-bootstrap intervals, over each product's own record, the Argo era and
   the 1993-2017 window shared by all three products; hatched total bars where
   the total trend is not significant.
c  Transport share of the change over every non-overlapping epoch pair:
   median, 90th and 95th percentiles and maximum.
d  Transport share with 95% intervals under each robustness test.

Writes PAPER_3_v2/figures_r1/Fig3_attribution.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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

AN = REPO / "PAPER_3_v2" / "analysis"
OUT = REPO / "PAPER_3_v2" / "figures_r1" / "Fig3_attribution"
PROD = [
    ("oras5", "ORAS5", C_ORAS5),
    ("glorys12", "GLORYS12V1", C_GLORYS),
    ("ecco", "ECCO-V4r4", C_ECCO),
]


def main() -> None:
    set_style()
    d = json.loads((AN / "attribution.json").read_text())
    cells = json.loads((AN / "r1_cells.json").read_text())
    grid = json.loads((AN / "r1_grid_sensitivity.json").read_text())
    step = json.loads((AN / "r1_step_adjusted.json").read_text())

    fig = plt.figure(figsize=(TWO_COL, 4.9))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.15],
        hspace=0.42,
        left=0.08,
        right=0.99,
        top=0.95,
        bottom=0.07,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.55, 1.45], wspace=0.75)
    gs = {(0, 0): top[0], (0, 1): top[1], (1, 0): bottom[0], (1, 1): bottom[1]}

    # a: factors
    ax = fig.add_subplot(gs[0, 0])
    for key, label, colour in PROD:
        r = d[key]
        yrs = np.array(r["years"])
        t = np.array(r["T_limb_Sv"])
        s = np.array(r["dS_limb_PSU"])
        lw = 0.9 if key != "ecco" else 0.6
        ax.plot(
            yrs,
            100 * (t - t.mean()) / t.mean(),
            color=colour,
            lw=lw,
            label=f"{label} $T$",
        )
        ax.plot(
            yrs,
            100 * (s - s.mean()) / s.mean(),
            color=colour,
            lw=lw,
            ls=(0, (2.5, 1.2)),
            label=f"{label} $\\Delta S$",
        )
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly from record mean (%)")
    ax.set_xlim(1958, 2026)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        handlelength=2.0,
        columnspacing=0.8,
    )
    panel_letter(ax, "a", x=-0.14, y=1.17)

    # Inset: fractional rates of F_ovS and T over each record (percent per decade).
    rates = json.loads((AN / "stepfit.json").read_text())
    ins = ax.inset_axes([0.12, 0.57, 0.3, 0.39])
    for i, (key, _label, colour) in enumerate(PROD[:2]):
        fr = rates[key]["fractional_rates"]["own record"]
        ins.bar(i - 0.18, fr["F_ov"]["fractional_pct_per_decade"], 0.34, color=colour)
        ins.bar(
            i + 0.18,
            fr["T_limb"]["fractional_pct_per_decade"],
            0.34,
            color="white",
            edgecolor=colour,
            lw=0.7,
        )
    ins.axhline(0, color="black", lw=0.4)
    ins.set_xticks([-0.18, 0.18, 0.82, 1.18])
    ins.set_xticklabels(["$F$", "$T$", "$F$", "$T$"], fontsize=5.5)
    ins.tick_params(axis="both", labelsize=5.5, length=1.5, pad=1)
    ins.set_ylabel("% per decade", fontsize=5.5, labelpad=1)
    ins.patch.set_alpha(0.9)

    # b: trend split by window
    ax = fig.add_subplot(gs[0, 1])
    windows = {
        "oras5": (
            ("own record", "1958–2025"),
            ("consolidated", "1958–2014"),
            ("argo", "2004–2025"),
            ("common", "1993–2017"),
        ),
        "glorys12": (
            ("own record", "1993–2025"),
            ("argo", "2004–2025"),
            ("argo to 2014", "2004–2014"),
            ("common", "1993–2017"),
        ),
        "ecco": (("common", "1993–2017"),),
    }
    rows = []
    for key, label, _ in PROD:
        for win, wl in windows[key]:
            if win == "consolidated":
                c = d[key]["consolidated_only"]["continuous"]
            else:
                c = d[key]["continuous"].get(win)
            if c is not None:
                rows.append((f"{label}, {wl}", c))
    ypos = np.arange(len(rows))[::-1].astype(float)
    h = 0.26
    xlim_b = (-9.0, 3.0)
    for k, (vk, ck, col, _name) in enumerate(
        (
            ("trend_total_mSv_per_yr", "ci_total_mSv_per_yr", C_TOTAL, "total"),
            ("trend_T_mSv_per_yr", "ci_T_mSv_per_yr", C_TRANSPORT, "transport term"),
            ("trend_S_mSv_per_yr", "ci_S_mSv_per_yr", C_SALINITY, "salinity term"),
        )
    ):
        off = (1 - k) * h
        vals = np.array([r[1][vk] for r in rows])
        lo = np.array([r[1][ck][0] for r in rows])
        hi = np.array([r[1][ck][1] for r in rows])
        for y, v, (_, c) in zip(ypos + off, vals, rows, strict=False):
            hatch = "/////" if (k == 0 and c["p_total"] >= 0.05) else None
            ax.barh(
                y,
                v,
                height=h * 0.9,
                color=col if hatch is None else "white",
                edgecolor=col,
                lw=0.6,
                hatch=hatch,
            )
        lo_c = np.clip(lo, *xlim_b)
        hi_c = np.clip(hi, *xlim_b)
        ax.errorbar(
            vals,
            ypos + off,
            xerr=[vals - lo_c, hi_c - vals],
            fmt="none",
            ecolor="0.2",
            elinewidth=0.5,
            capsize=1.0,
        )
        # Intervals that run past the axis are drawn to its edge with an arrow.
        for y, a, b in zip(ypos + off, lo, hi, strict=False):
            if a < xlim_b[0]:
                ax.annotate(
                    "",
                    xy=(xlim_b[0], y),
                    xytext=(xlim_b[0] + 0.45, y),
                    arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "0.2"},
                )
            if b > xlim_b[1]:
                ax.annotate(
                    "",
                    xy=(xlim_b[1], y),
                    xytext=(xlim_b[1] - 0.45, y),
                    arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "0.2"},
                )
    ax.set_xlim(*xlim_b)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    ax.set_xlabel("Trend in $F_{\\mathrm{ovS}}$ (mSv yr$^{-1}$)")
    ax.legend(
        handles=[
            Patch(fc=C_TOTAL, label="total"),
            Patch(fc=C_TRANSPORT, label="transport term"),
            Patch(fc=C_SALINITY, label="salinity term"),
            Patch(
                fc="white", ec=C_TOTAL, hatch="/////", label="total, not significant"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.45, 1.0),
        ncol=2,
        handlelength=1.2,
    )
    panel_letter(ax, "b", x=-0.62, y=1.17)

    # c: census
    ax = fig.add_subplot(gs[1, 0])
    for i, (key, _label, colour) in enumerate(PROD[:2]):
        c = d[key]["census"]
        st = [
            c["share_T_sym_median"],
            c["share_T_sym_p90"],
            c["share_T_sym_p95"],
            c["share_T_sym_max"],
        ]
        ax.plot([i, i], [st[0], st[-1]], color=colour, lw=0.8, alpha=0.5)
        ax.plot(i, st[0], "o", color=colour, ms=3.5)
        ax.plot([i, i], st[1:3], marker="_", ms=10, mew=1.0, ls="none", color=colour)
        ax.plot(i, st[3], "x", color=colour, ms=3.5, mew=0.8)
    ax.axhline(0.25, color="0.4", lw=0.5, ls=(0, (2, 1.5)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([p[1] for p in PROD[:2]])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Transport share of the change")
    ax.legend(
        handles=[
            plt.Line2D(
                [], [], color="0.3", marker="o", ls="none", ms=3.5, label="median"
            ),
            plt.Line2D(
                [],
                [],
                color="0.3",
                marker="_",
                ls="none",
                ms=8,
                mew=1.0,
                label="90th and\n95th percentiles",
            ),
            plt.Line2D(
                [], [], color="0.3", marker="x", ls="none", ms=3.5, label="maximum"
            ),
        ],
        loc="upper right",
    )
    panel_letter(ax, "c", x=-0.32)

    # d: robustness
    ax = fig.add_subplot(gs[1, 1])
    tests = [
        ("baseline, own record", lambda k: d[k]["continuous"]["own record"]),
        ("Argo era", lambda k: d[k]["continuous"]["argo"]),
        (
            "upper cell only",
            lambda k: cells[k]["windows"]["own"]["upper_cell_attribution"],
        ),
        (
            "$\\Psi_{\\max}$ from surface",
            lambda k: d[k]["psi_definition_sensitivity"][
                "streamfunction max, surface start"
            ],
        ),
        (
            "$\\Psi_{\\max}$ below 250 m",
            lambda k: d[k]["psi_definition_sensitivity"][
                "streamfunction max, 250 m start"
            ],
        ),
        ("removal in proportion to $|V|$", lambda k: grid[k]["_absV"]),
        ("thickness from level centres", lambda k: grid[k]["_centres"]),
        ("bottom cells at one fifth", lambda k: grid[k]["_bottom0p2"]),
        ("observing-system step fitted", lambda k: step[k]),
        (
            "consolidated stream only",
            lambda k: d[k]["consolidated_only"]["continuous"] if k == "oras5" else None,
        ),
    ]
    tests.append(("start years to 2001", lambda k: "scan" if k == "oras5" else None))
    xmax = 0.45

    def resolved(key: str, r: dict) -> bool:
        """Shares are quoted only where the trend of the total is resolved."""
        if "total_resolved" in r:
            return bool(r["total_resolved"])
        if "F_trend_resolved" in r:
            return bool(r["F_trend_resolved"])
        if "p_F_ov" in r:
            return bool(r["p_F_ov"] < 0.05)
        # Streamfunction definitions share the total of the baseline.
        return bool(d[key]["continuous"]["own record"]["total_resolved"])

    yy = np.arange(len(tests))[::-1].astype(float)
    for j, (key, _label, colour) in enumerate(PROD[:2]):
        off = 0.16 * (1 - 2 * j)
        for y, (_name, getter) in zip(yy, tests, strict=False):
            try:
                r = getter(key)
            except KeyError:
                r = None
            if r is None:
                continue
            if r == "scan":
                sh = [
                    s["share_T"]
                    for s in d[key]["start_year_scan"]
                    if s["p_total"] < 0.05
                ]
                ax.plot([min(sh), max(sh)], [y + off, y + off], color=colour, lw=2.2)
                continue
            p = r.get("share_T_point", r.get("share_T"))
            if resolved(key, r):
                ci = r["share_T_ci"]
                ax.plot(
                    [ci[0], min(ci[1], xmax)], [y + off, y + off], color=colour, lw=0.9
                )
                ax.plot(p, y + off, "o", color=colour, ms=2.8)
            else:
                ax.plot(
                    min(p, xmax), y + off, "o", mfc="white", mec=colour, mew=0.7, ms=2.8
                )
    ax.axvline(0.25, color="0.4", lw=0.5, ls=(0, (2, 1.5)))
    ax.set_yticks(yy)
    ax.set_yticklabels([t[0] for t in tests], fontsize=5.5)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Transport share of the $F_{\\mathrm{ovS}}$ trend")
    ax.legend(
        handles=[
            plt.Line2D([], [], color=c, marker="o", ms=2.8, lw=0.9, label=lab)
            for _, lab, c in PROD[:2]
        ]
        + [
            plt.Line2D(
                [],
                [],
                color="0.4",
                marker="o",
                mfc="white",
                ls="none",
                ms=2.8,
                label="total not resolved",
            )
        ],
        loc="lower right",
    )
    panel_letter(ax, "d", x=-0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=600)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
