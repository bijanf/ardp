#!/usr/bin/env python3
"""Figure 3: what is changing in the salinity field, and how robust it is.

(a) Transport-weighted salinity of the two limbs. The northward limb salinifies
    in both eddy-permitting products while the southward limb is flat, which is
    the whole of the F_ovS decline.
(b) Zonal-mean salinity trend against depth at the section, on one axis, with
    the limb boundary marked. Solid where the Santer-adjusted p is below 0.05.
(c) The transport share of the F_ovS trend under every robustness test applied:
    three definitions of the overturning strength, the Argo era alone, and, for
    ORAS5, the homogeneous consolidated stream alone.

Reads the archived section files and PAPER_3_v2/analysis/attribution.json.
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
from paper3v2_style import (  # noqa: E402
    C_GLORYS,
    C_ORAS5,
    TWO_COL,
    panel_letter,
    save,
    set_style,
)

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis" / "attribution.json"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig3_mechanism"

PRODUCTS = [
    ("oras5", "ORAS5", C_ORAS5, "paper3v2_section_oras5.nc"),
    ("glorys12", "GLORYS12V1", C_GLORYS, "paper3v2_section_glorys12.nc"),
]
TREND_WINDOW = (1993, 2025)


def main() -> None:
    set_style()
    d = json.loads(ANALYSIS.read_text())

    fig = plt.figure(figsize=(TWO_COL, 2.6))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 0.72, 1.05],
        wspace=0.62,
        left=0.06,
        right=0.995,
        top=0.90,
        bottom=0.17,
    )

    # ── a: limb salinities ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    for _key, _label, colour, fname in PRODUCTS:
        with xr.open_dataset(RESULTS / fname) as ds:
            a = ds.groupby("time.year").mean()
        yrs = a["year"].values
        for var, style in (("S_north", "-"), ("S_south", (0, (2.5, 1.2)))):
            v = a[var].values
            ax.plot(yrs, v - v.mean(), color=colour, ls=style, lw=0.9)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Limb salinity anomaly (PSU)")
    ax.set_xlim(1958, 2026)
    handles = [
        plt.Line2D([], [], color=c, lw=0.9, label=lab) for _, lab, c, _ in PRODUCTS
    ] + [
        plt.Line2D([], [], color="0.35", lw=0.9, label="northward limb"),
        plt.Line2D(
            [], [], color="0.35", lw=0.9, ls=(0, (2.5, 1.2)), label="southward limb"
        ),
    ]
    ax.legend(handles=handles, loc="upper left")
    panel_letter(ax, "a")

    # ── b: salinity trend profile ─────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    for _key, _label, colour, fname in PRODUCTS:
        with xr.open_dataset(RESULTS / fname) as ds:
            a = ds.groupby("time.year").mean()
            depth = ds["depth"].values
        yrs = a["year"].values
        m = (yrs >= TREND_WINDOW[0]) & (yrs <= TREND_WINDOW[1])
        prof = a["S_bar"].values[m]
        x = yrs[m].astype(float)
        trend = np.full(len(depth), np.nan)
        sig = np.zeros(len(depth), bool)
        for k in range(len(depth)):
            col = prof[:, k]
            if np.isfinite(col).all() and np.nanstd(col) > 0:
                f = ols_santer(x, col)
                trend[k] = f["slope"] * 10
                sig[k] = f["p_santer"] < 0.05
        ok = depth <= 3000
        ax.plot(trend[ok], depth[ok], color=colour, lw=0.8, alpha=0.45)
        t2 = np.where(sig[ok], trend[ok], np.nan)
        ax.plot(t2, depth[ok], color=colour, lw=1.3)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.invert_yaxis()
    ax.set_ylim(3000, 0)
    ax.set_xlabel("Salinity trend\n(PSU per decade)")
    ax.set_ylabel("Depth (m)")
    panel_letter(ax, "b", x=-0.46)

    # ── c: robustness of the transport share ──────────────────────────
    ax = fig.add_subplot(gs[2])
    # One row per robustness test, the two products offset within each row so
    # the labels are written once.
    tests: list[tuple[str, str]] = [
        ("full record", "own record"),
        ("2004 onwards", "argo"),
        ("$\\Psi_{\\max}$, surface", "psi_surface"),
        ("$\\Psi_{\\max}$, 250 m", "psi_250"),
        ("consolidated only", "consolidated"),
    ]

    def lookup(r: dict, tag: str):
        if tag in ("own record", "argo"):
            c = r["continuous"].get(tag)
        elif tag == "consolidated":
            c = r.get("consolidated_only", {}).get("continuous")
        else:
            name = (
                "streamfunction max, surface start"
                if tag == "psi_surface"
                else "streamfunction max, 250 m start"
            )
            c = r["psi_definition_sensitivity"].get(name)
        if c is None:
            return None
        return c["share_T_point"], c["share_T_ci"]

    ypos = np.arange(len(tests))[::-1]
    for j, (key, label, colour, _) in enumerate(PRODUCTS):
        off = 0.17 * (1 - 2 * j)
        for y, (_, tag) in zip(ypos, tests, strict=False):
            got = lookup(d[key], tag)
            if got is None:
                continue
            val, ci = got
            ax.plot([ci[0], ci[1]], [y + off, y + off], color=colour, lw=1.0)
            ax.plot(
                val,
                y + off,
                "o",
                color=colour,
                ms=2.8,
                label=label if y == ypos[0] else None,
            )
    ax.axvline(0.25, color="0.4", lw=0.5, ls=(0, (2, 1.5)))
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[0] for t in tests])
    ax.set_ylim(-0.6, len(tests) - 0.4)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("Transport share of the trend")
    ax.legend(loc="lower right")
    panel_letter(ax, "c", x=-0.62)

    save(fig, OUTBASE)


if __name__ == "__main__":
    main()
