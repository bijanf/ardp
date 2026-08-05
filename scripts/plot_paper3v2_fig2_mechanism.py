#!/usr/bin/env python3
"""PAPER_3_v2 Figure 2: what actually moves F_ovS.

(a) Annual upper-cell overturning strength at the 34.5 S section for ORAS5 and
    GLORYS12V1, with least-squares trends (solid where the Santer N_eff
    adjusted p is below 0.05, dashed otherwise).
(b) The three terms of the variation identity
    dF_ovS = -(1/S0) (DvS dPsi + Psi dDvS + dPsi dDvS)
    for four early-to-late epoch pairs. The overturning-strength term is the
    one associated with the salt-advection feedback.
(c) The same four epoch pairs split by the profile decomposition into velocity,
    salinity and cross terms, with the velocity bar further split into the part
    explained by a pure rescaling of the overturning (amplitude) and the part
    that comes from a change in its vertical structure.

Everything is read from PAPER_3_v2/analysis/feedback_identity.json, which is
produced by scripts/analysis_paper3v2_feedback_identity.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "PAPER_3_v2" / "analysis" / "feedback_identity.json"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig2_mechanism"

FIG_W, FIG_H = 7.09, 3.05

C_ORAS5 = "#20558a"
C_GLORYS = "#b3541e"
C_PSI = "#1B5E8C"
C_SAL = "#F0A93B"
C_CRS = "#9A9A9A"
C_TOT = "#111111"

ROWS = [
    ("oras5", "pre_registered", "ORAS5"),
    ("oras5", "long_epochs", "ORAS5"),
    ("glorys12", "pre_registered", "GLORYS12"),
    ("glorys12", "halves", "GLORYS12"),
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
            "legend.fontsize": 5,
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


def panel_letter(ax: plt.Axes, letter: str, x: float) -> None:
    ax.text(
        x,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def panel_a(ax: plt.Axes, res: dict) -> None:
    for key, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        r = res[key]
        years = np.asarray(r["years"], dtype=float)
        psi = np.asarray(r["psi_Sv"], dtype=float)
        ax.plot(years, psi, color=colour, lw=0.7, alpha=0.85, label=r["label"])
        trend = r["psi_trends"]["own record"]
        slope, inter = np.polyfit(years, psi, 1)
        style = "-" if trend["significant"] else (0, (2.4, 1.4))
        ax.plot(
            years,
            slope * years + inter,
            color=colour,
            lw=1.4,
            ls=style,
            zorder=4,
        )
    ax.set_xlim(1956, 2027)
    ax.set_xticks(np.arange(1960, 2030, 10))
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Overturning at 34.5$^\circ$S (Sv)")
    ax.legend(
        loc="lower left",
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _row_labels(ax: plt.Axes, rows: list[dict], names: list[str]) -> None:
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (r, name) in enumerate(zip(rows, names, strict=True)):
        ax.text(
            -0.04, i - 0.19, name, fontsize=6, ha="right", va="center", transform=trans
        )
        ax.text(
            -0.04,
            i + 0.19,
            f"{r['early_period']} vs {r['late_period']}",
            fontsize=5,
            color="0.30",
            ha="right",
            va="center",
            transform=trans,
        )


def _bar_axes(ax: plt.Axes, n: int, xlim: tuple[float, float]) -> None:
    ax.axvline(0.0, color="0.35", lw=0.5, zorder=2)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.grid(axis="x", color="0.90", lw=0.3, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_b(ax: plt.Axes, rows: list[dict], names: list[str]) -> None:
    height = 0.22
    for i, r in enumerate(rows):
        ax.barh(i - 0.25, r["term_psi_mSv"], height, color=C_PSI, zorder=3)
        ax.barh(i, r["term_dvs_mSv"], height, color=C_SAL, zorder=3)
        ax.barh(i + 0.25, r["term_cross_mSv"], height, color=C_CRS, zorder=3)
        ax.plot(
            [r["total_mSv"]] * 2,
            [i - 0.45, i + 0.45],
            color=C_TOT,
            lw=1.0,
            solid_capstyle="butt",
            zorder=4,
        )
    _bar_axes(ax, len(rows), (-108.0, 22.0))
    ax.set_xticks([-100, -75, -50, -25, 0])
    ax.set_xlabel(r"$\Delta F_{ovS}$ term (mSv)")
    _row_labels(ax, rows, names)
    ax.legend(
        handles=[
            Patch(facecolor=C_PSI, label=r"overturning $\Psi$"),
            Patch(facecolor=C_SAL, label=r"salinity contrast $\Delta_v S$"),
            Patch(facecolor=C_CRS, label="cross"),
            Line2D([0], [0], color=C_TOT, lw=1.0, label="total"),
        ],
        loc="upper left",
        labelspacing=0.35,
        handlelength=1.2,
        handletextpad=0.5,
        borderaxespad=0.2,
    )


def panel_c(ax: plt.Axes, rows: list[dict]) -> None:
    height = 0.22
    for i, r in enumerate(rows):
        d = r["decomposition"]
        split = r["velocity_split"]
        # Velocity bar, with the amplitude piece drawn on top as a hatched
        # segment anchored at zero.
        ax.barh(i - 0.25, d["delta_v_mSv"], height, color=C_PSI, zorder=3)
        ax.barh(
            i - 0.25,
            split["amplitude_mSv"],
            height,
            color="white",
            edgecolor=C_PSI,
            hatch="////",
            lw=0.4,
            zorder=4,
        )
        ax.barh(i, d["delta_s_mSv"], height, color=C_SAL, zorder=3)
        ax.barh(i + 0.25, d["delta_cross_mSv"], height, color=C_CRS, zorder=3)
        ax.plot(
            [d["delta_total_mSv"]] * 2,
            [i - 0.45, i + 0.45],
            color=C_TOT,
            lw=1.0,
            solid_capstyle="butt",
            zorder=5,
        )
    _bar_axes(ax, len(rows), (-108.0, 22.0))
    ax.set_xticks([-100, -75, -50, -25, 0])
    ax.set_xlabel(r"$\Delta F_{ovS}$ term (mSv)")
    ax.legend(
        handles=[
            Patch(facecolor=C_PSI, label="velocity"),
            Patch(
                facecolor="white",
                edgecolor=C_PSI,
                hatch="////",
                lw=0.4,
                label="of which amplitude",
            ),
            Patch(facecolor=C_SAL, label="salinity"),
            Patch(facecolor=C_CRS, label="cross"),
        ],
        loc="upper left",
        labelspacing=0.35,
        handlelength=1.2,
        handletextpad=0.5,
        borderaxespad=0.2,
    )


def main() -> None:
    res = json.loads(ANALYSIS.read_text())
    rows = [res[p]["epochs"][tag] for p, tag, _ in ROWS]
    names = [name for _, _, name in ROWS]

    # The amplitude piece of the profile velocity term must equal the identity's
    # overturning term exactly; both are (Psi_late/Psi_early - 1) * F_ov_early.
    for r in rows:
        assert abs(r["velocity_split"]["amplitude_mSv"] - r["term_psi_mSv"]) < 1e-9

    set_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.30, 1.16, 1.00],
        left=0.062,
        right=0.995,
        bottom=0.145,
        top=0.915,
        wspace=0.62,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, res)
    panel_b(ax_b, rows, names)
    panel_c(ax_c, rows)

    panel_letter(ax_a, "a", -0.14)
    panel_letter(ax_b, "b", -0.42)
    panel_letter(ax_c, "c", -0.10)

    OUTBASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUTBASE}.pdf")
    fig.savefig(f"{OUTBASE}.png", dpi=300)
    plt.close(fig)

    print(f"wrote {OUTBASE}.pdf and {OUTBASE}.png")
    for (_p, _tag, name), r in zip(ROWS, rows, strict=True):
        print(
            f"  {name:9s} {r['early_period']} vs {r['late_period']}: "
            f"Psi term {r['term_psi_mSv']:+6.2f}  DvS term {r['term_dvs_mSv']:+7.2f}"
            f"  total {r['total_mSv']:+7.2f} mSv"
        )
    for key in ("oras5", "glorys12"):
        t = res[key]["psi_trends"]["own record"]
        print(
            f"  {res[key]['label']:11s} Psi trend {t['trend']:+.3f} Sv/decade "
            f"(p={t['p_santer']:.4f}, significant={t['significant']})"
        )


if __name__ == "__main__":
    main()
