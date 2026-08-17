#!/usr/bin/env python3
"""PAPER_3_v2 Figure 5: the observation-only test of the limb asymmetry.

The velocity structure is frozen at each reanalysis record-mean profile and
only the EN4.2.2 objective analysis of profile observations is allowed to
vary, so every trend shown here is a trend in observed water masses under a
fixed circulation.

(a) Transport-weighted limb salinities from EN4 under the ORAS5 frozen
    velocity structure, northward (solid) and southward (dashed), as
    anomalies from their own means, with least-squares trends.
(b) The same for the GLORYS12V1 frozen velocity structure.
(c) Trend in the limb contrast dS over the matched 2005-2024 window: the two
    observation-only estimates against each reanalysis's own dS over exactly
    the same years, with 95% confidence intervals inflated by the Santer
    effective-sample-size factor.

Panels (a) and (b) are read from PAPER_3_v2/analysis/en4.json, produced by
scripts/analysis_paper3v2_en4.py. The reanalysis bars in (c) are recomputed
here from the dS_limb_PSU series in PAPER_3_v2/analysis/attribution.json
restricted to the EN4 window, so that every trend in the panel is formed the
same way; the point values are checked against the ones en4.json carries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
EN4 = REPO / "PAPER_3_v2" / "analysis" / "en4.json"
ATTR = REPO / "PAPER_3_v2" / "analysis" / "attribution.json"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig5_en4"

FIG_W, FIG_H = 7.09, 2.55

C_ORAS5 = "#20558a"
C_GLORYS = "#b3541e"
C_OBS = "#2e7d5b"
C_NORTH = "#20558a"
C_SOUTH = "#8a8a8a"


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


def santer_trend(years: np.ndarray, values: np.ndarray) -> dict:
    """OLS trend per decade with a Santer et al. (2000) N_eff adjustment."""
    n = years.size
    slope, intercept = np.polyfit(years, values, 1)
    resid = values - (slope * years + intercept)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    n_eff = n * (1.0 - r1) / (1.0 + r1)
    n_eff = float(np.clip(n_eff, 3.0, n))
    sxx = float(np.sum((years - years.mean()) ** 2))
    se = float(np.sqrt(np.sum(resid**2) / (n - 2) / sxx))
    se_adj = se * np.sqrt((n - 2) / (n_eff - 2))
    tcrit = stats.t.ppf(0.975, n_eff - 2)
    tstat = slope / se_adj
    return {
        "trend": slope * 10.0,
        "ci": (10.0 * (slope - tcrit * se_adj), 10.0 * (slope + tcrit * se_adj)),
        "p": float(2.0 * stats.t.sf(abs(tstat), n_eff - 2)),
        "n_eff": n_eff,
    }


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.16) -> None:
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


def limb_panel(ax: plt.Axes, years: np.ndarray, block: dict, title: str) -> None:
    lo, hi = np.inf, -np.inf
    for row, (key, colour, style, label) in enumerate(
        (
            ("S_north", C_NORTH, "-", "northward limb"),
            ("S_south", C_SOUTH, (0, (2.4, 1.4)), "southward limb"),
        )
    ):
        series = np.asarray(block[key]["series"], dtype=float)
        anom = series - series.mean()
        lo, hi = min(lo, anom.min()), max(hi, anom.max())
        ax.plot(years, anom, color=colour, lw=0.7, alpha=0.9, ls=style)
        fit = santer_trend(years, anom)
        slope = fit["trend"] / 10.0
        inter = anom.mean() - slope * years.mean()
        ax.plot(
            years,
            slope * years + inter,
            color=colour,
            lw=1.4,
            ls="-" if fit["p"] < 0.05 else (0, (2.4, 1.4)),
            alpha=0.95,
        )
        sig = "" if fit["p"] < 0.05 else " (n.s.)"
        ax.text(
            0.03,
            0.965 - 0.075 * row,
            f"{label}: {fit['trend']:+.3f} PSU per decade{sig}",
            transform=ax.transAxes,
            color=colour,
            fontsize=5,
            ha="left",
            va="top",
        )
    ax.axhline(0.0, color="0.55", lw=0.4, zorder=0)
    span = hi - lo
    ax.set_ylim(lo - 0.10 * span, hi + 0.32 * span)
    ax.set_xlabel("Year")
    ax.set_ylabel("Limb salinity anomaly (PSU)")
    ax.set_title(title, fontsize=6, pad=3)


def main() -> None:
    set_style()
    en4 = json.loads(EN4.read_text())
    attr = json.loads(ATTR.read_text())

    years = np.asarray(en4["years"], dtype=float)
    y0, y1 = en4["record"]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))

    limb_panel(axes[0], years, en4["oras5"], "EN4 salinity, ORAS5 velocity structure")
    panel_letter(axes[0], "a")
    limb_panel(
        axes[1], years, en4["glorys12"], "EN4 salinity, GLORYS12V1 velocity structure"
    )
    panel_letter(axes[1], "b")

    # ---- panel c: dS trends over the matched window --------------------
    ax = axes[2]
    rows = []
    for key, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        series = np.asarray(en4[key]["dS"]["series"], dtype=float)
        fit = santer_trend(years, series)
        rows.append((f"EN4, {en4[key]['label'].split()[0]} velocity", fit, C_OBS))

    for key, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        block = attr[key]
        yrs = np.asarray(block["years"], dtype=float)
        vals = np.asarray(block["dS_limb_PSU"], dtype=float)
        sel = (yrs >= y0) & (yrs <= y1)
        fit = santer_trend(yrs[sel], vals[sel])
        published = en4["reanalysis_same_window"][key]["dS_trend_per_decade"]
        if abs(fit["trend"] - published) > 5e-4:
            raise SystemExit(
                f"{key}: recomputed dS trend {fit['trend']:.5f} disagrees with "
                f"en4.json value {published:.5f}"
            )
        rows.append((f"{block['label']}, own salinity", fit, colour))

    ypos = np.arange(len(rows))[::-1]
    for y, (label, fit, colour) in zip(ypos, rows):
        lo, hi = fit["ci"]
        ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt")
        ax.plot([lo, lo], [y - 0.13, y + 0.13], color=colour, lw=1.0)
        ax.plot([hi, hi], [y - 0.13, y + 0.13], color=colour, lw=1.0)
        ax.plot(
            [fit["trend"]],
            [y],
            marker="o",
            ms=3.0,
            color=colour,
            mec="white",
            mew=0.4,
            zorder=3,
        )
    ax.axvline(0.0, color="0.55", lw=0.4, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"Trend in $\Delta S$ (PSU per decade)")
    ax.set_title(f"Limb contrast, {int(y0)}–{int(y1)}", fontsize=6, pad=3)
    panel_letter(ax, "c", x=-0.62)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    OUTBASE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTBASE.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUTBASE.with_suffix('.pdf')}")
    for label, fit, _ in rows:
        lo, hi = fit["ci"]
        print(
            f"  {label:34s} {fit['trend']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"p={fit['p']:.4f} n_eff={fit['n_eff']:.1f}"
        )


if __name__ == "__main__":
    main()
