#!/usr/bin/env python3
"""PAPER_3_v2 Figure 5: the observation-only test of the limb asymmetry.

The velocity structure is frozen at the ORAS5 record-mean profile and only the
EN4.2.2 objective analysis of profile observations is allowed to vary, so every
trend shown is a trend in observed water masses under a fixed circulation.

The obvious objection to that test is a sampling one. The northward limb is the
upper cell and lies inside the core-Argo layer; most of the southward limb's
transport weight sits below 2000 m, where core Argo does not sample and EN4
relaxes towards climatology. A relaxed field has its variance and any trend
damped, so the reported asymmetry could in principle be produced by the
observing system rather than by the ocean. Panel (b) answers that by repeating
the test with both limbs redefined inside the upper 2000 m, where both are
observed.

(a) Full-column limb salinities, northward (solid) and southward (dashed), as
    anomalies from their own means, with least-squares trends.
(b) The same with both limbs restricted to the upper 2000 m.
(c) Trend in the limb contrast dS over 2005-2024: the two observation-only
    estimates against each reanalysis's own dS over exactly the same years,
    with 95% confidence intervals inflated by the Santer effective-sample-size
    factor.

Reads PAPER_3_v2/analysis/en4_depth_test.json (produced by
scripts/analysis_paper3v2_en4_depth_test.py) and, for the reanalysis bars in
(c), the dS_limb_PSU series in PAPER_3_v2/analysis/attribution.json restricted
to the EN4 window, so that every trend in the panel is formed the same way.
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
DEPTH = REPO / "PAPER_3_v2" / "analysis" / "en4_depth_test.json"
ATTR = REPO / "PAPER_3_v2" / "analysis" / "attribution.json"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig5_en4"

FIG_W, FIG_H = 7.09, 2.65

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
    n_eff = float(np.clip(n * (1.0 - r1) / (1.0 + r1), 3.0, n))
    sxx = float(np.sum((years - years.mean()) ** 2))
    se = float(np.sqrt(np.sum(resid**2) / (n - 2) / sxx))
    se_adj = se * np.sqrt((n - 2) / (n_eff - 2))
    tcrit = stats.t.ppf(0.975, n_eff - 2)
    return {
        "trend": slope * 10.0,
        "ci": (10.0 * (slope - tcrit * se_adj), 10.0 * (slope + tcrit * se_adj)),
        "p": float(2.0 * stats.t.sf(abs(slope / se_adj), n_eff - 2)),
    }


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.17) -> None:
    ax.text(x, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def limb_panel(ax, years, block, title, note) -> None:
    n_note = note.count("\n") + 1
    lo, hi = np.inf, -np.inf
    for row, (key, colour, style, label) in enumerate(
        (("S_north", C_NORTH, "-", "northward limb"),
         ("S_south", C_SOUTH, (0, (2.4, 1.4)), "southward limb"))
    ):
        series = np.asarray(block[key]["series"], dtype=float)
        anom = series - series.mean()
        lo, hi = min(lo, anom.min()), max(hi, anom.max())
        ax.plot(years, anom, color=colour, lw=0.7, alpha=0.9, ls=style)
        fit = santer_trend(years, anom)
        slope = fit["trend"] / 10.0
        ax.plot(years, slope * years + (anom.mean() - slope * years.mean()),
                color=colour, lw=1.4,
                ls="-" if fit["p"] < 0.05 else (0, (2.4, 1.4)), alpha=0.95)
        sig = "" if fit["p"] < 0.05 else " (n.s.)"
        ax.text(0.03, 0.965 - 0.075 * row,
                f"{label}: {fit['trend']:+.3f} PSU per decade{sig}",
                transform=ax.transAxes, color=colour, fontsize=5,
                ha="left", va="top")
    ax.axhline(0.0, color="0.55", lw=0.4, zorder=0)
    # Reserve a clear band under the data for the note rather than writing it
    # over the series.
    span = hi - lo
    pad_bot = 0.12 + 0.11 * n_note
    ax.set_ylim(lo - pad_bot * span, hi + 0.34 * span)
    ax.text(0.03, 0.018, note, transform=ax.transAxes, fontsize=4.6,
            color="0.35", ha="left", va="bottom")
    ax.set_xlabel("Year")
    ax.set_ylabel("Limb salinity anomaly (PSU)")
    ax.set_title(title, fontsize=6, pad=3)


def main() -> None:
    set_style()
    d = json.loads(DEPTH.read_text())
    attr = json.loads(ATTR.read_text())
    blk = d["oras5"]
    y0, y1 = d["record"]
    years = np.asarray(blk["full_column"]["years"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))

    fc = blk["full_column"]
    limb_panel(
        axes[0], years, fc, "EN4 salinity, full column",
        f"southward limb: {fc['frac_south_below_argo_mean']*100:.0f}% of transport weight below 2000 m\n"
        f"mean EN4 observation weight  north {fc['obsw_north']['mean']:.2f}   south {fc['obsw_south']['mean']:.2f}",
    )
    panel_letter(axes[0], "a")

    up = blk["upper_2000m"]
    limb_panel(
        axes[1], years, up, "EN4 salinity, both limbs within upper 2000 m",
        "both limbs inside the core-Argo layer",
    )
    panel_letter(axes[1], "b")

    # ---- panel c: dS trends over the matched window ----------------------
    ax = axes[2]
    rows = []
    for tag, lab in (("full_column", "EN4, full column"),
                     ("upper_2000m", "EN4, upper 2000 m")):
        s = np.asarray(blk[tag]["dS"]["series"], dtype=float)
        rows.append((lab, santer_trend(years, s), C_OBS))
    for key, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        b = attr[key]
        yrs = np.asarray(b["years"], dtype=float)
        vals = np.asarray(b["dS_limb_PSU"], dtype=float)
        sel = (yrs >= y0) & (yrs <= y1)
        rows.append((f"{b['label']}, own salinity", santer_trend(yrs[sel], vals[sel]), colour))

    ypos = np.arange(len(rows))[::-1]
    for y, (label, fit, colour) in zip(ypos, rows):
        lo, hi = fit["ci"]
        ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.13, y + 0.13], color=colour, lw=1.0)
        ax.plot([fit["trend"]], [y], marker="o", ms=3.0, color=colour,
                mec="white", mew=0.4, zorder=3)
    ax.axvline(0.0, color="0.55", lw=0.4, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"Trend in $\Delta S$ (PSU per decade)")
    ax.set_title(f"Limb contrast, {int(y0)}–{int(y1)}", fontsize=6, pad=3)
    panel_letter(ax, "c", x=-0.72)

    fig.tight_layout(pad=0.5, w_pad=1.7)
    OUTBASE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTBASE.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUTBASE.with_suffix('.pdf')}")
    for label, fit, _ in rows:
        lo, hi = fit["ci"]
        print(f"  {label:30s} {fit['trend']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={fit['p']:.4f}")


if __name__ == "__main__":
    main()
