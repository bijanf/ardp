#!/usr/bin/env python3
"""PAPER_3_v2 Figure 4: the CMIP6 F_ovS bias and the AMOC-rate null result.

Panel a: mean F_ovS of 25 CMIP6 models, sorted, against the four observational
estimates (ORAS5, GLORYS12, ECCO, SODA).
Panel b: model mean F_ovS against the model AMOC(26.5 N) trend over the
2005-2023 RAPID overlap window, with the observed rates as reference lines.

No stability classification appears anywhere in this figure: there is no
two-colour split of the models, no threshold shading and no class labels.
F_ovS = 0 is drawn as a thin grey reference line and nothing more.

Inputs
    data/results/fovs_decomposition_cmip6_summary.csv   (column F_ov_baseline)
    revision/rev_papaer3_02/results/WP8_scatter_stats.json

Outputs
    PAPER_3_v2/figures/Fig4_cmip6.pdf
    PAPER_3_v2/figures/Fig4_cmip6.png
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parent.parent
CMIP6_CSV = REPO / "data" / "results" / "fovs_decomposition_cmip6_summary.csv"
WP8_JSON = REPO / "revision" / "rev_papaer3_02" / "results" / "WP8_scatter_stats.json"
OUT_BASE = REPO / "PAPER_3_v2" / "figures" / "Fig4_cmip6"

# Observational mean F_ovS (Sv), CANONICAL_FACTS section 1. `sig` records
# whether the 5-year block-bootstrap 95% CI excludes zero.
OBS_FOVS = [
    ("ORAS5", -0.0239, "#20558a", True),
    ("GLORYS12", -0.0709, "#b3541e", True),
    ("ECCO", -0.1161, "#3f7f5f", True),
    ("SODA", -0.0091, "#8a6bbf", False),
]

# Observed AMOC(26.5 N) rate lines for panel b. Dash patterns differ so the
# three lines stay separable in greyscale; solid marks the one significant
# trend, following the house convention.
OBS_RATE_STYLE = {
    "RAPID": ("#1a1a1a", (0, (4.0, 1.8)), 0.9),
    "ORAS5": ("#20558a", "solid", 0.9),
    "GLORYS12": ("#b3541e", (0, (1.0, 1.4)), 1.1),
}

BAR_FACE = "#a9a9a9"
BAR_EDGE = "#5f5f5f"
DOT_COLOR = "#3a3a3a"

# Manual label offsets (points) for crowded dots in panel b.
LABEL_OFFSETS: dict[int, tuple[float, float, str, str]] = {
    1: (-3.0, 0.0, "right", "center"),
    2: (3.0, 0.0, "left", "center"),
    3: (0.0, -3.2, "center", "top"),
    8: (3.0, -1.8, "left", "top"),
    9: (-3.0, -1.4, "right", "top"),
    16: (3.0, -1.8, "left", "top"),
}
DEFAULT_OFFSET = (3.0, 1.8, "left", "bottom")

XLIM = (-0.245, 0.445)


def minus(value: float, fmt: str = "+.3f") -> str:
    """Format a number with a typographic minus sign for figure text."""
    return format(value, fmt).replace("-", "−")


def load_models() -> list[tuple[str, float]]:
    """Return (model, mean F_ovS) sorted by increasing F_ovS."""
    with CMIP6_CSV.open() as fh:
        rows = [(r["model"], float(r["F_ov_baseline"])) for r in csv.DictReader(fh)]
    rows.sort(key=lambda t: t[1])
    return rows


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
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_a(ax, models: list[tuple[str, float]]) -> None:
    names = [m for m, _ in models]
    values = [v for _, v in models]
    ypos = list(range(len(models)))

    ax.axvline(0.0, color="0.72", linewidth=0.5, zorder=1)
    ax.barh(
        ypos,
        values,
        height=0.72,
        facecolor=BAR_FACE,
        edgecolor=BAR_EDGE,
        linewidth=0.35,
        zorder=2,
    )

    for name, value, color, sig in OBS_FOVS:
        ax.axvline(
            value,
            color=color,
            linewidth=0.9,
            linestyle="solid" if sig else (0, (2.6, 1.4)),
            zorder=3,
            label=f"{name} {minus(value)} Sv",
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_ylim(-0.8, len(models) - 0.2)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Mean $F_{ovS}$ (Sv)")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", width=0.4, length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=2.2, borderaxespad=0.4)


def panel_b(ax, wp8: dict) -> None:
    rows = wp8["models"]
    obs = wp8["observations"]

    ax.axvline(0.0, color="0.72", linewidth=0.5, zorder=1)
    ax.axhline(0.0, color="0.88", linewidth=0.4, zorder=1)

    for name, (color, dashes, lw) in OBS_RATE_STYLE.items():
        rate = obs[name]["rate_Sv_dec"]
        ax.axhline(
            rate,
            color=color,
            linewidth=lw,
            linestyle=dashes,
            zorder=2,
            label=f"{name} {minus(rate, '+.2f')} Sv decade$^{{-1}}$",
        )

    ax.scatter(
        [r["fovs_Sv"] for r in rows],
        [r["rate_Sv_dec"] for r in rows],
        s=11,
        color=DOT_COLOR,
        linewidth=0,
        zorder=4,
    )
    for r in rows:
        dx, dy, ha, va = LABEL_OFFSETS.get(r["label"], DEFAULT_OFFSET)
        ax.annotate(
            str(r["label"]),
            (r["fovs_Sv"], r["rate_Sv_dec"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=5,
            color=DOT_COLOR,
            ha=ha,
            va=va,
            zorder=5,
        )

    ax.set_xlim(*XLIM)
    ax.set_ylim(-1.95, 0.95)
    ax.set_xlabel("Model mean $F_{ovS}$ (Sv)")
    ax.set_ylabel("AMOC(26.5$\\degree$N) trend (Sv decade$^{-1}$)")
    ax.tick_params(width=0.4, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Upper left is the only region no reference line crosses.
    ax.legend(frameon=False, loc="upper left", handlelength=2.4, borderaxespad=0.2)


def main() -> None:
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    models = load_models()
    wp8 = json.loads(WP8_JSON.read_text())

    set_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.6), gridspec_kw={"width_ratios": [1.0, 1.12]}
    )
    panel_a(axes[0], models)
    panel_b(axes[1], wp8)

    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.11, top=0.955, wspace=0.30)
    for xpos, letter in ((0.010, "a"), (0.500, "b")):
        fig.text(
            xpos, 0.972, letter, fontsize=7, fontweight="bold", ha="left", va="bottom"
        )
    fig.savefig(OUT_BASE.with_suffix(".pdf"))
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=300)
    plt.close(fig)

    values = [v for _, v in models]
    n_neg = sum(1 for v in values if v < 0)
    print(f"Saved {OUT_BASE}.pdf / .png")
    n_pos = len(models) - n_neg
    print(f"panel a: n = {len(models)} models, {n_neg} negative, {n_pos} positive")
    print(f"panel a: min {min(values):+.4f}, max {max(values):+.4f}")
    for i, (name, value) in enumerate(models, start=1):
        print(f"  a{i:2d}  {name:16s} {value:+.4f}")
    print(f"panel b: window {wp8['window']}, n = {wp8['n_models']} models")
    for r in wp8["models"]:
        print(
            f"  b{r['label']:2d}  {r['model']:16s} "
            f"F_ovS {r['fovs_Sv']:+.3f}  rate {r['rate_Sv_dec']:+.3f}  p {r['p']:.3f}"
        )
    for name in OBS_RATE_STYLE:
        o = wp8["observations"][name]
        print(f"  obs {name:9s} rate {o['rate_Sv_dec']:+.3f} Sv/dec  p {o['p']:.3f}")


if __name__ == "__main__":
    main()
