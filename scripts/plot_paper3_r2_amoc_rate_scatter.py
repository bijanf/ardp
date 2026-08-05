#!/usr/bin/env python3
"""PAPER_3 round-2 WP8: fig:amoc_rate rebuilt as a continuous scatter.

Replaces the bistable/monostable bar chart with a plain scatter of model mean
F_ovS against the AMOC(26.5 N) trend over the RAPID overlap window. There is no
stability classification anywhere: no red/blue split, no threshold shading, no
class labels. The F_ovS = 0 line is drawn only as a thin grey reference.

Loaders are reused from scripts/plot_amoc_rate_comparison.py (`load_cmip6`,
`compute_rate`) and the window is taken from data/results/rapid_amoc26n.npz, so
the numbers match the figure this replaces.

Outputs:
    revision/rev_papaer3_02/figures/WP8_amoc_rate_scatter.pdf (+ .png)
    revision/rev_papaer3_02/results/WP8_scatter_stats.md
    revision/rev_papaer3_02/results/WP8_scatter_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from plot_amoc_rate_comparison import compute_rate, load_cmip6  # noqa: E402

from ardp.models import models_sorted_by_fovs  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_FIG = REPO / "revision" / "rev_papaer3_02" / "figures"
OUT_RES = REPO / "revision" / "rev_papaer3_02" / "results"

# Grayscale-distinguishable: different dash patterns, all neutral colours.
OBS_STYLE = {
    "RAPID": ("-", "#111111", 1.0),
    "ORAS5": ("--", "#444444", 1.0),
    "GLORYS12": (":", "#777777", 1.2),
}


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    rapid = np.load(RESULTS / "rapid_amoc26n.npz")
    y0, y1 = int(rapid["years"][0]), int(rapid["years"][-1])
    oras5 = np.load(RESULTS / "yearly_amoc26n_oras5.npz")
    glorys = np.load(RESULTS / "yearly_amoc26n_glorys12.npz")

    obs = {}
    for name, d in (("RAPID", rapid), ("ORAS5", oras5), ("GLORYS12", glorys)):
        rate, pval = compute_rate(d["years"], d["amoc"], y0, y1)
        obs[name] = {"rate_Sv_dec": float(rate), "p": float(pval)}

    cmip6 = load_cmip6(RESULTS)
    rows = []
    for model, fovs in models_sorted_by_fovs():
        if model not in cmip6:
            continue
        d = cmip6[model]
        rate, pval = compute_rate(d["years"], d["amoc"], y0, y1)
        if not np.isfinite(rate):
            continue
        rows.append(
            {
                "model": model,
                "fovs_Sv": float(fovs),
                "rate_Sv_dec": float(rate),
                "p": float(pval),
            }
        )
    rows.sort(key=lambda r: r["fovs_Sv"])
    for i, r in enumerate(rows, start=1):
        r["label"] = i

    x = np.array([r["fovs_Sv"] for r in rows])
    y = np.array([r["rate_Sv_dec"] for r in rows])

    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)

    # ── figure ────────────────────────────────────────────────────────
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

    fig, ax = plt.subplots(figsize=(3.46, 3.0))

    # F_ovS = 0 reference, thin and grey, no shading and no class labels.
    ax.axvline(0.0, color="0.72", linewidth=0.5, zorder=1)
    ax.axhline(0.0, color="0.85", linewidth=0.4, zorder=1)

    for name, (ls, col, lw) in OBS_STYLE.items():
        ax.axhline(
            obs[name]["rate_Sv_dec"],
            color=col,
            linewidth=lw,
            linestyle=ls,
            zorder=2,
            label=f"{name} {obs[name]['rate_Sv_dec']:+.2f} Sv dec$^{{-1}}$",
        )

    ax.scatter(
        x, y, s=22, facecolor="white", edgecolor="#20558a", linewidth=0.8, zorder=4
    )
    for r in rows:
        ax.annotate(
            str(r["label"]),
            (r["fovs_Sv"], r["rate_Sv_dec"]),
            fontsize=4.2,
            color="#20558a",
            ha="center",
            va="center",
            zorder=5,
        )

    ax.set_xlabel("Model mean $F_{ovS}$ (Sv)")
    ax.set_ylabel("AMOC(26.5°N) trend (Sv decade$^{-1}$)")
    # Upper left is the only empty quadrant; lower left would sit on the ORAS5
    # and GLORYS12 reference lines.
    ax.legend(frameon=False, loc="upper left", fontsize=5, handlelength=2.4)
    ymin = min(y.min(), *(o["rate_Sv_dec"] for o in obs.values()))
    ymax = max(y.max(), 0.0)
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + 3.2 * pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.4, length=2)

    out = OUT_FIG / "WP8_amoc_rate_scatter"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.pdf / .png")

    # ── stats ─────────────────────────────────────────────────────────
    payload = {
        "window": f"{y0}-{y1} (RAPID overlap)",
        "n_models": len(rows),
        "pearson": {"r": float(pear.statistic), "p": float(pear.pvalue)},
        "spearman": {"rho": float(spear.statistic), "p": float(spear.pvalue)},
        "observations": obs,
        "models": rows,
    }
    (OUT_RES / "WP8_scatter_stats.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# WP8. AMOC rate versus F_ovS, continuous scatter",
        "",
        f"Window: {y0}-{y1}, the RAPID overlap, taken from",
        "`data/results/rapid_amoc26n.npz`. x is each model's mean F_ovS (Sv),",
        "y is its AMOC(26.5 N) trend (Sv/decade) over that window, computed with",
        "`compute_rate` from `scripts/plot_amoc_rate_comparison.py`.",
        "",
        "The figure carries no stability dichotomy: no bistable/monostable",
        "colouring, no threshold shading, no class labels. F_ovS = 0 appears only",
        "as a thin grey reference line.",
        "",
        "## Correlation of x against y",
        "",
        "| Statistic | Value | p |",
        "|---|---|---|",
        f"| Pearson r | {pear.statistic:+.4f} | {pear.pvalue:.4f} |",
        f"| Spearman rho | {spear.statistic:+.4f} | {spear.pvalue:.4f} |",
        "",
        f"n = {len(rows)} CMIP6 models.",
        "",
        "## Observed reference lines",
        "",
        "| Product | Trend (Sv/decade) | p |",
        "|---|---|---|",
    ]
    for name in ("RAPID", "ORAS5", "GLORYS12"):
        lines.append(
            f"| {name} | {obs[name]['rate_Sv_dec']:+.3f} | {obs[name]['p']:.3f} |"
        )
    lines += [
        "",
        "## Key to the numbered dots",
        "",
        "Dots are numbered in order of increasing F_ovS. The key lives here",
        "rather than on the canvas because 16 model names are not legible at",
        "88 mm width.",
        "",
        "| # | Model | Mean F_ovS (Sv) | AMOC trend (Sv/decade) | p |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['model']} | {r['fovs_Sv']:+.4f} | "
            f"{r['rate_Sv_dec']:+.3f} | {r['p']:.3f} |"
        )
    lines.append("")
    (OUT_RES / "WP8_scatter_stats.md").write_text("\n".join(lines))
    print(
        f"Pearson r={pear.statistic:+.4f} (p={pear.pvalue:.4f}); "
        f"Spearman rho={spear.statistic:+.4f} (p={spear.pvalue:.4f})"
    )
    print(f"Saved {OUT_RES / 'WP8_scatter_stats.md'}")


if __name__ == "__main__":
    main()
