#!/usr/bin/env python3
"""Compare observed vs modeled AMOC weakening rates at 26.5°N.

Two panels:
  (a) Full ORAS5 period (1958–2025): ORAS5 vs CMIP6 models
  (b) RAPID overlap period (2004–2023): ORAS5 + GLORYS12 + RAPID vs CMIP6

Uses yearly AMOC(26.5°N) — directly comparable to RAPID observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import linregress

from ardp.models import models_sorted_by_fovs
from ardp.viz.style import apply_nature_style

CMIP6_MODELS = models_sorted_by_fovs()


def compute_rate(years, values, y0, y1):
    mask = (years >= y0) & (years <= y1)
    if mask.sum() < 5:
        return np.nan, np.nan
    res = linregress(years[mask], values[mask])
    return res.slope * 10, res.pvalue


def load_cmip6(results_dir):
    d = np.load(results_dir / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    models = d["models"]
    result = {}
    for model in models:
        result[str(model)] = {
            "years": d[f"{model}_years"],
            "amoc": d[f"{model}_amoc"],
            "fovs": float(d[f"{model}_fovs"][0]),
        }
    return result


def plot_panel(ax, model_rates, obs_lines, title, xlabel):
    model_rates.sort(key=lambda x: x[2])

    names = [m[0] for m in model_rates]
    rates = [m[2] for m in model_rates]
    fovs_vals = [m[1] for m in model_rates]
    colors = ["#CC3333" if f < 0 else "#3366AA" for f in fovs_vals]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, rates, height=0.7, color=colors, edgecolor="white",
            linewidth=0.5, zorder=3)

    for rate, style, color, lw in obs_lines:
        if np.isfinite(rate):
            ax.axvline(rate, color=color, linewidth=lw, linestyle=style, zorder=5)

    ax.axvline(0, color="0.5", linewidth=0.5, linestyle="-", zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12)
    del title  # npj style: caption supplies the title text
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    all_vals = rates + [r for r, _, _, _ in obs_lines if np.isfinite(r)]
    ax.set_xlim(min(all_vals) - 0.3, max(all_vals) + 0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/grl/fig_amoc_rate_comparison.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    apply_nature_style()

    # Load all data
    oras5 = np.load(args.results_dir / "yearly_amoc26n_oras5.npz")
    oras5_yr, oras5_am = oras5["years"], oras5["amoc"]
    print(f"ORAS5: {oras5_yr[0]}–{oras5_yr[-1]}, {len(oras5_yr)} years")

    glorys = np.load(args.results_dir / "yearly_amoc26n_glorys12.npz")
    glorys_yr, glorys_am = glorys["years"], glorys["amoc"]
    print(f"GLORYS12: {glorys_yr[0]}–{glorys_yr[-1]}, {len(glorys_yr)} years")

    rapid = np.load(args.results_dir / "rapid_amoc26n.npz")
    rapid_yr, rapid_am = rapid["years"], rapid["amoc"]
    print(f"RAPID: {rapid_yr[0]}–{rapid_yr[-1]}, {len(rapid_yr)} years")

    cmip6 = load_cmip6(args.results_dir)

    # ── Panel (a): Full ORAS5 period ──
    y0a, y1a = int(oras5_yr[0]), int(oras5_yr[-1])
    oras5_rate_a, oras5_p_a = compute_rate(oras5_yr, oras5_am, y0a, y1a)
    print(f"\n(a) ORAS5 {y0a}–{y1a}: {oras5_rate_a:+.2f} Sv/dec (p={oras5_p_a:.4f})")

    model_rates_a = []
    for model, fovs in CMIP6_MODELS:
        if model not in cmip6:
            continue
        d = cmip6[model]
        rate, pval = compute_rate(d["years"], d["amoc"], y0a, y1a)
        model_rates_a.append((model, fovs, rate, pval))

    # ── Panel (b): RAPID overlap period ──
    y0b, y1b = int(rapid_yr[0]), int(rapid_yr[-1])
    oras5_rate_b, oras5_p_b = compute_rate(oras5_yr, oras5_am, y0b, y1b)
    glorys_rate_b, glorys_p_b = compute_rate(glorys_yr, glorys_am, y0b, y1b)
    rapid_rate, rapid_p = compute_rate(rapid_yr, rapid_am, y0b, y1b)
    print(f"\n(b) Period {y0b}–{y1b}:")
    print(f"    ORAS5:   {oras5_rate_b:+.2f} Sv/dec (p={oras5_p_b:.4f})")
    print(f"    GLORYS12:{glorys_rate_b:+.2f} Sv/dec (p={glorys_p_b:.4f})")
    print(f"    RAPID:   {rapid_rate:+.2f} Sv/dec (p={rapid_p:.4f})")

    model_rates_b = []
    for model, fovs in CMIP6_MODELS:
        if model not in cmip6:
            continue
        d = cmip6[model]
        rate, pval = compute_rate(d["years"], d["amoc"], y0b, y1b)
        model_rates_b.append((model, fovs, rate, pval))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 9))

    plot_panel(
        ax, model_rates_b,
        obs_lines=[
            (oras5_rate_b, "-", "k", 2.5),
            (glorys_rate_b, "--", "#228B22", 2.5),
            (rapid_rate, "-", "#FF6600", 3.0),
        ],
        title=f"AMOC(26.5°N) trend, {y0b}–{y1b}",
        xlabel="AMOC(26.5°N) trend (Sv/decade)",
    )

    # Legend
    legend_elements = [
        Patch(facecolor="#CC3333", label="Bistable (F$_{ovS}$ < 0)"),
        Patch(facecolor="#3366AA", label="Monostable (F$_{ovS}$ > 0)"),
        Line2D([0], [0], color="#FF6600", linewidth=3,
               label=f"RAPID obs: {rapid_rate:+.2f} Sv/dec"),
        Line2D([0], [0], color="k", linewidth=2.5, linestyle="-",
               label=f"ORAS5: {oras5_rate_b:+.2f} Sv/dec"),
        Line2D([0], [0], color="#228B22", linewidth=2.5, linestyle="--",
               label=f"GLORYS12: {glorys_rate_b:+.2f} Sv/dec"),
    ]

    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11,
               framealpha=0.9, edgecolor="0.7", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 1.0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
