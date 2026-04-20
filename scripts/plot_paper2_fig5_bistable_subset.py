#!/usr/bin/env python3
"""Figure 5: mechanism-conditional AMOC projection, bistable-baseline subset.

Because the salt-advection feedback is physically active only when
F_ovS < 0 at baseline, the CMIP6 sub-ensemble that is *directly*
comparable to the observational reanalyses is the 6 models with a
bistable baseline. Of those 6, 4 show a forced F_ovS decline (and are
therefore eligible for the mechanism-conditional projection analysis of
Fig. 4): CNRM-CM6-1, MIROC6, MPI-ESM1-2-HR (salinity-dominant) and
NESM3 (velocity-dominant).

This figure shows the same boxplot-vs-Portmann comparison as Fig. 4
but restricted to the bistable subset. The sample sizes are small,
hence quantitative medians should be interpreted with caution — but
this is the cleanest like-with-like comparison between observational
reanalyses (all bistable) and CMIP6.

Reads: data/results/fovs_decomposition_cmip6_summary.csv,
       data/results/yearly_amoc26n_cmip6.npz
Writes: figures/paper2/fig5_bistable_subset.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from ardp.viz.style import apply_nature_style, save_publication_figure


def _classify(row):
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


CLASS_COLORS = {
    "v-dominant": "#E69F00",
    "s-dominant": "#56B4E9",
    "mixed":      "#009E73",
    "increasing": "0.5",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/paper2/fig5_bistable_subset"),
    )
    args = parser.parse_args()

    apply_nature_style()

    df = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_summary.csv")
    df["class"] = df.apply(_classify, axis=1)
    df["bistable"] = df["F_ov_baseline"] < 0

    amoc = np.load(args.results_dir / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    amoc_models = [str(m) for m in amoc["models"]]

    full = df[df["class"] != "increasing"]
    bistable_weak = df[(df["class"] != "increasing") & (df["bistable"])]

    def _pct_weakening(model_list):
        out = []
        for m in model_list:
            if m not in amoc_models:
                continue
            y = amoc[f"{m}_years"]
            a = amoc[f"{m}_amoc"]
            base = np.nanmean(a[(y >= 1950) & (y <= 1980)])
            end = np.nanmean(a[(y >= 2081) & (y <= 2100)])
            if np.isfinite(base) and np.isfinite(end) and base > 0:
                out.append(100 * (1 - end / base))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # ── Panel (a): full ensemble (same as Fig 4b but reproduced for reference) ──
    full_data = [_pct_weakening(full[full["class"] == c]["model"].tolist())
                 for c in ("v-dominant", "s-dominant", "mixed")]
    labels_full = [f"v-dom\n(n={len(full_data[0])})",
                   f"s-dom\n(n={len(full_data[1])})",
                   f"mixed\n(n={len(full_data[2])})"]
    bp_full = ax1.boxplot(full_data, tick_labels=labels_full, patch_artist=True,
                          widths=0.6,
                          medianprops={"color": "black", "linewidth": 1.0})
    for patch, cls in zip(bp_full["boxes"],
                          ("v-dominant", "s-dominant", "mixed")):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.55)
    ax1.fill_between([-0.5, 2.5], 43, 59, color="#CC3333", alpha=0.12, zorder=1)
    ax1.axhline(51, color="#CC3333", lw=1.3, zorder=2)
    ax1.text(0.02, 0.98, r"$\!$Portmann et al.$\,$2026: 51 $\pm$ 8%",
             transform=ax1.transAxes, fontsize=6.5, color="#CC3333",
             ha="left", va="top", fontweight="bold",
             bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                   "edgecolor": "#CC3333", "linewidth": 0.5, "alpha": 0.85})
    ax1.set_ylabel("AMOC weakening by 2100 (%, 2081–2100 vs 1950–1980)")
    ax1.set_title("(a) Full forced-weakening ensemble (n=12)",
                  fontweight="bold")
    ax1.set_ylim(None, 95)
    ax1.grid(axis="y", alpha=0.3, lw=0.3)

    # ── Panel (b): bistable-only subset (n=4, the observation-analogous set) ──
    bis_data = [_pct_weakening(bistable_weak[bistable_weak["class"] == c]["model"].tolist())
                for c in ("v-dominant", "s-dominant", "mixed")]
    labels_bis = [f"v-dom\n(n={len(bis_data[0])})",
                  f"s-dom\n(n={len(bis_data[1])})",
                  f"mixed\n(n={len(bis_data[2])})"]

    # Use individual points + small boxplot since sample sizes are small
    positions = [0, 1, 2]
    for pos, vals, cls in zip(positions, bis_data,
                              ("v-dominant", "s-dominant", "mixed")):
        if len(vals) == 0:
            continue
        ax2.scatter([pos] * len(vals), vals, color=CLASS_COLORS[cls],
                    edgecolor="0.2", linewidth=0.6, s=80, zorder=5)
        if len(vals) >= 2:
            ax2.hlines(np.median(vals), pos - 0.25, pos + 0.25,
                       color="black", lw=1.5, zorder=6)

    ax2.fill_between([-0.5, 2.5], 43, 59, color="#CC3333", alpha=0.12, zorder=1)
    ax2.axhline(51, color="#CC3333", lw=1.3, zorder=2)
    ax2.text(0.02, 0.98, r"$\!$Portmann et al.$\,$2026: 51 $\pm$ 8%",
             transform=ax2.transAxes, fontsize=6.5, color="#CC3333",
             ha="left", va="top", fontweight="bold",
             bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                   "edgecolor": "#CC3333", "linewidth": 0.5, "alpha": 0.85})
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_bis)
    ax2.set_ylabel("AMOC weakening by 2100 (%)")
    ax2.set_title("(b) Bistable-baseline subset only (n=4)",
                  fontweight="bold")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(-0.5, 2.5)
    ax2.grid(axis="y", alpha=0.3, lw=0.3)

    # Annotate bistable models directly. Use adjust_text so that
    # ties (e.g. two s-dominant models with nearly identical
    # projected weakening) do not stack on top of each other.
    bis_texts = []
    for pos, vals, cls, models in zip(
        positions, bis_data, ("v-dominant", "s-dominant", "mixed"),
        [bistable_weak[bistable_weak["class"] == c]["model"].tolist()
         for c in ("v-dominant", "s-dominant", "mixed")]):
        for v, m in zip(vals, models):
            short = m.replace("-CM6-1", "").replace("-0-LL", "").replace("-GC31-LL", "")
            bis_texts.append(ax2.text(pos, v, short,
                                      fontsize=5.5, color="0.2",
                                      zorder=7))
    if bis_texts:
        adjust_text(
            bis_texts, ax=ax2,
            expand=(1.3, 1.6),
            only_move={"text": "xy", "static": "xy"},
            arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.35},
        )

    fig.suptitle("Mechanism-conditional AMOC weakening: full vs bistable-only ensemble",
                 y=1.00, fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_publication_figure(fig, args.output)

    print("Per-class projected AMOC weakening (%):")
    for ens_name, data in [("FULL", full_data), ("BISTABLE-ONLY", bis_data)]:
        print(f"  {ens_name}:")
        for c, vals in zip(("v-dominant", "s-dominant", "mixed"), data):
            if vals:
                print(f"    {c:<12s} n={len(vals):>2d}  median={np.median(vals):+.1f}%"
                      f"  IQR=[{np.percentile(vals, 25):+.1f}, "
                      f"{np.percentile(vals, 75):+.1f}]%")


if __name__ == "__main__":
    main()
