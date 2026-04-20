#!/usr/bin/env python3
"""Figure 4 (new): mechanism-conditional AMOC projections.

Instead of a naive single-variable emergent constraint, this figure
conditions CMIP6 projections on which mechanism dominates the F_ovS
trend in each model:
  - v-dominant subset (velocity-share > 60%)
  - s-dominant subset (salinity-share > 60%)
  - mixed subset

Headline scientific message:
  "The observed AMOC trajectory bounds depend on which reanalysis
   interpretation is correct (ORAS5 v-dominant vs GLORYS12 s-dominant).
   CMIP6 models with s-dominant mechanism (matching GLORYS12) project
   [X]% AMOC weakening by 2100; v-dominant models (matching ORAS5)
   project [Y]%. Recent observational-constraint studies (e.g. Portmann
   et al. 2026) average over this mechanistic diversity and should be
   interpreted conditional on the true physical regime."

Panels:
  (a) CMIP6 AMOC26N trajectories coloured by mechanism class (spaghetti),
      with subset ensemble means and ±1σ ribbons.
  (b) Boxplots of projected %AMOC weakening (2080-2100 vs 1950-1980) by
      subset, with Portmann et al. 2026 (51 ± 8%) overplot.

Reads:
  data/results/fovs_decomposition_cmip6_summary.csv  (mechanism classes)
  data/results/yearly_amoc26n_cmip6.npz               (AMOC time series)

Outputs: figures/paper2/fig4_mechanism_conditional.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ardp.viz.style import apply_nature_style, save_publication_figure


def _classify(row: pd.Series) -> str:
    if row["delta_total"] >= -0.01:
        return "stable"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/fig4_mechanism_conditional"))
    args = parser.parse_args()

    apply_nature_style()

    # Mechanism classes
    summary = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_summary.csv")
    summary["class"] = summary.apply(_classify, axis=1)
    classes = {c: summary[summary["class"] == c]["model"].tolist()
               for c in ("v-dominant", "s-dominant", "mixed", "stable")}

    # AMOC data
    amoc = np.load(args.results_dir / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    amoc_models = [str(m) for m in amoc["models"]]

    CLASS_COLORS = {
        "v-dominant": "#E69F00",
        "s-dominant": "#56B4E9",
        "mixed":      "#009E73",
        "stable":     "0.6",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))

    # ── Panel (a): AMOC trajectories by class ──
    trajectories: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in classes}
    for cls, model_list in classes.items():
        for m in model_list:
            if m not in amoc_models:
                continue
            y = amoc[f"{m}_years"]
            a = amoc[f"{m}_amoc"]
            ax1.plot(y, a, color=CLASS_COLORS[cls], alpha=0.25, lw=0.5, zorder=2)
            trajectories[cls].append((y, a))

    # Ensemble means (interpolate to common grid 1850-2100)
    common_years = np.arange(1850, 2101)
    for cls in ("v-dominant", "s-dominant", "mixed"):
        if not trajectories[cls]:
            continue
        matrix = np.full((len(trajectories[cls]), len(common_years)), np.nan)
        for j, (y, a) in enumerate(trajectories[cls]):
            for k, yr in enumerate(common_years):
                idx = np.where(y == yr)[0]
                if idx.size:
                    matrix[j, k] = a[idx[0]]
        with np.errstate(all="ignore"):
            mean_a = np.nanmean(matrix, axis=0)
            std_a = np.nanstd(matrix, axis=0)
        n = len(trajectories[cls])
        ax1.plot(common_years, mean_a, color=CLASS_COLORS[cls], lw=2.2, zorder=6,
                 label=f"{cls}  (n={n})")
        ax1.fill_between(common_years, mean_a - std_a, mean_a + std_a,
                         color=CLASS_COLORS[cls], alpha=0.15, zorder=4)

    ax1.axvline(2014, color="0.5", lw=0.4, ls=":", zorder=1)
    ax1.text(2014, ax1.get_ylim()[1] if ax1.get_ylim()[1] else 25,
             " hist→ssp585", fontsize=5.5, color="0.4", va="top")
    ax1.set_xlim(1850, 2100)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("AMOC at 26.5°N  (Sv)")
    ax1.set_title("(a) AMOC trajectories by F$_{ovS}$ mechanism class",
                  fontweight="bold")
    ax1.legend(loc="lower left", fontsize=6, frameon=False)

    # ── Panel (b): boxplot of % weakening by class ──
    weakening_pct = {c: [] for c in ("v-dominant", "s-dominant", "mixed")}
    for cls, model_list in classes.items():
        if cls == "stable":
            continue
        for m in model_list:
            if m not in amoc_models:
                continue
            y = amoc[f"{m}_years"]
            a = amoc[f"{m}_amoc"]
            base_mask = (y >= 1950) & (y <= 1980)
            end_mask = (y >= 2081) & (y <= 2100)
            if base_mask.sum() >= 10 and end_mask.sum() >= 10:
                base = float(np.nanmean(a[base_mask]))
                end = float(np.nanmean(a[end_mask]))
                if base > 0:
                    weakening_pct[cls].append(100 * (base - end) / base)

    bp_data = [weakening_pct[c] for c in ("v-dominant", "s-dominant", "mixed")]
    bp_labels = [f"{c}\n(n={len(weakening_pct[c])})"
                 for c in ("v-dominant", "s-dominant", "mixed")]
    bp = ax2.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.6,
                     medianprops={"color": "black", "linewidth": 1.0})
    for patch, cls in zip(bp["boxes"], ("v-dominant", "s-dominant", "mixed")):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.65)

    # Overlay Portmann et al. 2026 constraint
    port_central = 51
    port_err = 8
    ax2.axhline(port_central, color="#CC3333", lw=1.5, zorder=5)
    ax2.fill_between([-0.5, len(bp_data) - 0.5],
                     port_central - port_err, port_central + port_err,
                     color="#CC3333", alpha=0.15, zorder=4)
    ax2.text(len(bp_data) - 0.55, port_central + 1,
             f"Portmann et al. 2026\n51 ± 8%",
             fontsize=5.8, color="#CC3333", ha="right", va="bottom")

    ax2.set_ylabel("AMOC weakening by 2100  (%, 2081–2100 vs 1950–1980)")
    ax2.set_title("(b) Projected weakening conditional on mechanism",
                  fontweight="bold")
    ax2.set_ylim(None, 95)
    ax2.grid(axis="y", alpha=0.3, lw=0.3)

    fig.tight_layout()
    save_publication_figure(fig, args.output)

    # Stats for the caption
    print("Per-class projected AMOC weakening (%):")
    for c in ("v-dominant", "s-dominant", "mixed"):
        vals = weakening_pct[c]
        if vals:
            print(f"  {c:12s} n={len(vals)}  median={np.median(vals):.1f}%  "
                  f"IQR=[{np.percentile(vals, 25):.1f}%, {np.percentile(vals, 75):.1f}%]")


if __name__ == "__main__":
    main()
