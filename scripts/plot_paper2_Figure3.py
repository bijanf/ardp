#!/usr/bin/env python3
"""Paper 2 Figure 3: AMOC projections for the bistable-only subset.

Single panel: projected AMOC weakening by 2100 for CMIP6 models with a
bistable baseline (F_ovS < 0 in 1950-1980), partitioned by mechanism
class. This is the like-with-like comparison to the observational
reanalyses. The full weakening ensemble is NOT repeated here (it is
already shown in the preceding figure); the caption should
cross-reference that panel instead (R2.m5 / R3.11).

Outputs:
  figures/paper2/Figure3.{png,pdf}        single-panel figure

Reads:
  data/results/fovs_decomposition_cmip6_summary.csv
  data/results/yearly_amoc26n_cmip6.npz
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


def _load_data(results_dir: Path):
    df = pd.read_csv(results_dir / "fovs_decomposition_cmip6_summary.csv")
    df["class"] = df.apply(_classify, axis=1)
    df["bistable"] = df["F_ov_baseline"] < 0
    amoc = np.load(results_dir / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    amoc_models = [str(m) for m in amoc["models"]]
    full = df[df["class"] != "increasing"]
    bistable_weak = df[(df["class"] != "increasing") & df["bistable"]]

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

    _ = full  # full-ensemble panel removed (duplicated the previous figure)
    bis_data = [
        _pct_weakening(bistable_weak[bistable_weak["class"] == c]["model"].tolist())
        for c in ("v-dominant", "s-dominant", "mixed")]
    bis_models = [bistable_weak[bistable_weak["class"] == c]["model"].tolist()
                  for c in ("v-dominant", "s-dominant", "mixed")]
    return bis_data, bis_models


def _draw_panel_b(ax, bis_data, bis_models, *, ylim=(None, 95)):
    labels_bis = [f"v-dom\n(n={len(bis_data[0])})",
                  f"s-dom\n(n={len(bis_data[1])})",
                  f"mixed\n(n={len(bis_data[2])})"]
    positions = [0, 1, 2]
    for pos, vals, cls in zip(positions, bis_data,
                              ("v-dominant", "s-dominant", "mixed"),
                              strict=True):
        if not vals:
            continue
        ax.scatter([pos] * len(vals), vals, color=CLASS_COLORS[cls],
                   edgecolor="0.2", linewidth=0.6, s=85, zorder=5)
        # Class median as a short tick to the LEFT of the cluster so it
        # never overlaps a model marker or label.
        if len(vals) >= 2:
            ax.hlines(np.median(vals), pos - 0.42, pos - 0.28,
                      color="black", lw=1.5, zorder=6)
    ax.fill_between([-0.5, 2.5], 43, 59, color="#CC3333", alpha=0.12, zorder=1)
    ax.axhline(51, color="#CC3333", lw=1.3, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_bis)
    ax.set_ylabel("AMOC weakening by 2100 (%)")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(axis="y", alpha=0.3, lw=0.3)

    bis_texts = []
    for pos, vals, models in zip(positions, bis_data, bis_models, strict=True):
        # vals may be shorter than models when a model lacks AMOC data,
        # so this zip is deliberately non-strict.
        for v, m in zip(vals, models, strict=False):
            short = (m.replace("-CM6-1", "")
                     .replace("-0-LL", "")
                     .replace("-GC31-LL", ""))
            bis_texts.append(ax.text(pos, v, short, fontsize=7.5,
                                     color="0.2", zorder=7))
    if bis_texts:
        adjust_text(bis_texts, ax=ax, expand=(1.3, 1.6),
                    only_move={"text": "xy", "static": "xy"},
                    arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.35})


def _render(bis_data, bis_models, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    _draw_panel_b(ax, bis_data, bis_models)
    fig.tight_layout()
    save_publication_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure3"))
    args = parser.parse_args()

    apply_nature_style()
    bis_data, bis_models = _load_data(args.results_dir)
    _render(bis_data, bis_models, args.output)


if __name__ == "__main__":
    main()
