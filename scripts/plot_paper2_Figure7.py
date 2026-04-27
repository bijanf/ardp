#!/usr/bin/env python3
"""Paper 2 Figure 7: SMILE robustness of the mechanism classification.

Two panels (one combined PDF):

  (a) F_ovS velocity-share vs salinity-share for all 50 initial-
      condition members of the MPI-ESM1-2-LR Grand Ensemble (CMIP6
      contribution), overlaid on the 13-model multi-model CMIP6
      weakening ensemble.
  (b) AMOC time series at 26.5°N (max of msftmz below 500 m), 1850-
      2100, hist+ssp585. 50-member spaghetti, ensemble mean, 1σ
      envelope, and per-member-weakening histogram inset.

Outputs (default --mode both):
  figures/paper2/Figure7.{png,pdf}        single combined PDF
  figures/paper2/Figure7{a,b}.{png,pdf}   two standalone panels

Reads:
  data/results/fovs_decomposition_smile_esgf.csv      (50 rows)
  data/results/fovs_decomposition_cmip6_summary.csv   (15 multi-model rows)
  data/results/smile_amoc26n_mpi_lr.npz               (50 trajectories)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _panel_label(ax, label: str, x: float = 0.0, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def _load_data(results_dir: Path):
    smile = pd.read_csv(results_dir / "fovs_decomposition_smile_esgf.csv")
    smile["class"] = smile.apply(_classify, axis=1)
    cmip6 = pd.read_csv(results_dir / "fovs_decomposition_cmip6_summary.csv")
    cmip6["class"] = cmip6.apply(_classify, axis=1)
    npz = np.load(results_dir / "smile_amoc26n_mpi_lr.npz", allow_pickle=True)
    members = [str(m) for m in npz["members"]]
    common_years = np.arange(1850, 2101)
    matrix = np.full((len(members), len(common_years)), np.nan)
    for i, m in enumerate(members):
        y = npz[f"{m}_years"].astype(float)
        a = npz[f"{m}_amoc"].astype(float)
        for j, yr in enumerate(common_years):
            idx = np.where(y == yr)[0]
            if idx.size:
                matrix[i, j] = a[idx[0]]
    return smile, cmip6, members, common_years, matrix


def _draw_panel_a(ax, smile, cmip6):
    # Backdrop: multi-model CMIP6 ensemble
    for cls in ("v-dominant", "s-dominant", "mixed", "increasing"):
        sub = cmip6[cmip6["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                   s=70, color=CLASS_COLORS[cls], alpha=0.30,
                   edgecolor="0.4", linewidth=0.4, zorder=3,
                   label=f"CMIP6 {cls} ({len(sub)})")
    # SMILE members
    for cls in ("v-dominant", "s-dominant", "mixed", "increasing"):
        sub = smile[smile["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                   s=40, marker="x", color=CLASS_COLORS[cls],
                   linewidth=1.5, zorder=5,
                   label=f"MPI-ESM1-2-LR SMILE {cls} ({len(sub)})")
    # SMILE mean
    if len(smile):
        ax.scatter([smile["velocity_share_pct"].mean()],
                   [smile["salinity_share_pct"].mean()],
                   s=70, marker="D", c="black", edgecolor="white",
                   linewidth=0.8, zorder=8, label="SMILE mean")
    # Class boundaries + diagonal
    ax.plot([0, 100], [100, 0], color="0.6", lw=0.6, ls="--", zorder=1)
    ax.axvline(60, color="#E69F00", lw=0.4, ls=":", alpha=0.6)
    ax.axhline(60, color="#56B4E9", lw=0.4, ls=":", alpha=0.6)
    ax.set_xlabel(r"Velocity share $f_v$  (%)")
    ax.set_ylabel(r"Salinity share $f_s$  (%)")
    ax.set_xlim(-80, 220)
    ax.set_ylim(-150, 220)
    ax.legend(loc="lower left", fontsize=5.5, frameon=False, ncol=1,
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.2)


def _draw_panel_b(ax, members, years, matrix, ax_hist=None):
    base_mask = (years >= 1950) & (years <= 1980)
    end_mask = (years >= 2081) & (years <= 2100)
    base = np.nanmean(matrix[:, base_mask], axis=1)
    end = np.nanmean(matrix[:, end_mask], axis=1)
    weakening_pct = 100 * (1 - end / base)

    # Spaghetti
    for i in range(len(members)):
        ax.plot(years, matrix[i], color="#E69F00", alpha=0.20, lw=0.5,
                zorder=2)
    mean_t = np.nanmean(matrix, axis=0)
    std_t = np.nanstd(matrix, axis=0)
    ax.fill_between(years, mean_t - std_t, mean_t + std_t,
                    color="#E69F00", alpha=0.30, zorder=4,
                    label="$\\pm 1\\sigma$ envelope")
    ax.plot(years, mean_t, color="#B45F00", lw=2.0, zorder=6,
            label="Ensemble mean")
    ax.axvline(2014, color="0.4", lw=0.8, ls="--", zorder=1,
               label="hist $\\rightarrow$ ssp585")
    ax.axvspan(1950, 1980, color="0.85", alpha=0.4, zorder=0)
    ax.axvspan(2081, 2100, color="0.85", alpha=0.4, zorder=0)
    ax.set_xlim(1850, 2100)
    ax.set_xlabel("Year")
    ax.set_ylabel("AMOC at 26.5°N  (Sv)")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)

    if ax_hist is not None:
        ax_hist.hist(weakening_pct, bins=12, orientation="horizontal",
                     color="#E69F00", alpha=0.7, edgecolor="0.2",
                     linewidth=0.4)
        ax_hist.axhline(weakening_pct.mean(), color="#B45F00", lw=1.5)
        ax_hist.set_xlabel("# members")
        ax_hist.set_ylabel("Weakening 2100  (%)")


def _render_combined(smile, cmip6, members, years, matrix, output: Path):
    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(1, 8, wspace=0.7)
    ax_a = fig.add_subplot(gs[0, 0:3])    # 38% — wider so legend fits
    ax_b = fig.add_subplot(gs[0, 3:7])    # 50%
    ax_hist = fig.add_subplot(gs[0, 7:8]) # 12%
    _draw_panel_a(ax_a, smile, cmip6)
    _panel_label(ax_a, "(a)")
    _draw_panel_b(ax_b, members, years, matrix, ax_hist=ax_hist)
    _panel_label(ax_b, "(b)")
    fig.tight_layout()
    save_publication_figure(fig, output)


def _render_split(smile, cmip6, members, years, matrix, output: Path):
    base = output.parent / output.name
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    _draw_panel_a(ax, smile, cmip6)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "a"))
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(1, 4, wspace=0.55)
    ax_b = fig.add_subplot(gs[0, 0:3])
    ax_hist = fig.add_subplot(gs[0, 3])
    _draw_panel_b(ax_b, members, years, matrix, ax_hist=ax_hist)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "b"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure7"))
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    smile, cmip6, members, years, matrix = _load_data(args.results_dir)

    if args.mode in ("combined", "both"):
        _render_combined(smile, cmip6, members, years, matrix, args.output)
    if args.mode in ("split", "both"):
        _render_split(smile, cmip6, members, years, matrix, args.output)


if __name__ == "__main__":
    main()
