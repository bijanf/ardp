#!/usr/bin/env python3
"""Paper 2 SI Figure S9 — SMILE AMOC trajectory band at 26.5°N.

Plots the 50-member MPI-ESM1-2-LR Grand Ensemble AMOC time series at
26.5°N (1850–2100, hist+ssp585), with the ensemble mean ± 1σ envelope
overlaid. Inset histogram shows the 2080-2100-vs-1950-1980 percent
weakening per member.

This complements Supp Fig S7 (mechanism class) — together they confirm
the SMILE mechanism class is robust AND the projected AMOC weakening
range is internally consistent across initial-condition realisations.

Reads:
  data/results/smile_amoc26n_mpi_lr.npz

Outputs:
  figures/paper2/diagSMILE_amoc.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ardp.viz.style import apply_nature_style, save_publication_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/results/smile_amoc26n_mpi_lr.npz"))
    parser.add_argument("--out-fig", type=Path,
                        default=Path("figures/paper2/diagSMILE_amoc"))
    args = parser.parse_args()

    apply_nature_style()
    npz = np.load(args.input, allow_pickle=True)
    members = [str(m) for m in npz["members"]]
    print(f"\nSMILE members loaded: {len(members)}")

    # Build a common-year matrix (1850-2100)
    common_years = np.arange(1850, 2101)
    matrix = np.full((len(members), len(common_years)), np.nan)
    for i, m in enumerate(members):
        y = npz[f"{m}_years"].astype(float)
        a = npz[f"{m}_amoc"].astype(float)
        for j, yr in enumerate(common_years):
            idx = np.where(y == yr)[0]
            if idx.size:
                matrix[i, j] = a[idx[0]]

    # Per-member 2080-2100 vs 1950-1980 weakening %
    base_mask = (common_years >= 1950) & (common_years <= 1980)
    end_mask = (common_years >= 2081) & (common_years <= 2100)
    base = np.nanmean(matrix[:, base_mask], axis=1)
    end = np.nanmean(matrix[:, end_mask], axis=1)
    weakening_pct = 100 * (1 - end / base)
    print(f"  AMOC 1950-1980 ensemble mean:   {np.nanmean(base):+.2f} Sv  "
          f"(σ = {np.nanstd(base):.2f})")
    print(f"  AMOC 2081-2100 ensemble mean:   {np.nanmean(end):+.2f} Sv  "
          f"(σ = {np.nanstd(end):.2f})")
    print("  Weakening 2100 vs 1950-80:")
    print(f"     mean: {weakening_pct.mean():+.1f}%   "
          f"σ: {weakening_pct.std():.1f}%   "
          f"range: [{weakening_pct.min():+.1f}%, {weakening_pct.max():+.1f}%]")

    # ── Plot ──
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(1, 4, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0:3])
    ax_hist = fig.add_subplot(gs[0, 3])

    # Spaghetti
    for i in range(len(members)):
        ax.plot(common_years, matrix[i], color="#E69F00", alpha=0.20,
                lw=0.5, zorder=2)
    # Ensemble mean ± 1σ
    mean_t = np.nanmean(matrix, axis=0)
    std_t = np.nanstd(matrix, axis=0)
    ax.fill_between(common_years, mean_t - std_t, mean_t + std_t,
                    color="#E69F00", alpha=0.30, zorder=4,
                    label=f"$\\pm 1\\sigma$ envelope (n={len(members)})")
    ax.plot(common_years, mean_t, color="#B45F00", lw=2.0, zorder=6,
            label="Ensemble mean")
    ax.axvline(2014, color="0.4", lw=0.8, ls="--", zorder=1,
               label="hist $\\rightarrow$ ssp585")
    ax.axvspan(1950, 1980, color="0.85", alpha=0.4, zorder=0)
    ax.axvspan(2081, 2100, color="0.85", alpha=0.4, zorder=0)
    ax.set_xlim(1850, 2100)
    ax.set_xlabel("Year")
    ax.set_ylabel("AMOC at 26.5°N  (Sv)")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    # Numerical summary (mean weakening, σ, range) is reported in the
    # figure caption rather than on the axes.

    # Histogram of weakening %
    ax_hist.hist(weakening_pct, bins=12, orientation="horizontal",
                 color="#E69F00", alpha=0.7, edgecolor="0.2",
                 linewidth=0.4)
    ax_hist.axhline(weakening_pct.mean(), color="#B45F00", lw=1.5,
                    label=f"mean={weakening_pct.mean():+.1f}%")
    ax_hist.set_xlabel("# members")
    ax_hist.set_ylabel("Weakening 2100 vs 1950-80  (%)")
    ax_hist.legend(loc="upper right", fontsize=7, frameon=False)

    fig.tight_layout()
    save_publication_figure(fig, args.out_fig)


if __name__ == "__main__":
    main()
