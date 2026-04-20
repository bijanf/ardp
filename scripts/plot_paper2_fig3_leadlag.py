#!/usr/bin/env python3
"""Figure 3: CMIP6 F_ovS-AMOC lead-lag relationships.

Three panels:
  (a) Cross-correlation curves for each CMIP6 model (grey spaghetti) with
      collapsing-model ensemble mean and ±1σ ribbon (red), plus stable-model
      ensemble mean (blue). X-axis: lag [years], positive = F_ovS leads AMOC.
  (b) Scatter of peak-lag vs. peak-|r| per model, coloured by end-century
      AMOC decline fraction.
  (c) Histogram of peak lags, separated by collapsing vs. stable.

Reads: data/results/cmip6_fovs_amoc_leadlag.nc
Outputs: figures/paper2/fig3_leadlag.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

COLLAPSE_THRESHOLD = 0.30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/results/cmip6_fovs_amoc_leadlag.nc"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/paper2/fig3_leadlag"),
    )
    args = parser.parse_args()

    apply_nature_style()

    ds = xr.open_dataset(args.input)
    lags = ds["lag"].values
    ccf = ds["ccf"].values  # (model, lag)
    peak_lag = ds["peak_lag"].values
    peak_r = ds["peak_r"].values
    decline = ds["amoc_decline_frac"].values
    models = [str(m) for m in ds["model"].values]
    ds.close()

    collapsing = decline > COLLAPSE_THRESHOLD
    stable = ~collapsing

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.3))

    # ── Panel (a): CCF spaghetti + ensemble means ──
    for i in range(len(models)):
        c = "#CC3333" if collapsing[i] else "#3366AA"
        ax1.plot(lags, ccf[i, :], color=c, alpha=0.25, lw=0.6, zorder=2)

    if collapsing.sum() > 1:
        mean_c = np.nanmean(ccf[collapsing, :], axis=0)
        std_c = np.nanstd(ccf[collapsing, :], axis=0)
        ax1.fill_between(lags, mean_c - std_c, mean_c + std_c,
                         color="#CC3333", alpha=0.20, zorder=3)
        ax1.plot(lags, mean_c, color="#CC3333", lw=2.2, zorder=5,
                 label=f"Collapsing  (n={int(collapsing.sum())})")
    if stable.sum() > 0:
        mean_s = np.nanmean(ccf[stable, :], axis=0)
        ax1.plot(lags, mean_s, color="#3366AA", lw=1.8, zorder=5,
                 ls="--", label=f"Stable  (n={int(stable.sum())})")

    ax1.axvline(0, color="0.6", lw=0.6, zorder=1)
    ax1.axhline(0, color="0.6", lw=0.6, zorder=1)
    ax1.set_xlabel("Lag τ  (years)   — positive = F$_{ovS}$ leads AMOC")
    ax1.set_ylabel(r"$r(\mathrm{F}_{ovS}(t),\ \mathrm{AMOC}(t+\tau))$")
    ax1.set_title("(a) Cross-correlation curves", fontweight="bold")
    ax1.set_xlim(lags[0], lags[-1])
    ax1.set_ylim(-1, 1)
    ax1.legend(loc="upper right", fontsize=5.8, frameon=False)

    # ── Panel (b): peak-lag vs peak-r, coloured by decline ──
    finite = np.isfinite(peak_lag) & np.isfinite(peak_r) & np.isfinite(decline)
    sc = ax2.scatter(peak_lag[finite], np.abs(peak_r[finite]),
                     c=decline[finite] * 100, cmap="RdBu_r",
                     vmin=0, vmax=70, s=35, edgecolors="0.3", linewidths=0.3, zorder=4)
    ax2.axvline(0, color="0.6", lw=0.6, zorder=1)
    ax2.set_xlabel("Peak lag τ* (years)")
    ax2.set_ylabel(r"Peak |r|")
    ax2.set_title("(b) Per-model CCF peak", fontweight="bold")
    ax2.set_xlim(lags[0], lags[-1])
    ax2.set_ylim(0, 1)
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02, shrink=0.85)
    cbar.set_label("AMOC decline by 2100 (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # ── Panel (c): histogram of peak lags ──
    bins = np.arange(-50, 51, 5)
    if collapsing.sum() > 0:
        ax3.hist(peak_lag[collapsing], bins=bins, color="#CC3333",
                 alpha=0.7, edgecolor="0.2", linewidth=0.3,
                 label=f"Collapsing  (n={int(collapsing.sum())})")
    if stable.sum() > 0:
        ax3.hist(peak_lag[stable], bins=bins, color="#3366AA",
                 alpha=0.5, edgecolor="0.2", linewidth=0.3,
                 label=f"Stable  (n={int(stable.sum())})")
    ax3.axvline(0, color="0.6", lw=0.6, zorder=1)

    if collapsing.sum() > 0:
        med = float(np.nanmedian(peak_lag[collapsing]))
        ax3.axvline(med, color="#CC3333", lw=1.2, ls=":",
                    label=f"Median (collapsing) = {med:+.0f}y")
    ax3.set_xlabel("Peak lag τ* (years)")
    ax3.set_ylabel("Number of models")
    ax3.set_title("(c) Distribution of peak lags", fontweight="bold")
    ax3.set_xlim(-50, 50)
    ax3.legend(loc="upper left", fontsize=5.8, frameon=False)

    fig.tight_layout()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
