#!/usr/bin/env python3
"""Figure S4: signal-to-noise — forced ΔF_ovS vs piControl null distribution.

Histogram of |ΔF_total| from:
  - 550 piControl bootstrap draws (13 CMIP6 models, 30-year segments)
  - 18 CMIP6 forced (historical 1950–1980 → ssp585 2080–2100)
  - 2 observational reanalyses with decomposition (ORAS5, GLORYS12V1)

The forced signal must rise clearly above the internal-variability null
for mechanism attribution to be meaningful.

Reads:
  data/results/fovs_decomposition_cmip6_null.csv
  data/results/fovs_decomposition_cmip6_summary.csv
  data/results/fovs_decomposition_{oras5,glorys12,ecco,soda}.nc (optional)

Outputs: figures/paper2/figS4_signal_noise.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

REANALYSIS_PRODUCTS = [
    ("ORAS5",      "fovs_decomposition_oras5.nc",    "#1f77b4"),
    ("GLORYS12V1", "fovs_decomposition_glorys12.nc", "#2ca02c"),
    ("SODA3.15.2", "fovs_decomposition_soda.nc",     "#e377c2"),
    ("ECCO-V4r4",  "fovs_decomposition_ecco.nc",     "#d62728"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/figS4_signal_noise"))
    args = parser.parse_args()

    apply_nature_style()

    null_df = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_null.csv")
    forced_df = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_summary.csv")

    null_abs = np.abs(null_df["delta_total"].values) * 1000  # mSv
    forced_abs = np.abs(forced_df["delta_total"].values) * 1000

    reanalysis_vals = []
    for label, fname, color in REANALYSIS_PRODUCTS:
        p = args.results_dir / fname
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        dF = float(ds.attrs["delta_total_Sv"]) * 1000
        ds.close()
        reanalysis_vals.append((label, abs(dF), color))

    fig, ax = plt.subplots(figsize=(6.4, 3.5))

    # piControl histogram (null)
    bins = np.linspace(0, max(null_abs.max(), forced_abs.max()) + 10, 40)
    ax.hist(null_abs, bins=bins, color="0.55", alpha=0.75, edgecolor="0.25",
            linewidth=0.3, density=True, zorder=3,
            label=f"piControl null  (n={len(null_abs)} bootstrap draws,\n 13 CMIP6 models × 30-yr random segments)")

    # Forced histogram
    ax.hist(forced_abs, bins=bins, color="#CC3333", alpha=0.6, edgecolor="0.25",
            linewidth=0.3, density=True, zorder=4,
            label=f"CMIP6 forced  (n={len(forced_abs)} models, hist→ssp585)")

    # Reanalysis markers
    y_arrow = ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] else 0.02
    for label, val, color in reanalysis_vals:
        ax.axvline(val, color=color, lw=2.0, zorder=6)
        ax.annotate(
            f"{label}\n{val:.0f} mSv", xy=(val, y_arrow),
            xytext=(0, 5), textcoords="offset points",
            fontsize=6, color=color, ha="center", fontweight="bold", zorder=7,
        )

    # Thresholds
    null_p95 = float(np.percentile(null_abs, 95))
    ax.axvline(null_p95, color="0.3", ls=":", lw=0.8, zorder=2,
               label=f"piControl p95 = {null_p95:.0f} mSv")

    ax.set_xlabel(r"$|\Delta F_{ovS}|$  between baseline and forced periods (mSv)")
    ax.set_ylabel("Density")
    # Title removed — Nature/Science style relies on the LaTeX caption.
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax.set_xlim(0, None)

    fig.tight_layout()
    save_publication_figure(fig, args.output)

    # Print summary
    print("\nSignal-to-noise summary:")
    print(f"  piControl |ΔF| median: {np.median(null_abs):.1f} mSv, "
          f"p95: {null_p95:.1f} mSv")
    print(f"  CMIP6 forced |ΔF| median: {np.median(forced_abs):.1f} mSv "
          f"({np.sum(forced_abs > null_p95)} of {len(forced_abs)} above null p95)")
    for label, val, _ in reanalysis_vals:
        mult = val / np.median(null_abs)
        print(f"  {label}: {val:.1f} mSv = {mult:.1f}x piControl median")


if __name__ == "__main__":
    main()
