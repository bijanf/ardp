#!/usr/bin/env python3
"""Figure 2: Mechanism decomposition of F_ovS trend across reanalyses.

Two panels:
  (a) Stacked bar chart per product showing ΔF_total = ΔF_v + ΔF_s + ΔF_cross
      (Sv between the two reference periods, e.g. 1993-2005 and 2013-2025).
  (b) Per-depth contribution profiles (Sv/m cumulative) for ORAS5 and
      GLORYS12 side-by-side, showing WHERE in the water column the
      different components are concentrated.

Reads: data/results/fovs_decomposition_{oras5,glorys12,soda,ecco}.nc
Outputs: figures/paper2/fig2_decomposition.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

PRODUCTS = [
    ("ORAS5",      "oras5",    "#1f77b4"),
    ("GLORYS12V1", "glorys12", "#2ca02c"),
    ("SODA3.15.2", "soda",     "#e377c2"),
    ("ECCO-V4r4",  "ecco",     "#d62728"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path, default=Path("figures/paper2/fig2_decomposition"))
    args = parser.parse_args()

    apply_nature_style()

    # Load what exists
    data = {}
    for label, key, color in PRODUCTS:
        path = args.results_dir / f"fovs_decomposition_{key}.nc"
        if path.exists():
            data[label] = (xr.open_dataset(path), color)

    if len(data) == 0:
        print("No decomposition files found. Run scripts/compute_fovs_decomposition.py first.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # ── Panel (a): stacked bar per product ──
    labels = list(data.keys())
    x = np.arange(len(labels))
    dv = np.array([float(data[lab][0].attrs["delta_v_Sv"]) * 1000 for lab in labels])
    ds_ = np.array([float(data[lab][0].attrs["delta_s_Sv"]) * 1000 for lab in labels])
    dc = np.array([float(data[lab][0].attrs["delta_cross_Sv"]) * 1000 for lab in labels])
    dtot = dv + ds_ + dc

    width = 0.55
    # Flag products with near-zero trend (|ΔF_total| < 10 mSv): mechanism
    # decomposition is ill-defined there.
    trend_threshold = 10.0  # mSv
    has_trend = np.abs(dtot) >= trend_threshold

    # Bars (with hatching if no trend)
    ax1.bar(x[has_trend], dv[has_trend], width=width, color="#E69F00",
            label=r"$\Delta F_v$  (velocity)")
    ax1.bar(x[has_trend], ds_[has_trend], width=width,
            bottom=dv[has_trend], color="#56B4E9",
            label=r"$\Delta F_s$  (salinity)")
    ax1.bar(x[has_trend], dc[has_trend], width=width,
            bottom=dv[has_trend] + ds_[has_trend], color="0.6",
            label=r"$\Delta F_\mathrm{cross}$")
    if (~has_trend).any():
        ax1.bar(x[~has_trend], dtot[~has_trend], width=width,
                color="0.8", edgecolor="0.4", hatch="///",
                label=r"$|\Delta F_\mathrm{total}| < 10$ mSv (mechanism ill-defined)")

    ax1.scatter(x, dtot, color="black", s=25, marker="D", zorder=5,
                label=r"$\Delta F_\mathrm{total}$")

    ax1.axhline(0, color="0.6", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=6.5)
    ax1.set_ylabel(r"$\Delta\mathrm{F}_{ovS}$ (mSv, late − early period)")
    ax1.set_title("(a) Mechanism decomposition", fontweight="bold")
    # Legend OUTSIDE the axes (below the title, above the plot area) so it
    # never overlaps bars. Horizontal single row.
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38),
               fontsize=6, frameon=False, ncol=3, handlelength=1.5,
               columnspacing=1.2)
    # Extend y-axis to -90 as requested, keep top modest
    ax1.set_ylim(-95, 25)

    # Annotate percentages INSIDE the bar for downward (negative) bars,
    # just below x-axis for upward (positive) bars, with enough margin
    # to avoid overlapping the legend and the bar edges.
    for i, lab in enumerate(labels):
        if abs(dtot[i]) < 10:  # ill-defined — label "no trend"
            ax1.text(i, -2, "no\ntrend", ha="center", va="top",
                     fontsize=5.5, color="0.3", style="italic")
            continue
        v_pct = 100 * dv[i] / dtot[i]
        s_pct = 100 * ds_[i] / dtot[i]
        if dtot[i] < 0:
            # place label a few units ABOVE the bar top (which is 0)
            yy = 3
            va = "bottom"
        else:
            # place label a few units BELOW the bar bottom (which is 0)
            yy = -3
            va = "top"
        ax1.text(i, yy, f"v:{v_pct:+.0f}%\ns:{s_pct:+.0f}%",
                 ha="center", va=va, fontsize=5.5, color="0.3")

    # ── Panel (b): depth profiles ──
    for lab in labels:
        ds_obj, color = data[lab]
        depth = ds_obj["depth"].values
        v_prof = ds_obj["depth_Sv_v"].values * 1000  # to mSv
        s_prof = ds_obj["depth_Sv_s"].values * 1000
        ax2.plot(v_prof, depth, color=color, lw=1.5, label=f"{lab} (v)")
        ax2.plot(s_prof, depth, color=color, lw=1.5, ls="--", label=f"{lab} (s)")

    ax2.axvline(0, color="0.6", lw=0.6)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Per-depth $\Delta F$ (mSv)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("(b) Depth distribution", fontweight="bold")
    # Move legend OUT of the data area — to the right of the axes.
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
               fontsize=5.5, frameon=False, ncol=1,
               handlelength=1.5)
    ax2.set_ylim(5500, 0)

    fig.tight_layout()
    save_publication_figure(fig, args.output)

    for _, (ds_obj, _) in data.items():
        ds_obj.close()


if __name__ == "__main__":
    main()
