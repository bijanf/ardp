#!/usr/bin/env python3
"""Compare F_ovS model rankings across three studies.

Three-panel figure:
  (a) van Westen & Dijkstra 2024 — 39 CMIP6 models, 1994–2020
  (b) This study — 22 CMIP6 models, 1850–2014 (raw, no bias correction)
  (c) Sgubin et al. 2022 — 10 CMIP5 models, piControl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure


# ── van Westen & Dijkstra 2024, Table A1 (1994–2020) ──
VAN_WESTEN_2024 = {
    "ACCESS-CM2": +0.07, "ACCESS-ESM1-5": +0.12, "BCC-CSM2-MR": +0.09,
    "CAMS-CSM1-0": -0.05, "CanESM5": -0.06, "CanESM5-CanOE": -0.06,
    "CAS-ESM2-0": +0.31, "CESM2": +0.18, "CESM2-FV2": +0.20,
    "CESM2-WACCM": +0.17, "CIESM": -0.08, "CMCC-CM2-SR5": +0.09,
    "CMCC-ESM2": +0.10, "CNRM-CM6-1": -0.11, "CNRM-CM6-1-HR": -0.23,
    "CNRM-ESM2-1": -0.13, "EC-Earth3": -0.04, "EC-Earth3-CC": 0.00,
    "EC-Earth3-Veg": -0.02, "EC-Earth3-Veg-LR": +0.01, "FGOALS-f3-L": +0.49,
    "FGOALS-g3": +0.31, "FIO-ESM-2-0": +0.19, "GFDL-CM4": +0.06,
    "GISS-E2-1-G": +0.24, "GISS-E2-2-G": +0.27, "HadGEM3-GC31-LL": +0.11,
    "HadGEM3-GC31-MM": +0.01, "IPSL-CM6A-LR": -0.18, "MCM-UA-1-0": -0.08,
    "MIROC-ES2L": -0.20, "MIROC6": -0.10, "MPI-ESM1-2-HR": -0.06,
    "MRI-ESM2-0": -0.21, "NESM3": -0.24, "NorESM2-LM": +0.23,
    "NorESM2-MM": +0.19, "TaiESM1": +0.34, "UKESM1-0-LL": +0.08,
}
VAN_WESTEN_REANALYSIS = -0.10  # Sv (their reanalysis value)

# ── Sgubin et al. 2022, Table 1 (piControl, mSv → Sv) ──
SGUBIN_2022 = {
    "BCC-CSM1": +0.145, "BNU-ESM": +0.702, "CCSM4": +0.145,
    "CSIRO-Mk3": +0.274, "GFDL-ESM2G": +0.220,
    "CMCC-CM": -0.107, "CMCC-CMS": -0.165, "FIO-ESM": -0.101,
    "IPSL-LR": -0.056, "IPSL-MR": -0.015,
}


def load_this_study(results_dir: Path) -> dict[str, float]:
    """Load raw historical-mean F_ovS from our computed results."""
    cmip6_dir = results_dir / "cmip6"
    model_means = {}
    for f in sorted(cmip6_dir.glob("fovs_*_historical.nc")):
        model = f.stem.replace("fovs_", "").replace("_historical", "")
        da = xr.open_dataarray(f)
        vals = da.values[np.isfinite(da.values)]
        if len(vals) > 20:
            model_means[model] = float(np.mean(vals))
        da.close()
    return model_means


def plot_panel(ax, data: dict[str, float], title: str, reanalysis_val: float | None,
               reanalysis_label: str = "ORAS5"):
    """Plot one horizontal bar chart panel."""
    sorted_models = sorted(data.keys(), key=lambda m: data[m])
    values = [data[m] for m in sorted_models]
    colors = ["#CC3333" if v < 0 else "#3366AA" for v in values]

    n_neg = sum(1 for v in values if v < 0)
    n_pos = sum(1 for v in values if v >= 0)

    y_pos = np.arange(len(sorted_models))
    ax.barh(y_pos, values, height=0.7, color=colors, edgecolor="white",
            linewidth=0.3, zorder=3)

    ax.axvline(0, color="black", lw=0.5, ls=":", zorder=2)

    if reanalysis_val is not None:
        ax.axvline(reanalysis_val, color="#228833", lw=1.5, ls="--", zorder=5)
        ax.text(reanalysis_val + 0.01, len(sorted_models) - 0.5,
                reanalysis_label, fontsize=4.5, color="#228833", va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=4)
    ax.set_title(title, fontsize=7, fontweight="bold", pad=6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, linewidth=0.3)

    # Bistable/monostable count
    ax.text(0.98, 0.02,
            f"{n_neg} bistable / {n_pos} monostable",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="bottom",
            color="0.4")


def main():
    parser = argparse.ArgumentParser(
        description="Compare F_ovS rankings across studies."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/grl/fig_fovs_multi_study"))
    args = parser.parse_args()

    apply_nature_style()

    # Load our data
    this_study = load_this_study(args.results_dir)
    print(f"This study: {len(this_study)} models")
    print(f"van Westen: {len(VAN_WESTEN_2024)} models")
    print(f"Sgubin:     {len(SGUBIN_2022)} models")

    # ORAS5 mean
    oras5_path = args.results_dir / "oras5_f_ovs.nc"
    oras5_mean = None
    if oras5_path.exists():
        da = xr.open_dataarray(oras5_path)
        oras5_mean = float(da.mean())
        da.close()
        print(f"ORAS5 mean: {oras5_mean:.3f} Sv")

    # Shared x-axis range
    all_vals = list(VAN_WESTEN_2024.values()) + list(this_study.values()) + list(SGUBIN_2022.values())
    x_max = max(abs(min(all_vals)), abs(max(all_vals))) * 1.15

    # Figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.73, 7.0),
                                         sharey=False)

    plot_panel(ax1, VAN_WESTEN_2024,
               "van Westen & Dijkstra 2024\nCMIP6, 1994\u20132020",
               VAN_WESTEN_REANALYSIS, "Reanalysis")

    plot_panel(ax2, this_study,
               "This study\nCMIP6, 1850\u20132014 (raw)",
               oras5_mean, "ORAS5")

    plot_panel(ax3, SGUBIN_2022,
               "Sgubin et al. 2022\nCMIP5, piControl",
               None)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-x_max, x_max)
        ax.set_xlabel("F$_{ovS}$ (Sv)", fontsize=6)

    # Panel labels
    for ax, label in zip([ax1, ax2, ax3], "abc"):
        ax.text(0.03, 1.02, f"({label})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    fig.tight_layout(w_pad=1.0)
    save_publication_figure(fig, args.output)

    # Summary
    for name, data in [("van Westen", VAN_WESTEN_2024),
                       ("This study", this_study),
                       ("Sgubin", SGUBIN_2022)]:
        n_neg = sum(1 for v in data.values() if v < 0)
        print(f"{name}: {n_neg}/{len(data)} bistable")


if __name__ == "__main__":
    main()
