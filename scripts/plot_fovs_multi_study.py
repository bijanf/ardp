#!/usr/bin/env python3
"""Compare F_ovS model rankings across studies and time periods.

Four-panel figure:
  (a) van Westen & Dijkstra 2024 — 39 CMIP6 models, 1994–2020
  (b) This study — CMIP6 models, 1994–2014 (same period as van Westen)
  (c) This study — CMIP6 models, 1850–2014 (full historical)
  (d) Sgubin et al. 2022 — 10 CMIP5 models, piControl
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
VAN_WESTEN_REANALYSIS = -0.10

# ── Sgubin et al. 2022, Table 1 (piControl, mSv → Sv) ──
SGUBIN_2022 = {
    "BCC-CSM1": +0.145, "BNU-ESM": +0.702, "CCSM4": +0.145,
    "CSIRO-Mk3": +0.274, "GFDL-ESM2G": +0.220,
    "CMCC-CM": -0.107, "CMCC-CMS": -0.165, "FIO-ESM": -0.101,
    "IPSL-LR": -0.056, "IPSL-MR": -0.015,
}


def load_this_study(results_dir: Path, year_start: int | None = None,
                    year_end: int | None = None) -> dict[str, float]:
    """Load F_ovS means from our computed results, optionally for a sub-period."""
    cmip6_dir = results_dir / "cmip6"
    model_means = {}
    for f in sorted(cmip6_dir.glob("fovs_*_historical.nc")):
        model = f.stem.replace("fovs_", "").replace("_historical", "")
        da = xr.open_dataarray(f)

        if year_start is not None or year_end is not None:
            try:
                s = str(year_start) if year_start else None
                e = str(year_end) if year_end else None
                da = da.sel(time=slice(s, e))
            except Exception:
                pass

        vals = da.values[np.isfinite(da.values)]
        if len(vals) > 10:
            model_means[model] = float(np.mean(vals))
        da.close()
    return model_means


def plot_panel(ax, data: dict[str, float], title: str,
               reanalysis_val: float | None = None,
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
        ax.text(reanalysis_val, len(sorted_models) + 0.3,
                f"{reanalysis_label} ({reanalysis_val:+.2f})",
                fontsize=4, color="#228833", ha="center", va="bottom")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=3.5)
    ax.set_title(title, fontsize=6.5, fontweight="bold", pad=6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, linewidth=0.3)

    ax.text(0.98, 0.02,
            f"{n_neg} bistable / {n_pos} monostable",
            transform=ax.transAxes, fontsize=4, ha="right", va="bottom",
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

    # Load our data for two periods
    this_study_present = load_this_study(args.results_dir, 1994, 2014)
    this_study_full = load_this_study(args.results_dir)

    # ORAS5
    oras5_path = args.results_dir / "oras5_f_ovs.nc"
    oras5_mean = None
    oras5_present = None
    if oras5_path.exists():
        da = xr.open_dataarray(oras5_path)
        oras5_mean = float(da.mean())
        try:
            oras5_present = float(da.sel(time=slice("1994", "2014")).mean())
        except Exception:
            oras5_present = oras5_mean
        da.close()

    print(f"van Westen 2024:       {len(VAN_WESTEN_2024)} models (1994-2020)")
    print(f"This study (1994-2014): {len(this_study_present)} models")
    print(f"This study (1850-2014): {len(this_study_full)} models")
    print(f"Sgubin 2022:           {len(SGUBIN_2022)} models (piControl)")
    if oras5_mean:
        print(f"ORAS5 full mean: {oras5_mean:.3f} Sv, present-day: {oras5_present:.3f} Sv")

    # Check for piControl data
    picontrol = {}
    picontrol_dir = Path("data/cmip6_fullfield")
    for f in sorted(picontrol_dir.glob("*_piControl_vo_zonal.nc")):
        model = f.name.replace("_piControl_vo_zonal.nc", "")
        picontrol[model] = f
    has_picontrol = len(picontrol) > 0
    if has_picontrol:
        print(f"piControl data:        {len(picontrol)} models")

    # Shared x-axis range
    all_vals = (list(VAN_WESTEN_2024.values()) + list(this_study_full.values())
                + list(SGUBIN_2022.values()))
    x_max = max(abs(min(all_vals)), abs(max(all_vals))) * 1.15

    # Figure — 4 panels (or 5 if piControl available)
    n_panels = 5 if has_picontrol else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(6.73, 7.5), sharey=False)

    idx = 0

    # (a) van Westen
    plot_panel(axes[idx], VAN_WESTEN_2024,
               "(a) van Westen & Dijkstra 2024\nCMIP6, 1994\u20132020",
               VAN_WESTEN_REANALYSIS, "Reanalysis")
    idx += 1

    # (b) This study, same period
    plot_panel(axes[idx], this_study_present,
               "(b) This study\nCMIP6, 1994\u20132014",
               oras5_present, "ORAS5")
    idx += 1

    # (c) This study, full historical
    plot_panel(axes[idx], this_study_full,
               "(c) This study\nCMIP6, 1850\u20132014",
               oras5_mean, "ORAS5")
    idx += 1

    # (d) or (e) Sgubin CMIP5
    if has_picontrol:
        # (d) Our piControl
        # TODO: compute piControl F_ovS means when data is ready
        plot_panel(axes[idx], {},
                   "(d) This study\nCMIP6, piControl",
                   None)
        idx += 1

    plot_panel(axes[idx], SGUBIN_2022,
               f"({chr(ord('a') + idx)}) Sgubin et al. 2022\nCMIP5, piControl",
               None)

    for ax in axes:
        ax.set_xlim(-x_max, x_max)
        ax.set_xlabel("F$_{ovS}$ (Sv)", fontsize=5)

    fig.tight_layout(w_pad=0.5)
    save_publication_figure(fig, args.output)

    # Summary
    for name, data in [("van Westen 2024", VAN_WESTEN_2024),
                       ("This study 1994-2014", this_study_present),
                       ("This study 1850-2014", this_study_full),
                       ("Sgubin 2022", SGUBIN_2022)]:
        n_neg = sum(1 for v in data.values() if v < 0)
        print(f"  {name}: {n_neg}/{len(data)} bistable")


if __name__ == "__main__":
    main()
