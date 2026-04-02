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
                    year_end: int | None = None,
) -> tuple[dict[str, float], set[str]]:
    """Load F_ovS means from our computed results, optionally for a sub-period.

    If year_end > 2014, uses hist+ssp585 where available, falls back to
    historical-only for models without SSP data.

    Returns (model_means dict, set of models that used historical-only fallback).
    """
    cmip6_dir = results_dir / "cmip6"
    model_means = {}
    fallback_models = set()

    use_ssp = year_end is not None and year_end > 2014

    # First pass: try hist+ssp585 for full coverage
    if use_ssp:
        for f in sorted(cmip6_dir.glob("fovs_*_hist_ssp585.nc")):
            model = f.stem.replace("fovs_", "").replace("_hist_ssp585", "")
            da = xr.open_dataarray(f)
            try:
                da = da.sel(time=slice(str(year_start), str(year_end)))
            except Exception:
                pass
            vals = da.values[np.isfinite(da.values)]
            if len(vals) > 10:
                model_means[model] = float(np.mean(vals))
            da.close()

    # Second pass: fill in missing models from historical-only (up to 2014)
    for f in sorted(cmip6_dir.glob("fovs_*_historical.nc")):
        model = f.stem.replace("fovs_", "").replace("_historical", "")
        if model in model_means:
            continue  # already have this model from hist+ssp
        da = xr.open_dataarray(f)
        s = str(year_start) if year_start else None
        e = str(min(year_end, 2014)) if year_end else None
        try:
            da = da.sel(time=slice(s, e))
        except Exception:
            pass
        vals = da.values[np.isfinite(da.values)]
        if len(vals) > 10:
            model_means[model] = float(np.mean(vals))
            if use_ssp:
                fallback_models.add(model)
        da.close()

    return model_means, fallback_models


def plot_panel(ax, data: dict[str, float], title: str,
               reanalysis_val: float | None = None,
               reanalysis_label: str = "ORAS5",
               starred: set[str] | None = None):
    """Plot one horizontal bar chart panel with fixed bar width.

    Models in `starred` get a star (*) appended to their name.
    """
    if starred is None:
        starred = set()
    sorted_models = sorted(data.keys(), key=lambda m: data[m])
    values = [data[m] for m in sorted_models]
    colors = ["#CC3333" if v < 0 else "#3366AA" for v in values]
    labels = [f"{m} *" if m in starred else m for m in sorted_models]

    n_neg = sum(1 for v in values if v < 0)

    # Fixed bar height = 0.8, spacing = 1.0 per model (same in every panel)
    y_pos = np.arange(len(sorted_models), dtype=float)

    ax.barh(y_pos, values, height=0.8, color=colors, edgecolor="white",
            linewidth=0.3, zorder=3)

    ax.axvline(0, color="black", lw=0.5, ls=":", zorder=2)

    if reanalysis_val is not None:
        ax.axvline(reanalysis_val, color="#228833", lw=1.5, ls="--", zorder=5)
        # Place reanalysis label at the bottom of the data, not at top
        ax.text(reanalysis_val, len(sorted_models) + 1,
                f"{reanalysis_label} ({reanalysis_val:+.2f} Sv)",
                fontsize=5, color="#228833", ha="center", va="top",
                fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5, fontweight="bold")
    ax.set_ylim(max(len(sorted_models), 39) - 0.5, -2)
    ax.set_title(title, fontsize=6, fontweight="bold", pad=4)
    ax.grid(axis="x", alpha=0.2, linewidth=0.3)

    # Bistable count — large, near the top
    ax.text(0.5, 0.02,
            f"{n_neg}/{len(data)} bistable (F$_{{ovS}}$ < 0)",
            transform=ax.transAxes, fontsize=6, ha="center", va="bottom",
            color="#CC3333", fontweight="bold")


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
    this_study_preindustrial, _ = load_this_study(args.results_dir, 1850, 1900)
    this_study_present, fallback_models = load_this_study(args.results_dir, 1994, 2020)

    # ORAS5
    oras5_path = args.results_dir / "oras5_f_ovs.nc"
    oras5_mean = None
    oras5_present = None
    if oras5_path.exists():
        da = xr.open_dataarray(oras5_path)
        oras5_mean = float(da.mean())
        try:
            oras5_present = float(da.sel(time=slice("1994", "2020")).mean())
        except Exception:
            oras5_present = oras5_mean
        da.close()

    print(f"van Westen 2024:       {len(VAN_WESTEN_2024)} models (1994-2020)")
    print(f"This study (1850-1900): {len(this_study_preindustrial)} models")
    print(f"This study (1994-2020): {len(this_study_present)} models")
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
    all_vals = (list(VAN_WESTEN_2024.values()) + list(this_study_present.values())
                + list(this_study_preindustrial.values()))
    x_max = max(abs(min(all_vals)), abs(max(all_vals))) * 1.15

    # 3 panels, equal size — bars will have same physical height
    # Panel (a) has 39 models, (b,c) have 22 — set shared ylim to 39
    max_n = max(len(VAN_WESTEN_2024), len(this_study_present),
                len(this_study_preindustrial))

    fig, axes = plt.subplots(1, 3, figsize=(6.73, max_n * 0.22 + 1.5),
                              sharey=False)

    plot_panel(axes[0], VAN_WESTEN_2024,
               "(a) van Westen & Dijkstra 2024\nCMIP6, 1994\u20132020",
               VAN_WESTEN_REANALYSIS, "GLORYS12")

    plot_panel(axes[1], this_study_present,
               "(b) CMIP6, 1994\u20132020\n(hist+SSP585)",
               starred=fallback_models)

    plot_panel(axes[2], this_study_preindustrial,
               "(c) CMIP6, 1850\u20131900")

    for ax in axes:
        ax.set_xlim(-x_max, x_max)
        ax.set_xlabel("F$_{ovS}$ (Sv)", fontsize=5)

    # Footnote for starred models
    if fallback_models:
        fig.text(0.5, 0.005,
                 "* historical only (1994\u20132014), no SSP585 data available",
                 ha="center", fontsize=5, color="0.4", style="italic")

    fig.tight_layout(w_pad=0.5, rect=[0, 0.02, 1, 1])
    # Save at high resolution (600 DPI)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out.with_suffix('.png')} (600 DPI)")
    print(f"Saved: {out.with_suffix('.pdf')}")

    # Summary
    for name, data in [("van Westen 2024", VAN_WESTEN_2024),
                       ("This study 1994-2020", this_study_present),
                       ("This study 1850-1900", this_study_preindustrial)]:
        n_neg = sum(1 for v in data.values() if v < 0)
        print(f"  {name}: {n_neg}/{len(data)} bistable")


if __name__ == "__main__":
    main()
