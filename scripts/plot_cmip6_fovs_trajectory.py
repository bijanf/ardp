#!/usr/bin/env python3
"""Plot CMIP6 F_ovS trajectories alongside reanalysis observations.

Creates a publication-quality figure showing:
  Panel (a): F_ovS 30-year epoch box plots — CMIP6 models with observed ORAS5,
             GLORYS12, published reanalyses, and hydrographic estimates overlaid
  Panel (b): piControl mean F_ovS regime classification (bar chart)

This replaces the previous dot-plot (fig4_cmip6_comparison) with a trajectory
figure that shows the observed F_ovS diverging from the model envelope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from ardp.viz.style import (
    COLORS,
    add_panel_label,
    apply_nature_style,
    save_publication_figure,
    GRL_FULL_WIDTH,
    GRL_MAX_HEIGHT,
)

# Published piControl F_ovS means (Weijer et al., 2019; van Westen et al., 2024)
# For models without published piControl, we use their 1850-1900 historical mean.
CMIP6_PI_MEAN = {
    # --- Published piControl (Weijer et al. 2019) ---
    "CESM2": -0.05,
    "MPI-ESM1-2-LR": -0.10,
    "MPI-ESM1-2-HR": -0.02,
    "UKESM1-0-LL": +0.15,
    "CNRM-CM6-1": -0.08,
    "EC-Earth3": +0.01,
    "GFDL-ESM4": +0.05,
    "CanESM5": +0.12,
    "IPSL-CM6A-LR": -0.15,
    "ACCESS-ESM1-5": +0.08,
    # --- Published (van Westen & Dijkstra 2024, Ocean Science) ---
    "MIROC6": -0.10,
    "GFDL-CM4": +0.06,
    "ACCESS-CM2": +0.08,
    "CMCC-CM2-SR5": +0.09,
    "HadGEM3-GC31-LL": +0.11,
    "CESM2-WACCM": +0.17,
    "NorESM2-LM": +0.23,
    "GISS-E2-1-G": +0.24,
    "MRI-ESM2-0": -0.05,   # estimated from historical mean
    # --- New models (historical mean as proxy) ---
    "NESM3": -0.17,
    "CNRM-ESM2-1": -0.10,  # similar to CNRM-CM6-1
    "CanESM5-CanOE": +0.10,  # similar to CanESM5
    "EC-Earth3-AerChem": -0.03,
    "FGOALS-g3": +0.36,
    "FIO-ESM-2-0": +0.19,
    "GISS-E2-1-G-CC": +0.25,
    "SAM0-UNICON": +0.15,
    "TaiESM1": +0.28,
}

# Model colors — bistable (warm), near-zero (grey), monostable (cool)
MODEL_COLORS = {
    # Bistable (F_ovS < 0)
    "IPSL-CM6A-LR": "#ff7f0e",   # orange
    "NESM3": "#e6550d",          # dark orange
    "CNRM-CM6-1": "#d62728",     # red
    "CNRM-ESM2-1": "#e45756",    # light red
    "MPI-ESM1-2-LR": "#e377c2",  # pink
    "MIROC6": "#8c564b",         # brown
    "CESM2": "#d62728",          # red
    "CanESM5": "#1f77b4",        # blue
    # Near-zero
    "MPI-ESM1-2-HR": "#999999",  # grey
    "EC-Earth3": "#7f7f7f",      # dark grey
    "EC-Earth3-AerChem": "#636363", # darker grey
    "CMCC-CM2-SR5": "#c5b0d5",  # light purple
    # Monostable (F_ovS > 0)
    "UKESM1-0-LL": "#17becf",   # cyan
    "GFDL-ESM4": "#9467bd",     # purple
    "ACCESS-ESM1-5": "#2ca02c", # green
    "ACCESS-CM2": "#98df8a",    # light green
    "GFDL-CM4": "#aec7e8",     # light blue
    "HadGEM3-GC31-LL": "#c49c94", # light brown
    "CESM2-WACCM": "#f7b6d2",  # light pink
    "CanESM5-CanOE": "#6baed6", # medium blue
    "SAM0-UNICON": "#74c476",   # medium green
    "NorESM2-LM": "#c7c7c7",   # silver
    "FIO-ESM-2-0": "#fdae6b",  # peach
    "GISS-E2-1-G": "#dbdb8d",  # light olive
    "GISS-E2-1-G-CC": "#b5cf6b", # yellow-green
    "TaiESM1": "#9e9ac8",      # lavender
    "FGOALS-g3": "#e7cb94",    # tan
}

# Published reanalysis F_ovS estimates (Weijer et al. 2019 and individual papers)
# (mean F_ovS in Sv, approx coverage start, approx coverage end, source)
PUBLISHED_REANALYSIS_FOVS = {
    "SODA 2.2.4":  (+0.02, 1980, 2010, "Weijer2019"),
    "GECCO2":      (-0.16, 1952, 2001, "Weijer2019"),
    "NCEP GODAS":  (-0.11, 1980, 2020, "Weijer2019"),
    "ECDA (GFDL)": (-0.20, 1961, 2010, "Weijer2019"),
}

# Published hydrographic point estimates at ~34.5°S
# (F_ovS Sv, error Sv or None, approx year, source)
HYDRO_ESTIMATES = {
    "Garzoli et al. 2011": (-0.10, 0.10, 2005, "GarzoliMatano2011"),
    "Meinen et al. 2018":  (-0.09, None, 2015, "Meinen2018"),
}

SSP_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}


def _to_annual(da: xr.DataArray, window: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Convert monthly time series to annual running mean.

    Returns (years, values) arrays.
    """
    try:
        annual = da.groupby("time.year").mean()
        years = annual.year.values.astype(float)
        values = annual.values
    except Exception:
        # Fallback for cftime
        times = da.time.values
        if hasattr(times[0], "year"):
            yrs = np.array([t.year for t in times])
        else:
            ts = pd.DatetimeIndex(times)
            yrs = ts.year.values
        unique_yrs = np.unique(yrs)
        values = np.array([float(da.values[yrs == y].mean()) for y in unique_yrs])
        years = unique_yrs.astype(float)
    return years, values


def _running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Simple centered running mean with NaN handling."""
    result = np.full_like(values, np.nan)
    half = window // 2
    for i in range(half, len(values) - half):
        chunk = values[i - half : i + half + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 0:
            result[i] = valid.mean()
    return result


def load_cmip6_timeseries(results_dir: Path) -> dict[str, xr.DataArray]:
    """Load all CMIP6 F_ovS time series from results directory.

    Applies bias correction: shifts each model's time series so that its
    early-historical mean (1850-1900) matches the published piControl mean
    from Weijer et al. (2019). This corrects for section-latitude offsets
    introduced by our simplified grid extraction.
    """
    cmip6_dir = results_dir / "cmip6"
    series = {}
    for f in sorted(cmip6_dir.glob("fovs_*.nc")):
        key = f.stem.replace("fovs_", "")
        ds = xr.open_dataset(f)
        series[key] = ds["F_ovS"]
        ds.close()

    # Bias-correct: shift so early-historical mean matches published piControl
    for key, da in list(series.items()):
        # Find model name
        model = None
        for m in CMIP6_PI_MEAN:
            if key.startswith(m):
                model = m
                break
        if model is None:
            continue

        # Compute early-historical mean (1850-1900) from the historical series
        hist_key = f"{model}_historical"
        if hist_key not in series:
            continue

        hist_da = series[hist_key]
        try:
            early = hist_da.sel(time=slice(None, "1900"))
            if len(early) > 0:
                computed_mean = float(early.mean())
                published_mean = CMIP6_PI_MEAN[model]
                offset = published_mean - computed_mean
                series[key] = da + offset
        except Exception:
            pass

    return series


def _epoch_model_means(
    cmip6: dict[str, xr.DataArray],
    models: list[str],
    epoch_start: int,
    epoch_end: int,
    ssp: str = "ssp585",
) -> list[float]:
    """Compute per-model 30-year mean F_ovS for a given epoch.

    For historical epochs (end <= 2014) uses historical series.
    For future epochs uses hist+ssp concatenated series.
    Returns a list of finite model means (one per model that has data).
    """
    means = []
    for model in models:
        # Pick the best available series for this epoch
        if epoch_start >= 2100:
            key = f"{model}_{ssp}ext"
            if key not in cmip6:
                continue
        elif epoch_end <= 2014:
            key = f"{model}_historical"
        else:
            key = f"{model}_hist_{ssp}"
            if key not in cmip6:
                key = f"{model}_historical"

        if key not in cmip6:
            continue

        da = cmip6[key]
        years, vals = _to_annual(da)
        mask = (years >= epoch_start) & (years < epoch_end)
        epoch_vals = vals[mask]
        finite = epoch_vals[np.isfinite(epoch_vals)]
        if len(finite) > 10:  # need at least 10 years
            means.append(float(finite.mean()))
    return means


def plot_trajectory_figure(
    results_dir: Path,
    output_path: Path,
) -> None:
    """Create the F_ovS trajectory figure with 30-year epoch box plots."""
    apply_nature_style()

    # Load reanalysis data
    oras5_path = results_dir / "oras5_f_ovs.nc"
    glorys12_path = results_dir / "glorys12_f_ovs.nc"

    if not oras5_path.exists():
        print(f"Missing: {oras5_path}")
        return

    oras5 = xr.open_dataarray(oras5_path)
    glorys12 = xr.open_dataarray(glorys12_path) if glorys12_path.exists() else None

    # Load CMIP6 time series
    cmip6 = load_cmip6_timeseries(results_dir)
    if not cmip6:
        print("No CMIP6 time series found. Run compute_cmip6_fovs_timeseries.py first.")
        return

    # Identify available CMIP6 models
    known_models = set(CMIP6_PI_MEAN.keys())
    models = set()
    for key in cmip6:
        for m in known_models:
            if key.startswith(m + "_"):
                models.add(m)
                break
    models = sorted(models)
    print(f"CMIP6 models with time series: {models}")

    # Create figure
    fig, axes = plt.subplots(
        1, 2,
        figsize=(GRL_FULL_WIDTH, GRL_FULL_WIDTH * 0.45),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.35},
    )

    ax_ts, ax_bar = axes

    # ── Panel (a): 30-year epoch box plots ──

    # Define 30-year epochs
    epochs = [
        (1850, 1880), (1880, 1910), (1910, 1940), (1940, 1970),
        (1970, 2000), (2000, 2030), (2030, 2060), (2060, 2090),
    ]
    epoch_labels = [f"{s}" for s, e in epochs]
    epoch_centers = [(s + e) / 2 for s, e in epochs]

    # Collect box plot data for SSP585
    bp_data_585 = []
    for s, e in epochs:
        means = _epoch_model_means(cmip6, models, s, e, ssp="ssp585")
        bp_data_585.append(means)

    # Collect box plot data for SSP245 (future epochs only)
    bp_data_245 = []
    for s, e in epochs:
        if s >= 2030:
            means = _epoch_model_means(cmip6, models, s, e, ssp="ssp245")
            bp_data_245.append(means)
        else:
            bp_data_245.append([])

    # F_ovS = 0 threshold line
    ax_ts.axhline(0, color="black", lw=0.5, ls=":", alpha=0.5, zorder=1)

    # Box width in year-units
    box_w = 12

    # Plot SSP585 boxes
    bp585 = ax_ts.boxplot(
        bp_data_585,
        positions=epoch_centers,
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        zorder=3,
        medianprops={"color": "black", "lw": 1.0},
        whiskerprops={"color": "0.4", "lw": 0.6},
        capprops={"color": "0.4", "lw": 0.6},
    )
    for i, (patch, (s, e)) in enumerate(zip(bp585["boxes"], epochs)):
        if s >= 2015:
            patch.set_facecolor(COLORS["red"])
            patch.set_alpha(0.25)
            patch.set_edgecolor(COLORS["red"])
        else:
            patch.set_facecolor(COLORS["grey"])
            patch.set_alpha(0.35)
            patch.set_edgecolor("0.5")

    # Plot SSP245 boxes (offset slightly right, only for future)
    future_idx = [i for i, (s, e) in enumerate(epochs) if s >= 2030]
    if future_idx:
        future_data = [bp_data_245[i] for i in future_idx]
        future_pos = [epoch_centers[i] + box_w + 2 for i in future_idx]
        bp245 = ax_ts.boxplot(
            future_data,
            positions=future_pos,
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            zorder=3,
            medianprops={"color": "black", "lw": 1.0},
            whiskerprops={"color": "0.4", "lw": 0.6, "ls": "--"},
            capprops={"color": "0.4", "lw": 0.6},
        )
        for patch in bp245["boxes"]:
            patch.set_facecolor(COLORS["yellow"])
            patch.set_alpha(0.25)
            patch.set_edgecolor(COLORS["yellow"])

    # Overlay reanalysis observations as continuous time series
    oras5_yr, oras5_val = _to_annual(oras5)
    oras5_rm = _running_mean(oras5_val, window=10)
    ax_ts.plot(oras5_yr, oras5_val, color=COLORS["blue"], alpha=0.15, lw=0.4, zorder=8)
    ax_ts.plot(oras5_yr, oras5_rm, color=COLORS["blue"], lw=2.0, zorder=10,
               label="ORAS5")

    if glorys12 is not None:
        g12_yr, g12_val = _to_annual(glorys12)
        g12_rm = _running_mean(g12_val, window=10)
        ax_ts.plot(g12_yr, g12_val, color=COLORS["green"], alpha=0.15, lw=0.4, zorder=8)
        ax_ts.plot(g12_yr, g12_rm, color=COLORS["green"], lw=2.0, zorder=10,
                   label="GLORYS12")


    # "now" line
    ax_ts.axvline(2025, color="black", lw=0.5, ls="--", alpha=0.3, zorder=1)

    ylim = ax_ts.get_ylim()

    ax_ts.set_xlabel("Year")
    ax_ts.set_ylabel("F$_{ovS}$ (Sv)")
    ax_ts.set_xlim(1840, 2110)
    ax_ts.set_xticks(list(range(1850, 2101, 50)))
    ax_ts.set_xticklabels([str(y) for y in range(1850, 2101, 50)])

    # Count of models in each epoch
    for i, (s, e) in enumerate(epochs):
        n_total = len(bp_data_585[i])
        ax_ts.text(epoch_centers[i], ylim[0] + 0.01 * (ylim[1] - ylim[0]),
                   f"n={n_total}", fontsize=3.5, ha="center", va="bottom", color="0.5")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["grey"], alpha=0.35, edgecolor="0.5",
              label="CMIP6 Historical"),
        Patch(facecolor=COLORS["red"], alpha=0.25, edgecolor=COLORS["red"],
              label="SSP5-8.5"),
        Patch(facecolor=COLORS["yellow"], alpha=0.25, edgecolor=COLORS["yellow"],
              label="SSP2-4.5"),
        Line2D([0], [0], color=COLORS["blue"], lw=2.0, label="ORAS5"),
        Line2D([0], [0], color=COLORS["green"], lw=2.0, label="GLORYS12"),
    ]
    ax_ts.legend(handles=legend_elements, fontsize=4.0, loc="upper center",
                 framealpha=0.9, handlelength=1.2, ncol=5)

    add_panel_label(ax_ts, "(a)")

    # ── Panel (b): Historical mean F_ovS — regime classification ──
    # Compute mean and std of annual F_ovS over the full historical period
    model_means = {}
    model_stds = {}
    for model in models:
        hist_key = f"{model}_historical"
        if hist_key not in cmip6:
            continue
        da = cmip6[hist_key]
        years, vals = _to_annual(da)
        finite = vals[np.isfinite(vals)]
        if len(finite) > 20:
            model_means[model] = float(finite.mean())
            model_stds[model] = float(finite.std())

    # ORAS5 mean and std
    oras5_finite = oras5_val[np.isfinite(oras5_val)]
    oras5_mean = float(oras5_finite.mean())
    oras5_std = float(oras5_finite.std())

    # Sort models by mean F_ovS
    sorted_models_b = sorted(model_means.keys(), key=lambda m: model_means[m])
    mean_values = [model_means[m] for m in sorted_models_b]
    std_values = [model_stds[m] for m in sorted_models_b]
    bar_colors = ["0.5"] * len(sorted_models_b)

    y_pos = np.arange(len(sorted_models_b))
    ax_bar.barh(y_pos, mean_values, xerr=std_values, color=bar_colors,
                edgecolor="white", height=0.7, alpha=0.8,
                error_kw={"ecolor": "0.3", "lw": 0.6, "capsize": 1.5})

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(sorted_models_b, fontsize=4)
    ax_bar.set_xlabel("Historical mean F$_{ovS}$ (Sv)")
    ax_bar.axvline(0, color="black", lw=0.5, ls=":")

    # Mark observed ORAS5 mean
    ax_bar.axvline(oras5_mean, color=COLORS["blue"], lw=1.5, ls="--", alpha=0.8)
    ax_bar.text(oras5_mean + 0.01, len(sorted_models_b) - 0.3,
                "ORAS5",
                fontsize=4.5, ha="left", va="bottom", color=COLORS["blue"])

    add_panel_label(ax_bar, "(b)", x=-0.25)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.93, wspace=0.35)
    save_publication_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CMIP6 F_ovS trajectories with reanalysis observations."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output", default="figures/grl/fig4_cmip6_comparison")
    args = parser.parse_args()

    plot_trajectory_figure(
        results_dir=Path(args.results_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
