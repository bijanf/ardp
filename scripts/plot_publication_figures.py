#!/usr/bin/env python3
"""Generate all publication-quality figures for Nature Communications.

Produces:
  Figure 1: F_ovS time series with trend and uncertainty
  Figure 2: 2x2 fingerprint summary panel
  Figure 3: Gulf Stream destabilization spatial analysis
  Figure 4: Multi-product comparison (when multiple products available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

from ardp.viz.style import (
    COLORS,
    FINGERPRINT_COLORS,
    add_panel_label,
    add_trend_annotation,
    figure_double_col,
    figure_single_col,
    format_year_axis,
    save_publication_figure,
)


def _time_to_years(time: xr.DataArray) -> np.ndarray:
    """Convert xarray time coordinate to fractional years."""
    import pandas as pd
    try:
        ts = pd.DatetimeIndex(time.values)
        return np.array([t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
                         for t in ts])
    except Exception:
        if hasattr(time.values[0], "year"):
            return np.array([t.year + (t.month - 1) / 12.0
                             for t in time.values])
        return time.values.astype(float)


def _compute_trend(ts: xr.DataArray) -> dict:
    """Compute linear trend with confidence interval and p-value."""
    years = _time_to_years(ts["time"])
    values = ts.values.ravel()
    valid = np.isfinite(values)

    if valid.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "pvalue": np.nan,
                "stderr": np.nan, "trend_line": np.full_like(values, np.nan)}

    result = stats.linregress(years[valid], values[valid])
    trend_line = result.slope * years + result.intercept

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "pvalue": result.pvalue,
        "stderr": result.stderr,
        "trend_line": trend_line,
        "r_squared": result.rvalue ** 2,
    }


def figure1_f_ovs(results_dir: Path, output_dir: Path) -> None:
    """Figure 1: F_ovS time series — the critical tipping indicator."""
    path = results_dir / "f_ovs.nc"
    if not path.exists():
        print(f"Skipping Figure 1: {path} not found")
        return

    f_ovs = xr.open_dataarray(path)
    trend = _compute_trend(f_ovs)

    fig, ax = figure_single_col(height_ratio=0.65)

    # Convert to mSv for readability
    scale = 1e3 if np.nanmax(np.abs(f_ovs.values)) < 1 else 1.0
    unit = "mSv" if scale == 1e3 else "Sv"
    data = f_ovs * scale

    ax.plot(f_ovs.time.values, data.values, color=FINGERPRINT_COLORS["f_ovs"],
            linewidth=0.8, label="$F_{ovS}$")
    ax.plot(f_ovs.time.values, trend["trend_line"] * scale,
            color=COLORS["red"], linewidth=1.0, linestyle="--",
            label="Linear trend")

    # 12-month running mean if enough data
    if len(data) > 24:
        rolling = data.rolling(time=12, center=True).mean()
        ax.plot(f_ovs.time.values, rolling.values,
                color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5, alpha=0.8,
                label="12-month mean")

    ax.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
    ax.set_xlabel("")
    ax.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax.legend(loc="lower left")

    trend_unit = f"m{unit}" if unit == "Sv" else unit
    add_trend_annotation(
        ax,
        slope=trend["slope"] * scale,
        unit=trend_unit,
        pvalue=trend["pvalue"],
    )

    ax.set_title("")
    fig.tight_layout()
    save_publication_figure(fig, output_dir / "fig1_f_ovs")


def figure2_fingerprints(results_dir: Path, output_dir: Path) -> None:
    """Figure 2: 2x2 panel — all four AMOC fingerprints."""
    files = {
        "f_ovs": ("f_ovs.nc", "$F_{ovS}$", "Sv", FINGERPRINT_COLORS["f_ovs"]),
        "pileup": ("salinity_pileup.nc", "Salinity pile-up", "PSU",
                    FINGERPRINT_COLORS["salinity_pileup"]),
        "nawh": ("nawh.nc", "NAWH index", r"$\degree$C",
                 FINGERPRINT_COLORS["nawh"]),
        "gs": ("gulf_stream_destab.nc", "GS destabilization",
               r"$\degree$lon", FINGERPRINT_COLORS["gulf_stream"]),
    }

    data = {}
    for key, (fname, label, unit, color) in files.items():
        path = results_dir / fname
        if path.exists():
            data[key] = (xr.open_dataarray(path), label, unit, color)

    if not data:
        print("Skipping Figure 2: no fingerprint results found")
        return

    n = len(data)
    nrows = 2 if n > 2 else 1
    ncols = 2 if n > 1 else 1

    fig, axes = figure_double_col(nrows=nrows, ncols=ncols, height_ratio=0.45)
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    panels = "abcdefgh"
    for i, (key, (ts, label, unit, color)) in enumerate(data.items()):
        ax = axes[i]
        trend = _compute_trend(ts)

        ax.plot(ts.time.values, ts.values, color=color, linewidth=0.7)
        ax.plot(ts.time.values, trend["trend_line"],
                color="0.3", linewidth=0.8, linestyle="--")

        # Rolling mean if enough data
        if len(ts) > 24:
            rolling = ts.rolling(time=12, center=True).mean()
            ax.plot(ts.time.values, rolling.values, color=color,
                    linewidth=1.3, alpha=0.9)

        ax.set_ylabel(f"{label} [{unit}]")
        add_panel_label(ax, panels[i])
        add_trend_annotation(ax, trend["slope"], unit, trend["pvalue"],
                             position="upper left")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save_publication_figure(fig, output_dir / "fig2_fingerprints")


def figure3_gulf_stream(results_dir: Path, output_dir: Path) -> None:
    """Figure 3: Gulf Stream destabilization point analysis."""
    path = results_dir / "gulf_stream_destab.nc"
    if not path.exists():
        print(f"Skipping Figure 3: {path} not found")
        return

    destab = xr.open_dataarray(path)
    trend = _compute_trend(destab)

    fig = plt.figure(figsize=(7.09, 3.0))

    # Left: Atlantic map with Gulf Stream region
    import cartopy.crs as ccrs
    from ardp.constants import GULF_STREAM_REGION
    from ardp.viz.maps import plot_region_box
    from ardp.viz.style import apply_nature_style

    apply_nature_style()

    ax_map = fig.add_axes([0.05, 0.1, 0.38, 0.8],
                          projection=ccrs.PlateCarree())
    ax_map.coastlines(linewidth=0.5, color="0.4")
    ax_map.set_extent([-85, 5, 20, 60], crs=ccrs.PlateCarree())
    plot_region_box(ax_map, GULF_STREAM_REGION, color=COLORS["red"])
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    add_panel_label(ax_map, "a", x=-0.05, y=1.05)

    # Right: destabilization longitude time series
    ax_ts = fig.add_axes([0.55, 0.15, 0.42, 0.75])
    ax_ts.plot(destab.time.values, destab.values,
               color=FINGERPRINT_COLORS["gulf_stream"], linewidth=0.8)
    ax_ts.plot(destab.time.values, trend["trend_line"],
               color="0.3", linewidth=0.8, linestyle="--")

    if len(destab) > 24:
        rolling = destab.rolling(time=12, center=True).mean()
        ax_ts.plot(destab.time.values, rolling.values,
                   color=FINGERPRINT_COLORS["gulf_stream"], linewidth=1.3)

    ax_ts.set_ylabel(r"Destabilization longitude [$\degree$E]")
    add_panel_label(ax_ts, "b")
    add_trend_annotation(ax_ts, trend["slope"], r"$\degree$", trend["pvalue"])

    save_publication_figure(fig, output_dir / "fig3_gulf_stream")


def figure4_multi_product(results_dir: Path, output_dir: Path) -> None:
    """Figure 4: Multi-product comparison of F_ovS (if available)."""
    products = ["glorys12", "oras5", "cglors"]
    product_colors = {
        "glorys12": COLORS["blue"],
        "oras5": COLORS["red"],
        "cglors": COLORS["green"],
    }
    product_labels = {
        "glorys12": "GLORYS12V1",
        "oras5": "ORAS5",
        "cglors": "C-GLORS",
    }

    found = {}
    for prod in products:
        path = results_dir.parent / f"results_{prod}" / "f_ovs.nc"
        if not path.exists():
            path = results_dir / f"f_ovs_{prod}.nc"
        if path.exists():
            found[prod] = xr.open_dataarray(path)

    if len(found) < 2:
        print("Skipping Figure 4: need at least 2 products for comparison")
        return

    fig, ax = figure_double_col(height_ratio=0.4)

    for prod, ts in found.items():
        color = product_colors[prod]
        label = product_labels[prod]
        ax.plot(ts.time.values, ts.values, color=color, linewidth=0.8,
                alpha=0.5, label=f"{label} (monthly)")
        if len(ts) > 24:
            rolling = ts.rolling(time=12, center=True).mean()
            ax.plot(ts.time.values, rolling.values, color=color,
                    linewidth=1.5, label=f"{label} (12-mo mean)")

    ax.set_ylabel("$F_{ovS}$ [Sv]")
    ax.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("")

    fig.tight_layout()
    save_publication_figure(fig, output_dir / "fig4_multi_product")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Nature Communications publication figures."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="figures/publication")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication figures...")
    figure1_f_ovs(results_dir, output_dir)
    figure2_fingerprints(results_dir, output_dir)
    figure3_gulf_stream(results_dir, output_dir)
    figure4_multi_product(results_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
