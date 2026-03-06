#!/usr/bin/env python3
"""Plot Gulf Stream SSH gradient map and destabilization point time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from ardp.constants import GULF_STREAM_REGION
from ardp.viz.maps import plot_region_box, plot_timeseries_with_trend, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Gulf Stream destabilization analysis."
    )
    parser.add_argument(
        "--results-dir", default="data/results",
        help="Directory with computed results.",
    )
    parser.add_argument(
        "--output-dir", default="figures",
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    destab_path = results_dir / "gulf_stream_destab.nc"
    if not destab_path.exists():
        print(f"Error: {destab_path} not found. Run compute_fingerprints.py first.")
        return

    destab = xr.open_dataarray(destab_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: region box on Atlantic map
    import cartopy.crs as ccrs

    ax_map = fig.add_axes(
        axes[0].get_position(),
        projection=ccrs.PlateCarree(),
    )
    axes[0].remove()

    ax_map.coastlines()
    ax_map.set_extent([-90, 10, 20, 60], crs=ccrs.PlateCarree())
    ax_map.set_title("Gulf Stream Region")
    plot_region_box(ax_map, GULF_STREAM_REGION)
    ax_map.gridlines(draw_labels=True, alpha=0.3)

    # Right panel: destabilization longitude time series
    plot_timeseries_with_trend(
        destab,
        title="Gulf Stream Destabilization Point",
        ylabel="Longitude [deg E]",
        ax=axes[1],
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "gulf_stream_analysis.png")


if __name__ == "__main__":
    main()
