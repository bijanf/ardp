#!/usr/bin/env python3
"""Plot F_ovS time series with linear trend."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from ardp.viz.maps import plot_timeseries_with_trend, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot F_ovS time series.")
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

    f_ovs = xr.open_dataarray(results_dir / "f_ovs.nc")

    ax = plot_timeseries_with_trend(
        f_ovs,
        title="F_ovS: Overturning Freshwater Transport at 34.5S",
        ylabel="F_ovS [Sv]",
    )

    save_figure(ax.figure, output_dir / "f_ovs_timeseries.png")


if __name__ == "__main__":
    main()
