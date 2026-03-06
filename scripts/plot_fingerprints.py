#!/usr/bin/env python3
"""Plot 2x2 summary panel of all AMOC fingerprints."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from ardp.viz.maps import plot_multi_panel_fingerprints, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all fingerprints summary.")
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

    fingerprints: dict[str, xr.DataArray] = {}

    files = {
        "F_ovS [Sv]": "f_ovs.nc",
        "Salinity pile-up [PSU]": "salinity_pileup.nc",
        "NAWH index [degC]": "nawh.nc",
        "GS destab. longitude [deg]": "gulf_stream_destab.nc",
    }

    for label, fname in files.items():
        path = results_dir / fname
        if path.exists():
            fingerprints[label] = xr.open_dataarray(path)
        else:
            print(f"Warning: {path} not found, skipping {label}")

    if not fingerprints:
        print("No fingerprint results found. Run compute_fingerprints.py first.")
        return

    fig = plot_multi_panel_fingerprints(fingerprints)
    save_figure(fig, output_dir / "fingerprints_summary.png")


if __name__ == "__main__":
    main()
