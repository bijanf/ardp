#!/usr/bin/env python3
"""Download reanalysis data for AMOC diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ardp.ingestion.download import download_cglors, download_glorys12, download_oras5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download monthly reanalysis data for AMOC analysis."
    )
    parser.add_argument(
        "--product",
        choices=["glorys12", "oras5", "cglors"],
        required=True,
        help="Reanalysis product to download.",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM for Copernicus, YYYY for ORAS5).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM for Copernicus, YYYY for ORAS5).")
    parser.add_argument("--output-dir", default="data", help="Root output directory (default: data).")
    parser.add_argument("--lon-min", type=float, default=-80.0)
    parser.add_argument("--lon-max", type=float, default=30.0)
    parser.add_argument("--lat-min", type=float, default=-60.0)
    parser.add_argument("--lat-max", type=float, default=70.0)
    parser.add_argument(
        "--include-indopacific",
        action="store_true",
        help="Extend longitude to 290E for salinity pile-up comparison.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lon_max = 290.0 if args.include_indopacific else args.lon_max

    if args.product == "glorys12":
        download_glorys12(
            output_dir, args.start, args.end,
            lon_min=args.lon_min, lon_max=lon_max,
            lat_min=args.lat_min, lat_max=args.lat_max,
        )
    elif args.product == "oras5":
        download_oras5(
            output_dir, int(args.start), int(args.end),
            lon_min=args.lon_min, lon_max=lon_max,
            lat_min=args.lat_min, lat_max=args.lat_max,
        )
    elif args.product == "cglors":
        download_cglors(
            output_dir, args.start, args.end,
            lon_min=args.lon_min, lon_max=lon_max,
            lat_min=args.lat_min, lat_max=args.lat_max,
        )

    print("Download complete.")


if __name__ == "__main__":
    main()
