#!/usr/bin/env python3
"""Download ORAS5 data with parallel CDS requests.

Two-phase strategy:
  Phase 1: Single-level (SST, SSS, SSH) — small, fast, covers 3/4 fingerprints
  Phase 2: All-levels (salinity + velocity only) — larger, needed for F_ovS

Phase 1 is ~300 MB/year. Phase 2 is ~8 GB/year (2 variables instead of 5).
"""

from __future__ import annotations

import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi


SINGLE_LEVEL_VARS = [
    "sea_surface_temperature",
    "sea_surface_salinity",
    "sea_surface_height",
]

ALL_LEVEL_VARS = [
    "salinity",
    "meridional_velocity",
]


def download_one_year(
    year: int,
    dest: Path,
    variables: list[str],
    vertical_resolution: str,
    tag: str,
    months: list[str] | None = None,
) -> str:
    """Download a single year of ORAS5 data."""
    # Check for existing files
    outfile_nc = dest / f"oras5_{tag}_{year}.nc"
    if outfile_nc.exists():
        return f"{year} ({tag}): already exists"

    # Also check for extracted files from zip
    existing = list(dest.glob(f"*{tag}*{year}*"))
    if existing:
        return f"{year} ({tag}): already exists ({existing[0].name})"

    product = "consolidated" if year <= 2021 else "operational"
    if months is None:
        months = [f"{m:02d}" for m in range(1, 13)]
    outfile_zip = dest / f"oras5_{tag}_{year}.zip"

    try:
        client = cdsapi.Client(quiet=True)
        client.retrieve(
            "reanalysis-oras5",
            {
                "product_type": [product],
                "vertical_resolution": vertical_resolution,
                "variable": variables,
                "year": [str(year)],
                "month": months,
            },
            str(outfile_zip),
        )

        # Extract zip
        if outfile_zip.exists() and zipfile.is_zipfile(outfile_zip):
            with zipfile.ZipFile(outfile_zip, "r") as zf:
                zf.extractall(dest)
            outfile_zip.unlink()
            return f"{year} ({tag}): extracted"
        elif outfile_zip.exists():
            outfile_zip.rename(outfile_nc)
            return f"{year} ({tag}): saved"
        else:
            return f"{year} ({tag}): no output file"

    except Exception as e:
        outfile_zip.unlink(missing_ok=True)
        return f"{year} ({tag}): FAILED - {e}"


def run_phase(
    years: list[int],
    dest: Path,
    variables: list[str],
    vertical_resolution: str,
    tag: str,
    workers: int,
    months: list[str] | None = None,
) -> None:
    """Run parallel downloads for a phase."""
    print(f"\n{'='*60}")
    print(f"Phase: {tag} ({vertical_resolution})")
    print(f"Variables: {variables}")
    print(f"Months: {months or 'all 12'}")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Workers: {workers}")
    print(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_one_year, y, dest, variables, vertical_resolution, tag,
                months,
            ): y
            for y in years
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            print(f"  [{completed}/{len(years)}] {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ORAS5 in parallel.")
    parser.add_argument("--start", type=int, default=1958)
    parser.add_argument("--end", type=int, default=2023)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel CDS requests (default: 5, CDS max ~10)")
    parser.add_argument("--phase", choices=["surface", "depth", "both"],
                        default="both",
                        help="Which phase to run (default: both)")
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated months to download, e.g. '06' for "
                        "June only, '01,04,07,10' for quarterly (default: all 12)")
    args = parser.parse_args()

    dest = Path(args.output_dir) / "oras5"
    dest.mkdir(parents=True, exist_ok=True)

    years = list(range(args.start, args.end + 1))
    months = args.months.split(",") if args.months else None

    if args.phase in ("surface", "both"):
        run_phase(years, dest, SINGLE_LEVEL_VARS, "single_level",
                  "sfc", args.workers)

    if args.phase in ("depth", "both"):
        run_phase(years, dest, ALL_LEVEL_VARS, "all_levels",
                  "3d", args.workers, months=months)

    print(f"\nDone. Files in {dest}:")
    for f in sorted(dest.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
