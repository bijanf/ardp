#!/usr/bin/env python3
"""Download missing ORAS5 3D monthly data (velocity + salinity) for full MOC analysis.

Checks which months already exist and only downloads missing ones.
Runs serially (1 request at a time) to respect CDS API rate limits.

Target: all months for 2004-2023 (RAPID overlap), plus 2013-2017 fill for SAMBA.
Each file is ~800 MB, total estimated: ~170 GB for ~220 missing months.

Usage:
  # Download all missing months for RAPID overlap (2004-2023):
  python scripts/download_oras5_monthly_batch.py --start 2004 --end 2023

  # Download everything 1958-2023 (warning: ~600 GB):
  python scripts/download_oras5_monthly_batch.py --start 1958 --end 2023

  # Dry run to see what's missing:
  python scripts/download_oras5_monthly_batch.py --start 2004 --end 2023 --dry-run

  # Resume after interruption (just re-run, it skips existing files):
  python scripts/download_oras5_monthly_batch.py --start 2004 --end 2023
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import cdsapi

DEST = Path("data/oras5")
VARIABLES = ["meridional_velocity", "salinity"]
VARIABLE_PREFIXES = ["vomecrty", "vosaline"]


def file_exists(dest: Path, year: int, month: int) -> bool:
    """Check if both velocity and salinity files exist for a given month."""
    yyyymm = f"{year}{month:02d}"
    for prefix in VARIABLE_PREFIXES:
        matches = list(dest.glob(f"{prefix}_*_3D_{yyyymm}_*.nc"))
        if not matches:
            return False
    return True


def find_missing_months(
    dest: Path, start: int, end: int
) -> list[tuple[int, int]]:
    """Find all (year, month) pairs where data is missing."""
    missing = []
    for year in range(start, end + 1):
        for month in range(1, 13):
            if not file_exists(dest, year, month):
                missing.append((year, month))
    return missing


def download_month(year: int, month: int, dest: Path) -> str:
    """Download one month of ORAS5 3D data (velocity + salinity)."""
    yyyymm = f"{year}{month:02d}"
    product = "consolidated" if year <= 2014 else "operational"

    # Handle boundary year: 2014 has consolidated for most months,
    # but operational may be needed. Try consolidated first.
    if year == 2014:
        product = "consolidated"
    if year == 2015 and month <= 6:
        # Some early 2015 months may be consolidated
        product = "operational"

    outfile_zip = dest / f"oras5_3d_{yyyymm}.zip"

    try:
        client = cdsapi.Client(quiet=True)
        client.retrieve(
            "reanalysis-oras5",
            {
                "product_type": [product],
                "vertical_resolution": "all_levels",
                "variable": VARIABLES,
                "year": [str(year)],
                "month": [f"{month:02d}"],
            },
            str(outfile_zip),
        )

        # Extract
        if outfile_zip.exists() and zipfile.is_zipfile(outfile_zip):
            with zipfile.ZipFile(outfile_zip, "r") as zf:
                zf.extractall(dest)
            outfile_zip.unlink()
            return f"{yyyymm}: extracted OK"
        elif outfile_zip.exists():
            # Single file (not zipped)
            outfile_zip.rename(dest / f"oras5_3d_{yyyymm}.nc")
            return f"{yyyymm}: saved (single file)"
        else:
            return f"{yyyymm}: no output"

    except Exception as e:
        outfile_zip.unlink(missing_ok=True)
        # If consolidated fails for boundary years, try operational
        if product == "consolidated" and year >= 2014:
            try:
                client = cdsapi.Client(quiet=True)
                client.retrieve(
                    "reanalysis-oras5",
                    {
                        "product_type": ["operational"],
                        "vertical_resolution": "all_levels",
                        "variable": VARIABLES,
                        "year": [str(year)],
                        "month": [f"{month:02d}"],
                    },
                    str(outfile_zip),
                )
                if outfile_zip.exists() and zipfile.is_zipfile(outfile_zip):
                    with zipfile.ZipFile(outfile_zip, "r") as zf:
                        zf.extractall(dest)
                    outfile_zip.unlink()
                    return f"{yyyymm}: extracted OK (operational fallback)"
                elif outfile_zip.exists():
                    outfile_zip.rename(dest / f"oras5_3d_{yyyymm}.nc")
                    return f"{yyyymm}: saved (operational fallback)"
            except Exception as e2:
                outfile_zip.unlink(missing_ok=True)
                return f"{yyyymm}: FAILED both product_types - {e2}"
        return f"{yyyymm}: FAILED - {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download missing ORAS5 3D monthly data serially."
    )
    parser.add_argument("--start", type=int, default=2004,
                        help="Start year (default: 2004 for RAPID overlap)")
    parser.add_argument("--end", type=int, default=2023,
                        help="End year (default: 2023)")
    parser.add_argument("--output-dir", type=str, default="data/oras5")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list missing months, don't download")
    parser.add_argument("--delay", type=int, default=5,
                        help="Seconds between requests (default: 5)")
    args = parser.parse_args()

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Find what's missing
    missing = find_missing_months(dest, args.start, args.end)
    total_months = (args.end - args.start + 1) * 12
    existing = total_months - len(missing)

    print("ORAS5 3D monthly batch download")
    print(f"  Range: {args.start}-{args.end} ({total_months} months)")
    print(f"  Existing: {existing}")
    print(f"  Missing: {len(missing)}")
    print(f"  Estimated download: ~{len(missing) * 0.8:.0f} GB")
    est_hours = len(missing) * 3 / 60  # ~3 min per file
    print(f"  Estimated time: ~{est_hours:.0f} hours")

    if args.dry_run:
        print("\nMissing months:")
        for year, month in missing:
            print(f"  {year}-{month:02d}")
        return

    if not missing:
        print("\nAll months present!")
        return

    print(f"\nStarting serial download ({args.delay}s delay between requests)...")
    print("Press Ctrl+C to stop (re-run to resume from where you left off)\n")

    successes = 0
    failures = 0
    for i, (year, month) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] Downloading {year}-{month:02d} ...", end=" ",
              flush=True)
        result = download_month(year, month, dest)
        print(result)

        if "FAILED" in result:
            failures += 1
        else:
            successes += 1

        if i < len(missing):
            time.sleep(args.delay)

    print(f"\nDone: {successes} downloaded, {failures} failed")


if __name__ == "__main__":
    main()
