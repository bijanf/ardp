#!/usr/bin/env python3
"""Download missing ORAS5 3D monthly data with parallel CDS requests.

Resumable: checks which months already exist and only downloads missing ones.
Uses ThreadPoolExecutor for parallel downloads (default: 5 workers).

Usage:
  python scripts/download_oras5_3d_parallel.py --start 1958 --end 2023
  python scripts/download_oras5_3d_parallel.py --start 1958 --end 2023 --dry-run
  python scripts/download_oras5_3d_parallel.py --start 1958 --end 2023 --workers 8
"""

from __future__ import annotations

import argparse
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # Determine product type
    if year <= 2014:
        product = "consolidated"
    else:
        product = "operational"

    outfile_zip = dest / f"oras5_3d_{yyyymm}.zip"

    def _try_download(prod: str) -> str | None:
        """Attempt download with given product type. Returns None on success."""
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(
                "reanalysis-oras5",
                {
                    "product_type": [prod],
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
                return None  # success
            elif outfile_zip.exists():
                # Not a valid zip - might be corrupt
                outfile_zip.unlink()
                return f"invalid zip with {prod}"
            else:
                return f"no output with {prod}"

        except Exception as e:
            outfile_zip.unlink(missing_ok=True)
            return str(e)

    # Try primary product type
    err = _try_download(product)
    if err is None:
        return f"{yyyymm}: OK"

    # Fallback for boundary years
    if year >= 2014 and product == "consolidated":
        err2 = _try_download("operational")
        if err2 is None:
            return f"{yyyymm}: OK (operational)"
        return f"{yyyymm}: FAILED - consolidated: {err}; operational: {err2}"

    if year >= 2014 and product == "operational":
        err2 = _try_download("consolidated")
        if err2 is None:
            return f"{yyyymm}: OK (consolidated)"
        return f"{yyyymm}: FAILED - operational: {err}; consolidated: {err2}"

    return f"{yyyymm}: FAILED - {err}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download missing ORAS5 3D monthly data in parallel."
    )
    parser.add_argument("--start", type=int, default=1958)
    parser.add_argument("--end", type=int, default=2023)
    parser.add_argument("--output-dir", type=str, default="data/oras5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel downloads (default: 5, CDS allows ~10)")
    args = parser.parse_args()

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    missing = find_missing_months(dest, args.start, args.end)
    total_months = (args.end - args.start + 1) * 12
    existing = total_months - len(missing)

    print("ORAS5 3D parallel download")
    print(f"  Range: {args.start}-{args.end} ({total_months} months)")
    print(f"  Existing: {existing}")
    print(f"  Missing: {len(missing)}")
    print(f"  Workers: {args.workers}")
    est_gb = len(missing) * 0.8
    est_hours = len(missing) * 3 / 60 / args.workers
    print(f"  Estimated: ~{est_gb:.0f} GB, ~{est_hours:.0f} hours")

    if args.dry_run:
        print("\nMissing months:")
        for year, month in missing:
            print(f"  {year}-{month:02d}")
        return

    if not missing:
        print("\nAll months present!")
        return

    print(f"\nStarting {args.workers} parallel downloads...")
    print("(Re-run to resume from where you left off)\n")

    successes = 0
    failures = 0
    failed_months: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_month, year, month, dest): (year, month)
            for year, month in missing
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            year, month = futures[future]
            elapsed = time.time() - t0
            rate = elapsed / i  # seconds per download
            remaining = rate * (len(missing) - i) / 3600

            if "FAILED" in result:
                failures += 1
                failed_months.append(result)
                print(f"  [{i}/{len(missing)}] {result}")
            else:
                successes += 1
                if i % 10 == 0 or i <= 5:
                    print(f"  [{i}/{len(missing)}] {result}  "
                          f"(~{remaining:.1f}h remaining)")

    elapsed_h = (time.time() - t0) / 3600
    print(f"\nDone in {elapsed_h:.1f}h: {successes} downloaded, {failures} failed")

    if failed_months:
        print(f"\nFailed months ({failures}):")
        for fm in failed_months:
            print(f"  {fm}")


if __name__ == "__main__":
    main()
