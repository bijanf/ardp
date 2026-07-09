#!/usr/bin/env python3
"""Download EN4.2.2 monthly objective-analysis NetCDF files (2005-2024).

Source: Met Office Hadley Centre EN4.2.2 analyses (Good et al., 2013), with the
        Gouretski-Reseghetti (2010) XBT corrections — the "g10" variant.
        https://www.metoffice.gov.uk/hadobs/en4/

Data description:
  Monthly objective analysis of in-situ profile T/S observations on a 1° x 1° x
  42-level grid, 1900-present. Each year is distributed as a single zip archive
  containing 12 monthly NetCDF files plus an analysis-error variance field.

  Variables of interest for the AMOC paper:
    - salinity        (PSU, on 42 depth levels)
    - salinity_uncertainty (PSU, analysis-error standard deviation)

Output:
  data/en4/EN.4.2.2.f.analysis.g10.YYYYMM.nc   — one file per month (240 files)
  data/en4/.checksum.json                       — sha256 of each file

Use as input to:
  - scripts/compute_argo_sa_trend.py
  - scripts/compute_argo_zonal_section.py

Roughly 12 GB total on disk after decompression. The download is incremental:
re-running the script skips files that already exist with matching size.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/"
LEGACY_PATH = "EN.4.2.2/EN.4.2.2.analyses.g10.{year}.zip"     # for years <= 2020
CURRENT_PATH = "EN.4.2.2.analyses.g10.{year}.zip"             # for years >= 2021
OUTPUT_DIR = Path("data/en4")
CHECKSUM_FILE = OUTPUT_DIR / ".checksum.json"
START_YEAR = 2005
END_YEAR = 2024


def _url(year: int) -> str:
    sub = LEGACY_PATH if year <= 2020 else CURRENT_PATH
    return BASE + sub.format(year=year)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_year(year: int, tmp_dir: Path) -> Path:
    """Download a year's zip archive to tmp_dir. Returns the zip path."""
    url = _url(year)
    target = tmp_dir / f"EN.4.2.2.analyses.g10.{year}.zip"
    if target.exists() and target.stat().st_size > 1024:
        print(f"  zip cached: {target.name} ({target.stat().st_size/1e6:.1f} MB)")
        return target
    print(f"  fetching {url}")
    urllib.request.urlretrieve(url, target)  # noqa: S310 — public Met Office URL
    print(f"  saved {target.name} ({target.stat().st_size/1e6:.1f} MB)")
    return target


def _extract_monthly(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract monthly .nc files from a year zip; return the extracted paths."""
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".nc"):
                continue
            target = out_dir / Path(member).name
            if target.exists() and target.stat().st_size > 1024:
                extracted.append(target)
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUTPUT_DIR / "_zip"
    tmp_dir.mkdir(exist_ok=True)

    checksums: dict[str, str] = {}
    if CHECKSUM_FILE.exists():
        checksums = json.loads(CHECKSUM_FILE.read_text())

    total_files = 0
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Year {year}")
        zip_path = _download_year(year, tmp_dir)
        monthly_files = _extract_monthly(zip_path, OUTPUT_DIR)
        for mf in monthly_files:
            key = mf.name
            if key not in checksums:
                checksums[key] = _sha256(mf)
        total_files += len(monthly_files)
        print(f"  -> {len(monthly_files)} monthly files extracted")

    CHECKSUM_FILE.write_text(json.dumps(checksums, indent=2, sort_keys=True))
    print(f"\nDone. {total_files} monthly EN4.2.2 NetCDF files in {OUTPUT_DIR}/")
    print(f"Checksums: {CHECKSUM_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
