#!/usr/bin/env python3
"""Download Roemmich-Gilson Argo Climatology salinity fields (2004-2024).

Source: Scripps Institution of Oceanography (Roemmich & Gilson, 2009).
        https://sio-argo.ucsd.edu/RG_Climatology.html

Data description:
  Monthly 1° x 1° x 58-level objective analysis of Argo profiles 2004-present.
  Two components:
    - RG_ArgoClim_Salinity_2019.nc       — long-term mean + 2004-2018 anomalies
    - RG_ArgoClim_YYYYMM_2019.nc         — monthly extension anomalies thereafter

  Pure-Argo (no XBT/MBT/CTD outside of Argo era), so it provides a clean
  robustness foil for the EN4.2.2 product, which includes XBT corrections.

Output:
  data/argo_rg09/RG_ArgoClim_Salinity_2019.nc
  data/argo_rg09/RG_ArgoClim_YYYYMM_2019.nc  (one per monthly extension)
  data/argo_rg09/.checksum.json

About 500 MB total. Re-running skips files that already exist.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://sio-argo.ucsd.edu/RG/"
OUTPUT_DIR = Path("data/argo_rg09")
CHECKSUM_FILE = OUTPUT_DIR / ".checksum.json"
CLIMATOLOGY_FILE = "RG_ArgoClim_Salinity_2019.nc.gz"
START_YEAR = 2019
END_YEAR = 2024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_and_unzip(url: str, target_gz: Path) -> bool:
    """Download .gz file and decompress to a .nc alongside.

    Returns True if the target was newly fetched.
    """
    target_nc = target_gz.with_suffix("")  # strip the .gz
    if target_nc.exists() and target_nc.stat().st_size > 1024:
        print(f"  cached: {target_nc.name} ({target_nc.stat().st_size/1e6:.1f} MB)")
        return False
    print(f"  fetching {url}")
    try:
        urllib.request.urlretrieve(url, target_gz)  # noqa: S310 — public Scripps URL
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"  not yet published: {target_gz.name}")
            return False
        raise
    # Decompress
    with gzip.open(target_gz, "rb") as src, target_nc.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    target_gz.unlink()  # keep only the .nc
    print(f"  saved {target_nc.name} ({target_nc.stat().st_size/1e6:.1f} MB)")
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}
    if CHECKSUM_FILE.exists():
        checksums = json.loads(CHECKSUM_FILE.read_text())

    print("Climatology + 2004-2018 anomalies")
    target_gz = OUTPUT_DIR / CLIMATOLOGY_FILE
    _download_and_unzip(BASE_URL + CLIMATOLOGY_FILE, target_gz)
    nc_name = target_gz.with_suffix("").name
    if nc_name not in checksums and (OUTPUT_DIR / nc_name).exists():
        checksums[nc_name] = _sha256(OUTPUT_DIR / nc_name)

    print(f"\nMonthly extensions {START_YEAR}-{END_YEAR}")
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            name = f"RG_ArgoClim_{year}{month:02d}_2019.nc.gz"
            target_gz = OUTPUT_DIR / name
            _download_and_unzip(BASE_URL + name, target_gz)
            nc_name = target_gz.with_suffix("").name
            if nc_name not in checksums and (OUTPUT_DIR / nc_name).exists():
                checksums[nc_name] = _sha256(OUTPUT_DIR / nc_name)

    CHECKSUM_FILE.write_text(json.dumps(checksums, indent=2, sort_keys=True))
    print(f"\nDone. {len(checksums)} files in {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
