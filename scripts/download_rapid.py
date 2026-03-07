#!/usr/bin/env python3
"""Download RAPID-MOCHA-WBTS MOC transport time series at 26.5N.

Source: National Oceanography Centre, UK
        RAPID-MOCHA-WBTS array, continuous since April 2004.
        DOI: 10.5285/48d0bf43-0598-ceb2-e063-7086abc062f1
        Version: v2024.1a (Apr 2004 - Mar 2024)

Data description:
  Sub-daily (~12-hourly) transport time series at 26.5N decomposed into:
  - t_gs10: Florida Straits transport (~31 Sv)
  - t_ek10: Ekman transport (~3 Sv)
  - t_umo10: Upper mid-ocean transport
  - moc_mar_hc10: Total MOC overturning transport (~17 Sv mean)
  - Layer transports: thermocline, intermediate, upper/lower NADW, AABW

Output:
  data/external/rapid_moc_transports.nc  — full sub-daily dataset
  data/external/rapid_moc_monthly.nc     — monthly means for reanalysis comparison
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

URL = "https://rapid.ac.uk/sites/default/files/rapid_data/moc_transports.nc"
OUTPUT_DIR = Path("data/external")
OUTPUT_RAW = OUTPUT_DIR / "rapid_moc_transports.nc"
OUTPUT_MONTHLY = OUTPUT_DIR / "rapid_moc_monthly.nc"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    if OUTPUT_RAW.exists():
        print(f"Already exists: {OUTPUT_RAW}")
    else:
        print(f"Downloading RAPID MOC data from {URL} ...")
        try:
            urllib.request.urlretrieve(URL, OUTPUT_RAW)
            print(f"Saved: {OUTPUT_RAW}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nManual download:")
            print(f"  1. Visit https://rapid.ac.uk/data/data-download")
            print(f"  2. Download MOC transports (NetCDF)")
            print(f"  3. Save as {OUTPUT_RAW}")
            return

    # Inspect and create monthly means
    ds = xr.open_dataset(OUTPUT_RAW)
    moc = ds["moc_mar_hc10"]
    n_valid = int(np.isfinite(moc).sum())
    print(f"\nRAPID MOC data:")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  N records: {len(ds.time)} ({n_valid} valid)")
    print(f"  MOC mean: {float(moc.mean()):.1f} Sv")
    print(f"  MOC std: {float(moc.std()):.1f} Sv")
    print(f"  Variables: {list(ds.data_vars)}")

    # Monthly means
    ds_monthly = ds.resample(time="MS").mean()
    ds_monthly.attrs = ds.attrs.copy()
    ds_monthly.attrs["temporal_resolution"] = "monthly mean"
    ds_monthly.to_netcdf(OUTPUT_MONTHLY)
    print(f"\nSaved monthly: {OUTPUT_MONTHLY} ({len(ds_monthly.time)} months)")

    ds.close()


if __name__ == "__main__":
    main()
