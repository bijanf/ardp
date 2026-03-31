#!/usr/bin/env python3
"""Fix corrupted time coordinates in CMIP6 SSP585 extension files.

The download script's cftime-to-numeric conversion produced garbage values
for some time types. This script reconstructs correct time coordinates
from the known file metadata (model, start year, number of timesteps).
"""

from pathlib import Path

import numpy as np
import xarray as xr

EXT_DIR = Path("data/cmip6_fullfield")

# Expected start years and durations from ESGF file listings
MODEL_INFO = {
    "ACCESS-CM2":    {"start_year": 2251, "n_months": 600},   # 2251-2300
    "ACCESS-ESM1-5": {"start_year": 2211, "n_months": 1080},  # 2211-2300
    "MRI-ESM2-0":    {"start_year": 2101, "n_months": 2400},  # 2101-2300
    "IPSL-CM6A-LR":  {"start_year": 2101, "n_months": 2400},  # 2101-2300
    "CESM2-WACCM":   {"start_year": 2101, "n_months": 2388},  # 2101-2299 (partial)
}


def fix_file(model: str) -> None:
    fpath = EXT_DIR / f"{model}_ssp585ext_vo_zonal.nc"
    if not fpath.exists():
        print(f"  SKIP: {fpath.name} not found")
        return

    info = MODEL_INFO[model]
    ds = xr.open_dataset(fpath, decode_times=False)
    n_time = ds.sizes["time"]

    # Check if time is already correct
    t = ds["time"].values
    expected_first = (info["start_year"] - 1850) * 365.25 + 15
    if abs(t[0] - expected_first) < 50:
        print(f"  OK: {fpath.name} — time looks correct (first={t[0]:.1f}, expected~{expected_first:.0f})")
        ds.close()
        return

    print(f"  FIX: {fpath.name} — {n_time} timesteps, first time={t[0]:.4g} (expected~{expected_first:.0f})")

    # Reconstruct time: monthly values from start_year
    months = []
    yr = info["start_year"]
    mo = 1
    for _ in range(n_time):
        days = (yr - 1850) * 365.25 + (mo - 1) * 30.4375 + 15
        months.append(days)
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1

    new_time = np.array(months, dtype=np.float64)
    ds["time"] = xr.Variable("time", new_time, attrs={
        "units": "days since 1850-01-01",
        "calendar": "proleptic_gregorian",
    })

    tmp = fpath.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    ds.close()
    tmp.replace(fpath)

    last_yr = 1850 + new_time[-1] / 365.25
    print(f"       Fixed: {info['start_year']}–{last_yr:.0f} ({n_time} months)")


def main():
    print("Fixing time coordinates in CMIP6 extension files...\n")
    for model in MODEL_INFO:
        print(f"{model}:")
        fix_file(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
