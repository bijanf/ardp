#!/usr/bin/env python3
"""Download SAMBA array MOC transport data from Kersale et al. (2020).

Source: Kersale et al. (2020), Science Advances, doi:10.1126/sciadv.aba7573
        NOAA FTP: ftp://ftp.aoml.noaa.gov/phod/pub/SAM/2020_Kersale_etal_ScienceAdvances/

Data description:
  Daily volume transport of the upper and abyssal overturning cells at 34.5S
  from the SAMBA mooring array (Sep 2013 - Jul 2017).

  - Upper cell: basin-wide transport from surface to 1315 dbar (northward positive)
  - Abyssal cell: transport from 3155 to 4295 dbar (southward positive)
  - Units: Sv (10^6 m^3/s)
  - Accuracy: ~6.4 Sv (upper), ~6.3 Sv (abyssal) daily

Important note:
  This is MOC *volume* transport, not freshwater transport (F_ov). The F_ov
  requires salinity profiles which moorings don't directly measure. For
  validation of ORAS5 F_ovS, the upper-cell MOC transport serves as a
  dynamically related proxy — a weakening MOC (reduced northward upper-cell
  transport) is expected to correlate with declining F_ovS.

Output:
  data/external/samba_moc_daily.nc    — daily upper & abyssal cell transport
  data/external/samba_moc_monthly.nc  — monthly means for ORAS5 comparison
"""

from __future__ import annotations

from ftplib import FTP
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FTP_HOST = "ftp.aoml.noaa.gov"
FTP_DIR = "/phod/pub/SAM/2020_Kersale_etal_ScienceAdvances/"
OUTPUT_DIR = Path("data/external")
OUTPUT_DAILY = OUTPUT_DIR / "samba_moc_daily.nc"
OUTPUT_MONTHLY = OUTPUT_DIR / "samba_moc_monthly.nc"
TRANSPORT_FILE = "Upper_Abyssal_Transport_Anomalies.txt"


def _download_transport() -> str:
    """Download the transport anomalies file from SAMBA FTP."""
    print(f"Connecting to {FTP_HOST} ...")
    ftp = FTP(FTP_HOST)
    ftp.login()
    ftp.cwd(FTP_DIR)

    # List directory contents
    files = ftp.nlst()
    print(f"FTP directory contents: {files}")

    lines: list[str] = []
    print(f"Downloading {TRANSPORT_FILE} ...")
    ftp.retrlines(f"RETR {TRANSPORT_FILE}", lines.append)
    ftp.quit()
    return "\n".join(lines)


def _parse_transport(text: str) -> xr.Dataset:
    """Parse SAMBA transport data.

    Format (tab-delimited):
      year month day hour minute upper_cell_Sv abyssal_cell_Sv

    Missing values indicated by NaN.
    """
    lines = text.strip().split("\n")

    # Skip comment lines (starting with %)
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        data_lines.append(stripped)

    # Show sample
    print("\nFirst 5 data lines:")
    for line in data_lines[:5]:
        print(f"  {line}")

    # Parse
    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            upper = float(parts[5])
            abyssal = float(parts[6])
            rows.append((year, month, day, upper, abyssal))
        except (ValueError, IndexError):
            continue

    arr = np.array(rows)
    dates = pd.to_datetime({
        "year": arr[:, 0].astype(int),
        "month": arr[:, 1].astype(int),
        "day": arr[:, 2].astype(int),
    })

    ds = xr.Dataset(
        {
            "upper_cell": xr.DataArray(
                arr[:, 3], dims=["time"], coords={"time": dates},
                attrs={
                    "long_name": "Upper overturning cell transport at 34.5S",
                    "units": "Sv",
                    "positive": "northward in upper limb",
                    "integration_depth": "surface to 1315 dbar",
                    "accuracy": "6.4 Sv daily",
                },
            ),
            "abyssal_cell": xr.DataArray(
                arr[:, 4], dims=["time"], coords={"time": dates},
                attrs={
                    "long_name": "Abyssal overturning cell transport at 34.5S",
                    "units": "Sv",
                    "positive": "southward in upper limb",
                    "integration_depth": "3155 to 4295 dbar",
                    "accuracy": "6.3 Sv daily",
                },
            ),
        },
        attrs={
            "source": "SAMBA mooring array, Kersale et al. (2020) Science Advances",
            "doi": "10.1126/sciadv.aba7573",
            "latitude": -34.5,
            "description": "MOC volume transport (NOT freshwater transport)",
        },
    )
    return ds


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        text = _download_transport()
    except Exception as e:
        print(f"FTP connection failed: {e}")
        print("\nManual download instructions:")
        print(f"  1. Connect to ftp://{FTP_HOST}{FTP_DIR}")
        print(f"  2. Download {TRANSPORT_FILE}")
        print(f"  3. Place in {OUTPUT_DIR}/")
        print("\nAlternatively, visit:")
        print("  https://www.aoml.noaa.gov/phod/SAMOC_international/")
        return

    ds = _parse_transport(text)

    # Replace NaN strings that survived
    for var in ds.data_vars:
        ds[var] = ds[var].where(np.isfinite(ds[var]))

    n_valid = int(np.isfinite(ds["upper_cell"]).sum())
    print(f"\nParsed {len(ds.time)} daily records ({n_valid} valid)")
    print(f"Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"Upper cell: mean={float(ds['upper_cell'].mean()):.1f} Sv, "
          f"std={float(ds['upper_cell'].std()):.1f} Sv")

    # Save daily
    ds.to_netcdf(OUTPUT_DAILY)
    print(f"\nSaved daily: {OUTPUT_DAILY}")

    # Monthly means
    ds_monthly = ds.resample(time="MS").mean()
    ds_monthly.attrs = ds.attrs.copy()
    ds_monthly.attrs["temporal_resolution"] = "monthly mean"
    ds_monthly.to_netcdf(OUTPUT_MONTHLY)
    print(f"Saved monthly: {OUTPUT_MONTHLY} ({len(ds_monthly.time)} months)")


if __name__ == "__main__":
    main()
