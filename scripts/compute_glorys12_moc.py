#!/usr/bin/env python3
"""Compute GLORYS12 MOC upper-cell transport at 26.5N for RAPID validation.

Memory-efficient: processes one year file at a time, extracting only the
j-row at 26.5N latitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import RAPID_LAT

# Atlantic basin at 26.5N (Florida Straits to Africa)
ATLANTIC_LON_MIN = -80.5
ATLANTIC_LON_MAX = -13.0


def compute_glorys12_moc(data_dir: Path, workers: int = 1) -> None:
    """Compute MOC at 26.5N from GLORYS12 monthly vo fields."""
    files = sorted(data_dir.glob("glorys12_*.nc"))
    if not files:
        raise FileNotFoundError(f"No GLORYS12 files in {data_dir}")
    print(f"Found {len(files)} yearly files")

    # Get grid info from first file
    ds0 = xr.open_dataset(files[0])
    lat = ds0["latitude"].values
    lon = ds0["longitude"].values
    depth = ds0["depth"].values

    # Find j-index for 26.5N
    j_idx = int(np.abs(lat - RAPID_LAT).argmin())
    actual_lat = float(lat[j_idx])
    print(f"  Section at j={j_idx}, lat={actual_lat:.2f} (target: {RAPID_LAT})")

    # Atlantic mask
    atlantic_mask = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    n_atl = atlantic_mask.sum()
    print(f"  Atlantic x-points: {n_atl}")

    # Grid spacings
    dlon = np.abs(lon[1] - lon[0])
    e1t = dlon * 111000.0 * np.cos(np.deg2rad(actual_lat))  # constant at this lat
    e3t = np.diff(depth, prepend=0.0)

    ds0.close()

    # Process all files
    all_times = []
    all_moc = []

    for fi, f in enumerate(files):
        ds = xr.open_dataset(f)
        vo = ds["vo"].isel(latitude=j_idx).values  # (time, depth, lon)
        times = ds["time"].values
        ds.close()

        nt = vo.shape[0]
        for t in range(nt):
            v_section = vo[t][:, atlantic_mask]  # (depth, n_atl)
            nz = v_section.shape[0]

            # Compute zonally integrated transport at each depth
            nz_use = min(nz, len(e3t))
            v_transport = np.zeros(nz_use)
            for k in range(nz_use):
                v_k = np.nan_to_num(v_section[k, :], nan=0.0)
                v_transport[k] = v_k.sum() * e1t * e3t[k]  # m³/s

            # Streamfunction and upper-cell MOC
            psi_sv = np.cumsum(v_transport) / 1e6
            moc_upper = float(np.max(psi_sv))

            all_times.append(times[t])
            all_moc.append(moc_upper)

        if (fi + 1) % 5 == 0 or fi == len(files) - 1:
            print(f"  {fi+1}/{len(files)} files done")

    # Assemble
    timestamps = np.array(all_times)
    values = np.array(all_moc)

    da = xr.DataArray(
        values,
        dims=("time",),
        coords={"time": timestamps},
        name="moc_upper",
        attrs={
            "units": "Sv",
            "long_name": f"Upper-cell MOC transport at {actual_lat:.1f}N",
            "section_latitude": actual_lat,
            "source": "GLORYS12 meridional velocity",
        },
    )

    print(f"\n  MOC range: {values.min():.1f} to {values.max():.1f} Sv")
    print(f"  MOC mean: {values.mean():.1f} Sv")

    return da


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GLORYS12 MOC at 26.5N for RAPID validation."
    )
    parser.add_argument("--data-dir", default="data/glorys12")
    parser.add_argument("--results-dir", default="data/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    da = compute_glorys12_moc(Path(args.data_dir))

    outfile = results_dir / "glorys12_moc_26N.nc"
    da.to_netcdf(outfile)
    print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
