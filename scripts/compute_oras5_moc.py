#!/usr/bin/env python3
"""Compute ORAS5 MOC upper-cell transport at 26.5N (RAPID) and 34.5S (SAMBA).

Memory-efficient: processes one velocity file at a time, extracting only the
j-row at each target latitude (~430 KB per timestep per latitude).

The MOC upper-cell strength is the maximum of the depth-space overturning
streamfunction, computed as:
  1. Zonally integrate meridional velocity × dx × dz at each depth level
  2. Cumulate from the ocean bottom upward
  3. Take the maximum = upper-cell MOC transport [Sv]

This is the same quantity measured by the RAPID array (moc_mar_hc10) and
the SAMBA array (upper_cell transport, surface to 1315 dbar).

Output:
  data/results/oras5_moc_26N.nc   — MOC time series at 26.5N (for RAPID validation)
  data/results/oras5_moc_34S.nc   — MOC time series at 34.5S (for SAMBA validation)
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from ardp.constants import RAPID_LAT, SAMBA_LAT


# Atlantic basin longitude bounds (latitude-dependent)
ATLANTIC_BOUNDS: dict[str, tuple[float, float]] = {
    "26N": (-100.0, 0.0),   # Florida Straits through to Africa
    "34S": (-70.0, 20.0),   # South Atlantic
}


def find_j_index_and_metrics(
    data_dir: Path, target_lat: float
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Find j-index for target latitude and compute grid metrics.

    Returns (j_idx, e1t_atlantic, e3t, atlantic_mask, actual_lat).
    """
    v_file = sorted(data_dir.glob("vomecrty_*.nc"))[0]
    ds = xr.open_dataset(v_file)

    nav_lat = ds["nav_lat"].values  # (y, x)
    nav_lon = ds["nav_lon"].values  # (y, x)

    # Mean latitude per j-row
    lat_1d = np.nanmean(nav_lat, axis=1)
    j_idx = int(np.abs(lat_1d - target_lat).argmin())
    actual_lat = float(lat_1d[j_idx])

    # Atlantic mask at this latitude
    lon_row = nav_lon[j_idx, :]
    # Pick appropriate Atlantic bounds
    if target_lat > 0:
        lon_min, lon_max = ATLANTIC_BOUNDS["26N"]
    else:
        lon_min, lon_max = ATLANTIC_BOUNDS["34S"]
    atlantic_mask = (lon_row >= lon_min) & (lon_row <= lon_max)

    # e1t: zonal grid spacing from longitude differences
    lat_row = nav_lat[j_idx, :]
    dlon = np.diff(lon_row)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dlon = np.append(dlon, dlon[-1])
    cos_lat = np.cos(np.deg2rad(lat_row))
    e1t = np.abs(dlon) * 111000.0 * cos_lat
    e1t = np.clip(e1t, 1.0, None)

    # e3t from depth coordinate
    depth = ds["depthv"].values
    e3t = np.diff(depth, prepend=0.0)

    ds.close()
    return j_idx, e1t[atlantic_mask], e3t, atlantic_mask, actual_lat


def _extract_yyyymm(path: Path) -> str:
    """Extract YYYYMM from filename like vomecrty_..._3D_195801_CONS_v0.1.nc."""
    m = re.search(r"_3D_(\d{6})_", path.name)
    if not m:
        raise ValueError(f"Cannot extract YYYYMM from {path.name}")
    return m.group(1)


def list_velocity_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Find all velocity files and extract YYYYMM."""
    files = sorted(data_dir.glob("vomecrty_*_3D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No velocity files in {data_dir}")
    return [(f, _extract_yyyymm(f)) for f in files]


def compute_moc_one_month(
    v_file: Path,
    j_idx: int,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    atlantic_mask: np.ndarray,
) -> tuple[np.datetime64, float]:
    """Compute MOC upper-cell strength for one month at one latitude.

    MOC = max of streamfunction psi(z), where:
      psi(z) = cumsum from bottom of V_transport(z)
      V_transport(z) = sum_x(v(z,x) * e1(x)) * e3(z)

    Returns (timestamp, moc_sv).
    """
    ds = xr.open_dataset(v_file)
    v_section = ds["vomecrty"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]
    timestamp = ds["time_counter"].values[0]
    ds.close()

    nz = v_section.shape[0]

    # Compute zonally integrated transport at each depth level
    v_transport = np.zeros(nz)
    for k in range(nz):
        v_k = np.nan_to_num(v_section[k, :], nan=0.0)
        v_transport[k] = (v_k * e1t_atl).sum() * e3t[k]  # m³/s

    # Streamfunction: cumulative sum from surface downward
    # psi(z) = integral from surface to z of V_transport
    # Upper-cell MOC = max(psi) — the maximum northward overturning
    psi_sv = np.cumsum(v_transport) / 1e6  # Sv

    # Upper-cell MOC = maximum of streamfunction
    moc_upper = float(np.max(psi_sv))

    return (timestamp, moc_upper)


def compute_moc_at_latitude(
    data_dir: Path,
    target_lat: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute MOC time series at a given latitude.

    Returns (timestamps, moc_values, actual_lat).
    """
    print(f"\n--- Computing MOC at {target_lat:.1f}N ---")

    j_idx, e1t_atl, e3t, atlantic_mask, actual_lat = find_j_index_and_metrics(
        data_dir, target_lat
    )
    print(f"  Section at j={j_idx}, lat={actual_lat:.2f}")
    print(f"  Atlantic x-points: {atlantic_mask.sum()}")

    v_files = list_velocity_files(data_dir)
    print(f"  Found {len(v_files)} velocity files")

    results: list[tuple[np.datetime64, float]] = []
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                compute_moc_one_month, v_file, j_idx, e1t_atl, e3t, atlantic_mask
            ): yyyymm
            for v_file, yyyymm in v_files
        }
        for i, future in enumerate(as_completed(futures), 1):
            yyyymm = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                errors.append(f"{yyyymm}: {e}")
                print(f"  WARNING: {yyyymm} failed: {e}")
            if i % 20 == 0 or i == len(v_files):
                print(f"  {i}/{len(v_files)} done")

    if errors:
        print(f"  {len(errors)} months failed")

    results.sort(key=lambda x: x[0])
    timestamps = np.array([r[0] for r in results])
    values = np.array([r[1] for r in results])

    print(f"  MOC range: {values.min():.1f} to {values.max():.1f} Sv")
    print(f"  MOC mean: {values.mean():.1f} Sv")

    return timestamps, values, actual_lat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ORAS5 MOC at RAPID (26.5N) and SAMBA (34.5S) latitudes."
    )
    parser.add_argument("--data-dir", default="data/oras5")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for target_lat, label, outname in [
        (RAPID_LAT, "RAPID 26.5N", "oras5_moc_26N.nc"),
        (SAMBA_LAT, "SAMBA 34.5S", "oras5_moc_34S.nc"),
    ]:
        timestamps, values, actual_lat = compute_moc_at_latitude(
            data_dir, target_lat, args.workers
        )

        da = xr.DataArray(
            values,
            dims=("time",),
            coords={"time": timestamps},
            name="moc_upper",
            attrs={
                "units": "Sv",
                "long_name": f"Upper-cell MOC transport at {actual_lat:.1f}",
                "section_latitude": actual_lat,
                "method": "max of depth-space overturning streamfunction",
                "source": "ORAS5 meridional velocity",
                "validation_target": label,
            },
        )

        outfile = results_dir / outname
        da.to_netcdf(outfile)
        print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
