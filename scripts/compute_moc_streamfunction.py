#!/usr/bin/env python3
"""Compute the time-mean Atlantic MOC streamfunction Ψ(lat, depth).

Processes ORAS5 monthly vomecrty files, zonally integrates across the
Atlantic at each latitude, and cumulatively integrates from the bottom
to produce the overturning streamfunction in Sv.

Result is cached as a .npz file for fast replotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def atlantic_lon_bounds(lat: float) -> tuple[float, float]:
    """Return (lon_min, lon_max) for Atlantic at given latitude."""
    if lat < -34:
        return (-70.0, 20.0)
    elif lat < 0:
        return (-70.0, 20.0)
    elif lat < 10:
        return (-90.0, 10.0)
    elif lat < 30:
        return (-100.0, 0.0)
    elif lat < 45:
        return (-82.0, -5.0)    # exclude Mediterranean
    elif lat < 65:
        return (-70.0, 0.0)
    else:
        return (-60.0, 10.0)    # Nordic Seas


def compute_mean_streamfunction(
    data_dir: Path,
    start_year: int = 2005,
    end_year: int = 2024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-mean MOC streamfunction from ORAS5 vomecrty files.

    Returns
    -------
    psi : ndarray (nz, ny) — streamfunction in Sv
    lat_1d : ndarray (ny,) — latitude at each y index
    depth : ndarray (nz,) — depth levels in metres
    """
    files = sorted(data_dir.glob("vomecrty_control_monthly_highres_3D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No vomecrty files in {data_dir}")

    # Filter by year range
    selected = []
    for f in files:
        # Extract YYYYMM from filename
        parts = f.stem.split("_")
        for p in parts:
            if len(p) == 6 and p.isdigit():
                yr = int(p[:4])
                if start_year <= yr <= end_year:
                    selected.append(f)
                break
    print(f"  Selected {len(selected)} files ({start_year}–{end_year})")

    if not selected:
        raise ValueError(f"No files found for {start_year}–{end_year}")

    # Read grid from first file
    ds0 = xr.open_dataset(selected[0])
    nav_lon = ds0["nav_lon"].values   # (y, x)
    nav_lat = ds0["nav_lat"].values   # (y, x)
    depth = ds0["depthv"].values      # (nz,)
    ny, nx = nav_lon.shape
    nz = len(depth)
    ds0.close()

    # Representative latitude for each y-index (mean across x)
    lat_1d = np.nanmean(nav_lat, axis=1)

    # Precompute Atlantic mask and dx at each latitude
    print("  Precomputing Atlantic mask and grid metrics...")
    # dx = spacing in metres along x at each (y, x) point
    # Must handle longitude wrapping at ±180° on NEMO ORCA grid
    dlon = np.diff(nav_lon, axis=1)  # (y, x-1)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    # Pad to same shape
    dlon = np.concatenate([dlon, dlon[:, -1:]], axis=1)  # (y, x)
    cos_lat = np.cos(np.deg2rad(nav_lat))
    dx = np.abs(dlon) * 111000.0 * cos_lat  # metres
    dx = np.clip(dx, 1.0, None)  # avoid zero/negative

    # Atlantic mask: 2D boolean (y, x)
    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        lat_j = lat_1d[j]
        if lat_j < -55 or lat_j > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(lat_j)
        atl_mask[j, :] = (nav_lon[j, :] >= lon_min) & (nav_lon[j, :] <= lon_max)

    # Depth spacing (dz) — prepend 0 so first level has its own thickness
    dz = np.diff(depth, prepend=0.0)  # (nz,)

    # Accumulate zonally integrated V transport: sum_x(v * dx) at each (z, y)
    v_zonal_sum = np.zeros((nz, ny), dtype=np.float64)
    count = 0

    for i, f in enumerate(selected):
        if (i + 1) % 24 == 0 or i == 0:
            print(f"  Processing file {i+1}/{len(selected)}: {f.name}")
        ds = xr.open_dataset(f)
        v = ds["vomecrty"].values[0]  # (nz, y, x), float32
        ds.close()

        # Replace fill values with 0 (land)
        v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)

        # Zonally integrate: sum_x(v * dx) for Atlantic points
        # v has shape (nz, y, x), dx has shape (y, x), atl_mask has shape (y, x)
        # Broadcast: v * dx[None, :, :] * atl_mask[None, :, :]
        v_dx = v * dx[np.newaxis, :, :] * atl_mask[np.newaxis, :, :]
        v_zonal = np.nansum(v_dx, axis=2)  # (nz, y)

        v_zonal_sum += v_zonal
        count += 1

    # Time mean
    v_zonal_mean = v_zonal_sum / count  # (nz, y) in m²/s

    # Multiply by dz to get transport per level: m³/s
    transport = v_zonal_mean * dz[:, np.newaxis]  # (nz, y)

    # Streamfunction: cumulative integral from surface downward
    # psi(z, y) = integral from surface to z of transport
    # Convention: positive = clockwise (northward in upper ocean)
    # psi peaks at ~1000m = AMOC upper cell strength
    psi = np.cumsum(transport, axis=0)  # m³/s
    psi /= 1e6  # convert to Sv

    print(f"  MOC streamfunction computed: shape {psi.shape}")
    print(f"  Max upper cell: {np.nanmax(psi):.1f} Sv")
    print(f"  Min lower cell: {np.nanmin(psi):.1f} Sv")

    return psi, lat_1d, depth


def main():
    parser = argparse.ArgumentParser(description="Compute MOC streamfunction")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oras5"))
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    cache_file = args.cache_dir / f"moc_streamfunction_{args.start_year}_{args.end_year}.npz"

    if cache_file.exists():
        print(f"Cache exists: {cache_file}")
        cached = np.load(cache_file)
        psi = cached["psi"]
        print(f"  Shape: {psi.shape}, max: {np.nanmax(psi):.1f} Sv")
        return

    print("Computing MOC streamfunction...")
    psi, lat_1d, depth = compute_mean_streamfunction(
        args.data_dir, args.start_year, args.end_year
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file, psi=psi, lat=lat_1d, depth=depth,
        start_year=args.start_year, end_year=args.end_year,
    )
    print(f"  Saved to {cache_file}")


if __name__ == "__main__":
    main()
