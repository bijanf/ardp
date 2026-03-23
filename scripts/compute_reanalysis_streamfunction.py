#!/usr/bin/env python3
"""Compute windowed Atlantic MOC streamfunction snapshots from reanalysis.

Produces Ψ(lat, depth) for rolling time windows, saved as .npz for animation.
Supports ORAS5 (monthly NEMO ORCA files) and GLORYS12 (annual regular-grid files).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


WINDOW_YEARS = 10
STEP_YEARS = 3


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
        return (-82.0, -5.0)
    elif lat < 65:
        return (-70.0, 0.0)
    else:
        return (-60.0, 10.0)


# ── ORAS5 ──────────────────────────────────────────────────────────────

def compute_oras5(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute windowed streamfunction from ORAS5 vomecrty files.

    Returns (psi, lat_1d, depth, center_years).
    psi shape: (n_frames, nz, ny)
    """
    files = sorted(data_dir.glob("vomecrty_control_monthly_highres_3D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No vomecrty files in {data_dir}")

    # Parse year from each filename
    file_years = {}
    for f in files:
        for p in f.stem.split("_"):
            if len(p) == 6 and p.isdigit():
                file_years[f] = int(p[:4])
                break

    year_min = min(file_years.values())
    year_max = max(file_years.values())
    print(f"  ORAS5 data: {year_min}–{year_max}, {len(files)} monthly files")

    # Read grid from first file
    ds0 = xr.open_dataset(files[0])
    nav_lon = ds0["nav_lon"].values
    nav_lat = ds0["nav_lat"].values
    depth = ds0["depthv"].values
    ny, nx = nav_lon.shape
    nz = len(depth)
    ds0.close()

    lat_1d = np.nanmean(nav_lat, axis=1)

    # Grid metrics
    dlon = np.diff(nav_lon, axis=1)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dlon = np.concatenate([dlon, dlon[:, -1:]], axis=1)
    cos_lat = np.cos(np.deg2rad(nav_lat))
    dx = np.abs(dlon) * 111000.0 * cos_lat
    dx = np.clip(dx, 1.0, None)

    # Atlantic mask
    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        lat_j = lat_1d[j]
        if lat_j < -55 or lat_j > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(lat_j)
        atl_mask[j, :] = (nav_lon[j, :] >= lon_min) & (nav_lon[j, :] <= lon_max)

    dz = np.diff(depth, prepend=0.0)

    # Generate center years
    half_w = WINDOW_YEARS // 2
    center_years = np.arange(
        year_min + half_w, year_max - 1, STEP_YEARS
    )
    print(f"  Windows: {len(center_years)} frames, {center_years[0]}–{center_years[-1]}")

    psi_all = np.zeros((len(center_years), nz, ny), dtype=np.float32)

    for fi, cy in enumerate(center_years):
        y0, y1 = cy - half_w, cy + half_w - 1
        window_files = [f for f, yr in file_years.items() if y0 <= yr <= y1]
        print(f"  Frame {fi+1}/{len(center_years)}: {cy} ({y0}–{y1}), {len(window_files)} months", flush=True)

        v_zonal_sum = np.zeros((nz, ny), dtype=np.float64)
        count = 0
        for f in window_files:
            ds = xr.open_dataset(f)
            v = ds["vomecrty"].values[0]
            ds.close()
            v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)
            v_dx = v * dx[np.newaxis, :, :] * atl_mask[np.newaxis, :, :]
            v_zonal_sum += np.nansum(v_dx, axis=2)
            count += 1

        v_zonal_mean = v_zonal_sum / count
        transport = v_zonal_mean * dz[:, np.newaxis]
        psi = np.cumsum(transport, axis=0) / 1e6
        psi_all[fi] = psi.astype(np.float32)

    return psi_all, lat_1d, depth, center_years


# ── GLORYS12 ───────────────────────────────────────────────────────────

def compute_glorys12(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute windowed streamfunction from GLORYS12 annual files.

    Returns (psi, lat_1d, depth, center_years).
    """
    files = sorted(data_dir.glob("glorys12_*.nc"))
    if not files:
        raise FileNotFoundError(f"No glorys12 files in {data_dir}")

    file_years = {f: int(f.stem.split("_")[1]) for f in files}
    year_min = min(file_years.values())
    year_max = max(file_years.values())
    print(f"  GLORYS12 data: {year_min}–{year_max}, {len(files)} annual files")

    # Read grid from first file (only vo)
    ds0 = xr.open_dataset(files[0])[["vo"]]
    lon = ds0["longitude"].values
    lat = ds0["latitude"].values
    depth = ds0["depth"].values
    nz = len(depth)
    ny = len(lat)
    ds0.close()

    # Regular grid: dx = dlon * 111km * cos(lat)
    dlon = np.abs(np.mean(np.diff(lon)))
    dx_1d = dlon * 111000.0 * np.cos(np.deg2rad(lat))

    # Atlantic mask: (ny, nx) boolean
    nx = len(lon)
    atl_mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        if lat[j] < -55 or lat[j] > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(float(lat[j]))
        atl_mask[j, :] = (lon >= lon_min) & (lon <= lon_max)

    # Precompute Atlantic longitude indices per latitude to avoid full-grid ops
    atl_x_slices = {}
    for j in range(ny):
        cols = np.where(atl_mask[j])[0]
        if len(cols) > 0:
            atl_x_slices[j] = (cols[0], cols[-1] + 1)

    dz = np.diff(depth, prepend=0.0)

    half_w = WINDOW_YEARS // 2
    center_years = np.arange(
        year_min + half_w, year_max - 1, STEP_YEARS
    )
    print(f"  Windows: {len(center_years)} frames, {center_years[0]}–{center_years[-1]}")

    psi_all = np.zeros((len(center_years), nz, ny), dtype=np.float32)

    for fi, cy in enumerate(center_years):
        y0, y1 = cy - half_w, cy + half_w - 1
        window_files = [f for f, yr in file_years.items() if y0 <= yr <= y1]
        print(f"  Frame {fi+1}/{len(center_years)}: {cy} ({y0}–{y1}), {len(window_files)} files", flush=True)

        v_zonal_sum = np.zeros((nz, ny), dtype=np.float64)
        count = 0
        for f in window_files:
            # Only load vo to avoid OOM on 9.3 GB files
            ds = xr.open_dataset(f)[["vo"]]
            vo = ds["vo"].values  # (12, nz, ny, nx)
            ds.close()

            for m in range(vo.shape[0]):
                v = vo[m]
                v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)
                v_masked = v * atl_mask[np.newaxis, :, :]
                v_zonal = np.nansum(v_masked, axis=2) * dx_1d[np.newaxis, :]
                v_zonal_sum += v_zonal
                count += 1
            del vo  # free memory

        v_zonal_mean = v_zonal_sum / count
        transport = v_zonal_mean * dz[:, np.newaxis]
        psi = np.cumsum(transport, axis=0) / 1e6
        psi_all[fi] = psi.astype(np.float32)

    return psi_all, lat, depth, center_years


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute windowed MOC streamfunction from reanalysis"
    )
    parser.add_argument(
        "--product", required=True, choices=["oras5", "glorys12"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outfile = args.output_dir / f"streamfunction_{args.product}_{WINDOW_YEARS}yr.npz"

    print(f"Computing {args.product} windowed streamfunction ({WINDOW_YEARS}yr windows, {STEP_YEARS}yr steps)...")

    if args.product == "oras5":
        psi, lat, depth, center_years = compute_oras5(Path("data/oras5"))
    else:
        psi, lat, depth, center_years = compute_glorys12(Path("data/glorys12"))

    np.savez_compressed(outfile, psi=psi, lat=lat, depth=depth, center_years=center_years)
    print(f"  Saved: {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")
    print(f"  Shape: {psi.shape}, max Ψ: {np.nanmax(psi):.1f} Sv")


if __name__ == "__main__":
    main()
