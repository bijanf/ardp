#!/usr/bin/env python3
"""Compute windowed Gulf Stream SSH gradient maps from ORAS5.

Produces |grad(SSH)| fields, jet axis, and destabilization point for each
10-year window, matching the streamfunction animation frame grid.

Output: data/results/gulfstream_ssh_grad_10yr.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

WINDOW_YEARS = 10
STEP_YEARS = 3
NORTH_ATLANTIC_REGION = (-85.0, -10.0, 25.0, 55.0)  # wider view for gradient map
JET_SEARCH_REGION = (-80.0, -45.0, 25.0, 50.0)     # corridor for jet axis finding
DESTAB_THRESHOLD = 0.5  # fraction of peak gradient


def approximate_grid_metrics(nav_lon, nav_lat):
    """Compute approximate e1 (zonal) and e2 (meridional) grid metrics in meters."""
    # Zonal: dlon * 111km * cos(lat)
    dlon = np.diff(nav_lon, axis=1)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    cos_lat = np.cos(np.deg2rad(nav_lat[:, :-1]))
    e1 = np.abs(dlon) * 111000.0 * cos_lat
    e1 = np.clip(e1, 1.0, None)

    # Meridional: dlat * 111km
    dlat = np.diff(nav_lat, axis=0)
    e2 = np.abs(dlat) * 111000.0
    e2 = np.clip(e2, 1.0, None)

    return e1, e2


def compute_ssh_gradient_np(ssh, e1, e2):
    """Compute |grad(SSH)| from numpy arrays.

    Returns gradient on interior grid (trimmed by 1 on each edge).
    """
    dssh_dx = np.diff(ssh, axis=1) / e1  # (ny, nx-1)
    dssh_dy = np.diff(ssh, axis=0) / e2  # (ny-1, nx)

    # Align to common interior grid
    dssh_dx = dssh_dx[1:, :]   # (ny-1, nx-1)
    dssh_dy = dssh_dy[:, 1:]   # (ny-1, nx-1)

    return np.sqrt(dssh_dx**2 + dssh_dy**2)


def find_jet_axis_np(grad_mag, lon_sub, lat_sub, search_region=None):
    """Find jet axis (lat of max gradient at each longitude).

    Only searches within search_region (lon_min, lon_max, lat_min, lat_max)
    to avoid picking up non-Gulf-Stream gradients.

    Returns jet_lat, jet_lon, jet_grad arrays of shape (nx,).
    """
    nx = grad_mag.shape[1]
    jet_lat = np.full(nx, np.nan)
    jet_grad = np.full(nx, np.nan)
    jet_lon = lon_sub[0, :]

    for i in range(nx):
        lon_i = jet_lon[i]
        # Skip columns outside search region
        if search_region is not None:
            if lon_i < search_region[0] or lon_i > search_region[1]:
                continue

        col = grad_mag[:, i].copy()
        # Mask latitudes outside search region
        if search_region is not None:
            lat_col = lat_sub[:, i]
            outside = (lat_col < search_region[2]) | (lat_col > search_region[3])
            col[outside] = np.nan

        valid = np.isfinite(col) & (col > 0)
        if np.any(valid):
            j = np.nanargmax(col)
            jet_lat[i] = lat_sub[j, i]
            jet_grad[i] = col[j]

    return jet_lat, jet_lon, jet_grad


def find_destabilization_point(jet_grad, jet_lon, threshold_frac=DESTAB_THRESHOLD):
    """Find the destabilization point longitude.

    Easternmost longitude where gradient exceeds threshold_frac * peak.
    """
    valid = np.isfinite(jet_grad) & (jet_grad > 0)
    if not np.any(valid):
        return np.nan, np.nan

    peak = np.nanmax(jet_grad)
    threshold = peak * threshold_frac
    above = (jet_grad >= threshold) & valid

    if not np.any(above):
        return np.nan, np.nan

    # Last index where gradient is above threshold
    indices = np.where(above)[0]
    last_idx = indices[-1]
    return float(jet_lon[last_idx]), float(np.nan)  # lat from jet_lat separately


def main():
    parser = argparse.ArgumentParser(
        description="Compute windowed Gulf Stream SSH gradient from ORAS5"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/oras5"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find SSH files
    files = sorted(args.data_dir.glob("sossheig_*.nc"))
    if not files:
        raise FileNotFoundError(f"No sossheig files in {args.data_dir}")

    # Parse years
    file_years = {}
    for f in files:
        for p in f.stem.split("_"):
            if len(p) == 6 and p.isdigit():
                file_years[f] = int(p[:4])
                break

    year_min = min(file_years.values())
    year_max = max(file_years.values())
    print(f"ORAS5 SSH: {year_min}–{year_max}, {len(files)} monthly files")

    # Read grid from first file
    ds0 = xr.open_dataset(files[0])
    nav_lon = ds0["nav_lon"].values
    nav_lat = ds0["nav_lat"].values
    ds0.close()

    # Approximate grid metrics
    e1, e2 = approximate_grid_metrics(nav_lon, nav_lat)

    # Find North Atlantic subregion indices
    lon_min, lon_max, lat_min, lat_max = NORTH_ATLANTIC_REGION
    lat_1d = np.nanmean(nav_lat, axis=1)
    lon_1d = np.nanmean(nav_lon, axis=0)

    j_mask = np.where((lat_1d >= lat_min - 1) & (lat_1d <= lat_max + 1))[0]
    i_mask = np.where((lon_1d >= lon_min - 1) & (lon_1d <= lon_max + 1))[0]
    j0, j1 = j_mask[0], j_mask[-1] + 1
    i0, i1 = i_mask[0], i_mask[-1] + 1
    print(f"North Atlantic subregion: j=[{j0}:{j1}], i=[{i0}:{i1}]")

    lon_sub_full = nav_lon[j0:j1, i0:i1]
    lat_sub_full = nav_lat[j0:j1, i0:i1]
    e1_sub = e1[j0:j1, i0:i1-1]  # e1 is (ny, nx-1)
    e2_sub = e2[j0:j1-1, i0:i1]  # e2 is (ny-1, nx)

    # Interior grid after gradient computation (trimmed by 1 on each edge)
    lon_interior = lon_sub_full[1:, 1:]
    lat_interior = lat_sub_full[1:, 1:]

    # Generate center years
    half_w = WINDOW_YEARS // 2
    center_years = np.arange(
        year_min + half_w, year_max - 1, STEP_YEARS
    )
    print(f"Windows: {len(center_years)} frames, {center_years[0]}–{center_years[-1]}")

    ny_int, nx_int = lon_interior.shape
    n_frames = len(center_years)

    grad_all = np.zeros((n_frames, ny_int, nx_int), dtype=np.float32)
    jet_lat_all = np.zeros((n_frames, nx_int), dtype=np.float32)
    jet_grad_all = np.zeros((n_frames, nx_int), dtype=np.float32)
    destab_lon_all = np.full(n_frames, np.nan, dtype=np.float32)
    destab_lat_all = np.full(n_frames, np.nan, dtype=np.float32)

    for fi, cy in enumerate(center_years):
        y0, y1 = cy - half_w, cy + half_w - 1
        window_files = [f for f, yr in file_years.items() if y0 <= yr <= y1]
        print(f"  Frame {fi+1}/{n_frames}: {cy} ({y0}–{y1}), {len(window_files)} months", flush=True)

        # Compute mean SSH in subregion
        ssh_sum = np.zeros((j1 - j0, i1 - i0), dtype=np.float64)
        count = 0
        for f in window_files:
            ds = xr.open_dataset(f)
            ssh = ds["sossheig"].values[0, j0:j1, i0:i1]
            ds.close()
            valid = np.isfinite(ssh)
            ssh = np.where(valid, ssh, 0.0)
            ssh_sum += ssh
            count += 1

        ssh_mean = ssh_sum / count

        # Compute gradient
        grad = compute_ssh_gradient_np(ssh_mean, e1_sub, e2_sub)
        # Mask land (where SSH was 0)
        land_interior = (ssh_mean[1:, 1:] == 0)
        grad = np.where(land_interior, np.nan, grad)

        grad_all[fi] = grad.astype(np.float32)

        # Find jet axis (search only in Gulf Stream corridor)
        jet_lat, jet_lon, jet_grad = find_jet_axis_np(
            grad, lon_interior, lat_interior, search_region=JET_SEARCH_REGION
        )
        jet_lat_all[fi] = jet_lat
        jet_grad_all[fi] = jet_grad

        # Find destabilization point
        d_lon, _ = find_destabilization_point(jet_grad, jet_lon)
        destab_lon_all[fi] = d_lon
        # Get destab lat from jet_lat at the destab longitude
        if np.isfinite(d_lon):
            idx = np.argmin(np.abs(jet_lon - d_lon))
            destab_lat_all[fi] = jet_lat[idx]

    # Print gradient statistics for colorbar calibration
    valid_grad = grad_all[np.isfinite(grad_all) & (grad_all > 0)]
    print(f"\nGradient stats: min={valid_grad.min():.2e}, median={np.median(valid_grad):.2e}, "
          f"95th={np.percentile(valid_grad, 95):.2e}, max={valid_grad.max():.2e}")
    print(f"Destab lon range: {np.nanmin(destab_lon_all):.1f} to {np.nanmax(destab_lon_all):.1f}")

    outfile = args.output_dir / "gulfstream_ssh_grad_10yr.npz"
    np.savez_compressed(
        outfile,
        grad_ssh=grad_all,
        lon_sub=lon_interior,
        lat_sub=lat_interior,
        jet_lat=jet_lat_all,
        jet_lon=jet_lon,  # same for all frames
        jet_grad=jet_grad_all,
        destab_lon=destab_lon_all,
        destab_lat=destab_lat_all,
        center_years=center_years,
    )
    print(f"Saved: {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
