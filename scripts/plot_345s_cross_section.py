#!/usr/bin/env python3
"""Visualize freshwater transport structure at 34.5S from ORAS5.

Three-panel figure:
  (a) Early period (1960-1990): v*(S-S0) cross-section — the F_ovS integrand
  (b) Late period (2005-2025): same quantity
  (c) Difference (late - early): where freshwater transport changed

Shows the physical mechanism: northward surface flow carrying fresh water
into the Atlantic (negative F_ovS = freshwater import).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0, SAMBA_LAT
from ardp.viz.style import apply_nature_style, save_publication_figure


def load_mean_section(
    data_dir: Path,
    j_idx: int,
    atlantic_mask: np.ndarray,
    year_start: int,
    year_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and average v and S sections over a year range.

    Returns (v_mean, s_mean) arrays of shape (nz, n_atlantic).
    """
    v_files = sorted(data_dir.glob("vomecrty_*_3D_*.nc"))
    s_files = {
        re.search(r"_3D_(\d{6})_", f.name).group(1): f
        for f in sorted(data_dir.glob("vosaline_*_3D_*.nc"))
    }

    v_sum = None
    s_sum = None
    count = 0

    for vf in v_files:
        m = re.search(r"_3D_(\d{6})_", vf.name)
        if not m:
            continue
        yyyymm = m.group(1)
        year = int(yyyymm[:4])
        if year < year_start or year > year_end:
            continue
        if yyyymm not in s_files:
            continue

        ds_v = xr.open_dataset(vf)
        ds_s = xr.open_dataset(s_files[yyyymm])

        v = ds_v["vomecrty"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]
        s = ds_s["vosaline"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]

        ds_v.close()
        ds_s.close()

        v = np.where(np.isfinite(v) & (np.abs(v) < 10), v, 0.0)
        s = np.where(np.isfinite(s) & (s > 0) & (s < 50), s, np.nan)

        if v_sum is None:
            v_sum = np.zeros_like(v, dtype=np.float64)
            s_sum = np.zeros_like(s, dtype=np.float64)

        v_sum += v
        s_sum += np.nan_to_num(s, nan=0.0)
        count += 1

        if count % 60 == 0:
            print(f"    {count} months processed ({year})", flush=True)

    print(f"    Total: {count} months for {year_start}-{year_end}", flush=True)
    return v_sum / count, s_sum / count


def main():
    parser = argparse.ArgumentParser(
        description="Visualize freshwater transport at 34.5S."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/oras5"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/grl/fig_345s_cross_section"))
    args = parser.parse_args()

    apply_nature_style()

    # Get grid info
    v0 = sorted(args.data_dir.glob("vomecrty_*_3D_*.nc"))[0]
    ds = xr.open_dataset(v0)
    nav_lat = ds["nav_lat"].values
    nav_lon = ds["nav_lon"].values
    depth = ds["depthv"].values
    lat_1d = np.nanmean(nav_lat, axis=1)
    j_idx = int(np.abs(lat_1d - SAMBA_LAT).argmin())
    actual_lat = lat_1d[j_idx]
    lon = nav_lon[j_idx, :]
    atlantic_mask = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    lon_atl = lon[atlantic_mask]
    ds.close()

    nz = len(depth)
    print(f"Section at j={j_idx}, lat={actual_lat:.2f}, {atlantic_mask.sum()} Atlantic points")

    # Load two periods
    print("Loading early period (1960-1990)...")
    v_early, s_early = load_mean_section(args.data_dir, j_idx, atlantic_mask, 1960, 1990)
    print("Loading late period (2005-2025)...")
    v_late, s_late = load_mean_section(args.data_dir, j_idx, atlantic_mask, 2005, 2025)

    # Compute F_ovS integrand: v * (S - S0)
    integrand_early = v_early * (s_early - S0)
    integrand_late = v_late * (s_late - S0)
    integrand_diff = integrand_late - integrand_early

    # Depth limit for plotting (upper 2000m shows the action)
    depth_max = 2000
    z_mask = depth <= depth_max
    depth_plot = depth[z_mask]

    # Create meshgrid for pcolormesh
    lon_2d, depth_2d = np.meshgrid(lon_atl, depth_plot)

    # Symmetric colorbar
    vmax = np.nanpercentile(np.abs(integrand_early[z_mask, :]), 98)
    vmax = np.ceil(vmax * 100) / 100
    vmax_diff = np.nanpercentile(np.abs(integrand_diff[z_mask, :]), 95)
    vmax_diff = np.ceil(vmax_diff * 1000) / 1000

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(6.73, 8.0), sharex=True)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("0.85")

    for ax, data, tag, vm in [
        (axes[0], integrand_early[z_mask, :], "early", vmax),
        (axes[1], integrand_late[z_mask, :], "late", vmax),
        (axes[2], integrand_diff[z_mask, :], "diff", vmax_diff),
    ]:
        n_levels = 16
        bounds = np.linspace(-vm, vm, n_levels + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax.pcolormesh(lon_2d, depth_2d, data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        ax.set_ylim(depth_max, 0)
        ax.set_xlim(-60, 20)  # No ocean data west of ~-60° at 34.5°S
        ax.set_ylabel("Depth (m)", fontsize=8)
        cb = fig.colorbar(im, ax=ax, extend="both", shrink=0.8, pad=0.02)
        cb.set_label("$v \\cdot (S - S_0)$ [m/s \u00b7 PSU]", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Add contour at v=0 (level of no motion) for early/late periods.
        if tag != "diff":
            v_data = v_early[z_mask, :] if tag == "early" else v_late[z_mask, :]
            ax.contour(lon_2d, depth_2d, v_data, levels=[0],
                       colors="black", linewidths=0.8, linestyles="--")

    axes[2].set_xlabel("Longitude (\u00b0E)", fontsize=8)

    # Panel labels (a, b, c). Editable vector text via pdf.fonttype=42.
    for ax, lab in zip(axes, ("a", "b", "c")):
        ax.text(0.015, 0.93, lab, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    save_publication_figure(fig, args.output)

    # Print summary: compute actual F_ovS for each period
    print("\nSummary:")
    dz = np.diff(depth, prepend=0)
    dx = np.abs(np.diff(lon_atl, append=lon_atl[-1])) * 111000 * np.cos(np.deg2rad(actual_lat))

    for label, v, s in [("Early", v_early, s_early), ("Late", v_late, s_late)]:
        fovs = 0.0
        for k in range(nz):
            ocean = np.isfinite(s[k, :]) & (s[k, :] > 0)
            if ocean.sum() == 0:
                continue
            v_k = np.where(ocean, v[k, :], 0.0)
            v_int = (v_k * dx).sum()
            dx_ocean = np.where(ocean, dx, 0.0)
            s_k = np.where(ocean, s[k, :], 0.0)
            s_mean = (s_k * dx_ocean).sum() / dx_ocean.sum()
            fovs += v_int * (s_mean - S0) * dz[k]
        fovs = -(1.0 / S0) * fovs / 1e6
        print(f"  {label}: F_ovS = {fovs:.4f} Sv")


if __name__ == "__main__":
    main()
