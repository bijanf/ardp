#!/usr/bin/env python3
"""Decompose F_ovS at 34.5°S into velocity and salinity terms.

Six-panel figure (2 rows × 3 columns):
  Top row: Early period (1960-1990)
  Bottom row: Late period (2005-2025)
  Col 1: v(z, lon) — meridional velocity
  Col 2: S(z, lon) - S0 — salinity anomaly
  Col 3: v*(S-S0) — the F_ovS integrand

Shows WHY F_ovS is negative: the northward flow is salty (importing salt)
and the southward flow is fresh (exporting freshwater).
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
    data_dir: Path, j_idx: int, atlantic_mask: np.ndarray,
    year_start: int, year_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and average v and S sections. Returns (v_mean, s_mean) of shape (nz, n_atl)."""
    v_files = sorted(data_dir.glob("vomecrty_*_3D_*.nc"))
    s_files = {
        re.search(r"_3D_(\d{6})_", f.name).group(1): f
        for f in sorted(data_dir.glob("vosaline_*_3D_*.nc"))
    }
    v_sum, s_sum, s_count, count = None, None, None, 0

    for vf in v_files:
        m = re.search(r"_3D_(\d{6})_", vf.name)
        if not m:
            continue
        yyyymm = m.group(1)
        year = int(yyyymm[:4])
        if year < year_start or year > year_end or yyyymm not in s_files:
            continue

        ds_v = xr.open_dataset(vf)
        ds_s = xr.open_dataset(s_files[yyyymm])
        v = ds_v["vomecrty"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]
        s = ds_s["vosaline"].isel(time_counter=0, y=j_idx).values[:, atlantic_mask]
        ds_v.close()
        ds_s.close()

        v = np.where(np.isfinite(v) & (np.abs(v) < 10), v, 0.0)
        s_valid = np.isfinite(s) & (s > 0) & (s < 50)

        if v_sum is None:
            v_sum = np.zeros_like(v, dtype=np.float64)
            s_sum = np.zeros_like(v, dtype=np.float64)
            s_count = np.zeros_like(v, dtype=np.float64)

        v_sum += v
        s_sum += np.where(s_valid, s, 0.0)
        s_count += s_valid.astype(float)
        count += 1

        if count % 60 == 0:
            print(f"    {count} months ({year})", flush=True)

    print(f"    Total: {count} months", flush=True)
    v_mean = v_sum / count
    s_mean = np.where(s_count > 0, s_sum / s_count, np.nan)
    return v_mean, s_mean


def main():
    parser = argparse.ArgumentParser(description="Decompose F_ovS at 34.5S.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oras5"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/grl/fig_fovs_decomposition"))
    parser.add_argument("--depth-max", type=float, default=2000)
    parser.add_argument("--lon-min", type=float, default=-60)
    parser.add_argument("--lon-max", type=float, default=20)
    args = parser.parse_args()

    apply_nature_style()

    # Grid info
    v0 = sorted(args.data_dir.glob("vomecrty_*_3D_*.nc"))[0]
    ds = xr.open_dataset(v0)
    nav_lon = ds["nav_lon"].values
    nav_lat = ds["nav_lat"].values
    depth = ds["depthv"].values
    lat_1d = np.nanmean(nav_lat, axis=1)
    j_idx = int(np.abs(lat_1d - SAMBA_LAT).argmin())
    actual_lat = lat_1d[j_idx]
    lon = nav_lon[j_idx, :]
    atlantic_mask = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    lon_atl = lon[atlantic_mask]
    ds.close()

    print(f"Section at lat={actual_lat:.2f}, {atlantic_mask.sum()} Atlantic points")

    # Load both periods
    print("Loading early (1960-1990)...")
    v_early, s_early = load_mean_section(args.data_dir, j_idx, atlantic_mask, 1960, 1990)
    print("Loading late (2005-2025)...")
    v_late, s_late = load_mean_section(args.data_dir, j_idx, atlantic_mask, 2005, 2025)

    # Depth and longitude masks
    z_mask = depth <= args.depth_max
    depth_p = depth[z_mask]
    x_mask = (lon_atl >= args.lon_min) & (lon_atl <= args.lon_max)
    lon_zoom = lon_atl[x_mask]
    lon_2d, depth_2d = np.meshgrid(lon_zoom, depth_p)

    # Apply lon zoom to all data
    v_early = v_early[:, x_mask]
    s_early = s_early[:, x_mask]
    v_late = v_late[:, x_mask]
    s_late = s_late[:, x_mask]

    # Compute terms (divide by S0 for freshwater flux)
    s_anom_early = (s_early - S0) / S0
    s_anom_late = (s_late - S0) / S0
    product_early = v_early * s_anom_early
    product_late = v_late * s_anom_late

    # Zonal integration: weight by dx at each longitude
    dlon_atl = np.abs(np.diff(lon_zoom, append=lon_zoom[-1]))
    dx = dlon_atl * 111000 * np.cos(np.deg2rad(actual_lat))  # meters

    # Zonally integrated profiles (as function of depth only)
    # V_int(z) = sum(v * dx) [m^2/s] — the overturning velocity
    v_int_early = np.nansum(v_early * dx[np.newaxis, :], axis=1)
    v_int_late = np.nansum(v_late * dx[np.newaxis, :], axis=1)

    # F_ovS integrand integrated zonally: sum(v * (S-S0)/S0 * dx) [m^2/s]
    fovs_int_early = np.nansum(product_early * dx[np.newaxis, :], axis=1)
    fovs_int_late = np.nansum(product_late * dx[np.newaxis, :], axis=1)

    # Convert to Sv per meter depth for nicer units
    v_int_early_sv = v_int_early / 1e6
    v_int_late_sv = v_int_late / 1e6
    fovs_int_early_sv = fovs_int_early / 1e6
    fovs_int_late_sv = fovs_int_late / 1e6

    # Salinity anomaly difference for maps
    s_diff = s_anom_late - s_anom_early

    # ── Figure: 3 rows × 3 cols ──
    # Col 1: zonally-integrated v(z) profiles
    # Col 2: (S-S0)/S0 lon×depth maps
    # Col 3: zonally-integrated F_ovS integrand(z) profiles
    fig = plt.figure(figsize=(6.73, 7.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 1],
                          hspace=0.35, wspace=0.4,
                          left=0.08, right=0.95, top=0.94, bottom=0.06)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("0.85")

    col_blue = "#4477AA"
    col_red = "#CC3311"

    # Helper for profile plots. Panel labels are drawn as editable vector text.
    def plot_profile(ax, depth_arr, early_vals, late_vals, xlabel, _title, label_panel):
        ax.plot(early_vals, depth_arr, color=col_blue, lw=1.5, ls="--", label="1960\u20131990")
        ax.plot(late_vals, depth_arr, color=col_red, lw=1.5, label="2005\u20132025")
        ax.axvline(0, color="0.5", lw=0.5, ls=":")
        ax.set_ylim(args.depth_max, 0)
        ax.set_xlabel(xlabel, fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.04, 0.96, label_panel, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    # Helper for difference profile
    def plot_diff_profile(ax, depth_arr, diff_vals, xlabel, _title, label_panel):
        ax.fill_betweenx(depth_arr, 0, diff_vals,
                         where=(diff_vals > 0), color=col_red, alpha=0.3)
        ax.fill_betweenx(depth_arr, 0, diff_vals,
                         where=(diff_vals < 0), color=col_blue, alpha=0.3)
        ax.plot(diff_vals, depth_arr, color="0.2", lw=1.2)
        ax.axvline(0, color="0.5", lw=0.5, ls=":")
        ax.set_ylim(args.depth_max, 0)
        ax.set_xlabel(xlabel, fontsize=6)
        ax.tick_params(labelsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.04, 0.96, label_panel, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    # Helper for salinity map
    def plot_smap(ax, data, vm, _title, label_panel, show_xlabel=False):
        n_levels = 14
        bounds = np.linspace(-vm, vm, n_levels + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.pcolormesh(lon_2d, depth_2d, data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        ax.set_ylim(args.depth_max, 0)
        ax.tick_params(labelsize=5)
        if show_xlabel:
            ax.set_xlabel("Longitude (\u00b0E)", fontsize=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, extend="both")
        cb.ax.tick_params(labelsize=4)
        ax.text(0.04, 0.96, label_panel, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.75, boxstyle="round,pad=0.1",
                          edgecolor="none"))
        return im

    # ── Row 1: Early period ──
    ax_v1 = fig.add_subplot(gs[0, 0])
    ax_v1.set_ylabel("Depth (m)", fontsize=7)

    ax_s1 = fig.add_subplot(gs[0, 1])
    ax_f1 = fig.add_subplot(gs[0, 2])

    vmax_s = np.nanpercentile(np.abs(s_anom_early[z_mask, :]), 98)

    plot_profile(ax_v1, depth_p, v_int_early_sv[z_mask], v_int_late_sv[z_mask],
                 "Sv", "Zonally-integrated $V(z)$", "a")
    plot_smap(ax_s1, s_anom_early[z_mask, :], vmax_s,
              "$(S - S_0)/S_0$ (1960\u20131990)", "b")
    plot_profile(ax_f1, depth_p, fovs_int_early_sv[z_mask], fovs_int_late_sv[z_mask],
                 "Sv", "F$_{ovS}$ integrand $\\Sigma v(S-S_0)/S_0$", "c")

    # ── Row 2: Late period ──
    ax_v2 = fig.add_subplot(gs[1, 0], sharey=ax_v1)
    ax_v2.set_ylabel("Depth (m)", fontsize=7)

    ax_s2 = fig.add_subplot(gs[1, 1], sharey=ax_s1)
    ax_f2 = fig.add_subplot(gs[1, 2], sharey=ax_f1)

    # For row 2, show the same profiles (already in row 1 as solid lines)
    # Instead show late-only salinity map
    plot_smap(ax_s2, s_anom_late[z_mask, :], vmax_s,
              "$(S - S_0)/S_0$ (2005\u20132025)", "e")

    # Row 2 profiles: show cumulative streamfunction instead
    # Streamfunction psi(z) = cumsum(V_int * dz)
    dz = np.diff(depth, prepend=0)
    psi_early = np.cumsum(v_int_early * dz) / 1e6  # Sv
    psi_late = np.cumsum(v_int_late * dz) / 1e6
    plot_profile(ax_v2, depth_p, psi_early[z_mask], psi_late[z_mask],
                 "Sv", "Streamfunction $\\Psi(z)$", "d")

    # Cumulative F_ovS: how F_ovS builds up with depth
    fovs_cum_early = -np.cumsum(fovs_int_early * dz) / 1e6  # Sv (with -1/S0 already in)
    fovs_cum_late = -np.cumsum(fovs_int_late * dz) / 1e6
    plot_profile(ax_f2, depth_p, fovs_cum_early[z_mask], fovs_cum_late[z_mask],
                 "Sv", "Cumulative F$_{ovS}(z)$", "f")

    # ── Row 3: Difference ──
    ax_v3 = fig.add_subplot(gs[2, 0], sharey=ax_v1)
    ax_v3.set_ylabel("Depth (m)", fontsize=7)

    ax_s3 = fig.add_subplot(gs[2, 1], sharey=ax_s1)
    ax_f3 = fig.add_subplot(gs[2, 2], sharey=ax_f1)

    vmax_sd = np.nanpercentile(np.abs(s_diff[z_mask, :]), 95)

    plot_diff_profile(ax_v3, depth_p,
                      (v_int_late_sv - v_int_early_sv)[z_mask],
                      "Sv", "$\\Delta V(z)$ (late \u2212 early)", "g")
    plot_smap(ax_s3, s_diff[z_mask, :], vmax_sd,
              "$\\Delta (S - S_0)/S_0$", "h", show_xlabel=True)
    plot_diff_profile(ax_f3, depth_p,
                      (fovs_int_late_sv - fovs_int_early_sv)[z_mask],
                      "Sv", "$\\Delta$ F$_{ovS}$ integrand", "i")

    save_publication_figure(fig, args.output)

    # Summary
    print("\nSummary:")
    print(f"  Streamfunction max (early): {psi_early.max():.2f} Sv")
    print(f"  Streamfunction max (late):  {psi_late.max():.2f} Sv")
    print(f"  Cumulative F_ovS (early, full depth): {fovs_cum_early[-1]:.4f} Sv")
    print(f"  Cumulative F_ovS (late, full depth):  {fovs_cum_late[-1]:.4f} Sv")


if __name__ == "__main__":
    main()
