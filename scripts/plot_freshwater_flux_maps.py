#!/usr/bin/env python3
"""Map the Atlantic freshwater transport by upper and lower AMOC branches.

Four-panel figure showing depth-integrated meridional freshwater flux:
  (a) Upper branch (0-1000m) — early period
  (b) Upper branch (0-1000m) — late period
  (c) Lower branch (1000-4000m) — early period
  (d) Lower branch (1000-4000m) — late period

Freshwater flux = v * (S0 - S) / S0, integrated over depth [m²/s per unit width].
Positive = freshwater going north. Negative = salt going north.

When F_ovS < 0, the net column integral is negative at 34.5°S,
meaning the overturning imports salt into the Atlantic.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.constants import S0
from ardp.viz.style import apply_nature_style, save_publication_figure


def compute_fw_flux_maps(
    data_dir: Path,
    year_start: int,
    year_end: int,
    depth: np.ndarray,
    nav_lon: np.ndarray,
    nav_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute depth-integrated freshwater flux maps for upper and lower layers.

    Returns (upper_fw, lower_fw) of shape (ny, nx) in m²/s.
    """
    v_files = sorted(data_dir.glob("vomecrty_*_3D_*.nc"))
    s_files = {
        re.search(r"_3D_(\d{6})_", f.name).group(1): f
        for f in sorted(data_dir.glob("vosaline_*_3D_*.nc"))
    }

    dz = np.diff(depth, prepend=0.0)
    upper_mask = depth <= 1000
    lower_mask = (depth > 1000) & (depth <= 4000)

    ny, nx = nav_lon.shape
    upper_sum = np.zeros((ny, nx), dtype=np.float64)
    lower_sum = np.zeros((ny, nx), dtype=np.float64)
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

        v = ds_v["vomecrty"].values[0]  # (nz, ny, nx)
        s = ds_s["vosaline"].values[0]

        ds_v.close()
        ds_s.close()

        v = np.where(np.isfinite(v) & (np.abs(v) < 10), v, 0.0)
        s = np.where(np.isfinite(s) & (s > 0) & (s < 50), s, S0)

        # Freshwater flux: v * (S0 - S) / S0 * dz at each depth level
        # Positive = freshwater northward, Negative = salt northward
        fw = v * (S0 - s) / S0

        # Integrate over depth layers
        for k in range(len(depth)):
            if upper_mask[k]:
                upper_sum += fw[k, :, :] * dz[k]
            if lower_mask[k]:
                lower_sum += fw[k, :, :] * dz[k]

        count += 1
        if count % 12 == 0:
            print(f"    {count} months ({year})", flush=True)

    print(f"    Total: {count} months for {year_start}-{year_end}", flush=True)
    return upper_sum / count, lower_sum / count


def plot_panel(ax, data, lon, lat, title, vmax, cmap, proj):
    """Plot one map panel."""
    n_levels = 14
    bounds = np.linspace(-vmax, vmax, n_levels + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(lon, lat, data, transform=proj,
                       cmap=cmap, norm=norm, shading="auto", zorder=1)
    ax.add_feature(cfeature.LAND, facecolor="#a09e99", edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.3, color="0.4", zorder=3)
    ax.set_extent([-80, 25, -55, 65], crs=proj)

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.7", alpha=0.5,
                      linestyle=":", zorder=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 5, "color": "0.4"}
    gl.ylabel_style = {"size": 5, "color": "0.4"}

    ax.set_title(title, fontsize=8, pad=4)

    # Mark 34.5°S
    ax.plot([-80, 25], [-34.5, -34.5], color="black", linewidth=0.8,
            linestyle="--", transform=proj, zorder=4)

    return im


def main():
    parser = argparse.ArgumentParser(
        description="Map Atlantic freshwater transport by AMOC branches."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/oras5"))
    parser.add_argument("--early-start", type=int, default=1960)
    parser.add_argument("--early-end", type=int, default=1990)
    parser.add_argument("--late-start", type=int, default=2005)
    parser.add_argument("--late-end", type=int, default=2025)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/grl/fig_freshwater_flux_maps"))
    args = parser.parse_args()

    apply_nature_style()

    # Get grid info
    v0 = sorted(args.data_dir.glob("vomecrty_*_3D_*.nc"))[0]
    ds = xr.open_dataset(v0)
    nav_lon = ds["nav_lon"].values
    nav_lat = ds["nav_lat"].values
    depth = ds["depthv"].values
    ds.close()
    print(f"Grid: {nav_lat.shape}, {len(depth)} depth levels")

    # Compute for both periods
    print(f"\nEarly period ({args.early_start}-{args.early_end}):")
    upper_early, lower_early = compute_fw_flux_maps(
        args.data_dir, args.early_start, args.early_end, depth, nav_lon, nav_lat)

    print(f"\nLate period ({args.late_start}-{args.late_end}):")
    upper_late, lower_late = compute_fw_flux_maps(
        args.data_dir, args.late_start, args.late_end, depth, nav_lon, nav_lat)

    # Mask non-Atlantic regions for cleaner display
    # (pcolormesh will show global, but set_extent crops to Atlantic)

    proj = ccrs.PlateCarree()
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad("0.85")

    # Determine symmetric color limits
    vmax_upper = np.nanpercentile(np.abs(upper_early), 97)
    vmax_lower = np.nanpercentile(np.abs(lower_early), 97)

    fig = plt.figure(figsize=(6.73, 9.0))
    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.05,
                          left=0.02, right=0.92, top=0.94, bottom=0.03)

    # Row 1: Upper branch
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    im1 = plot_panel(ax1, upper_early, nav_lon, nav_lat,
                     f"Upper (0\u20131000 m), {args.early_start}\u2013{args.early_end}",
                     vmax_upper, cmap, proj)

    ax2 = fig.add_subplot(gs[0, 1], projection=proj)
    plot_panel(ax2, upper_late, nav_lon, nav_lat,
              f"Upper (0\u20131000 m), {args.late_start}\u2013{args.late_end}",
              vmax_upper, cmap, proj)

    # Row 2: Lower branch
    ax3 = fig.add_subplot(gs[1, 0], projection=proj)
    im3 = plot_panel(ax3, lower_early, nav_lon, nav_lat,
                     f"Lower (1000\u20134000 m), {args.early_start}\u2013{args.early_end}",
                     vmax_lower, cmap, proj)

    ax4 = fig.add_subplot(gs[1, 1], projection=proj)
    plot_panel(ax4, lower_late, nav_lon, nav_lat,
              f"Lower (1000\u20134000 m), {args.late_start}\u2013{args.late_end}",
              vmax_lower, cmap, proj)

    # Row 3: Difference (total column)
    total_early = upper_early + lower_early
    total_late = upper_late + lower_late
    total_diff = total_late - total_early
    vmax_diff = np.nanpercentile(np.abs(total_diff), 95)

    ax5 = fig.add_subplot(gs[2, 0], projection=proj)
    im5 = plot_panel(ax5, total_diff, nav_lon, nav_lat,
                     "Total change (late \u2212 early)",
                     vmax_diff, plt.cm.RdBu.copy(), proj)

    # Row 3 right: zonal mean profile
    ax6 = fig.add_subplot(gs[2, 1])
    lat_1d = np.nanmean(nav_lat, axis=1)
    # Zonal sum of freshwater flux (to get total transport per latitude)
    dlon = np.abs(np.diff(nav_lon, axis=1, append=nav_lon[:, -1:]))
    cos_lat = np.cos(np.deg2rad(nav_lat))
    dx = dlon * 111000 * cos_lat

    upper_zonal_early = np.nansum(upper_early * dx, axis=1) / 1e6  # Sv
    upper_zonal_late = np.nansum(upper_late * dx, axis=1) / 1e6
    lower_zonal_early = np.nansum(lower_early * dx, axis=1) / 1e6
    lower_zonal_late = np.nansum(lower_late * dx, axis=1) / 1e6

    atl_lat = (lat_1d >= -55) & (lat_1d <= 70)
    ax6.plot(upper_zonal_early[atl_lat], lat_1d[atl_lat],
             color="#d62728", lw=1.5, ls="--", label=f"Upper {args.early_start}\u2013{args.early_end}")
    ax6.plot(upper_zonal_late[atl_lat], lat_1d[atl_lat],
             color="#d62728", lw=1.5, label=f"Upper {args.late_start}\u2013{args.late_end}")
    ax6.plot(lower_zonal_early[atl_lat], lat_1d[atl_lat],
             color="#1f77b4", lw=1.5, ls="--", label=f"Lower {args.early_start}\u2013{args.early_end}")
    ax6.plot(lower_zonal_late[atl_lat], lat_1d[atl_lat],
             color="#1f77b4", lw=1.5, label=f"Lower {args.late_start}\u2013{args.late_end}")

    ax6.axvline(0, color="0.5", lw=0.5, ls=":")
    ax6.axhline(-34.5, color="black", lw=0.8, ls="--")
    ax6.set_xlabel("Freshwater transport (Sv)", fontsize=7)
    ax6.set_ylabel("Latitude (\u00b0N)", fontsize=7)
    ax6.set_title("Zonally integrated", fontsize=8)
    ax6.legend(fontsize=4.5, loc="upper left")
    ax6.set_ylim(-55, 70)
    ax6.tick_params(labelsize=6)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # Colorbars
    cax1 = fig.add_axes([0.93, 0.67, 0.015, 0.25])
    cb1 = fig.colorbar(im1, cax=cax1, extend="both")
    cb1.set_label("FW flux [m\u00b2/s]", fontsize=6)
    cb1.ax.tick_params(labelsize=5)

    cax3 = fig.add_axes([0.93, 0.37, 0.015, 0.25])
    cb3 = fig.colorbar(im3, cax=cax3, extend="both")
    cb3.set_label("FW flux [m\u00b2/s]", fontsize=6)
    cb3.ax.tick_params(labelsize=5)

    cax5 = fig.add_axes([0.45, 0.01, 0.015, 0.25])
    cb5 = fig.colorbar(im5, cax=cax5, extend="both")
    cb5.set_label("\u0394 FW flux [m\u00b2/s]", fontsize=6)
    cb5.ax.tick_params(labelsize=5)

    # Panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5, ax6], "abcdef"):
        ax.text(0.02, 0.95, f"({label})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", color="black",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

    fig.suptitle(
        "Meridional freshwater transport by AMOC branches (ORAS5)\n"
        "Blue = freshwater northward | Red = salt northward (freshwater southward)",
        fontsize=9, fontweight="bold", y=0.98,
    )

    save_publication_figure(fig, args.output)

    # Summary statistics at 34.5°S
    j34 = int(np.abs(lat_1d - (-34.5)).argmin())
    print(f"\nAt 34.5°S (j={j34}):")
    print(f"  Upper FW transport (early): {upper_zonal_early[j34]:.4f} Sv")
    print(f"  Upper FW transport (late):  {upper_zonal_late[j34]:.4f} Sv")
    print(f"  Lower FW transport (early): {lower_zonal_early[j34]:.4f} Sv")
    print(f"  Lower FW transport (late):  {lower_zonal_late[j34]:.4f} Sv")
    print(f"  Total (early): {upper_zonal_early[j34]+lower_zonal_early[j34]:.4f} Sv")
    print(f"  Total (late):  {upper_zonal_late[j34]+lower_zonal_late[j34]:.4f} Sv")


if __name__ == "__main__":
    main()
