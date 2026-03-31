#!/usr/bin/env python3
"""Plot the Atlantic cross-section at 34.5°S showing velocity and salinity.

Visualises the two ingredients of the F_ovS calculation:
  (a) Meridional velocity — northward upper limb vs southward deep return
  (b) Salinity — saltier surface layer vs fresher deep water

Uses a time-mean over the GLORYS12 record (1993–2024) for a clean picture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm

from ardp.constants import SAMBA_LAT
from ardp.viz.style import (
    add_panel_label,
    apply_nature_style,
    save_publication_figure,
)

ATLANTIC_LON_MIN = -70.0
ATLANTIC_LON_MAX = 20.0


def load_mean_section(
    data_dir: Path,
    target_lat: float = SAMBA_LAT,
) -> dict:
    """Load time-mean velocity and salinity sections at target latitude."""
    files = sorted(data_dir.glob("glorys12_*.nc"))
    if not files:
        raise FileNotFoundError(f"No GLORYS12 files in {data_dir}")

    # Get grid info from first file
    ds0 = xr.open_dataset(files[0])
    lat = ds0["latitude"].values
    lon = ds0["longitude"].values
    depth = ds0["depth"].values
    j_idx = int(np.abs(lat - target_lat).argmin())
    actual_lat = float(lat[j_idx])

    # Atlantic mask
    atl_mask = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    lon_atl = lon[atl_mask]
    ds0.close()

    print(f"Section at j={j_idx}, lat={actual_lat:.2f}°")
    print(f"Atlantic points: {atl_mask.sum()}, lon range: {lon_atl[0]:.1f} to {lon_atl[-1]:.1f}")

    # Accumulate time-mean
    v_sum = np.zeros((len(depth), atl_mask.sum()), dtype=np.float64)
    s_sum = np.zeros_like(v_sum)
    v_count = np.zeros_like(v_sum)
    s_count = np.zeros_like(v_sum)

    for i, f in enumerate(files):
        print(f"  {i+1}/{len(files)}: {f.name}")
        ds = xr.open_dataset(f)
        n_times = ds.sizes["time"]

        for t in range(n_times):
            v_slice = ds["vo"].isel(time=t, latitude=j_idx).values[:, atl_mask]
            s_slice = ds["so"].isel(time=t, latitude=j_idx).values[:, atl_mask]

            v_valid = np.isfinite(v_slice)
            s_valid = np.isfinite(s_slice)

            v_sum[v_valid] += v_slice[v_valid]
            v_count[v_valid] += 1
            s_sum[s_valid] += s_slice[s_valid]
            s_count[s_valid] += 1

        ds.close()

    v_mean = np.where(v_count > 0, v_sum / v_count, np.nan)
    s_mean = np.where(s_count > 0, s_sum / s_count, np.nan)

    return {
        "v": v_mean,
        "s": s_mean,
        "lon": lon_atl,
        "depth": depth,
        "lat": actual_lat,
        "n_months": int(v_count.max()),
    }


def plot_cross_section(section: dict, output_dir: Path) -> None:
    """Plot 2-panel cross-section: velocity + salinity."""
    apply_nature_style()

    fig, (ax_v, ax_s) = plt.subplots(
        2, 1, figsize=(6.73, 5.5), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    lon = section["lon"]
    depth = section["depth"]
    v = section["v"] * 100  # convert m/s to cm/s for readability
    s = section["s"]

    # ── Panel (a): Meridional velocity ──────────────────────────────
    v_levels = np.array([-8, -6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6, 8])
    v_cmap = plt.get_cmap("RdBu_r")
    v_norm = BoundaryNorm(v_levels, v_cmap.N)

    cf_v = ax_v.pcolormesh(
        lon, depth, v,
        cmap=v_cmap, norm=v_norm, shading="nearest",
    )
    # Zero contour
    ax_v.contour(lon, depth, v, levels=[0], colors="k", linewidths=0.8)

    ax_v.set_ylim(5500, 0)
    ax_v.set_ylabel("Depth (m)", fontsize=7)
    ax_v.set_title(
        f"GLORYS12 time-mean meridional velocity at {abs(section['lat']):.1f}°S",
        fontsize=8, pad=6,
    )
    ax_v.tick_params(labelsize=6)

    cbar_v = fig.colorbar(cf_v, ax=ax_v, orientation="vertical",
                          shrink=0.85, pad=0.02, aspect=20)
    cbar_v.set_label("v (cm/s)", fontsize=7)
    cbar_v.ax.tick_params(labelsize=5)
    add_panel_label(ax_v, "a", x=-0.08, y=1.08)

    # Annotate upper/deep limbs with background boxes
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.85)
    ax_v.text(
        -25, 300, "Northward (upper limb)",
        fontsize=6, ha="center", va="center", color="#AA3377",
        fontweight="bold", bbox=bbox_props,
    )
    ax_v.text(
        -25, 2200, "Southward (deep return)",
        fontsize=6, ha="center", va="center", color="#4477AA",
        fontweight="bold", bbox=bbox_props,
    )

    # ── Panel (b): Salinity ─────────────────────────────────────────
    # Mask out near-zero values (land/fill artifacts)
    s_plot = np.where(s > 30, s, np.nan)
    s_levels = np.arange(34.0, 36.01, 0.2)
    s_cmap = plt.get_cmap("YlOrRd")
    s_norm = BoundaryNorm(s_levels, s_cmap.N)

    cf_s = ax_s.pcolormesh(
        lon, depth, s_plot,
        cmap=s_cmap, norm=s_norm, shading="nearest",
    )
    # S0 reference contour
    ax_s.contour(lon, depth, s_plot, levels=[35.0], colors="k",
                 linewidths=0.8, linestyles="--")

    ax_s.set_ylim(5500, 0)
    ax_s.set_xlabel("Longitude (°E)", fontsize=7)
    ax_s.set_ylabel("Depth (m)", fontsize=7)
    ax_s.set_title(
        f"GLORYS12 time-mean salinity at {abs(section['lat']):.1f}°S",
        fontsize=8, pad=6,
    )
    ax_s.tick_params(labelsize=6)

    cbar_s = fig.colorbar(cf_s, ax=ax_s, orientation="vertical",
                          shrink=0.85, pad=0.02, aspect=20)
    cbar_s.set_label("Salinity (PSU)", fontsize=7)
    cbar_s.ax.tick_params(labelsize=5)
    add_panel_label(ax_s, "b", x=-0.08, y=1.08)

    # Annotate salinity structure with background boxes
    ax_s.text(
        -25, 300, "Salty upper water (carried north by AMOC)",
        fontsize=6, ha="center", va="center", color="0.2",
        fontweight="bold", bbox=bbox_props,
    )
    ax_s.text(
        -25, 2200, "Fresher deep water (returns south)",
        fontsize=6, ha="center", va="center", color="0.2",
        fontweight="bold", bbox=bbox_props,
    )

    save_publication_figure(fig, output_dir / "fig_345s_cross_section")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Atlantic cross-section at 34.5°S"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/glorys12"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/grl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    cache_file = args.cache_dir / "glorys12_345s_section_mean.npz"

    if cache_file.exists():
        print(f"Loading cached section: {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        section = {k: cached[k] for k in cached.files}
        # Scalars stored as 0-d arrays
        section["lat"] = float(section["lat"])
        section["n_months"] = int(section["n_months"])
    else:
        print("Computing time-mean section (this takes a while)...")
        section = load_mean_section(args.data_dir)

        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            v=section["v"], s=section["s"],
            lon=section["lon"], depth=section["depth"],
            lat=section["lat"], n_months=section["n_months"],
        )
        print(f"Cached to: {cache_file}")

    print(f"Section: lat={section['lat']:.2f}°, {section['n_months']} months averaged")
    print(f"Velocity range: {np.nanmin(section['v']):.4f} to {np.nanmax(section['v']):.4f} m/s")
    print(f"Salinity range: {np.nanmin(section['s']):.2f} to {np.nanmax(section['s']):.2f} PSU")

    plot_cross_section(section, args.output_dir)


if __name__ == "__main__":
    main()
