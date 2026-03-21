#!/usr/bin/env python3
"""Side-by-side AMOC overturning streamfunction: ORAS5 vs GLORYS12.

Shows the latitude × depth structure of the Atlantic MOC from both
reanalysis products over their common period (1993–2024), with the
SAMBA (34.5°S) and RAPID (26.5°N) array latitudes marked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.viz.style import (
    add_panel_label,
    apply_nature_style,
    save_publication_figure,
)


# ── Atlantic basin longitude bounds (same as compute_moc_streamfunction.py) ──

def atlantic_lon_bounds(lat: float) -> tuple[float, float]:
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


# ── GLORYS12 streamfunction computation ──────────────────────────────────────

def compute_glorys12_streamfunction(
    data_dir: Path,
    start_year: int = 1993,
    end_year: int = 2024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-mean MOC streamfunction from GLORYS12 vo files.

    Returns (psi, lat, depth) where psi is in Sv.
    """
    files = sorted(data_dir.glob("glorys12_*.nc"))
    # Filter by year range
    selected = []
    for f in files:
        year = int(f.stem.split("_")[1])
        if start_year <= year <= end_year:
            selected.append(f)
    print(f"  GLORYS12: {len(selected)} files ({start_year}–{end_year})", flush=True)

    # Grid from first file
    ds0 = xr.open_dataset(selected[0])
    lat = ds0["latitude"].values
    lon = ds0["longitude"].values
    depth = ds0["depth"].values
    ny = len(lat)
    nz = len(depth)
    ds0.close()

    # Zonal grid spacing dx(lat, lon) — regular grid
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    cos_lat_2d = np.cos(np.deg2rad(lat))[:, np.newaxis]  # (ny, 1)
    dx_2d = np.abs(dlon)[np.newaxis, :] * 111000.0 * cos_lat_2d  # (ny, nlon)
    dx_2d = np.clip(dx_2d, 1.0, None)

    # Atlantic mask per latitude
    atl_mask = np.zeros((ny, len(lon)), dtype=bool)
    for j in range(ny):
        if lat[j] < -55 or lat[j] > 70:
            continue
        lon_min, lon_max = atlantic_lon_bounds(lat[j])
        atl_mask[j, :] = (lon >= lon_min) & (lon <= lon_max)

    # Depth spacing
    dz = np.diff(depth, prepend=0.0)

    # Accumulate zonally integrated V transport
    v_zonal_sum = np.zeros((nz, ny), dtype=np.float64)
    count = 0

    for i, f in enumerate(selected):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(selected)}: {f.name}", flush=True)
        # Only open 'vo' to avoid loading all 5 variables into memory
        ds = xr.open_dataset(f)[["vo"]]
        n_times = ds.sizes["time"]

        for t in range(n_times):
            # vo: (depth, latitude, longitude)
            v = ds["vo"].isel(time=t).values
            v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)

            # Zonally integrate: v * dx * atl_mask, sum over longitude
            v_dx = v * dx_2d[np.newaxis, :, :] * atl_mask[np.newaxis, :, :]
            v_zonal = np.nansum(v_dx, axis=2)  # (nz, ny)
            v_zonal_sum += v_zonal
            count += 1

        ds.close()

    print(f"  Averaged {count} monthly fields", flush=True)

    # Time mean
    v_zonal_mean = v_zonal_sum / count

    # Transport per level
    transport = v_zonal_mean * dz[:, np.newaxis]

    # Streamfunction: cumulative from surface
    psi = np.cumsum(transport, axis=0) / 1e6  # Sv

    print(f"  Max upper cell: {np.nanmax(psi):.1f} Sv")
    return psi, lat, depth


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(
    psi_oras5: np.ndarray, lat_oras5: np.ndarray, depth_oras5: np.ndarray,
    psi_glorys: np.ndarray, lat_glorys: np.ndarray, depth_glorys: np.ndarray,
    output_dir: Path,
    period_str: str = "1993–2024",
) -> None:
    """Plot side-by-side streamfunction comparison."""
    apply_nature_style()

    fig, (ax_o, ax_g) = plt.subplots(
        2, 1, figsize=(6.73, 5.5), sharex=True,
        gridspec_kw={"hspace": 0.25},
    )

    # Shared colormap settings
    levels = np.arange(-6, 20, 2)
    cmap = plt.cm.RdBu_r.copy()
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend="both")

    for ax, psi, lat_arr, depth_arr, title, label in [
        (ax_o, psi_oras5, lat_oras5, depth_oras5, "ORAS5", "a"),
        (ax_g, psi_glorys, lat_glorys, depth_glorys, "GLORYS12", "b"),
    ]:
        # Limit to Atlantic latitudes — extend south to show Southern Ocean connection
        lat_mask = (lat_arr >= -55) & (lat_arr <= 70)
        psi_plot = psi[:, lat_mask]
        lat_plot = lat_arr[lat_mask]

        cf = ax.contourf(
            lat_plot, depth_arr, psi_plot,
            levels=levels, cmap=cmap, norm=norm, extend="both",
        )
        ax.contour(
            lat_plot, depth_arr, psi_plot,
            levels=levels, colors="0.4", linewidths=0.3,
        )
        ax.contour(
            lat_plot, depth_arr, psi_plot,
            levels=[0], colors="k", linewidths=0.8,
        )

        ax.set_ylim(5500, 0)
        ax.set_xlim(-55, 70)
        ax.set_ylabel("Depth (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        if ax is ax_g:
            ax.set_xlabel("Latitude (°N)", fontsize=7)

        # Mark RAPID and SAMBA latitudes
        ax.axvline(26.5, color="0.3", linewidth=0.5, linestyle=":", zorder=5)
        ax.axvline(-34.5, color="0.3", linewidth=0.5, linestyle=":", zorder=5)
        ax.text(27.5, 200, "RAPID", fontsize=5, color="0.3", rotation=90,
                va="top")
        ax.text(-33.5, 200, "34.5°S", fontsize=5, color="0.3", rotation=90,
                va="top")

        # Max AMOC strength — in the interior (below 500m, 0–60°N)
        interior = (lat_plot >= 0) & (lat_plot <= 60)
        deep_mask = depth_arr >= 500
        psi_interior = psi_plot[np.ix_(deep_mask, interior)]
        psi_max = np.nanmax(psi_interior)
        ax.set_title(f"{title} ({period_str})\nAMOC max = {psi_max:.1f} Sv",
                      fontsize=8, pad=6)
        add_panel_label(ax, label, x=-0.08, y=1.08)

    # Single shared colorbar
    cbar = fig.colorbar(cf, ax=[ax_o, ax_g], orientation="vertical",
                        shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label("$\\Psi$ (Sv)", fontsize=7)
    cbar.ax.tick_params(labelsize=5)

    save_publication_figure(fig, output_dir / "fig_moc_streamfunction_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side AMOC streamfunction: ORAS5 vs GLORYS12"
    )
    parser.add_argument("--oras5-dir", type=Path, default=Path("data/oras5"))
    parser.add_argument("--glorys12-dir", type=Path, default=Path("data/glorys12"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/grl"))
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    # ── ORAS5 streamfunction ──
    # Try matching period first, fall back to any available cache
    oras5_cache = args.cache_dir / f"moc_streamfunction_oras5_{args.start_year}_{args.end_year}.npz"
    oras5_fallback = args.cache_dir / "moc_streamfunction_2005_2024.npz"
    if oras5_cache.exists():
        print(f"Loading ORAS5 cache: {oras5_cache}")
        cached = np.load(oras5_cache)
        psi_o, lat_o, depth_o = cached["psi"], cached["lat"], cached["depth"]
    elif oras5_fallback.exists():
        print(f"Loading ORAS5 fallback cache: {oras5_fallback}")
        cached = np.load(oras5_fallback)
        psi_o, lat_o, depth_o = cached["psi"], cached["lat"], cached["depth"]
        print("  NOTE: using 2005–2024 period (run compute_moc_streamfunction.py for other periods)")
    else:
        raise FileNotFoundError(
            "No ORAS5 streamfunction cache found. Run:\n"
            f"  python scripts/compute_moc_streamfunction.py --start-year {args.start_year} --end-year {args.end_year}"
        )

    # ── GLORYS12 streamfunction ──
    glorys_cache = args.cache_dir / f"moc_streamfunction_glorys12_{args.start_year}_{args.end_year}.npz"
    if glorys_cache.exists():
        print(f"Loading GLORYS12 cache: {glorys_cache}")
        cached = np.load(glorys_cache)
        psi_g, lat_g, depth_g = cached["psi"], cached["lat"], cached["depth"]
    else:
        print("Computing GLORYS12 streamfunction...")
        psi_g, lat_g, depth_g = compute_glorys12_streamfunction(
            args.glorys12_dir, args.start_year, args.end_year,
        )
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(glorys_cache, psi=psi_g, lat=lat_g, depth=depth_g)
        print(f"Cached: {glorys_cache}")

    print(f"\nORAS5:   max Ψ = {np.nanmax(psi_o):.1f} Sv")
    print(f"GLORYS12: max Ψ = {np.nanmax(psi_g):.1f} Sv")

    period_str = f"{args.start_year}–{args.end_year}"
    plot_comparison(
        psi_o, lat_o, depth_o,
        psi_g, lat_g, depth_g,
        args.output_dir,
        period_str=period_str,
    )


if __name__ == "__main__":
    main()
