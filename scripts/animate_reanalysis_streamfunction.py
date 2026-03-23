#!/usr/bin/env python3
"""Animate AMOC streamfunction from ORAS5 reanalysis.

Single-panel animation showing Ψ(lat, depth) evolving from 1963–2020.

Output: figures/animations/reanalysis_streamfunction.gif / .mp4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ardp.viz.style import apply_nature_style

PSI_LEVELS = np.arange(-6, 20, 2)
LAT_RANGE = (-40, 70)
DEPTH_RANGE = (4000, 0)


def get_amoc_max(psi, lat, depth):
    lat_mask = (lat >= 0) & (lat <= 60)
    depth_mask = (depth >= 500) & (depth <= 4000)
    return np.nanmax(psi[np.ix_(depth_mask, lat_mask)])


def create_animation(
    results_dir: Path, output_dir: Path, interval: int = 500, dpi: int = 200,
) -> None:
    apply_nature_style()

    sf_file = results_dir / "streamfunction_oras5_10yr.npz"
    if not sf_file.exists():
        print(f"ERROR: {sf_file} not found")
        return
    sf = np.load(sf_file)
    psi = sf["psi"]
    lat = sf["lat"]
    depth = sf["depth"]
    center_years = sf["center_years"]
    n_frames = len(center_years)
    print(f"  ORAS5: {center_years[0]}–{center_years[-1]}, {n_frames} frames")

    lat_mask = (lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1])
    depth_mask = depth <= 4000
    lat_plot = lat[lat_mask]
    depth_plot = depth[depth_mask]

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 0.03, 0.06],
        left=0.10, right=0.90, bottom=0.12, top=0.85,
    )
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 2])

    cmap = plt.cm.RdBu_r.copy()
    norm = mcolors.BoundaryNorm(PSI_LEVELS, cmap.N, extend="both")

    ax.set_ylim(*DEPTH_RANGE)
    ax.set_xlim(*LAT_RANGE)
    ax.set_xticks([-20, 0, 20, 40, 60])
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.set_xticklabels([f"{x}°" for x in [-20, 0, 20, 40, 60]], fontsize=14)
    ax.set_yticklabels(["0", "1", "2", "3", "4"], fontsize=14)
    ax.set_ylabel("Depth (km)", fontsize=16)
    ax.set_xlabel("Latitude", fontsize=16)
    ax.tick_params(axis="both", length=3, width=0.5, pad=2)

    # Initial frame
    psi_plot = psi[0][np.ix_(depth_mask, lat_mask)]
    ax.contourf(lat_plot, depth_plot, psi_plot,
                levels=PSI_LEVELS, cmap=cmap, norm=norm, extend="both")
    ax.contour(lat_plot, depth_plot, psi_plot,
               levels=PSI_LEVELS, colors="0.4", linewidths=0.2)
    ax.contour(lat_plot, depth_plot, psi_plot,
               levels=[0], colors="k", linewidths=0.5)

    amoc_max = get_amoc_max(psi[0], lat, depth)
    label = ax.text(
        0.5, 0.03, f"ORAS5\n$\\Psi_{{max}}$={amoc_max:.0f} Sv",
        transform=ax.transAxes, fontsize=20, color="white",
        fontweight="bold", va="bottom", ha="center",
        bbox=dict(facecolor="#2266AA", alpha=0.9, edgecolor="white",
                  linewidth=0.5, boxstyle="round,pad=0.3"),
        zorder=11, linespacing=1.4,
    )

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cbar_ax, orientation="vertical")
    cbar.set_label("$\\Psi$ (Sv)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    year_text = fig.suptitle(
        str(int(center_years[0])),
        fontsize=56, fontweight="bold", y=0.95,
    )

    def update(frame):
        for coll in list(ax.collections):
            coll.remove()

        psi_plot = psi[frame][np.ix_(depth_mask, lat_mask)]
        ax.contourf(lat_plot, depth_plot, psi_plot,
                    levels=PSI_LEVELS, cmap=cmap, norm=norm, extend="both")
        ax.contour(lat_plot, depth_plot, psi_plot,
                   levels=PSI_LEVELS, colors="0.4", linewidths=0.2)
        ax.contour(lat_plot, depth_plot, psi_plot,
                   levels=[0], colors="k", linewidths=0.5)

        amoc_max = get_amoc_max(psi[frame], lat, depth)
        label.set_text(f"ORAS5\n$\\Psi_{{max}}$={amoc_max:.0f} Sv")
        year_text.set_text(str(int(center_years[frame])))
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "reanalysis_streamfunction.gif"
    print(f"Saving GIF: {gif_path}")
    anim.save(str(gif_path), writer=PillowWriter(fps=1000 // interval), dpi=dpi)
    print(f"  GIF saved: {gif_path.stat().st_size / 1e6:.1f} MB")

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ffmpeg_exe:
        from matplotlib.animation import FFMpegWriter
        mp4_path = output_dir / "reanalysis_streamfunction.mp4"
        print(f"Saving MP4: {mp4_path}")
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
        anim.save(str(mp4_path), writer=FFMpegWriter(fps=1000 // interval), dpi=dpi)
        print(f"  MP4 saved: {mp4_path.stat().st_size / 1e6:.1f} MB")

    plt.close(fig)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate AMOC streamfunction from ORAS5 reanalysis."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/animations"))
    parser.add_argument("--interval", type=int, default=500, help="ms per frame")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    create_animation(args.results_dir, args.output_dir, args.interval, args.dpi)


if __name__ == "__main__":
    main()
