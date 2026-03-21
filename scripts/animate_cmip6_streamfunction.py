#!/usr/bin/env python3
"""Animate AMOC streamfunction for 16 CMIP6 models under SSP585.

Creates a 4x4 grid animation of Psi(lat, depth) evolving from ~1865 to 2085,
with models sorted by F_ovS (most bistable top-left to most monostable
bottom-right).

Output: figures/animations/cmip6_streamfunction_16models.gif
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from ardp.constants import RAPID_LAT, SAMBA_LAT
from ardp.viz.style import apply_nature_style

# Models sorted by F_ovS (most bistable first)
MODELS_SORTED = [
    ("NESM3", -0.187),
    ("IPSL-CM6A-LR", -0.171),
    ("CNRM-CM6-1", -0.119),
    ("MIROC6", -0.093),
    ("MPI-ESM1-2-HR", -0.044),
    ("CanESM5", -0.040),
    ("UKESM1-0-LL", +0.051),
    ("CMCC-CM2-SR5", +0.052),
    ("GFDL-CM4", +0.062),
    ("ACCESS-CM2", +0.072),
    ("MPI-ESM1-2-LR", +0.093),
    ("HadGEM3-GC31-LL", +0.095),
    ("CESM2", +0.162),
    ("FIO-ESM-2-0", +0.186),
    ("GISS-E2-1-G", +0.240),
    ("FGOALS-g3", +0.347),
]

LEVELS = np.arange(-6, 20, 2)
LAT_RANGE = (-40, 70)
DEPTH_RANGE = (4000, 0)


def load_streamfunction(results_dir: Path, model: str):
    f = results_dir / f"streamfunction_{model}_ssp585.npz"
    if not f.exists():
        return None
    data = np.load(f)
    return {
        "psi": data["psi"], "lat": data["lat"],
        "depth": data["depth"], "center_years": data["center_years"],
    }


def get_amoc_max(psi, lat, depth):
    lat_mask = (lat >= 0) & (lat <= 60)
    depth_mask = (depth >= 500) & (depth <= 4000)
    psi_interior = psi[np.ix_(depth_mask, lat_mask)]
    return np.nanmax(psi_interior)


def create_animation(
    results_dir: Path, output_dir: Path, interval: int = 400, dpi: int = 200,
) -> None:
    apply_nature_style()

    # Load all available models
    model_data = {}
    available = []
    for model, fovs in MODELS_SORTED:
        d = load_streamfunction(results_dir, model)
        if d is not None:
            model_data[model] = d
            available.append((model, fovs))
            print(f"  Loaded {model} (F_ovS = {fovs:+.3f})")
        else:
            print(f"  SKIP {model} (no data)")

    n_models = len(available)
    if n_models == 0:
        print("No models available!")
        return

    # Determine grid layout
    if n_models <= 4:
        nrows, ncols = 1, n_models
    elif n_models <= 8:
        nrows, ncols = 2, (n_models + 1) // 2
    elif n_models <= 12:
        nrows, ncols = 3, (n_models + 2) // 3
    else:
        nrows, ncols = 4, (n_models + 3) // 4

    # Build unified year grid from earliest to latest across all models
    all_years = set()
    for d in model_data.values():
        all_years.update(d["center_years"].tolist())
    unified_years = np.array(sorted(all_years))

    # For each model, build a mapping: unified frame index → model frame index
    # Models without data for a given year show their first available frame
    for model in model_data:
        d = model_data[model]
        year_to_idx = {int(y): i for i, y in enumerate(d["center_years"])}
        frame_map = []
        for y in unified_years:
            if y in year_to_idx:
                frame_map.append(year_to_idx[y])
            else:
                frame_map.append(None)  # no data for this year
        d["frame_map"] = frame_map

    n_frames = len(unified_years)
    print(f"\nAnimation: {n_models} models, {nrows}x{ncols} grid, {n_frames} frames")
    print(f"  Year range: {unified_years[0]}–{unified_years[-1]}")

    # Use full figure area — minimal margins
    fig = plt.figure(figsize=(ncols * 6, nrows * 4.5 + 1.5))
    gs = fig.add_gridspec(
        nrows, ncols + 2,
        width_ratios=[1] * ncols + [0.03, 0.08],
        hspace=0.0, wspace=0.0,
        left=0.06, right=0.94, bottom=0.07, top=0.93,
    )

    cmap = plt.cm.RdBu_r.copy()
    norm = mcolors.BoundaryNorm(LEVELS, cmap.N, extend="both")

    axes = {}
    labels = {}

    for idx, (model, fovs) in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        axes[model] = ax
        d = model_data[model]

        # Find first available frame for initial plot
        fi = d["frame_map"][0]
        has_data = fi is not None

        lat_mask = (d["lat"] >= LAT_RANGE[0]) & (d["lat"] <= LAT_RANGE[1])
        depth_mask = d["depth"] <= 4000
        lat_plot = d["lat"][lat_mask]
        depth_plot = d["depth"][depth_mask]

        ax.set_ylim(*DEPTH_RANGE)
        ax.set_xlim(*LAT_RANGE)
        ax.set_xticks([-20, 0, 20, 40, 60])
        ax.set_yticks([0, 1000, 2000, 3000, 4000])

        if has_data:
            psi_plot = d["psi"][fi][np.ix_(depth_mask, lat_mask)]
            ax.contourf(lat_plot, depth_plot, psi_plot,
                        levels=LEVELS, cmap=cmap, norm=norm, extend="both")
            ax.contour(lat_plot, depth_plot, psi_plot,
                       levels=LEVELS, colors="0.4", linewidths=0.2)
            ax.contour(lat_plot, depth_plot, psi_plot,
                       levels=[0], colors="k", linewidths=0.5)
        else:
            ax.set_facecolor("0.9")

        # Only show tick labels on edges
        if row < nrows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{x}°" for x in [-20, 0, 20, 40, 60]], fontsize=14)
        if col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0", "1", "2", "3", "4"], fontsize=14)

        ax.tick_params(axis="both", length=3, width=0.5, pad=2)

        # RAPID/SAMBA lines
        ax.axvline(RAPID_LAT, color="0.4", lw=0.3, ls=":")
        ax.axvline(SAMBA_LAT, color="0.4", lw=0.3, ls=":")

        # Label at the bottom center
        regime_color = "#CC3333" if fovs < 0 else "#3366AA"
        if has_data:
            amoc_max = get_amoc_max(d["psi"][fi], d["lat"], d["depth"])
            label_str = f"{model}\nF$_{{ovS}}$={fovs:+.2f} Sv,  $\\Psi_{{max}}$={amoc_max:.0f} Sv"
        else:
            label_str = f"{model}\nno data yet"
        txt = ax.text(
            0.5, 0.03, label_str,
            transform=ax.transAxes, fontsize=20, color="white",
            fontweight="bold", va="bottom", ha="center",
            bbox=dict(facecolor=regime_color, alpha=0.9, edgecolor="white",
                      linewidth=0.5, boxstyle="round,pad=0.3"),
            zorder=11, linespacing=1.4,
        )
        labels[model] = txt

    # Axis labels on edges
    for idx, (model, fovs) in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[model]
        if row == nrows - 1:
            ax.set_xlabel("Latitude", fontsize=16)
        if col == 0:
            ax.set_ylabel("Depth (km)", fontsize=16)

    # Colorbar in the last column
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, orientation="vertical",
    )
    cbar.set_label("$\\Psi$ (Sv)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Year label
    year_text = fig.suptitle(
        str(model_data[available[0][0]]["center_years"][0]),
        fontsize=56, fontweight="bold", y=0.98,
    )

    def update(frame):
        for idx, (model, fovs) in enumerate(available):
            ax = axes[model]
            d = model_data[model]
            fi = d["frame_map"][frame]

            for coll in list(ax.collections):
                coll.remove()

            lat_mask = (d["lat"] >= LAT_RANGE[0]) & (d["lat"] <= LAT_RANGE[1])
            depth_mask = d["depth"] <= 4000
            lat_plot = d["lat"][lat_mask]
            depth_plot = d["depth"][depth_mask]

            if fi is not None:
                psi_plot = d["psi"][fi][np.ix_(depth_mask, lat_mask)]
                ax.contourf(lat_plot, depth_plot, psi_plot,
                            levels=LEVELS, cmap=cmap, norm=norm, extend="both")
                ax.contour(lat_plot, depth_plot, psi_plot,
                           levels=LEVELS, colors="0.4", linewidths=0.2)
                ax.contour(lat_plot, depth_plot, psi_plot,
                           levels=[0], colors="k", linewidths=0.5)
                ax.set_facecolor("white")

                amoc_max = get_amoc_max(d["psi"][fi], d["lat"], d["depth"])
                labels[model].set_text(
                    f"{model}\nF$_{{ovS}}$={fovs:+.2f} Sv,  $\\Psi_{{max}}$={amoc_max:.0f} Sv"
                )
            else:
                ax.set_facecolor("0.9")
                labels[model].set_text(f"{model}\nno data yet")

            ax.axvline(RAPID_LAT, color="0.4", lw=0.3, ls=":")
            ax.axvline(SAMBA_LAT, color="0.4", lw=0.3, ls=":")

        year_text.set_text(str(unified_years[frame]))
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "cmip6_streamfunction_16models.gif"
    print(f"Saving GIF: {gif_path}")
    anim.save(str(gif_path), writer=PillowWriter(fps=1000 // interval), dpi=dpi)
    print(f"  GIF saved: {gif_path.stat().st_size / 1e6:.1f} MB")

    # Find ffmpeg — check PATH first, then imageio-ffmpeg
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ffmpeg_exe:
        from matplotlib.animation import FFMpegWriter
        mp4_path = output_dir / "cmip6_streamfunction_16models.mp4"
        print(f"Saving MP4: {mp4_path}")
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
        anim.save(str(mp4_path), writer=FFMpegWriter(fps=1000 // interval), dpi=dpi)
        print(f"  MP4 saved: {mp4_path.stat().st_size / 1e6:.1f} MB")

    plt.close(fig)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate AMOC streamfunction for CMIP6 ensemble."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/cmip6"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/animations"))
    parser.add_argument("--interval", type=int, default=400, help="ms per frame")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    create_animation(args.results_dir, args.output_dir, args.interval, args.dpi)


if __name__ == "__main__":
    main()
