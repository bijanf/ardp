#!/usr/bin/env python3
"""Paper 2 Figure 5: zonal structure of Δv and ΔS at 34.5°S.

Single combined PDF with four panels (2×2):

  (a) ORAS5      Δv  (late − early)
  (b) GLORYS12V1 Δv
  (c) ORAS5      ΔS
  (d) GLORYS12V1 ΔS

Tweaks vs the previous atomic figS3 script:
  - Colour-bar saturated at the 95th percentile of the absolute
    anomaly (was 99th) so the basin-scale pattern is visible.
  - Land and sub-surface topography rendered grey via cmap.set_bad
    ("0.7") + ax.set_facecolor("0.7").
  - Panel labels baked into matplotlib (no LaTeX subfigure
    environment in Main.tex anymore).

Reads the raw sections via the same helpers used by
``compute_fovs_decomposition.py``.

Outputs: figures/paper2/Figure5.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ardp.viz.style import apply_nature_style, save_publication_figure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_fovs_decomposition import (  # noqa: E402
    EARLY, LATE,
    _glorys12_period_mean, _oras5_period_mean,
)


def _load_pair(product: str):
    if product == "oras5":
        v1, s1, grid = _oras5_period_mean(Path("data/oras5"), EARLY)
        v2, s2, _ = _oras5_period_mean(Path("data/oras5"), LATE)
    elif product == "glorys12":
        v1, s1, grid = _glorys12_period_mean(Path("data/glorys12"), EARLY)
        v2, s2, _ = _glorys12_period_mean(Path("data/glorys12"), LATE)
    else:
        return None
    # Use salinity NaN as the topography mask. ORAS5's velocity field
    # has NaN→0 replacement applied during compute, so without this
    # masking step the land cells in panel (a) would render white
    # rather than the grey topography colour shown in panels (b)-(d).
    ocean = np.isfinite(s1) & np.isfinite(s2)
    dv_raw = v2 - v1
    dv = np.where(ocean, dv_raw, np.nan)
    with np.errstate(invalid="ignore"):
        ds = np.where(ocean, s2 - s1, np.nan)
    return grid, dv, ds


def _draw(ax, grid, field, cmap_name: str, vmax: float):
    cmap_obj = plt.colormaps[cmap_name].copy()
    cmap_obj.set_bad("0.7")  # NaN cells → light grey (topography)
    ax.set_facecolor("0.7")
    n_x = field.shape[1]
    lon = grid["lon_atl"][:n_x]
    depth = grid["depth"][: field.shape[0]]
    masked = np.ma.masked_invalid(field)
    im = ax.pcolormesh(lon, depth, masked, cmap=cmap_obj,
                       vmin=-vmax, vmax=vmax, shading="auto")
    ax.invert_yaxis()
    return im


def _panel_label(ax, label: str, x: float = 0.0, y: float = 1.04) -> None:
    """Draw the panel label ABOVE the axes (not over the data) so it
    cannot overlap with the section maps. ``y > 1`` puts the text
    outside the axes box; ``ha='left'`` aligns the label flush with
    the y-axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure5"))
    args = parser.parse_args()

    apply_nature_style()

    oras5 = _load_pair("oras5")
    glorys12 = _load_pair("glorys12")
    if oras5 is None or glorys12 is None:
        print("Need both ORAS5 and GLORYS12 sections.")
        return

    # Tighter colour scales — 95th percentile (user feedback: more
    # patterns should be visible; the previous 99th-percentile scaling
    # was dominated by boundary-current spikes).
    dv_vmax = max(
        float(np.nanpercentile(np.abs(oras5[1]), 95)),
        float(np.nanpercentile(np.abs(glorys12[1]), 95)),
    )
    ds_vmax = max(
        float(np.nanpercentile(np.abs(oras5[2]), 95)),
        float(np.nanpercentile(np.abs(glorys12[2]), 95)),
    )
    print(f"Δv saturation: ±{dv_vmax:.4f} m/s")
    print(f"ΔS saturation: ±{ds_vmax:.4f} PSU")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8),
                             gridspec_kw={"wspace": 0.18, "hspace": 0.55,
                                          "top": 0.94, "bottom": 0.10,
                                          "left": 0.10, "right": 0.88})

    im_v_a = _draw(axes[0, 0], oras5[0], oras5[1], "RdBu_r", dv_vmax)
    _draw(axes[0, 1], glorys12[0], glorys12[1], "RdBu_r", dv_vmax)
    im_s_a = _draw(axes[1, 0], oras5[0], oras5[2], "PiYG_r", ds_vmax)
    _draw(axes[1, 1], glorys12[0], glorys12[2], "PiYG_r", ds_vmax)

    for ax in axes.flat:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Depth  (m)")

    _panel_label(axes[0, 0], "(a) ORAS5  $\\Delta v$")
    _panel_label(axes[0, 1], "(b) GLORYS12V1  $\\Delta v$")
    _panel_label(axes[1, 0], "(c) ORAS5  $\\Delta S$")
    _panel_label(axes[1, 1], "(d) GLORYS12V1  $\\Delta S$")

    cbar_v = fig.colorbar(im_v_a, ax=axes[0, :], pad=0.015, shrink=0.85,
                          location="right")
    cbar_v.set_label(r"$\Delta v$  (m s$^{-1}$)")
    cbar_s = fig.colorbar(im_s_a, ax=axes[1, :], pad=0.015, shrink=0.85,
                          location="right")
    cbar_s.set_label(r"$\Delta S$  (PSU)")

    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
