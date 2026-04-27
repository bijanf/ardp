#!/usr/bin/env python3
"""Paper 2 Figure 5: zonal structure of Δv and ΔS at 34.5°S.

Four panels (2×2 in the combined layout):

  (a) ORAS5      Δv  (late − early)
  (b) GLORYS12V1 Δv
  (c) ORAS5      ΔS
  (d) GLORYS12V1 ΔS

Outputs (default --mode both):
  figures/paper2/Figure5.{png,pdf}            single 2x2 PDF (shared
                                              colorbars per row)
  figures/paper2/Figure5{a,b,c,d}.{png,pdf}   four standalone panels
                                              (each with its own
                                              colorbar)

Reads the raw sections via the same helpers used by
``compute_fovs_decomposition.py``.
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
    # ORAS5's velocity field has NaN→0 replacement; salinity NaN gives
    # the right topography mask for both fields.
    ocean = np.isfinite(s1) & np.isfinite(s2)
    dv = np.where(ocean, v2 - v1, np.nan)
    with np.errstate(invalid="ignore"):
        ds = np.where(ocean, s2 - s1, np.nan)
    return grid, dv, ds


def _draw(ax, grid, field, cmap_name: str, vmax: float):
    cmap_obj = plt.colormaps[cmap_name].copy()
    cmap_obj.set_bad("0.7")
    ax.set_facecolor("0.7")
    n_x = field.shape[1]
    lon = grid["lon_atl"][:n_x]
    depth = grid["depth"][: field.shape[0]]
    masked = np.ma.masked_invalid(field)
    im = ax.pcolormesh(lon, depth, masked, cmap=cmap_obj,
                       vmin=-vmax, vmax=vmax, shading="auto")
    ax.invert_yaxis()
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Depth  (m)")
    return im


def _panel_label(ax, label: str, x: float = 0.0, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


PANELS = [  # (letter, product_idx, field_idx, cmap, var_name, label_text)
    ("a", "oras5",    "dv", "RdBu_r", "Δv", r"ORAS5  $\Delta v$"),
    ("b", "glorys12", "dv", "RdBu_r", "Δv", r"GLORYS12V1  $\Delta v$"),
    ("c", "oras5",    "ds", "PiYG_r", "ΔS", r"ORAS5  $\Delta S$"),
    ("d", "glorys12", "ds", "PiYG_r", "ΔS", r"GLORYS12V1  $\Delta S$"),
]


def _render_combined(oras5, glorys12, dv_vmax, ds_vmax, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8),
                             gridspec_kw={"wspace": 0.18, "hspace": 0.55,
                                          "top": 0.94, "bottom": 0.10,
                                          "left": 0.10, "right": 0.88})
    im_v_a = _draw(axes[0, 0], oras5[0], oras5[1], "RdBu_r", dv_vmax)
    _draw(axes[0, 1], glorys12[0], glorys12[1], "RdBu_r", dv_vmax)
    im_s_a = _draw(axes[1, 0], oras5[0], oras5[2], "PiYG_r", ds_vmax)
    _draw(axes[1, 1], glorys12[0], glorys12[2], "PiYG_r", ds_vmax)
    _panel_label(axes[0, 0], r"(a) ORAS5  $\Delta v$")
    _panel_label(axes[0, 1], r"(b) GLORYS12V1  $\Delta v$")
    _panel_label(axes[1, 0], r"(c) ORAS5  $\Delta S$")
    _panel_label(axes[1, 1], r"(d) GLORYS12V1  $\Delta S$")
    cbar_v = fig.colorbar(im_v_a, ax=axes[0, :], pad=0.015, shrink=0.85,
                          location="right")
    cbar_v.set_label(r"$\Delta v$  (m s$^{-1}$)")
    cbar_s = fig.colorbar(im_s_a, ax=axes[1, :], pad=0.015, shrink=0.85,
                          location="right")
    cbar_s.set_label(r"$\Delta S$  (PSU)")
    save_publication_figure(fig, output)


def _render_split(oras5, glorys12, dv_vmax, ds_vmax, output: Path):
    base = output.parent / output.name
    fields = {
        ("oras5", "dv"):    (oras5[0],    oras5[1],    "RdBu_r", dv_vmax,
                              r"$\Delta v$  (m s$^{-1}$)"),
        ("glorys12", "dv"): (glorys12[0], glorys12[1], "RdBu_r", dv_vmax,
                              r"$\Delta v$  (m s$^{-1}$)"),
        ("oras5", "ds"):    (oras5[0],    oras5[2],    "PiYG_r", ds_vmax,
                              r"$\Delta S$  (PSU)"),
        ("glorys12", "ds"): (glorys12[0], glorys12[2], "PiYG_r", ds_vmax,
                              r"$\Delta S$  (PSU)"),
    }
    for letter, prod, field_key, _cmap, _var, _label in PANELS:
        grid, field, cmap, vmax, cbar_label = fields[(prod, field_key)]
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        im = _draw(ax, grid, field, cmap, vmax)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
        cbar.set_label(cbar_label)
        fig.tight_layout()
        save_publication_figure(fig, base.with_name(base.name + letter))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure5"))
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    oras5 = _load_pair("oras5")
    glorys12 = _load_pair("glorys12")
    if oras5 is None or glorys12 is None:
        print("Need both ORAS5 and GLORYS12 sections.")
        return
    dv_vmax = max(float(np.nanpercentile(np.abs(oras5[1]), 95)),
                  float(np.nanpercentile(np.abs(glorys12[1]), 95)))
    ds_vmax = max(float(np.nanpercentile(np.abs(oras5[2]), 95)),
                  float(np.nanpercentile(np.abs(glorys12[2]), 95)))
    print(f"Δv saturation: ±{dv_vmax:.4f} m/s")
    print(f"ΔS saturation: ±{ds_vmax:.4f} PSU")

    if args.mode in ("combined", "both"):
        _render_combined(oras5, glorys12, dv_vmax, ds_vmax, args.output)
    if args.mode in ("split", "both"):
        _render_split(oras5, glorys12, dv_vmax, ds_vmax, args.output)


if __name__ == "__main__":
    main()
