#!/usr/bin/env python3
"""Figure S3: zonal structure of Δv and ΔS at 34.5°S across reanalyses.

Shows where in the Atlantic basin the velocity and salinity anomalies
(between early and late periods) are concentrated. This reveals whether
the ORAS5 v-dominant vs GLORYS12 s-dominant contrast comes from:
  - boundary current differences (e.g. western boundary current strength)
  - interior gyre changes
  - thermocline structure adjustments

Four panels (2x2): ΔV(z, x) and ΔS(z, x) for ORAS5 and GLORYS12.

Reads the raw sections loaded on the fly (via compute_fovs_decomposition
helpers), for the same EARLY/LATE periods used in the main Fig 2.

Outputs: figures/paper2/figS3_zonal_structure.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.viz.style import apply_nature_style, save_publication_figure
from compute_fovs_decomposition import (  # noqa: E402
    EARLY, LATE,
    _glorys12_period_mean, _oras5_period_mean,
)


def _load_pair(product: str) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Return (grid_info, dv[z,x], ds[z,x]) for one reanalysis."""
    try:
        if product == "oras5":
            v1, s1, grid = _oras5_period_mean(Path("data/oras5"), EARLY)
            v2, s2, _ = _oras5_period_mean(Path("data/oras5"), LATE)
        elif product == "glorys12":
            v1, s1, grid = _glorys12_period_mean(Path("data/glorys12"), EARLY)
            v2, s2, _ = _glorys12_period_mean(Path("data/glorys12"), LATE)
        else:
            return None
    except Exception as e:
        print(f"  {product}: {e}")
        return None

    dv = v2 - v1
    # For salinity, mask where either period had NaN
    with np.errstate(invalid="ignore"):
        ds = np.where(np.isfinite(s1) & np.isfinite(s2), s2 - s1, np.nan)
    return grid, dv, ds


def _plot_panel(ax, grid, field, cmap, vmax, title=""):
    """Draw one Δv or ΔS panel on the supplied axis. Returns the QuadMesh.

    The ``title`` argument is retained for backward compatibility but is
    intentionally not rendered — Nature/Science style provides the panel
    label via LaTeX (\\textbf{(a)} etc.) below the figure.
    """
    e1t = grid["e1t_atl"]
    n_x = field.shape[1]
    x_km = np.concatenate([[0.0], np.cumsum(e1t[:-1]) / 1000.0])[:n_x]
    depth = grid["depth"][: field.shape[0]]

    im = ax.pcolormesh(x_km, depth, field, cmap=cmap, vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.invert_yaxis()
    _ = title  # kept for API compatibility, not drawn
    ax.set_xlabel("Distance from western boundary  (km)")
    ax.set_ylabel("Depth  (m)")
    return im


def _save_single_panel(grid, field, cmap, vmax, title, cbar_label, out_path):
    """Save a single panel (one Δv or ΔS subplot) as standalone PNG+PDF."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = _plot_panel(ax, grid, field, cmap, vmax, title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    save_publication_figure(fig, out_path, formats=["png", "pdf"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/figS3_zonal_structure"))
    parser.add_argument(
        "--split-panels", action="store_true",
        help="Also emit four standalone panels as figures/paper2/Fig5a.{png,pdf}"
             "...Fig5d.{png,pdf} for the new Main.tex zonal-structure figure.",
    )
    parser.add_argument(
        "--split-output-dir", type=Path,
        default=Path("figures/paper2"),
        help="Directory for split panel files (Fig5a..Fig5d).",
    )
    args = parser.parse_args()

    apply_nature_style()

    oras5 = _load_pair("oras5")
    glorys12 = _load_pair("glorys12")

    if oras5 is None or glorys12 is None:
        print("Need both ORAS5 and GLORYS12 sections.")
        return

    # Compute shared colour scales (used for both combined and split figures
    # so that the four individual panels keep a comparable scale).
    dv_vmax = max(
        float(np.nanpercentile(np.abs(oras5[1]), 99)),
        float(np.nanpercentile(np.abs(glorys12[1]), 99)),
    )
    ds_vmax = max(
        float(np.nanpercentile(np.abs(oras5[2]), 99)),
        float(np.nanpercentile(np.abs(glorys12[2]), 99)),
    )
    print(f"Δv saturation: ±{dv_vmax:.4f} m/s")
    print(f"ΔS saturation: ±{ds_vmax:.4f} PSU")

    # ── Combined 2x2 figure (legacy figS3) ──
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0),
                             gridspec_kw={"wspace": 0.20, "hspace": 0.28})
    im1 = _plot_panel(axes[0, 0], oras5[0], oras5[1], "RdBu_r", dv_vmax,
                      "(a) ORAS5  Δv  (late − early)")
    _plot_panel(axes[0, 1], glorys12[0], glorys12[1], "RdBu_r", dv_vmax,
                "(b) GLORYS12V1  Δv")
    im3 = _plot_panel(axes[1, 0], oras5[0], oras5[2], "PiYG_r", ds_vmax,
                     "(c) ORAS5  ΔS")
    _plot_panel(axes[1, 1], glorys12[0], glorys12[2], "PiYG_r", ds_vmax,
                "(d) GLORYS12V1  ΔS")
    cbar1 = fig.colorbar(im1, ax=axes[0, :], pad=0.015, shrink=0.85, location="right")
    cbar1.set_label(r"$\Delta v$  (m s$^{-1}$)")
    cbar3 = fig.colorbar(im3, ax=axes[1, :], pad=0.015, shrink=0.85, location="right")
    cbar3.set_label(r"$\Delta S$  (PSU)")
    # suptitle removed — Nature/Science style relies on the LaTeX caption.
    save_publication_figure(fig, args.output)

    # ── Optional: 4 standalone panels for the new Main.tex (Fig 5) ──
    if args.split_panels:
        out_dir = args.split_output_dir
        _save_single_panel(oras5[0], oras5[1], "RdBu_r", dv_vmax,
                           "ORAS5  Δv  (late − early)",
                           r"$\Delta v$  (m s$^{-1}$)",
                           out_dir / "Fig5a")
        _save_single_panel(glorys12[0], glorys12[1], "RdBu_r", dv_vmax,
                           "GLORYS12V1  Δv  (late − early)",
                           r"$\Delta v$  (m s$^{-1}$)",
                           out_dir / "Fig5b")
        _save_single_panel(oras5[0], oras5[2], "PiYG_r", ds_vmax,
                           "ORAS5  ΔS  (late − early)",
                           r"$\Delta S$  (PSU)",
                           out_dir / "Fig5c")
        _save_single_panel(glorys12[0], glorys12[2], "PiYG_r", ds_vmax,
                           "GLORYS12V1  ΔS  (late − early)",
                           r"$\Delta S$  (PSU)",
                           out_dir / "Fig5d")


if __name__ == "__main__":
    main()
