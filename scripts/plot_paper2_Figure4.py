#!/usr/bin/env python3
"""Paper 2 Figure 4: robustness of the mechanism disagreement.

Three panels:

  (a) Post-Argo decomposition bars (single window pair).
  (b) Sensitivity to period choice: velocity-share distributions
      across 25 early/late window pairs per product.
  (c) Signal-to-noise test: histogram of |ΔF_ovS| from 2600 piControl
      30-year segments, with reanalysis values marked as vertical
      lines.

Outputs (default --mode both):
  figures/paper2/Figure4.{png,pdf}            single combined PDF
  figures/paper2/Figure4{a,b,c}.{png,pdf}     three standalone panels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

PRODUCTS = [
    ("ORAS5",      "oras5",    "#1f77b4"),
    ("GLORYS12V1", "glorys12", "#2ca02c"),
    ("SODA3.15.2", "soda",     "#e377c2"),
    ("ECCO-V4r4",  "ecco",     "#d62728"),
]


def _load_postargo(results_dir: Path, key: str) -> dict | None:
    p = results_dir / f"fovs_decomposition_{key}_postargo.nc"
    if not p.exists():
        return None
    ds = xr.open_dataset(p)
    out = {
        "delta_total": float(ds.attrs["delta_total_Sv"]),
        "delta_v": float(ds.attrs["delta_v_Sv"]),
        "delta_s": float(ds.attrs["delta_s_Sv"]),
        "delta_cross": float(ds.attrs["delta_cross_Sv"]),
    }
    ds.close()
    return out


def _shares(res: dict) -> tuple[float, float]:
    dt = res["delta_total"]
    if abs(dt) < 0.010:
        return np.nan, np.nan
    return 100 * res["delta_v"] / dt, 100 * res["delta_s"] / dt


def _panel_label(ax, label: str, x: float = 0.02, y: float = 0.97) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def _load_data(results_dir: Path):
    rows = []
    for label, key, color in PRODUCTS:
        post = _load_postargo(results_dir, key)
        rows.append((label, key, color, post))
    sens = pd.read_csv(results_dir / "fovs_decomposition_sensitivity.csv")
    null_df = pd.read_csv(results_dir / "fovs_decomposition_cmip6_null.csv")
    forced_df = pd.read_csv(results_dir / "fovs_decomposition_cmip6_summary.csv")
    rean_vals = []
    for label, key, color in PRODUCTS:
        path = results_dir / f"fovs_decomposition_{key}.nc"
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        rean_vals.append((label, abs(float(ds.attrs["delta_total_Sv"]) * 1000),
                          color))
        ds.close()
    return rows, sens, null_df, forced_df, rean_vals


def _draw_panel_a(ax, rows, *, legend_loc: str = "lower center",
                  legend_anchor=(0.5, -0.32), legend_ncol: int = 5):
    x = np.arange(len(rows))
    dv = np.array([r[3]["delta_v"] * 1000 if r[3] else np.nan for r in rows])
    ds_ = np.array([r[3]["delta_s"] * 1000 if r[3] else np.nan for r in rows])
    dc = np.array([r[3]["delta_cross"] * 1000 if r[3] else np.nan for r in rows])
    dt = dv + ds_ + dc
    has_data = np.array([r[3] is not None for r in rows])
    has_trend = np.abs(dt) >= 10.0
    width = 0.55
    ax.bar(x[has_trend], dv[has_trend], width=width, color="#E69F00",
           label=r"$\Delta F_v$  (velocity)")
    ax.bar(x[has_trend], ds_[has_trend], width=width,
           bottom=dv[has_trend], color="#56B4E9",
           label=r"$\Delta F_s$  (salinity)")
    ax.bar(x[has_trend], dc[has_trend], width=width,
           bottom=dv[has_trend] + ds_[has_trend], color="0.6",
           label=r"$\Delta F_\mathrm{cross}$")
    no_trend_mask = has_data & ~has_trend
    if no_trend_mask.any():
        ax.bar(x[no_trend_mask], dt[no_trend_mask], width=width,
               color="0.85", edgecolor="0.4", hatch="///",
               label=r"ill-defined ($|\Delta F| < 10$ mSv)")
    no_data_mask = ~has_data
    if no_data_mask.any():
        for xi in x[no_data_mask]:
            ax.text(xi, 0.0, "pending", ha="center", va="center",
                    fontsize=8, color="0.45", style="italic")
    ax.scatter(x[has_data], dt[has_data], color="black", s=30,
               marker="D", zorder=5, label=r"$\Delta F_\mathrm{total}$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right",
                       fontsize=8)
    # Two-line label so the full text (quantity, units, differencing
    # convention) fits without being clipped at the figure edge (R3.10).
    ax.set_ylabel("$\\Delta F_{ovS}$ (mSv)\n(late minus early, post-Argo)")
    ax.set_ylim(-100, 30)
    ax.grid(axis="y", alpha=0.3, lw=0.3)
    for i, r in enumerate(rows):
        result = r[3]
        if result is None:
            continue
        fv, fs = _shares(result)
        if not np.isfinite(fv):
            ax.text(i, 14, "no\ntrend", ha="center", va="bottom",
                    fontsize=8, color="0.3", style="italic",
                    fontweight="bold")
            continue
        yy, va = (4, "bottom") if dt[i] < 0 else (-4, "top")
        ax.text(i, yy, f"v:{fv:+.0f}%\ns:{fs:+.0f}%",
                ha="center", va=va, fontsize=8, color="0.2",
                fontweight="bold")
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor,
              fontsize=8, frameon=False, ncol=legend_ncol,
              handlelength=1.4, columnspacing=1.2)


def _draw_panel_b(ax, sens, *, legend_loc: str = "upper right"):
    product_colors = {"oras5": "#1f77b4", "glorys12": "#2ca02c",
                      "soda": "#e377c2", "ecco": "#d62728"}
    labels_map = {"oras5": "ORAS5", "glorys12": "GLORYS12V1",
                  "soda": "SODA3.15.2", "ecco": "ECCO-V4r4"}
    products_in_sens = list(sens["product"].unique())
    rng = np.random.default_rng(42)
    for i, p in enumerate(products_in_sens):
        sub = sens[sens["product"] == p].dropna(subset=["velocity_share_pct"])
        if len(sub) == 0:
            continue
        x = sub["velocity_share_pct"].values
        y = np.full(len(x), i, dtype=float) + rng.normal(0, 0.1, len(x))
        ax.scatter(x, y, color=product_colors.get(p, "0.3"), alpha=0.65,
                   s=35, edgecolor="0.2", linewidth=0.3, zorder=4,
                   label=f"{labels_map.get(p, p)} (n={len(x)})")
    ax.axvline(60, color="#E69F00", ls=":", lw=0.7, alpha=0.7)
    ax.axvline(40, color="#56B4E9", ls=":", lw=0.7, alpha=0.7)
    ax.set_yticks(range(len(products_in_sens)))
    ax.set_yticklabels([labels_map.get(p, p) for p in products_in_sens],
                       fontsize=8)
    ax.set_xlabel(r"Velocity share $f_v$ (%)")
    ax.set_xlim(-100, 220)
    ax.legend(loc=legend_loc, fontsize=7, frameon=False)


def _draw_panel_c(ax, null_df, forced_df, rean_vals):
    null_abs = np.abs(null_df["delta_total"].values) * 1000
    forced_abs = np.abs(forced_df["delta_total"].values) * 1000
    bins = np.linspace(0, max(null_abs.max(), forced_abs.max()) + 10, 40)
    ax.hist(null_abs, bins=bins, color="0.55", alpha=0.7, edgecolor="0.25",
            linewidth=0.3, density=True, zorder=3,
            label=f"piControl null (n={len(null_abs)})")
    ax.hist(forced_abs, bins=bins, color="#CC3333", alpha=0.55,
            edgecolor="0.25", linewidth=0.3, density=True, zorder=4,
            label=f"CMIP6 forced (n={len(forced_abs)})")
    null_p95 = float(np.percentile(null_abs, 95))
    ax.axvline(null_p95, color="0.3", ls=":", lw=0.9, zorder=2,
               label=f"piControl p95 = {null_p95:.0f} mSv")
    for _label, val, color in rean_vals:
        ax.axvline(val, color=color, lw=2.0, zorder=6)
    ax.set_xlabel(r"$|\Delta F_{ovS}|$ (mSv)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, None)
    base_handles, base_labels = ax.get_legend_handles_labels()
    rean_handles = [
        plt.Line2D([], [], color=color, lw=2.0,
                   label=f"{label}: {val:.0f} mSv")
        for label, val, color in rean_vals
    ]
    ax.legend(base_handles + rean_handles,
              base_labels + [h.get_label() for h in rean_handles],
              loc="upper right", fontsize=7, frameon=False,
              handlelength=1.6)


def _render_combined(rows, sens, null_df, forced_df, rean_vals, output: Path):
    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.45, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    _draw_panel_a(ax_a, rows)
    _panel_label(ax_a, "a")
    _draw_panel_b(ax_b, sens)
    _panel_label(ax_b, "b")
    _draw_panel_c(ax_c, null_df, forced_df, rean_vals)
    _panel_label(ax_c, "c")
    fig.tight_layout()
    save_publication_figure(fig, output)


def _render_split(rows, sens, null_df, forced_df, rean_vals, output: Path):
    base = output.parent / output.name
    # Panel (a) — wider since it's the full-width subfigure when used in
    # LaTeX. Move the legend INSIDE the axes (lower-center inside) so
    # the standalone version still shows the legend.
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _draw_panel_a(ax, rows, legend_loc="lower center",
                  legend_anchor=(0.5, -0.32), legend_ncol=5)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "a"))
    # Panel (b)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    _draw_panel_b(ax, sens)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "b"))
    # Panel (c)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    _draw_panel_c(ax, null_df, forced_df, rean_vals)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "c"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure4"))
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    rows, sens, null_df, forced_df, rean_vals = _load_data(args.results_dir)

    if args.mode in ("combined", "both"):
        _render_combined(rows, sens, null_df, forced_df, rean_vals, args.output)
    if args.mode in ("split", "both"):
        _render_split(rows, sens, null_df, forced_df, rean_vals, args.output)


if __name__ == "__main__":
    main()
