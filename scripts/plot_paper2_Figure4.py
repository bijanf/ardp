#!/usr/bin/env python3
"""Paper 2 Figure 4: robustness of the mechanism disagreement.

Three panels:

  (a) Decomposition bars for the main windows (1993-2005 vs 2013-2025)
      and the Argo-era windows (2006-2012 vs 2018-2024), side by side.
  (b) Sensitivity to period choice: velocity-share distributions
      across 25 early/late window pairs per product.
  (c) Signal-to-noise test: histogram of |ΔF_ovS| from 2600 piControl
      30-year segments and from the CMIP6 historical to SSP5-8.5
      change, with reanalysis values marked as vertical lines and the
      control-run 95th percentiles for 30-year windows and for 13-year
      windows 20 years apart.

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


def _load_postargo(results_dir: Path, key: str, suffix: str = "_postargo") -> dict | None:
    p = results_dir / f"fovs_decomposition_{key}{suffix}.nc"
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


def _panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def _load_data(results_dir: Path, thresholds: Path | None = None):
    rows = []
    for label, key, color in PRODUCTS:
        post = _load_postargo(results_dir, key)
        main = _load_postargo(results_dir, key, suffix="")
        rows.append((label, key, color, post, main))
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
    thr_path = thresholds or results_dir / "picontrol_matched_thresholds.json"
    if thr_path.exists():
        import json
        thr = json.loads(thr_path.read_text())["pooled"]["13y_sep20_raw"]["p95_mSv"]
        null_df.attrs["p95_13y_sep20"] = thr
    return rows, sens, null_df, forced_df, rean_vals


def _draw_panel_a(ax, rows, *, legend_loc: str = "lower center",
                  legend_anchor=(0.5, -0.42), legend_ncol: int = 5):
    """Paired bars: main windows (light, left) and Argo-era windows (dark, right)."""
    colors = {"main": ("#F3CF8A", "#B9DFF4", "0.85"),
              "argo": ("#E69F00", "#56B4E9", "0.55")}
    x = np.arange(len(rows))
    width = 0.36
    for j, (tag, idx, off) in enumerate((("main", 4, -0.2), ("argo", 3, 0.2))):
        res = [r[idx] for r in rows]
        dv = np.array([r["delta_v"] * 1000 if r else np.nan for r in res])
        ds_ = np.array([r["delta_s"] * 1000 if r else np.nan for r in res])
        dc = np.array([r["delta_cross"] * 1000 if r else np.nan for r in res])
        dt = dv + ds_ + dc
        ok = np.isfinite(dt) & (np.abs(dt) >= 10.0)
        small = np.isfinite(dt) & ~ok
        cv, cs, cc = colors[tag]
        xs = x + off
        # Stack positive and negative parts separately so bars never overlap.
        pos_base = np.zeros(len(rows))
        neg_base = np.zeros(len(rows))
        for vals, col, lab in ((dv, cv, r"$\Delta F_v$"), (ds_, cs, r"$\Delta F_s$"),
                               (dc, cc, r"$\Delta F_\mathrm{cross}$")):
            v = np.where(ok, vals, 0.0)
            base = np.where(v >= 0, pos_base, neg_base)
            ax.bar(xs, v, width=width, bottom=base, color=col, edgecolor="0.25",
                   linewidth=0.3, label=lab if tag == "argo" else None)
            pos_base = pos_base + np.where(v >= 0, v, 0.0)
            neg_base = neg_base + np.where(v < 0, v, 0.0)
        if small.any():
            ax.bar(xs[small], dt[small], width=width, color="white", edgecolor="0.4",
                   hatch="///", linewidth=0.4,
                   label=r"$|\Delta F_{ovS}| < 10$ mSv" if tag == "main" else None)
        ax.scatter(xs[np.isfinite(dt)], dt[np.isfinite(dt)], color="black", s=22,
                   marker="D", zorder=5, label=r"$\Delta F_{ovS}$ (sum)" if tag == "main" else None)
        for i in range(len(rows)):
            if not np.isfinite(dt[i]):
                continue
            if ok[i]:
                txt = f"v {100 * dv[i] / dt[i]:.0f}%\ns {100 * ds_[i] / dt[i]:.0f}%"
            else:
                txt = "below\nfloor"
            top = max(pos_base[i], dt[i], 0.0)
            bot = min(neg_base[i], dt[i], 0.0)
            if dt[i] < 0:
                ax.text(xs[i], top + 3, txt, ha="center", va="bottom", fontsize=6.5, color="0.15")
            else:
                ax.text(xs[i], bot - 3, txt, ha="center", va="top", fontsize=6.5, color="0.15")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(np.concatenate([x - 0.2, x + 0.2]))
    ax.set_xticklabels(["from 1993"] * len(rows) + ["Argo era"] * len(rows), fontsize=7)
    ax.tick_params(axis="x", length=2, pad=2)
    for i, r in enumerate(rows):
        ax.text(x[i], -0.13, r[0], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("$\\Delta F_{ovS}$ (mSv)\n(late minus early)")
    ax.set_ylim(-105, 40)
    ax.grid(axis="y", alpha=0.3, lw=0.3)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor,
              fontsize=7, frameon=False, ncol=legend_ncol,
              handlelength=1.4, columnspacing=1.0)


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
    ax.set_ylabel("Reanalysis")
    ax.set_ylim(-0.6, len(products_in_sens) - 0.4)
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
            label=f"CMIP6 historical to SSP5-8.5 (n={len(forced_abs)})")
    null_p95 = float(np.percentile(null_abs, 95))
    ax.axvline(null_p95, color="0.3", ls=":", lw=0.9, zorder=2,
               label=f"piControl p95, 30-yr = {null_p95:.0f} mSv")
    p95_13 = null_df.attrs.get("p95_13y_sep20")
    if p95_13 is not None:
        ax.axvline(p95_13, color="0.3", ls="--", lw=0.9, zorder=2,
                   label=f"piControl p95, 13-yr = {p95_13:.0f} mSv")
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
              loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2,
              fontsize=6.5, frameon=False, handlelength=1.6, columnspacing=0.8)


def _render_combined(rows, sens, null_df, forced_df, rean_vals, output: Path):
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.62, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    _draw_panel_a(ax_a, rows)
    _panel_label(ax_a, "a", x=-0.06)
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
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="JSON written by compute_cmip6_picontrol_threshold.py")
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    rows, sens, null_df, forced_df, rean_vals = _load_data(args.results_dir, args.thresholds)

    if args.mode in ("combined", "both"):
        _render_combined(rows, sens, null_df, forced_df, rean_vals, args.output)
    if args.mode in ("split", "both"):
        _render_split(rows, sens, null_df, forced_df, rean_vals, args.output)


if __name__ == "__main__":
    main()
