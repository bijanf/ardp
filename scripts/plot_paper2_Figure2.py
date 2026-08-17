#!/usr/bin/env python3
"""Paper 2 Figure 2: CMIP6 mirrors the reanalysis split.

Three panels:

  (a) Velocity-share vs salinity-share scatter for 12 weakening CMIP6
      models, with the four reanalyses overplotted as diamonds.
  (b) Mechanism-conditional AMOC trajectories at 26.5°N.
  (c) Boxplot of projected AMOC weakening by 2100 per mechanism class
      (red ribbon: Portmann 2026 constraint).

Outputs (default --mode both):
  figures/paper2/Figure2.{png,pdf}            single combined PDF
  figures/paper2/Figure2{a,b,c}.{png,pdf}     three standalone panels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

REANALYSIS_PRODUCTS = [
    ("ORAS5",      "fovs_decomposition_oras5.nc",    "#1f77b4"),
    ("GLORYS12V1", "fovs_decomposition_glorys12.nc", "#2ca02c"),
    ("SODA3.15.2", "fovs_decomposition_soda.nc",     "#e377c2"),
    ("ECCO-V4r4",  "fovs_decomposition_ecco.nc",     "#d62728"),
]

CLASS_COLORS = {
    "v-dominant": "#E69F00",
    "s-dominant": "#56B4E9",
    "mixed":      "#009E73",
    "stable":     "0.6",
    "increasing": "0.5",
}
CLASS_ORDER = ["v-dominant", "s-dominant", "mixed", "increasing"]
# Display name for the legend: the "increasing" bucket also contains models
# whose |dF_ovS| falls below the 10 mSv classification floor (EC-Earth3 is
# -1.4 mSv), so "excluded" is the accurate public label.
CLASS_DISPLAY = {"increasing": "excluded"}


def _load_reanalysis_shares(path: Path):
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    dtot = float(ds.attrs["delta_total_Sv"])
    dv = float(ds.attrs["delta_v_Sv"])
    ds_ = float(ds.attrs["delta_s_Sv"])
    ds.close()
    has_trend = abs(dtot) >= 0.010
    v_pct = 100 * dv / dtot if has_trend else np.nan
    s_pct = 100 * ds_ / dtot if has_trend else np.nan
    return dtot * 1000, v_pct, s_pct, has_trend


def _classify(row):
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def _classify_amoc(row):
    if row["delta_total"] >= -0.01:
        return "stable"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def _panel_label(ax, label: str, x: float = 0.0, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def _load_data(results_dir: Path):
    cmip6 = pd.read_csv(results_dir / "fovs_decomposition_cmip6_summary.csv")
    cmip6["class"] = cmip6.apply(_classify, axis=1)
    cmip6["bistable"] = cmip6["F_ov_baseline"] < 0
    reanalysis_points = []
    for label, fname, color in REANALYSIS_PRODUCTS:
        r = _load_reanalysis_shares(results_dir / fname)
        if r is not None:
            dtot_msv, v_pct, s_pct, has_trend = r
            reanalysis_points.append((label, dtot_msv, v_pct, s_pct,
                                       has_trend, color))
    cmip6_amoc = cmip6.copy()
    cmip6_amoc["class"] = cmip6_amoc.apply(_classify_amoc, axis=1)
    classes = {c: cmip6_amoc[cmip6_amoc["class"] == c]["model"].tolist()
               for c in ("v-dominant", "s-dominant", "mixed", "stable")}
    amoc = np.load(results_dir / "yearly_amoc26n_cmip6.npz",
                   allow_pickle=True)
    amoc_models = [str(m) for m in amoc["models"]]
    return cmip6, reanalysis_points, classes, amoc, amoc_models


def _draw_panel_a(ax, cmip6, reanalysis_points):
    ax.plot([0, 100], [100, 0], color="0.5", lw=0.6, ls="--", zorder=2,
            label=r"$v+s=100$%")
    ax.axhline(60, color="#56B4E9", lw=0.5, ls=":", alpha=0.7, zorder=1)
    ax.axvline(60, color="#E69F00", lw=0.5, ls=":", alpha=0.7, zorder=1)
    for cls in CLASS_ORDER:
        for bistable, edge_color, edge_lw in [(True, "black", 1.2),
                                                (False, "0.75", 0.5)]:
            sub = cmip6[(cmip6["class"] == cls) & (cmip6["bistable"] == bistable)]
            if len(sub) == 0:
                continue
            tag = "bistable" if bistable else "monostable"
            ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                       s=55, color=CLASS_COLORS[cls], alpha=0.75,
                       edgecolor=edge_color, linewidth=edge_lw, zorder=4,
                       label=f"{CLASS_DISPLAY.get(cls, cls)}, {tag} (n={len(sub)})")
    # SODA3.15.2 sits very close to the v=s=50 line, so its label is pulled
    # well above the diamond (to ~y=60 data units) with a thin leader line
    # rather than the small offset used for the others.
    rean_offsets = {"ORAS5": (10, 6), "GLORYS12V1": (8, 10),
                    "SODA3.15.2": None, "ECCO-V4r4": (-10, -10)}
    for label, _dtot, v_pct, s_pct, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax.scatter(v_pct, s_pct, s=140, c=color, edgecolor="black",
                   linewidth=0.9, zorder=8, marker="D")
        if label == "SODA3.15.2":
            label_x = v_pct - 22
            label_y = 65
            ax.annotate(
                label, xy=(v_pct, s_pct),
                xytext=(label_x, label_y), textcoords="data",
                fontsize=8.5, color=color, fontweight="bold",
                ha="left", va="bottom", zorder=9,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5,
                            "shrinkA": 0.0, "shrinkB": 3.0},
            )
            continue
        dx, dy = rean_offsets.get(label, (8, 8))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(label, xy=(v_pct, s_pct),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="bold",
                    ha=ha, va=va, zorder=9)
    ax.set_xlabel(r"Velocity share $f_v$ (%)")
    ax.set_ylabel(r"Salinity share $f_s$ (%)")
    ax.set_xlim(-80, 220)
    ax.set_ylim(-150, 240)
    ax.legend(loc="lower left", fontsize=6.8, frameon=False,
              handlelength=1.3, handletextpad=0.5, borderaxespad=0.3)


def _draw_panel_b(ax, classes, amoc, amoc_models):
    trajectories = {c: [] for c in classes}
    for cls, model_list in classes.items():
        for m in model_list:
            if m not in amoc_models:
                continue
            y = amoc[f"{m}_years"]
            a = amoc[f"{m}_amoc"]
            ax.plot(y, a, color=CLASS_COLORS[cls], alpha=0.25, lw=0.5,
                    zorder=2)
            trajectories[cls].append((y, a))
    common_years = np.arange(1850, 2101)
    for cls in ("v-dominant", "s-dominant", "mixed"):
        if not trajectories[cls]:
            continue
        n = len(trajectories[cls])
        matrix = np.full((n, len(common_years)), np.nan)
        for j, (y, a) in enumerate(trajectories[cls]):
            for k, yr in enumerate(common_years):
                idx = np.where(y == yr)[0]
                if idx.size:
                    matrix[j, k] = a[idx[0]]
        with np.errstate(all="ignore"):
            n_valid = np.sum(np.isfinite(matrix), axis=0)
            mean_a = np.nanmean(matrix, axis=0)
            std_a = np.nanstd(matrix, axis=0)
        keep = n_valid >= n
        mean_a = np.where(keep, mean_a, np.nan)
        std_a = np.where(keep, std_a, np.nan)
        ax.plot(common_years, mean_a, color=CLASS_COLORS[cls], lw=2.2,
                zorder=6, label=f"{cls} (n={n})")
        ax.fill_between(common_years, mean_a - std_a, mean_a + std_a,
                        color=CLASS_COLORS[cls], alpha=0.15, zorder=4)
    ax.axvline(2014, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(1850, 2100)
    ax.set_xlabel("Year")
    ax.set_ylabel("AMOC at 26.5°N  (Sv)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)


def _draw_panel_c(ax, classes, amoc, amoc_models):
    weakening_pct = {c: [] for c in ("v-dominant", "s-dominant", "mixed")}
    for cls, model_list in classes.items():
        if cls == "stable":
            continue
        for m in model_list:
            if m not in amoc_models:
                continue
            y = amoc[f"{m}_years"]
            a = amoc[f"{m}_amoc"]
            base = float(np.nanmean(a[(y >= 1950) & (y <= 1980)]))
            end = float(np.nanmean(a[(y >= 2081) & (y <= 2100)]))
            if np.isfinite(base) and np.isfinite(end) and base > 0:
                weakening_pct[cls].append(100 * (base - end) / base)
    bp_data = [weakening_pct[c] for c in ("v-dominant", "s-dominant", "mixed")]
    bp_labels = [f"v-dom\n(n={len(weakening_pct['v-dominant'])})",
                 f"s-dom\n(n={len(weakening_pct['s-dominant'])})",
                 f"mixed\n(n={len(weakening_pct['mixed'])})"]
    bp = ax.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True,
                    widths=0.6,
                    medianprops={"color": "black", "linewidth": 1.0})
    for patch, cls in zip(bp["boxes"], ("v-dominant", "s-dominant", "mixed"),
                          strict=False):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.65)
    ax.fill_between([0.5, 3.5], 43, 59, color="#CC3333", alpha=0.12,
                    zorder=1)
    ax.axhline(51, color="#CC3333", lw=1.3, zorder=2)
    ax.set_ylabel("AMOC weakening by 2100 (%)")
    ax.set_ylim(None, 95)
    ax.set_xlim(0.5, 3.5)
    ax.grid(axis="y", alpha=0.3, lw=0.3)


def _render_combined(cmip6, reanalysis_points, classes, amoc, amoc_models,
                     output: Path):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(11.0, 3.8))
    _draw_panel_a(ax_a, cmip6, reanalysis_points)
    _panel_label(ax_a, "a")
    _draw_panel_b(ax_b, classes, amoc, amoc_models)
    _panel_label(ax_b, "b")
    _draw_panel_c(ax_c, classes, amoc, amoc_models)
    _panel_label(ax_c, "c")
    fig.tight_layout()
    save_publication_figure(fig, output)


def _render_split(cmip6, reanalysis_points, classes, amoc, amoc_models,
                  output: Path):
    base = output.parent / output.name
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    _draw_panel_a(ax, cmip6, reanalysis_points)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "a"))
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    _draw_panel_b(ax, classes, amoc, amoc_models)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "b"))
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    _draw_panel_c(ax, classes, amoc, amoc_models)
    fig.tight_layout()
    save_publication_figure(fig, base.with_name(base.name + "c"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure2"))
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    cmip6, rean_pts, classes, amoc, amoc_models = _load_data(args.results_dir)

    if args.mode in ("combined", "both"):
        _render_combined(cmip6, rean_pts, classes, amoc, amoc_models,
                         args.output)
    if args.mode in ("split", "both"):
        _render_split(cmip6, rean_pts, classes, amoc, amoc_models,
                      args.output)


if __name__ == "__main__":
    main()
