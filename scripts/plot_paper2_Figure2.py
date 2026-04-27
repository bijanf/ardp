#!/usr/bin/env python3
"""Paper 2 Figure 2: CMIP6 mirrors the reanalysis split.

Three panels (one combined PDF):

  (a) Velocity-share vs salinity-share scatter for 12 weakening CMIP6
      models, with the four reanalyses overplotted as diamonds.
      Per-model CMIP6 labels are intentionally omitted (they cannot
      be placed without overlap in a multi-panel figure); class
      colour + edge-colour encoding identifies each point's
      mechanism class and bistable-baseline state. Only the four
      reanalyses are labelled by name.
  (b) Mechanism-conditional AMOC trajectories at 26.5°N, with class
      ensemble means.
  (c) Boxplot of projected AMOC weakening by 2100 (2081-2100 vs
      1950-1980) per mechanism class. Red ribbon and line:
      observational constraint of Portmann et al. 2026 (51 ± 8\%);
      its numeric value is documented in the figure caption.

Reads:
  data/results/fovs_decomposition_cmip6_summary.csv
  data/results/fovs_decomposition_{oras5,glorys12,soda,ecco}.nc
  data/results/yearly_amoc26n_cmip6.npz

Outputs: figures/paper2/Figure2.{png,pdf}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure2"))
    args = parser.parse_args()

    apply_nature_style()

    cmip6 = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_summary.csv")
    cmip6["class"] = cmip6.apply(_classify, axis=1)
    cmip6["bistable"] = cmip6["F_ov_baseline"] < 0

    reanalysis_points = []
    for label, fname, color in REANALYSIS_PRODUCTS:
        r = _load_reanalysis_shares(args.results_dir / fname)
        if r is not None:
            dtot_mSv, v_pct, s_pct, has_trend = r
            reanalysis_points.append((label, dtot_mSv, v_pct, s_pct, has_trend, color))

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(11.0, 3.8))

    # ── Panel (a): v-share vs s-share scatter ──
    ax_a.plot([0, 100], [100, 0], color="0.5", lw=0.6, ls="--", zorder=2,
              label=r"$v+s=100$%")
    ax_a.axhline(60, color="#56B4E9", lw=0.5, ls=":", alpha=0.7, zorder=1)
    ax_a.axvline(60, color="#E69F00", lw=0.5, ls=":", alpha=0.7, zorder=1)
    for cls in CLASS_ORDER:
        for bistable, edge_color, edge_lw in [(True, "black", 1.2),
                                               (False, "0.75", 0.5)]:
            sub = cmip6[(cmip6["class"] == cls) & (cmip6["bistable"] == bistable)]
            if len(sub) == 0:
                continue
            tag = "bistable" if bistable else "monostable"
            ax_a.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                         s=55, color=CLASS_COLORS[cls], alpha=0.75,
                         edgecolor=edge_color, linewidth=edge_lw, zorder=4,
                         label=f"{cls}, {tag} (n={len(sub)})")

    # Reanalyses: large diamonds, labelled. To avoid overlap with the
    # CMIP6 cloud we anchor labels at fixed offsets around each diamond
    # along the four cardinal directions, choosing the side that lies
    # in a sparse area of the panel.
    rean_offsets = {"ORAS5": (10, 6), "GLORYS12V1": (-10, 6),
                    "SODA3.15.2": (10, -10), "ECCO-V4r4": (-10, -10)}
    for label, _dtot, v_pct, s_pct, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax_a.scatter(v_pct, s_pct, s=140, c=color, edgecolor="black",
                     linewidth=0.9, zorder=8, marker="D")
        dx, dy = rean_offsets.get(label, (8, 8))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax_a.annotate(label, xy=(v_pct, s_pct),
                      xytext=(dx, dy), textcoords="offset points",
                      fontsize=8.5, color=color, fontweight="bold",
                      ha=ha, va=va, zorder=9)

    ax_a.set_xlabel(r"Velocity share $f_v$ (%)")
    ax_a.set_ylabel(r"Salinity share $f_s$ (%)")
    ax_a.set_xlim(-80, 220)
    ax_a.set_ylim(-150, 220)
    ax_a.legend(loc="lower left", fontsize=6.8, frameon=False,
                handlelength=1.3, handletextpad=0.5, borderaxespad=0.3)
    _panel_label(ax_a, "(a)")

    # ── Panel (b): AMOC trajectories by mechanism class ──
    cmip6_amoc = cmip6.copy()
    cmip6_amoc["class"] = cmip6_amoc.apply(_classify_amoc, axis=1)
    classes = {c: cmip6_amoc[cmip6_amoc["class"] == c]["model"].tolist()
               for c in ("v-dominant", "s-dominant", "mixed", "stable")}
    amoc = np.load(args.results_dir / "yearly_amoc26n_cmip6.npz",
                   allow_pickle=True)
    amoc_models = [str(m) for m in amoc["models"]]
    trajectories = {c: [] for c in classes}
    for cls, model_list in classes.items():
        for m in model_list:
            if m not in amoc_models:
                continue
            y = amoc[f"{m}_years"]
            a = amoc[f"{m}_amoc"]
            ax_b.plot(y, a, color=CLASS_COLORS[cls], alpha=0.25, lw=0.5,
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
        ax_b.plot(common_years, mean_a, color=CLASS_COLORS[cls], lw=2.2,
                  zorder=6, label=f"{cls} (n={n})")
        ax_b.fill_between(common_years, mean_a - std_a, mean_a + std_a,
                          color=CLASS_COLORS[cls], alpha=0.15, zorder=4)
    ax_b.axvline(2014, color="0.35", lw=0.8, ls="--", zorder=1)
    ax_b.set_xlim(1850, 2100)
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("AMOC at 26.5°N  (Sv)")
    ax_b.legend(loc="lower left", fontsize=8, frameon=False)
    _panel_label(ax_b, "(b)")

    # ── Panel (c): boxplot of % weakening by class ──
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
    bp = ax_c.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True,
                      widths=0.6,
                      medianprops={"color": "black", "linewidth": 1.0})
    for patch, cls in zip(bp["boxes"], ("v-dominant", "s-dominant", "mixed")):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.65)
    ax_c.fill_between([0.5, 3.5], 43, 59, color="#CC3333", alpha=0.12,
                      zorder=1)
    ax_c.axhline(51, color="#CC3333", lw=1.3, zorder=2)
    ax_c.set_ylabel("AMOC weakening by 2100 (%)")
    ax_c.set_ylim(None, 95)
    ax_c.set_xlim(0.5, 3.5)
    ax_c.grid(axis="y", alpha=0.3, lw=0.3)
    _panel_label(ax_c, "(c)")

    fig.tight_layout()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
