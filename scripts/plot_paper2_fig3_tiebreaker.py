#!/usr/bin/env python3
"""Figure 3: mechanism tie-breaker across reanalyses + CMIP6.

All 15 CMIP6 models and all 4 reanalyses on one plane.

(a) Velocity-share vs salinity-share. CMIP6 circles coloured by class
    (v-dominant, s-dominant, mixed, F_ovS-increasing); reanalyses shown
    as fixed-size coloured diamonds with labels. Symbol areas are NOT
    weighted by ΔF_ovS magnitude (previous version was misleading —
    GLORYS12V1's large |ΔF| dominated the scatter). A compact note in
    the lower-right spells this out.

(b) Velocity-share distribution across all CMIP6 weakening models,
    with reanalysis v-shares marked as vertical lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from adjustText import adjust_text

from ardp.viz.style import apply_nature_style, save_publication_figure

REANALYSIS_PRODUCTS = [
    ("ORAS5",      "fovs_decomposition_oras5.nc",    "#1f77b4"),
    ("GLORYS12V1", "fovs_decomposition_glorys12.nc", "#2ca02c"),
    ("SODA3.15.2", "fovs_decomposition_soda.nc",     "#e377c2"),
    ("ECCO-V4r4",  "fovs_decomposition_ecco.nc",     "#d62728"),
]


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


CLASS_COLORS = {
    "v-dominant": "#E69F00",
    "s-dominant": "#56B4E9",
    "mixed":      "#009E73",
    "increasing": "0.5",
}
CLASS_ORDER = ["v-dominant", "s-dominant", "mixed", "increasing"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/fig3_tiebreaker"))
    args = parser.parse_args()

    apply_nature_style()

    cmip6 = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_summary.csv")
    cmip6["class"] = cmip6.apply(_classify, axis=1)

    reanalysis_points = []
    for label, fname, color in REANALYSIS_PRODUCTS:
        r = _load_reanalysis_shares(args.results_dir / fname)
        if r is not None:
            dtot_mSv, v_pct, s_pct, has_trend = r
            reanalysis_points.append((label, dtot_mSv, v_pct, s_pct, has_trend, color))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # ── Panel (a): v-share vs s-share scatter ──
    # Physical anchor lines first (so markers sit on top)
    ax1.plot([0, 100], [100, 0], color="0.5", lw=0.6, ls="--", zorder=2,
             label=r"$v+s=100\%$  (no cross term)")
    ax1.axhline(60, color="#56B4E9", lw=0.5, ls=":", alpha=0.7, zorder=1)
    ax1.axvline(60, color="#E69F00", lw=0.5, ls=":", alpha=0.7, zorder=1)

    # Add baseline-regime flag: bistable (F_ov_baseline < 0) vs monostable.
    cmip6["bistable"] = cmip6["F_ov_baseline"] < 0

    # CMIP6 circles, one per class. Marker edge colour encodes baseline
    # regime: BLACK = bistable baseline, GREY = monostable baseline.
    for cls in CLASS_ORDER:
        for bistable, edge_color, edge_lw in [(True, "black", 1.2),
                                              (False, "0.75", 0.5)]:
            sub = cmip6[(cmip6["class"] == cls) & (cmip6["bistable"] == bistable)]
            if len(sub) == 0:
                continue
            label_tag = "bistable" if bistable else "monostable"
            label = f"CMIP6 {cls}, {label_tag} (n={len(sub)})"
            ax1.scatter(
                sub["velocity_share_pct"], sub["salinity_share_pct"],
                s=55, color=CLASS_COLORS[cls], alpha=0.75,
                edgecolor=edge_color, linewidth=edge_lw, zorder=4,
                label=label,
            )

    # Reanalyses: FIXED-size diamond markers (NOT scaled by ΔF)
    rean_scatter = []
    for label, dtot, v_pct, s_pct, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax1.scatter(
            v_pct, s_pct, s=110, c=color, edgecolor="black",
            linewidth=0.9, zorder=8, marker="D",
        )
        rean_scatter.append((label, v_pct, s_pct, color))

    # Collect all labels in one text list so adjust_text can avoid overlap
    # globally across CMIP6 models + reanalyses.
    texts = []
    for _, row in cmip6.iterrows():
        name = (
            row["model"]
            .replace("-CM6-1", "")
            .replace("-0-LL", "")
            .replace("-GC31-LL", "")
            .replace("-ESM1-2-", "-")
        )
        texts.append(ax1.text(
            row["velocity_share_pct"], row["salinity_share_pct"],
            name, fontsize=4.8, color="0.25", zorder=5,
        ))
    for label, v_pct, s_pct, color in rean_scatter:
        texts.append(ax1.text(
            v_pct, s_pct, label,
            fontsize=7, fontweight="bold", color=color, zorder=9,
        ))

    # Non-overlapping layout — draws thin arrows back to each point.
    # Limit repulsion to keep labels within the panel so the saved
    # bounding box doesn't get bloated into a near-square shape.
    adjust_text(
        texts, ax=ax1,
        expand=(1.05, 1.15),
        force_text=(0.3, 0.4),
        arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.35},
    )

    ax1.set_xlabel(r"Velocity share: $100 \cdot \Delta F_v / \Delta F_\mathrm{total}$ (%)")
    ax1.set_ylabel(r"Salinity share: $100 \cdot \Delta F_s / \Delta F_\mathrm{total}$ (%)")
    ax1.set_title("(a) Mechanism classification plane", fontweight="bold")
    ax1.set_xlim(-80, 220)
    ax1.set_ylim(-150, 220)
    ax1.legend(loc="lower left", fontsize=5.5, frameon=False)

    # ── Panel (b): velocity-share histogram across weakening models ──
    weakening = cmip6[cmip6["class"] != "increasing"]
    bins = np.arange(-80, 201, 15)
    ax2.hist(weakening["velocity_share_pct"], bins=bins, color="0.65",
             edgecolor="0.2", linewidth=0.4, alpha=0.8,
             label=f"CMIP6 weakening (n={len(weakening)})")

    for label, _, v_pct, _, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax2.axvline(v_pct, color=color, lw=2.0, zorder=5,
                    label=f"{label}: {v_pct:+.0f}%")

    ax2.axvline(60, color="#E69F00", lw=0.6, ls=":", alpha=0.7)
    ax2.axvline(40, color="#56B4E9", lw=0.6, ls=":", alpha=0.7)
    ax2.set_xlabel("Velocity share (%)")
    ax2.set_ylabel("Number of CMIP6 models")
    ax2.set_title("(b) Distribution of CMIP6 velocity shares", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=5.8, frameon=False)

    fig.tight_layout()
    # bbox_inches=None: preserve the intended 10.5 x 4 landscape aspect.
    # adjustText leader lines that escape the axes will be clipped rather
    # than bloating the canvas vertically.
    save_publication_figure(fig, args.output, bbox_inches=None)


if __name__ == "__main__":
    main()
