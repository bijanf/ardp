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

    # CMIP6 circles, one per class for a clean legend
    for cls in CLASS_ORDER:
        sub = cmip6[cmip6["class"] == cls]
        if len(sub) == 0:
            continue
        ax1.scatter(
            sub["velocity_share_pct"], sub["salinity_share_pct"],
            s=45, color=CLASS_COLORS[cls], alpha=0.7,
            edgecolor="0.2", linewidth=0.5, zorder=4,
            label=f"CMIP6 {cls} (n={len(sub)})",
        )

    # CMIP6 labels — lightweight, only for readable-ratio models (|ΔF| > 20 mSv)
    for _, row in cmip6.iterrows():
        if abs(row["delta_total"]) < 0.020:
            continue
        name = row["model"].replace("-CM6-1", "").replace("-0-LL", "").replace("-GC31-LL", "")[:10]
        ax1.annotate(
            name, xy=(row["velocity_share_pct"], row["salinity_share_pct"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=4.8, color="0.3", zorder=5,
        )

    # Reanalyses: FIXED-size diamond markers (NOT scaled by ΔF)
    for label, dtot, v_pct, s_pct, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax1.scatter(
            v_pct, s_pct, s=110, c=color, edgecolor="black",
            linewidth=0.9, zorder=8, marker="D",
        )
        ax1.annotate(
            label, xy=(v_pct, s_pct), xytext=(8, -2),
            textcoords="offset points",
            fontsize=7, fontweight="bold", color=color, zorder=9,
            va="center",
        )

    # Zone guides
    ax1.text(100, 80, "s-dominant", color="#56B4E9",
             fontsize=6, style="italic", ha="right", va="top")
    ax1.text(100, 10, "v-dominant", color="#E69F00",
             fontsize=6, style="italic", ha="right", va="top")

    ax1.set_xlabel(r"Velocity share: $100 \cdot \Delta F_v / \Delta F_\mathrm{total}$ (%)")
    ax1.set_ylabel(r"Salinity share: $100 \cdot \Delta F_s / \Delta F_\mathrm{total}$ (%)")
    ax1.set_title("(a) Mechanism classification plane", fontweight="bold")
    ax1.set_xlim(-80, 220)
    ax1.set_ylim(-150, 220)
    ax1.legend(loc="lower left", fontsize=5.5, frameon=False)

    # Clarifying note bottom-right of panel (a)
    ax1.text(0.98, 0.02,
             "Symbol size fixed — not scaled by ΔF$_\\mathrm{total}$.\n"
             "Reanalyses with |ΔF$_\\mathrm{total}$| < 10 mSv excluded.",
             transform=ax1.transAxes, fontsize=5.0, color="0.4",
             ha="right", va="bottom", style="italic")

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

    # Summary line under both panels
    cls_counts = {c: int((weakening["class"] == c).sum())
                  for c in ("v-dominant", "s-dominant", "mixed")}
    fig.text(0.5, -0.03,
             f"CMIP6 forced-weakening ensemble (n={len(weakening)}):  "
             f"{cls_counts['v-dominant']} v-dominant, "
             f"{cls_counts['s-dominant']} s-dominant, "
             f"{cls_counts['mixed']} mixed.",
             ha="center", fontsize=7, color="0.3")

    fig.tight_layout()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
