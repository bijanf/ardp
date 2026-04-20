#!/usr/bin/env python3
"""Figure 3 (new): mechanism tie-breaker across reanalyses + CMIP6.

Two panels:
  (a) Scatter of (velocity share, salinity share) across all 4 reanalyses
      and all forced-weakening CMIP6 models. A clean x+y = 100% line shows
      the "no-cross-term" diagonal. Points above the diagonal are
      salinity-dominant; below are velocity-dominant.
  (b) Distribution of velocity-share % in CMIP6 weakening models,
      with reanalysis values shown as vertical lines.

The story: ORAS5 and GLORYS12 lie at opposite corners of the CMIP6 cloud.
The split among CMIP6 forced-weakening models (6 s-dominant, 4 v-dominant,
1 mixed) shows that the reanalysis disagreement mirrors real physical
diversity in CMIP6 — the mechanism is not a universal property of AMOC
weakening.

Reads:
  data/results/fovs_decomposition_cmip6_summary.csv
  data/results/fovs_decomposition_{oras5,glorys12,soda,ecco}.nc  (optional)

Outputs: figures/paper2/fig3_tiebreaker.{png,pdf}
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


def _load_reanalysis_shares(path: Path) -> tuple[float, float, float, bool] | None:
    """Return (|ΔF_mSv|, v_pct, s_pct, has_trend).

    has_trend=False when |ΔF_total| < 10 mSv (ratios unreliable).
    """
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    dtot = float(ds.attrs["delta_total_Sv"])
    dv = float(ds.attrs["delta_v_Sv"])
    dsal = float(ds.attrs["delta_s_Sv"])
    ds.close()
    has_trend = abs(dtot) >= 0.010  # 10 mSv threshold
    v_pct = 100 * dv / dtot if has_trend else np.nan
    s_pct = 100 * dsal / dtot if has_trend else np.nan
    return dtot * 1000, v_pct, s_pct, has_trend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/paper2/fig3_tiebreaker"),
    )
    args = parser.parse_args()

    apply_nature_style()

    # CMIP6 summary
    cmip6 = pd.read_csv(args.results_dir / "fovs_decomposition_cmip6_summary.csv")
    cmip6["weakening"] = cmip6["delta_total"] < -0.01
    forced = cmip6[cmip6["weakening"]].copy()

    # Reanalyses
    reanalysis_points = []
    for label, fname, color in REANALYSIS_PRODUCTS:
        r = _load_reanalysis_shares(args.results_dir / fname)
        if r is not None:
            dtot_mSv, v_pct, s_pct, has_trend = r
            reanalysis_points.append((label, dtot_mSv, v_pct, s_pct, has_trend, color))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # ── Panel (a): v-share vs s-share scatter ──
    # CMIP6 weakening models
    x_cmip = forced["velocity_share_pct"].values
    y_cmip = forced["salinity_share_pct"].values
    sizes = np.abs(forced["delta_total"]) * 800  # bubble scaled by |ΔF|
    sc = ax1.scatter(
        x_cmip, y_cmip, s=sizes, c="0.55", alpha=0.55, edgecolor="0.2",
        linewidth=0.4, zorder=4, label="CMIP6 weakening models",
    )
    for _, row in forced.iterrows():
        ax1.annotate(
            row["model"].replace("-CM6-1", "").replace("-0-LL", "").replace("-GC31-LL", "")[:11],
            xy=(row["velocity_share_pct"], row["salinity_share_pct"]),
            xytext=(2, 2), textcoords="offset points",
            fontsize=4.5, color="0.3", zorder=5,
        )

    # Reanalyses as coloured markers (skip no-trend products from this panel)
    for label, dtot, v_pct, s_pct, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax1.scatter(
            v_pct, s_pct, s=abs(dtot) * 8, c=color, edgecolor="black",
            linewidth=0.8, zorder=8, marker="D",
        )
        ax1.annotate(
            label, xy=(v_pct, s_pct), xytext=(6, 0), textcoords="offset points",
            fontsize=7, fontweight="bold", color=color, zorder=9, va="center",
        )

    # Physical anchor lines
    ax1.plot([0, 100], [100, 0], color="0.4", lw=0.7, ls="--", zorder=2,
             label="v + s = 100% (no cross term)")
    ax1.axhline(60, color="#56B4E9", lw=0.6, ls=":", alpha=0.7)
    ax1.axvline(60, color="#E69F00", lw=0.6, ls=":", alpha=0.7)
    ax1.text(30, 90, "salinity-dominant", color="#56B4E9", fontsize=6, style="italic")
    ax1.text(70, 10, "velocity-dominant", color="#E69F00", fontsize=6, style="italic",
             ha="left")

    ax1.set_xlabel(r"Velocity share: $100 \cdot \Delta F_v / \Delta F_{\mathrm{total}}$  (%)")
    ax1.set_ylabel(r"Salinity share: $100 \cdot \Delta F_s / \Delta F_{\mathrm{total}}$  (%)")
    ax1.set_title("(a) Mechanism classification", fontweight="bold")
    ax1.set_xlim(-60, 200)
    ax1.set_ylim(-60, 200)
    ax1.legend(loc="upper right", fontsize=5.5, frameon=False)

    # ── Panel (b): histogram of velocity-share across CMIP6 weakening + reanalyses ──
    bins = np.arange(-60, 201, 15)
    ax2.hist(x_cmip, bins=bins, color="0.65", edgecolor="0.2", linewidth=0.4,
             alpha=0.8, label=f"CMIP6 weakening (n={len(forced)})")
    for label, _, v_pct, _, has_trend, color in reanalysis_points:
        if not has_trend:
            continue
        ax2.axvline(v_pct, color=color, lw=2, zorder=5,
                    label=f"{label}: {v_pct:+.0f}%")
    ax2.axvline(60, color="#E69F00", lw=0.6, ls=":", alpha=0.7)
    ax2.text(65, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] else 3,
             "v-dominant >60%", fontsize=5.5, color="#E69F00", va="top")
    ax2.set_xlabel("Velocity share (%)")
    ax2.set_ylabel("Number of CMIP6 models")
    ax2.set_title("(b) Velocity-share distribution", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=5.8, frameon=False)

    # Summary line
    if len(forced) > 0:
        summary = (
            f"CMIP6 weakening models: "
            f"{(x_cmip > 60).sum()} v-dom, "
            f"{(y_cmip > 60).sum()} s-dom, "
            f"{((x_cmip <= 60) & (y_cmip <= 60)).sum()} mixed"
        )
        fig.text(0.5, -0.02, summary, ha="center", fontsize=7, color="0.3")

    fig.tight_layout()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
