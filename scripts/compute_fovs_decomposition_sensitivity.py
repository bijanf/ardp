#!/usr/bin/env python3
"""Sensitivity of the Δv/Δs reanalysis decomposition to period choice.

Vary the early and late period windows and see whether the dominant
mechanism (v vs s) is robust. If ORAS5 is 88% v-dominant for one period
choice but flips to s-dominant for another, our headline finding is
fragile. If v vs s stays stable, we have confidence.

For each reanalysis and a grid of (early_end, late_start) choices,
compute the mechanism shares.

Reads:   data/oras5/, data/glorys12/, data/ecco/ (via
         compute_fovs_decomposition.py helpers)
Writes:  data/results/fovs_decomposition_sensitivity.csv
         figures/paper2/figS2_decomposition_sensitivity.{png,pdf}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.constants import S0
from ardp.physics.fovs_decomposition import decompose_fovs_trend
from ardp.viz.style import apply_nature_style, save_publication_figure
from compute_fovs_decomposition import (  # noqa: E402
    _ecco_period_mean,
    _glorys12_period_mean,
    _oras5_period_mean,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")


def _compute_one(product: str, early: tuple[int, int],
                 late: tuple[int, int]) -> dict | None:
    try:
        if product == "oras5":
            v1, s1, grid = _oras5_period_mean(Path("data/oras5"), early)
            v2, s2, _ = _oras5_period_mean(Path("data/oras5"), late)
        elif product == "glorys12":
            v1, s1, grid = _glorys12_period_mean(Path("data/glorys12"), early)
            v2, s2, _ = _glorys12_period_mean(Path("data/glorys12"), late)
        elif product == "ecco":
            v1, s1, grid = _ecco_period_mean(Path("data/ecco"), early)
            v2, s2, _ = _ecco_period_mean(Path("data/ecco"), late)
        else:
            return None
    except Exception as e:
        log.warning(f"  {product} ({early[0]}-{early[1]} vs {late[0]}-{late[1]}): {e}")
        return None

    result = decompose_fovs_trend(
        v1, s1, v2, s2,
        e1t_atl=grid["e1t_atl"], e3t=grid["e3t"], s0=S0,
    )
    dtot = result["delta_total"]
    return {
        "product": product,
        "early_start": early[0], "early_end": early[1],
        "late_start": late[0], "late_end": late[1],
        "delta_total": dtot, "delta_v": result["delta_v"],
        "delta_s": result["delta_s"], "delta_cross": result["delta_cross"],
        "velocity_share_pct": 100 * result["delta_v"] / dtot if abs(dtot) > 1e-6 else np.nan,
        "salinity_share_pct": 100 * result["delta_s"] / dtot if abs(dtot) > 1e-6 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", nargs="+",
                        default=["oras5", "glorys12"])
    args = parser.parse_args()

    apply_nature_style()

    # Sensitivity grid: early ending year (9 cases from 2001 to 2009),
    # late starting year (9 cases from 2011 to 2019). Early always starts
    # at 1993 (GLORYS12 data range); late always ends at 2025.
    # For ORAS5 we could use 1958 start but keep 1993 for comparability.
    early_ends = [2001, 2003, 2005, 2007, 2009]
    late_starts = [2011, 2013, 2015, 2017, 2019]

    rows = []
    for product in args.products:
        log.info(f"=== {product.upper()} sensitivity grid ===")
        for e_end in early_ends:
            for l_start in late_starts:
                if e_end >= l_start - 1:
                    continue
                early = (1993, e_end)
                late = (l_start, 2025 if product != "ecco" else 2017)
                row = _compute_one(product, early, late)
                if row is None:
                    continue
                log.info(
                    f"  early {early[0]}-{early[1]}  late {late[0]}-{late[1]}  "
                    f"ΔF={row['delta_total'] * 1000:+.1f} mSv  "
                    f"v:{row['velocity_share_pct']:+.0f}%  "
                    f"s:{row['salinity_share_pct']:+.0f}%"
                )
                rows.append(row)

    if not rows:
        log.error("No rows computed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "fovs_decomposition_sensitivity.csv", index=False)
    log.info(f"Saved: {RESULTS_DIR / 'fovs_decomposition_sensitivity.csv'}")

    # Summary per product
    for p in args.products:
        sub = df[df["product"] == p]
        valid = sub["velocity_share_pct"].dropna()
        if len(valid) == 0:
            continue
        log.info(
            f"{p}: n={len(valid)} period-pair draws, "
            f"v_share median={np.median(valid):+.0f}%, "
            f"p5-p95 = [{np.percentile(valid, 5):+.0f}%, {np.percentile(valid, 95):+.0f}%]"
        )

    # ── Figure S2: sensitivity summary ──
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    product_colors = {"oras5": "#1f77b4", "glorys12": "#2ca02c",
                      "soda": "#e377c2", "ecco": "#d62728"}
    labels = {"oras5": "ORAS5", "glorys12": "GLORYS12V1",
              "soda": "SODA3.15.2", "ecco": "ECCO-V4r4"}

    for i, p in enumerate(args.products):
        sub = df[df["product"] == p]
        valid = sub.dropna(subset=["velocity_share_pct"])
        if len(valid) == 0:
            continue
        x = valid["velocity_share_pct"].values
        y = np.full(len(x), i, dtype=float) + np.random.default_rng(42).normal(0, 0.1, len(x))
        ax.scatter(x, y, color=product_colors[p], alpha=0.6, s=35,
                   edgecolor="0.2", linewidth=0.3, zorder=4,
                   label=f"{labels[p]} (n={len(x)})")
        ax.axhline(i, color="0.8", lw=0.4, zorder=1)

    ax.axvline(60, color="#56B4E9", ls=":", lw=0.6, alpha=0.7)
    ax.axvline(100 - 60, color="#E69F00", ls=":", lw=0.6, alpha=0.7)
    ax.text(62, -0.5, "s-dominant >60%",
            fontsize=6, color="#56B4E9", style="italic")
    ax.text(38, -0.5, "v-dominant >60%",
            fontsize=6, color="#E69F00", style="italic", ha="right")

    ax.set_yticks(range(len(args.products)))
    ax.set_yticklabels([labels[p] for p in args.products], fontsize=7)
    ax.set_xlabel("Velocity share (%) across period choices")
    ax.set_title("Fig. S2 — Sensitivity to period choice",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    ax.set_xlim(-100, 200)
    fig.tight_layout()
    save_publication_figure(fig, Path("figures/paper2/figS2_decomposition_sensitivity"))


if __name__ == "__main__":
    main()
