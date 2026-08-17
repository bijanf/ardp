#!/usr/bin/env python3
"""Joint sensitivity of the v-vs-s mechanism gap to threshold and |ΔF| floor.

Addresses the v3 reviewer concerns:
  BLOCKER 1 -- the manuscript cites a threshold sensitivity test that does
              not actually sweep thresholds (Fig 4(b) sweeps windows at
              fixed 60%).
  BLOCKER 2 -- f_v = 1006%, 193%, 139% in some CMIP6 models reflects sign
              cancellation at small |ΔF_total|; reporting a percentage
              share of a near-zero quantity is misleading.

For each (threshold, |ΔF| floor) on the discretised grid:
  - drop CMIP6 models with |ΔF_total| < floor (kinematic-share unreliable)
  - drop "increasing" models (delta_total > -1 mSv -- AMOC monotonic with
    F_ovS sign)
  - classify the rest as v-dominant / s-dominant / mixed at the threshold
  - compute median 2100 AMOC weakening per class
  - record n_v, n_s, gap = median_s - median_v.

Outputs:
  data/results/diagA4_joint_sensitivity.csv     -- full grid table
  figures/paper2/diagA4_joint_sensitivity.{pdf,png}  -- gap vs threshold,
                                                       coloured by floor
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ardp.viz.style import apply_nature_style, save_publication_figure

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "data" / "results"
FIG = REPO / "figures" / "paper2" / "diagA4_joint_sensitivity"
CSV_OUT = RESULTS / "diagA4_joint_sensitivity.csv"

THRESHOLDS_PCT = [40, 45, 50, 55, 60, 65, 70, 75]
FLOOR_MSV = [0, 10, 20, 30, 40, 50]


def _pct_weakening(model: str, amoc_models, amoc) -> float:
    if model not in amoc_models:
        return float("nan")
    y = amoc[f"{model}_years"]
    a = amoc[f"{model}_amoc"]
    base = float(np.nanmean(a[(y >= 1950) & (y <= 1980)]))
    end = float(np.nanmean(a[(y >= 2081) & (y <= 2100)]))
    if not (np.isfinite(base) and np.isfinite(end)) or base <= 0:
        return float("nan")
    return 100 * (1 - end / base)


def _classify(row, threshold_pct: float) -> str:
    if row["velocity_share_pct"] > threshold_pct:
        return "v"
    if row["salinity_share_pct"] > threshold_pct:
        return "s"
    return "mixed"


def main() -> None:
    df = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")
    df["abs_delta_msv"] = df["delta_total"].abs() * 1e3

    amoc_npz = np.load(RESULTS / "yearly_amoc26n_cmip6.npz",
                       allow_pickle=True)
    amoc_models = [str(m) for m in amoc_npz["models"]]
    df["weakening_pct"] = df["model"].apply(
        lambda m: _pct_weakening(m, amoc_models, amoc_npz)
    )

    # Hard "increasing" filter -- AMOC path actually goes the wrong way.
    weakening = df[df["delta_total"] < -0.001].copy()
    print(f"Weakening models in F_ovS sense (n={len(weakening)}):")
    print(weakening[["model", "abs_delta_msv", "velocity_share_pct",
                     "salinity_share_pct", "weakening_pct"]]
          .sort_values("abs_delta_msv").to_string(index=False))

    rows: list[dict] = []
    for floor in FLOOR_MSV:
        sub = weakening[weakening["abs_delta_msv"] >= floor].copy()
        for thr in THRESHOLDS_PCT:
            sub2 = sub.copy()
            sub2["cls"] = sub2.apply(lambda r: _classify(r, thr), axis=1)
            v = sub2[sub2["cls"] == "v"]["weakening_pct"].dropna()
            s = sub2[sub2["cls"] == "s"]["weakening_pct"].dropna()
            rows.append(dict(
                floor_msv=floor, threshold_pct=thr,
                n_total=len(sub2), n_v=len(v), n_s=len(s),
                median_v=float(v.median()) if len(v) else float("nan"),
                median_s=float(s.median()) if len(s) else float("nan"),
                gap_pp=(float(s.median()) - float(v.median()))
                       if (len(v) and len(s)) else float("nan"),
            ))
    grid = pd.DataFrame(rows)
    grid.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT} ({len(grid)} rows)")
    print("\nv-vs-s gap (percentage points) at floor=30 mSv:")
    print(grid[grid["floor_msv"] == 30]
          [["threshold_pct", "n_v", "n_s", "median_v", "median_s",
            "gap_pp"]].to_string(index=False))

    # Render: gap vs threshold, one line per floor
    apply_nature_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                      gridspec_kw={"wspace": 0.35})
    cmap = plt.get_cmap("viridis")
    for k, floor in enumerate(FLOOR_MSV):
        sub = grid[grid["floor_msv"] == floor]
        c = cmap(k / max(1, len(FLOOR_MSV) - 1))
        ax_a.plot(sub["threshold_pct"], sub["gap_pp"],
                  marker="o", lw=1.6, color=c,
                  label=rf"floor = {floor} mSv")
        ax_b.plot(sub["threshold_pct"], sub["n_v"] + sub["n_s"],
                  marker="s", lw=1.2, color=c,
                  label=rf"floor = {floor} mSv")
    ax_a.axhline(0, color="0.5", lw=0.6)
    ax_a.axvline(60, color="#CC3333", lw=0.6, ls="--", alpha=0.6)
    ax_a.set_xlabel(r"Classification threshold on $f_v$ or $f_s$  (%)")
    ax_a.set_ylabel(r"Gap = median$_s -$ median$_v$  (pp)")
    ax_a.text(0.02, 0.97, "a   v-vs-s AMOC-weakening gap",
              transform=ax_a.transAxes, fontweight="bold",
              fontsize=10, va="top", ha="left")
    ax_a.legend(fontsize=6.5, frameon=False, ncol=2,
                handlelength=1.5, handletextpad=0.4, columnspacing=0.7,
                loc="lower right")

    ax_b.axvline(60, color="#CC3333", lw=0.6, ls="--", alpha=0.6)
    ax_b.set_xlabel(r"Classification threshold on $f_v$ or $f_s$  (%)")
    ax_b.set_ylabel(r"$n$ classified (v + s)")
    ax_b.text(0.02, 0.97, "b   sample size used",
              transform=ax_b.transAxes, fontweight="bold",
              fontsize=10, va="top", ha="left")
    ax_b.legend(fontsize=6.5, frameon=False, ncol=2,
                handlelength=1.5, handletextpad=0.4, columnspacing=0.7)

    fig.tight_layout()
    save_publication_figure(fig, FIG)


if __name__ == "__main__":
    main()
