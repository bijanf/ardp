#!/usr/bin/env python3
"""Paper 2 SMILE supplementary figure.

Plots the velocity-share vs salinity-share of all available
MPI-ESM1-2-LR Grand Ensemble members against the multi-model CMIP6
weakening ensemble. If the SMILE members tightly cluster in one
mechanism quadrant, the binary classification is structural (and the
small-N criticism of the headline 10-percentage-point gap is
defused). If they scatter across the plane, the classification is
partly aliased onto internal variability and the headline must be
softened further.

Reads:
  data/results/fovs_decomposition_smile_esgf.csv (preferred — 50 members)
  data/results/fovs_decomposition_smile_mpi_esm1_2_lr.csv (Pangeo 10)
  data/results/fovs_decomposition_cmip6_summary.csv (multi-model bg)

Outputs: figures/paper2/diagSMILE.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ardp.viz.style import apply_nature_style, save_publication_figure


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smile-esgf", type=Path,
                        default=Path("data/results/fovs_decomposition_smile_esgf.csv"))
    parser.add_argument("--smile-pangeo", type=Path,
                        default=Path("data/results/fovs_decomposition_smile_mpi_esm1_2_lr.csv"))
    parser.add_argument("--cmip6", type=Path,
                        default=Path("data/results/fovs_decomposition_cmip6_summary.csv"))
    parser.add_argument("--out-fig", type=Path,
                        default=Path("figures/paper2/diagSMILE"))
    args = parser.parse_args()

    apply_nature_style()

    smile_frames = []
    if args.smile_esgf.exists():
        smile_frames.append(pd.read_csv(args.smile_esgf).assign(source="ESGF"))
    if args.smile_pangeo.exists():
        smile_frames.append(pd.read_csv(args.smile_pangeo).assign(source="Pangeo"))
    if not smile_frames:
        raise SystemExit(
            "No SMILE CSVs found. Run scripts/compute_smile_esgf.py "
            "or scripts/compute_cmip6_smile_decomposition.py first."
        )
    smile = pd.concat(smile_frames, ignore_index=True)
    # If a member appears in both ESGF and Pangeo, prefer ESGF.
    smile = smile.drop_duplicates(subset="member_id", keep="first")
    smile["class"] = smile.apply(_classify, axis=1)
    print(f"\nSMILE members: {len(smile)}")
    print(smile["class"].value_counts())

    cmip6 = pd.read_csv(args.cmip6) if args.cmip6.exists() else None
    if cmip6 is not None:
        cmip6["class"] = cmip6.apply(_classify, axis=1)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Backdrop: multi-model CMIP6 ensemble (greyscale, faded)
    if cmip6 is not None:
        for cls in ("v-dominant", "s-dominant", "mixed", "increasing"):
            sub = cmip6[cmip6["class"] == cls]
            if len(sub) == 0:
                continue
            ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                       s=70, color=CLASS_COLORS[cls], alpha=0.30,
                       edgecolor="0.4", linewidth=0.4, zorder=3,
                       label=f"CMIP6 {cls} ({len(sub)})")

    # SMILE members — coloured by class
    for cls in ("v-dominant", "s-dominant", "mixed", "increasing"):
        sub = smile[smile["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                   s=40, marker="x", color=CLASS_COLORS[cls],
                   linewidth=1.5, zorder=5,
                   label=f"MPI-ESM1-2-LR SMILE {cls} ({len(sub)})")

    # SMILE ensemble mean (large diamond)
    if len(smile):
        ax.scatter([smile["velocity_share_pct"].mean()],
                   [smile["salinity_share_pct"].mean()],
                   s=180, marker="D", c="black", edgecolor="white",
                   linewidth=1.5, zorder=8,
                   label=f"SMILE mean ({len(smile)} members)")

    # Class boundaries + diagonal
    ax.plot([0, 100], [100, 0], color="0.6", lw=0.6, ls="--", zorder=1)
    ax.axvline(60, color="#E69F00", lw=0.4, ls=":", alpha=0.6)
    ax.axhline(60, color="#56B4E9", lw=0.4, ls=":", alpha=0.6)

    # Annotations: SMILE summary
    n_smile = len(smile)
    weak_smile = smile[smile["delta_total"] < -0.01]
    n_weak = len(weak_smile)
    n_v = int(((weak_smile["velocity_share_pct"] > 60).sum()))
    n_s = int(((weak_smile["salinity_share_pct"] > 60).sum()))
    fv_mean = smile["velocity_share_pct"].mean()
    fv_sd = smile["velocity_share_pct"].std()
    text = (f"MPI-ESM1-2-LR Grand Ensemble  (n={n_smile})\n"
            f"Weakening: {n_weak}/{n_smile}    "
            f"v-dominant: {n_v}/{n_weak}    s-dominant: {n_s}/{n_weak}\n"
            f"$f_v$ ensemble: mean = {fv_mean:+.0f}%   $\\sigma$ = {fv_sd:.0f}%")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.6", "alpha": 0.92})

    ax.set_xlabel(r"Velocity share $f_v$  (%)")
    ax.set_ylabel(r"Salinity share $f_s$  (%)")
    ax.set_xlim(-80, 220)
    ax.set_ylim(-150, 220)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=2,
              handlelength=1.3, handletextpad=0.5)

    fig.tight_layout()
    save_publication_figure(fig, args.out_fig)


if __name__ == "__main__":
    main()
