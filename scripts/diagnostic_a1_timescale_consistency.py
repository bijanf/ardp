#!/usr/bin/env python3
"""Phase-A diagnostic A1 — decadal vs centennial CMIP6 mechanism class.

Question: in CMIP6 models that show forced AMOC weakening, is the
mechanism class (velocity-dominant vs salinity-dominant) of the
decadal historical+early-ssp585 trend (1993-2005 vs 2013-2025) the
same as the mechanism class of the centennial forced trend (1950-1980
vs 2080-2100)?

If YES: the kinematic mapping from reanalysis decadal mechanism class
to projected CMIP6 centennial AMOC weakening is physically defensible.
If NO: the mapping is timescale-dependent and the manuscript needs
restructuring.

Reads:
  data/results/fovs_decomposition_cmip6_summary.csv      (centennial)
  data/results/fovs_decomposition_cmip6_decadal.csv      (decadal)

Outputs:
  figures/paper2/diagA1_timescale.{png,pdf}
  data/results/diagA1_timescale_consistency.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from ardp.viz.style import apply_nature_style, save_publication_figure


def _classify(row):
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centennial", type=Path,
                        default=Path("data/results/fovs_decomposition_cmip6_summary.csv"))
    parser.add_argument("--decadal", type=Path,
                        default=Path("data/results/fovs_decomposition_cmip6_decadal.csv"))
    parser.add_argument("--out-fig", type=Path,
                        default=Path("figures/paper2/diagA1_timescale"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("data/results/diagA1_timescale_consistency.csv"))
    args = parser.parse_args()

    apply_nature_style()

    cen = pd.read_csv(args.centennial)
    dec = pd.read_csv(args.decadal)
    cen["class_centennial"] = cen.apply(_classify, axis=1)
    dec["class_decadal"] = dec.apply(_classify, axis=1)

    cen_c = cen.rename(columns={
        "delta_total": "delta_total_cen",
        "velocity_share_pct": "fv_cen",
        "salinity_share_pct": "fs_cen",
    })[["model", "delta_total_cen", "fv_cen", "fs_cen", "class_centennial"]]
    dec_c = dec.rename(columns={
        "delta_total": "delta_total_dec",
        "velocity_share_pct": "fv_dec",
        "salinity_share_pct": "fs_dec",
    })[["model", "delta_total_dec", "fv_dec", "fs_dec", "class_decadal"]]
    joined = pd.merge(cen_c, dec_c, on="model", how="inner")
    print(f"\nJoined ensemble: {len(joined)} models present in both CSVs")
    print(f"  Centennial-only models: "
          f"{set(cen['model']) - set(dec['model'])}")
    print(f"  Decadal-only models:    "
          f"{set(dec['model']) - set(cen['model'])}")

    # Class agreement: only meaningful for models that are weakening
    # in BOTH windows (otherwise the class is "increasing" and not
    # physically comparable to the velocity-vs-salinity dichotomy).
    weak_both = joined[
        (joined["class_centennial"] != "increasing") &
        (joined["class_decadal"] != "increasing")
    ].copy()
    weak_both["match"] = (
        weak_both["class_centennial"] == weak_both["class_decadal"]
    )
    weak_both["match_or_mixed"] = (
        weak_both["match"] |
        (weak_both["class_centennial"] == "mixed") |
        (weak_both["class_decadal"] == "mixed")
    )
    n_weak = len(weak_both)
    n_match = int(weak_both["match"].sum())
    n_match_lenient = int(weak_both["match_or_mixed"].sum())

    print(f"\nWeakening in both windows: n={n_weak}")
    print(f"  Strict class match (v=v, s=s):     {n_match}/{n_weak}"
          f" = {100*n_match/n_weak:.0f}%")
    print(f"  Lenient (allow 'mixed' bridge):    {n_match_lenient}/{n_weak}"
          f" = {100*n_match_lenient/n_weak:.0f}%")

    # Continuous correlation across all joined models
    rho, p = spearmanr(joined["fv_cen"], joined["fv_dec"])
    print(f"\nSpearman ρ(fv_cen, fv_dec) = {rho:+.3f}  (p = {p:.3g})")
    rho_w, p_w = spearmanr(weak_both["fv_cen"], weak_both["fv_dec"])
    print(f"Spearman ρ on weak-only ensemble    = {rho_w:+.3f}  (p = {p_w:.3g})")

    # Per-model table
    print("\nPer-model breakdown:")
    print(joined[["model", "fv_dec", "fv_cen", "class_decadal",
                  "class_centennial"]].to_string(index=False))

    joined.to_csv(args.out_csv, index=False)
    print(f"\nSaved table: {args.out_csv}")

    # ── Plot ──
    cls_color = {"v-dominant": "#E69F00", "s-dominant": "#56B4E9",
                 "mixed": "#009E73", "increasing": "0.6"}

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    # Diagonal + classification thresholds
    ax.plot([-200, 300], [-200, 300], color="0.6", lw=0.6, ls="--",
            zorder=1, label="diagonal")
    ax.axhline(60, color="#E69F00", lw=0.4, ls=":", alpha=0.6, zorder=1)
    ax.axhline(40, color="#56B4E9", lw=0.4, ls=":", alpha=0.6, zorder=1)
    ax.axvline(60, color="#E69F00", lw=0.4, ls=":", alpha=0.6, zorder=1)
    ax.axvline(40, color="#56B4E9", lw=0.4, ls=":", alpha=0.6, zorder=1)

    for _, row in joined.iterrows():
        # Colour by centennial class (the manuscript uses the
        # centennial class as the predictor)
        c = cls_color.get(row["class_centennial"], "0.4")
        ax.scatter(row["fv_cen"], row["fv_dec"], s=80, c=c,
                   edgecolor="0.2", linewidth=0.6, zorder=4)
        ax.annotate(
            row["model"]
                .replace("-CM6-1", "")
                .replace("-0-LL", "")
                .replace("-GC31-LL", "")
                .replace("-ESM1-2-", "-"),
            xy=(row["fv_cen"], row["fv_dec"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6.5, color="0.25", zorder=5,
        )

    ax.set_xlabel(r"Centennial $f_v$ (1950-1980 vs 2080-2100, %)")
    ax.set_ylabel(r"Decadal $f_v$ (1993-2005 vs 2013-2025, %)")
    ax.set_xlim(-50, 220)
    ax.set_ylim(-50, 220)

    # Annotate consistency stats
    ax.text(0.03, 0.97,
            f"Strict class match: {n_match}/{n_weak} = "
            f"{100*n_match/n_weak:.0f}%\n"
            f"Spearman ρ (weak): {rho_w:+.2f}  (p={p_w:.2g})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "0.7", "alpha": 0.85})

    # Legend for class colours
    handles = [plt.Line2D([], [], marker="o", linestyle="",
                          color=cls_color[k], markersize=8,
                          markeredgecolor="0.2",
                          label=f"{k} (centennial)")
               for k in ("v-dominant", "s-dominant", "mixed", "increasing")]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              frameon=False)

    fig.tight_layout()
    save_publication_figure(fig, args.out_fig)


if __name__ == "__main__":
    main()
