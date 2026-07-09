#!/usr/bin/env python3
"""Phase-A diagnostic A3 — continuous f_v vs AMOC weakening %.

The editorial critique attacks the binary 60% classification (n=5 vs
n=6) as statistically perilous. The fix: replace the binary threshold
with a continuous correlation across all 12 weakening models — if the
relationship between velocity-share and projected AMOC weakening is
significant on a continuous scale, the 10-pp gap claim is robust to
the choice of cutoff.

Reads:
  data/results/fovs_decomposition_cmip6_summary.csv
  data/results/yearly_amoc26n_cmip6.npz

Outputs:
  figures/paper2/diagA3_continuous.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ardp.viz.style import apply_nature_style, save_publication_figure


def _classify(row):
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def _pct_weakening(model: str, amoc_models, amoc):
    if model not in amoc_models:
        return np.nan
    y = amoc[f"{model}_years"]
    a = amoc[f"{model}_amoc"]
    base = float(np.nanmean(a[(y >= 1950) & (y <= 1980)]))
    end = float(np.nanmean(a[(y >= 2081) & (y <= 2100)]))
    if not (np.isfinite(base) and np.isfinite(end)) or base <= 0:
        return np.nan
    return 100 * (1 - end / base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path("data/results/fovs_decomposition_cmip6_summary.csv"))
    parser.add_argument("--amoc-npz", type=Path,
                        default=Path("data/results/yearly_amoc26n_cmip6.npz"))
    parser.add_argument("--out-fig", type=Path,
                        default=Path("figures/paper2/diagA3_continuous"))
    args = parser.parse_args()

    apply_nature_style()

    df = pd.read_csv(args.summary)
    df["class"] = df.apply(_classify, axis=1)
    amoc = np.load(args.amoc_npz, allow_pickle=True)
    amoc_models = [str(m) for m in amoc["models"]]
    df["weakening_pct"] = df["model"].apply(
        lambda m: _pct_weakening(m, amoc_models, amoc)
    )

    weakening = df[df["class"] != "increasing"].dropna(
        subset=["velocity_share_pct", "weakening_pct"]
    )
    print(f"\nWeakening ensemble: n={len(weakening)} models")

    rho, p_rho = stats.spearmanr(
        weakening["velocity_share_pct"],
        weakening["weakening_pct"],
    )
    r, p_r = stats.pearsonr(
        weakening["velocity_share_pct"],
        weakening["weakening_pct"],
    )
    res = stats.linregress(
        weakening["velocity_share_pct"],
        weakening["weakening_pct"],
    )
    slope = res.slope
    intercept = res.intercept
    R2 = res.rvalue ** 2
    p_lin = res.pvalue

    print(f"\nSpearman ρ(f_v, weakening %)  = {rho:+.3f}  (p = {p_rho:.3g})")
    print(f"Pearson  r(f_v, weakening %)  = {r:+.3f}    (p = {p_r:.3g})")
    print(f"Linear regression: slope = {slope:+.3f} %/% f_v, "
          f"intercept = {intercept:+.1f}%, R² = {R2:.3f}, p = {p_lin:.3g}")

    # Per-model table
    print("\nPer-model breakdown (sorted by f_v):")
    print(weakening[["model", "velocity_share_pct", "weakening_pct", "class"]]
          .sort_values("velocity_share_pct").to_string(index=False))

    # ── Plot ──
    # Clip the x-axis to a meaningful interpretation window; MRI-ESM2-0
    # at f_v ≈ -380% is reported in the caption rather than allowed to
    # blow the visual scale.
    XLIM = (-30, 130)
    weak_plot = weakening[(weakening["velocity_share_pct"] >= XLIM[0])
                          & (weakening["velocity_share_pct"] <= XLIM[1])]
    weak_offscale = weakening[(weakening["velocity_share_pct"] < XLIM[0])
                              | (weakening["velocity_share_pct"] > XLIM[1])]

    cls_color = {"v-dominant": "#E69F00", "s-dominant": "#56B4E9",
                 "mixed": "#009E73"}
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
    for cls, color in cls_color.items():
        sub = weak_plot[weak_plot["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["velocity_share_pct"], sub["weakening_pct"],
                   s=55, color=color, edgecolor="0.2", linewidth=0.5,
                   zorder=4, label=f"{cls} (n={len(sub)})")
    # Regression line on the FULL ensemble (so the slope is unchanged),
    # drawn only within the visible x-range
    xx = np.linspace(*XLIM, 50)
    ax.plot(xx, slope * xx + intercept, color="0.3", lw=1.2, zorder=3,
            label="OLS fit")
    ax.axvline(60, color="#E69F00", lw=0.5, ls=":", alpha=0.6)
    ax.axvline(40, color="#56B4E9", lw=0.5, ls=":", alpha=0.6)

    # Per-model labels with adjustText (collision-free)
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in weak_plot.iterrows():
            short = (row["model"]
                     .replace("-CM6-1", "")
                     .replace("-0-LL", "")
                     .replace("-GC31-LL", "")
                     .replace("-ESM1-2-", "-"))
            texts.append(ax.text(row["velocity_share_pct"],
                                 row["weakening_pct"],
                                 short, fontsize=5.5, color="0.25"))
        adjust_text(texts, ax=ax, expand=(1.4, 1.8),
                    only_move={"text": "xy", "static": "xy"},
                    arrowprops={"arrowstyle": "-", "color": "0.55",
                                "lw": 0.3})
    except ImportError:
        for _, row in weak_plot.iterrows():
            short = row["model"]
            ax.annotate(short,
                        xy=(row["velocity_share_pct"], row["weakening_pct"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=5.5, color="0.25")

    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"Velocity share $f_v$  (%)")
    ax.set_ylabel("AMOC weakening by 2100  (%)")
    # Legend in the corner opposite the data trend (top-right is mostly empty
    # because the regression slopes from upper-left to lower-right).
    ax.legend(loc="upper right", fontsize=6, frameon=False,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)
    # Footnote about off-scale models (in caption-equivalent position below
    # the x-axis; this is the ONLY allowed on-canvas text in this figure)
    if not weak_offscale.empty:
        off_list = ", ".join(
            f"{r['model']} ({r['velocity_share_pct']:+.0f}%)"
            for _, r in weak_offscale.iterrows()
        )
        ax.text(0.5, -0.20, f"Off-scale: {off_list}",
                transform=ax.transAxes, fontsize=5.5, color="0.4",
                ha="center", va="top")
    save_publication_figure(fig, args.out_fig)


if __name__ == "__main__":
    main()
