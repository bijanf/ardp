#!/usr/bin/env python3
"""Plot AMOC strength at 26.5°N: reanalyses vs CMIP6 envelope vs RAPID.

Single-panel figure showing:
  - CMIP6 model envelope (min–max, IQR, median) for historical + SSP585
  - ORAS5 and GLORYS12 reanalysis annual + 10-year running mean
  - RAPID array observations (yearly)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ardp.viz.style import apply_nature_style, save_publication_figure


def _running_mean(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Centered running mean with NaN padding at edges."""
    result = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(values) - half):
        chunk = values[i - half : i + half + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 0:
            result[i] = valid.mean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AMOC strength at 26.5N: reanalyses vs CMIP6 vs RAPID."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/grl/fig_amoc_strength_26N"))
    args = parser.parse_args()

    apply_nature_style()

    # ── Load data ──
    oras5 = np.load(args.results_dir / "yearly_amoc26n_oras5.npz")
    glorys12 = np.load(args.results_dir / "yearly_amoc26n_glorys12.npz")
    rapid = np.load(args.results_dir / "rapid_amoc26n.npz")
    cmip6 = np.load(args.results_dir / "yearly_amoc26n_cmip6.npz", allow_pickle=True)

    models = cmip6["models"]
    print(f"CMIP6 models: {len(models)}")

    # ── Build CMIP6 envelope on a common year grid ──
    all_years = set()
    for m in models:
        all_years.update(cmip6[f"{m}_years"].tolist())
    year_grid = np.array(sorted(all_years))

    # Interpolate each model onto the common grid
    model_matrix = np.full((len(models), len(year_grid)), np.nan)
    for i, m in enumerate(models):
        yrs = cmip6[f"{m}_years"]
        vals = cmip6[f"{m}_amoc"]
        for j, y in enumerate(yrs):
            idx = np.searchsorted(year_grid, y)
            if idx < len(year_grid) and year_grid[idx] == y:
                model_matrix[i, idx] = vals[j]

    # Compute envelope statistics (ignoring NaN)
    with np.errstate(all="ignore"):
        n_valid = np.sum(np.isfinite(model_matrix), axis=0)
        model_median = np.nanmedian(model_matrix, axis=0)
        model_p25 = np.nanpercentile(model_matrix, 25, axis=0)
        model_p75 = np.nanpercentile(model_matrix, 75, axis=0)
        model_min = np.nanmin(model_matrix, axis=0)
        model_max = np.nanmax(model_matrix, axis=0)

    # Only show where we have >= 5 models
    enough = n_valid >= 5

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6.73, 3.5))

    # CMIP6 envelope
    ax.fill_between(year_grid[enough], model_min[enough], model_max[enough],
                     color="0.85", alpha=0.5, linewidth=0, zorder=1,
                     label=f"CMIP6 range (n={len(models)})")
    ax.fill_between(year_grid[enough], model_p25[enough], model_p75[enough],
                     color="0.7", alpha=0.5, linewidth=0, zorder=2,
                     label="CMIP6 25th\u201375th pctl")
    ax.plot(year_grid[enough], model_median[enough],
            color="0.5", lw=0.8, zorder=3, label="CMIP6 median")

    # ORAS5
    o_yr, o_val = oras5["years"], oras5["amoc"]
    o_rm = _running_mean(o_val, window=10)
    ax.plot(o_yr, o_val, color="#4477AA", alpha=0.2, lw=0.5, zorder=8)
    ax.plot(o_yr, o_rm, color="#4477AA", lw=2.0, zorder=10, label="ORAS5")

    # GLORYS12
    g_yr, g_val = glorys12["years"], glorys12["amoc"]
    g_rm = _running_mean(g_val, window=10)
    ax.plot(g_yr, g_val, color="#228833", alpha=0.2, lw=0.5, zorder=8)
    ax.plot(g_yr, g_rm, color="#228833", lw=2.0, zorder=10, label="GLORYS12")

    # RAPID observations
    r_yr, r_val = rapid["years"], rapid["amoc"]
    ax.scatter(r_yr, r_val, color="black", s=12, zorder=12,
               marker="o", label="RAPID", edgecolors="white", linewidths=0.3)

    # Model names inside the plot, just above y=0 (two rows)
    sorted_names = sorted(str(m) for m in models)
    half = len(sorted_names) // 2
    row1 = "  |  ".join(sorted_names[:half])
    row2 = "  |  ".join(sorted_names[half:])
    ax.text(0.5, 0.07, row1,
            transform=ax.transAxes, fontsize=4.5, ha="center", va="bottom",
            color="black")
    ax.text(0.5, 0.02, row2,
            transform=ax.transAxes, fontsize=4.5, ha="center", va="bottom",
            color="black")

    # Formatting
    ax.set_xlim(1850, 2100)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year")
    ax.set_ylabel("AMOC strength at 26.5°N (Sv)")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_publication_figure(fig, args.output)
    print(f"Saved: {args.output}")

    # Print summary stats
    print(f"\nORAS5:   mean={o_val.mean():.1f} Sv, range={o_val.min():.1f}–{o_val.max():.1f}")
    print(f"GLORYS12: mean={g_val.mean():.1f} Sv, range={g_val.min():.1f}–{g_val.max():.1f}")
    print(f"RAPID:   mean={r_val.mean():.1f} Sv, range={r_val.min():.1f}–{r_val.max():.1f}")

    # Trend
    from scipy import stats
    sl, _, _, pv, _ = stats.linregress(o_yr, o_val)
    print(f"ORAS5 trend: {sl*10:.2f} Sv/decade (p={pv:.2e})")
    sl2, _, _, pv2, _ = stats.linregress(g_yr, g_val)
    print(f"GLORYS12 trend: {sl2*10:.2f} Sv/decade (p={pv2:.2e})")


if __name__ == "__main__":
    main()
