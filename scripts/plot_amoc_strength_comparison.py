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
    cmip6_585 = np.load(args.results_dir / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    cmip6_245_path = args.results_dir / "yearly_amoc26n_cmip6_ssp245.npz"
    cmip6_245 = np.load(cmip6_245_path, allow_pickle=True) if cmip6_245_path.exists() else None

    models = cmip6_585["models"]
    print(f"CMIP6 models (SSP585): {len(models)}")
    if cmip6_245 is not None:
        print(f"CMIP6 models (SSP245): {len(cmip6_245['models'])}")

    def build_envelope(cmip6_data):
        mods = cmip6_data["models"]
        all_years = set()
        for m in mods:
            all_years.update(cmip6_data[f"{m}_years"].tolist())
        year_grid = np.array(sorted(all_years))
        matrix = np.full((len(mods), len(year_grid)), np.nan)
        for i, m in enumerate(mods):
            yrs = cmip6_data[f"{m}_years"]
            vals = cmip6_data[f"{m}_amoc"]
            for j, y in enumerate(yrs):
                idx = np.searchsorted(year_grid, y)
                if idx < len(year_grid) and year_grid[idx] == y:
                    matrix[i, idx] = vals[j]
        with np.errstate(all="ignore"):
            n_valid = np.sum(np.isfinite(matrix), axis=0)
            median = np.nanmedian(matrix, axis=0)
            p25 = np.nanpercentile(matrix, 25, axis=0)
            p75 = np.nanpercentile(matrix, 75, axis=0)
            mn = np.nanmin(matrix, axis=0)
            mx = np.nanmax(matrix, axis=0)
        enough = n_valid >= 5
        return year_grid, median, p25, p75, mn, mx, enough

    year_585, med_585, p25_585, p75_585, min_585, max_585, ok_585 = build_envelope(cmip6_585)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6.73, 3.5))

    # SSP585 envelope (future only)
    fut_585 = ok_585 & (year_585 > 2014)
    ax.fill_between(year_585[fut_585], min_585[fut_585], max_585[fut_585],
                     color="#FFCCCC", alpha=0.4, linewidth=0, zorder=1,
                     label=f"SSP5-8.5 range (n={len(models)})")
    ax.fill_between(year_585[fut_585], p25_585[fut_585], p75_585[fut_585],
                     color="#FF9999", alpha=0.4, linewidth=0, zorder=2)
    ax.plot(year_585[fut_585], med_585[fut_585],
            color="#CC3333", lw=0.8, zorder=3, label="SSP5-8.5 median")

    # SSP245 envelope
    if cmip6_245 is not None:
        year_245, med_245, p25_245, p75_245, min_245, max_245, ok_245 = build_envelope(cmip6_245)
        # Only show SSP245 for future (>2014) to avoid overlap with historical
        future_245 = ok_245 & (year_245 > 2014)
        ax.fill_between(year_245[future_245], min_245[future_245], max_245[future_245],
                         color="#CCDDFF", alpha=0.4, linewidth=0, zorder=1,
                         label=f"SSP2-4.5 range (n={len(cmip6_245['models'])})")
        ax.fill_between(year_245[future_245], p25_245[future_245], p75_245[future_245],
                         color="#99BBFF", alpha=0.4, linewidth=0, zorder=2)
        ax.plot(year_245[future_245], med_245[future_245],
                color="#3366AA", lw=0.8, zorder=3, label="SSP2-4.5 median")

    # Historical part (grey, shared between scenarios)
    hist = ok_585 & (year_585 <= 2014)
    ax.fill_between(year_585[hist], min_585[hist], max_585[hist],
                     color="0.85", alpha=0.5, linewidth=0, zorder=0)
    ax.fill_between(year_585[hist], p25_585[hist], p75_585[hist],
                     color="0.7", alpha=0.5, linewidth=0, zorder=0)
    ax.plot(year_585[hist], med_585[hist],
            color="0.5", lw=0.8, zorder=1)

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

    # Model list moved to LaTeX caption; keep only data on the canvas.
    _ = models

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
