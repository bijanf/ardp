#!/usr/bin/env python3
"""AMOC anomalies at 26.5°N: four reanalyses + RAPID, Rahmstorf-style.

Shows anomalies relative to the 1950-2009 climatology. ORAS5 is the
long-record anchor (its 1958-2009 mean defines the zero line). Shorter
records are offset so their mean over the overlap with ORAS5 matches the
ORAS5 anomaly mean over the same period. RAPID has no overlap with
1950-2009, so it is offset to match ORAS5 over RAPID's own window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ardp.viz.style import apply_nature_style, save_publication_figure

CLIM_START, CLIM_END = 1950, 2009

DATASETS = [
    ("ORAS5",      "yearly_amoc26n_oras5.npz",    "#1f77b4"),
    ("GLORYS12V1", "yearly_amoc26n_glorys12.npz", "#2ca02c"),
    ("SODA3.15.2", "yearly_amoc26n_soda.npz",     "#e377c2"),
    ("ECCO-V4r4",  "yearly_amoc26n_ecco.npz",     "#d62728"),
]


def _running_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(values) - half):
        chunk = values[i - half : i + half + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 0:
            result[i] = valid.mean()
    return result


def _linear_trend(years: np.ndarray, values: np.ndarray) -> tuple[float, float, int]:
    """OLS trend with Santer et al. (2000) N_eff-adjusted p-value.

    Returns (slope_per_decade, p_value_adjusted, N_eff).
    """
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan, np.nan, 0
    y, v = years[mask], values[mask]
    n = len(v)
    res = stats.linregress(y, v)
    slope = res.slope * 10.0  # Sv/decade

    # Lag-1 autocorrelation of OLS residuals
    residuals = v - (res.slope * y + res.intercept)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    # Guard: clip r1 to [0, 0.99] — negative r1 means no correction needed
    r1 = np.clip(r1, 0.0, 0.99)
    # Santer et al. 2000: N_eff = N * (1 - r1) / (1 + r1)
    n_eff = max(3, int(n * (1 - r1) / (1 + r1)))

    # Re-compute t-statistic with N_eff degrees of freedom (Santer "AdjSE + AdjDF")
    se_slope = res.stderr  # OLS standard error of slope
    se_adj = se_slope * np.sqrt((n - 2) / (n_eff - 2))
    t_stat = res.slope / se_adj
    p_adj = 2 * stats.t.sf(abs(t_stat), df=n_eff - 2)

    return slope, p_adj, n_eff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/grl/fig_amoc_reanalysis_anomalies_santer"),
    )
    args = parser.parse_args()

    apply_nature_style()

    # ── Load ORAS5 (reference long record) ──
    oras5 = np.load(args.results_dir / "yearly_amoc26n_oras5.npz")
    o_yr = oras5["years"].astype(int)
    o_val = oras5["amoc"].astype(float)
    clim_mask = (o_yr >= CLIM_START) & (o_yr <= CLIM_END)
    oras5_clim = float(o_val[clim_mask].mean())
    print(f"ORAS5 climatology {CLIM_START}-{CLIM_END} "
          f"({clim_mask.sum()} yrs): {oras5_clim:.2f} Sv")

    anomalies: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    oras5_anom = o_val - oras5_clim
    anomalies["ORAS5"] = (o_yr, oras5_anom, DATASETS[0][2])

    # ── Load and offset each other dataset to ORAS5 anomaly ──
    for name, fname, color in DATASETS[1:]:
        d = np.load(args.results_dir / fname)
        yr = d["years"].astype(int)
        val = d["amoc"].astype(float)
        overlap = np.isin(yr, o_yr)
        o_overlap = np.isin(o_yr, yr)
        if overlap.sum() == 0:
            print(f"{name}: no ORAS5 overlap, skipping")
            continue
        offset = val[overlap].mean() - oras5_anom[o_overlap].mean()
        anom = val - offset
        anomalies[name] = (yr, anom, color)
        print(f"{name}: offset={offset:+.2f} Sv "
              f"(overlap {yr[overlap].min()}-{yr[overlap].max()}, "
              f"n={overlap.sum()})")

    # RAPID
    rapid = np.load(args.results_dir / "rapid_amoc26n.npz")
    r_yr = rapid["years"].astype(int)
    r_val = rapid["amoc"].astype(float)
    overlap = np.isin(r_yr, o_yr)
    o_overlap = np.isin(o_yr, r_yr)
    rapid_offset = r_val[overlap].mean() - oras5_anom[o_overlap].mean()
    r_anom = r_val - rapid_offset
    print(f"RAPID: offset={rapid_offset:+.2f} Sv "
          f"(adjusted to ORAS5 over {r_yr[overlap].min()}-{r_yr[overlap].max()})")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)

    trend_lines: list[tuple[str, str, float, float, int]] = []
    for name, (yr, anom, color) in anomalies.items():
        ax.plot(yr, anom, color=color, alpha=0.25, lw=0.6, zorder=3)
        rm = _running_mean(anom, window=5)
        ax.plot(yr, rm, color=color, lw=1.8, zorder=6, label=name)
        sl, pv, neff = _linear_trend(yr.astype(float), anom)
        trend_lines.append((name, color, sl, pv, neff))
        print(f"  {name}: {sl:+.2f} Sv/dec, p={pv:.4f}, N={len(yr)}, N_eff={neff}")

    # RAPID as markers + thin line
    ax.plot(r_yr, r_anom, color="black", lw=0.9, zorder=7)
    ax.scatter(r_yr, r_anom, color="black", s=10, zorder=8,
               marker="o", edgecolors="white", linewidths=0.3,
               label=f"RAPID (adj. {rapid_offset:+.1f} Sv)")
    sl, pv, neff = _linear_trend(r_yr.astype(float), r_val)  # RAPID trend from raw
    trend_lines.append(("RAPID", "black", sl, pv, neff))
    print(f"  RAPID: {sl:+.2f} Sv/dec, p={pv:.4f}, N={len(r_yr)}, N_eff={neff}")

    # ── Legend (horizontal, top) ──
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=5, frameon=False, fontsize=6.5, handlelength=1.8,
              columnspacing=1.2)

    # ── Trend table (lower-left) ──
    def _stars(p: float) -> str:
        if not np.isfinite(p):
            return ""
        if p < 0.01:
            return " **"
        if p < 0.05:
            return " *"
        return ""

    tx, ty = 0.015, 0.46
    ax.text(tx, ty, "Linear trends  (Santer N$_{eff}$)",
            transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="top", ha="left")
    for i, (name, color, sl, pv, neff) in enumerate(trend_lines):
        ax.text(tx, ty - 0.06 - i * 0.05,
                f"{name:<12s} {sl:+.2f} Sv/dec{_stars(pv)}  (N_eff={neff})",
                transform=ax.transAxes, fontsize=6, color=color,
                family="monospace", va="top", ha="left")
    ax.text(tx, ty - 0.06 - len(trend_lines) * 0.05 - 0.005,
            "** p < 0.01    * p < 0.05   (Santer et al. 2000)",
            transform=ax.transAxes, fontsize=5.5, color="0.4",
            va="top", ha="left")
    ax.text(tx, ty - 0.06 - len(trend_lines) * 0.05 - 0.05,
            f"Reference: ORAS5 {CLIM_START}–{CLIM_END} mean ({oras5_clim:.2f} Sv)",
            transform=ax.transAxes, fontsize=5.5, color="0.3",
            style="italic", va="top", ha="left")

    # ── Formatting ──
    ax.set_xlim(1958, 2025)
    ax.set_xlabel("Year")
    ax.set_ylabel("AMOC anomaly at 26.5°N (Sv)")
    fig.suptitle(
        f"AMOC anomalies at 26.5°N  (ref. climatology {CLIM_START}–{CLIM_END})",
        y=1.02, fontsize=9, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
