#!/usr/bin/env python3
"""Compare F_ovS between ORAS5 and GLORYS12 reanalysis products.

The GLORYS12 F_ovS shows NO trend (+0.002 mSv/yr, p=0.67) while ORAS5
shows a strong decline (-0.71 mSv/yr, p<0.001). This script investigates
the disagreement systematically:

1. Plot both F_ovS time series on common axes
2. Compare trends for the overlap period (1993-2023) only
3. Compute correlation between the two products
4. Test whether the disagreement is a time-span or product issue

The disagreement itself is a publishable finding for the reanalysis
assessment paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from ardp.viz.style import (
    COLORS,
    FINGERPRINT_COLORS,
    add_panel_label,
    figure_grl_full,
    save_publication_figure,
)


def _time_to_years(time: xr.DataArray) -> np.ndarray:
    """Convert xarray time coordinate to fractional years."""
    try:
        ts = pd.DatetimeIndex(time.values)
        return np.array([t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
                         for t in ts])
    except Exception:
        if hasattr(time.values[0], "year"):
            return np.array([t.year + (t.month - 1) / 12.0
                             for t in time.values])
        return time.values.astype(float)


def _compute_trend(years: np.ndarray, values: np.ndarray) -> dict:
    """Compute linear trend."""
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return {"slope": np.nan, "pvalue": np.nan, "intercept": np.nan}
    result = stats.linregress(years[valid], values[valid])
    return {
        "slope": result.slope,
        "pvalue": result.pvalue,
        "intercept": result.intercept,
        "stderr": result.stderr,
        "r_squared": result.rvalue ** 2,
    }


def load_fovs_products(results_dir: Path) -> dict[str, xr.DataArray]:
    """Load F_ovS from available products."""
    products = {}

    # ORAS5 F_ovS (from depth-based computation)
    for fname in ["oras5_f_ovs.nc", "f_ovs.nc"]:
        path = results_dir / fname
        if path.exists():
            products["ORAS5"] = xr.open_dataarray(path)
            print(f"  ORAS5 F_ovS: {len(products['ORAS5'])} months from {fname}")
            break

    # GLORYS12 F_ovS (from fingerprint computation)
    # Try GLORYS12-specific path first
    glorys_paths = [
        results_dir / "glorys12_f_ovs.nc",
        results_dir.parent / "glorys12_results" / "f_ovs.nc",
    ]
    # If compute_fingerprints was run with --product glorys12, it saves to f_ovs.nc
    # Check if we already loaded ORAS5 from f_ovs.nc; if so, GLORYS12 might be elsewhere
    for gpath in glorys_paths:
        if gpath.exists():
            products["GLORYS12"] = xr.open_dataarray(gpath)
            print(f"  GLORYS12 F_ovS: {len(products['GLORYS12'])} months from {gpath.name}")
            break

    return products


def compare_products(
    products: dict[str, xr.DataArray],
    output_dir: Path,
) -> None:
    """Generate multi-panel comparison figure."""
    if len(products) < 2:
        print("Need at least 2 products for comparison. Generating single-product analysis.")
        _single_product_analysis(products, output_dir)
        return

    fig, axes = figure_grl_full(nrows=2, ncols=2, height_ratio=0.5)
    axes = axes.ravel()

    product_colors = {
        "ORAS5": FINGERPRINT_COLORS["f_ovs"],
        "GLORYS12": COLORS["red"],
        "C-GLORS": COLORS["green"],
    }

    # --- Panel (a): Both F_ovS time series ---
    ax = axes[0]
    for name, ts in products.items():
        years = _time_to_years(ts["time"])
        color = product_colors.get(name, COLORS["grey"])

        # Scale to mSv if values are small
        scale = 1e3 if np.nanmax(np.abs(ts.values)) < 1 else 1.0
        vals = ts.values * scale

        ax.plot(years, vals, color=color, linewidth=0.5, alpha=0.5)

        # 12-month running mean
        if len(ts) > 24:
            rolling = pd.Series(vals).rolling(12, center=True).mean()
            ax.plot(years, rolling.values, color=color, linewidth=1.5, label=name)
        else:
            ax.plot(years, vals, color=color, linewidth=1.0, label=name)

    unit = "mSv" if scale == 1e3 else "Sv"
    ax.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
    ax.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=5)
    add_panel_label(ax, "a")

    # --- Panel (b): Overlap period trends ---
    ax = axes[1]
    # Find common time range
    all_starts = [pd.DatetimeIndex(ts.time.values).year.min() for ts in products.values()]
    all_ends = [pd.DatetimeIndex(ts.time.values).year.max() for ts in products.values()]
    overlap_start = max(all_starts)
    overlap_end = min(all_ends)

    trend_texts = []
    for name, ts in products.items():
        subset = ts.sel(time=slice(f"{overlap_start}-01", f"{overlap_end}-12"))
        years = _time_to_years(subset["time"])
        color = product_colors.get(name, COLORS["grey"])

        scale = 1e3 if np.nanmax(np.abs(subset.values)) < 1 else 1.0
        vals = subset.values * scale

        ax.plot(years, vals, color=color, linewidth=0.5, alpha=0.3)

        # Trend line
        trend = _compute_trend(years, vals)
        if not np.isnan(trend["slope"]):
            trend_line = trend["slope"] * years + trend["intercept"]
            ax.plot(years, trend_line, color=color, linewidth=1.5,
                    linestyle="--", label=name)

            p_str = "p < 0.001" if trend["pvalue"] < 0.001 else f"p = {trend['pvalue']:.2f}"
            trend_texts.append(f"{name}: {trend['slope']:+.2f} {unit}/yr ({p_str})")

    ax.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
    ax.set_title(f"Overlap period ({overlap_start}-{overlap_end})", fontsize=7)
    ax.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=5)
    add_panel_label(ax, "b")

    if trend_texts:
        ax.annotate(
            "\n".join(trend_texts),
            xy=(0.03, 0.95), xycoords="axes fraction",
            fontsize=5, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "0.7", "alpha": 0.9},
        )

    # --- Panel (c): Scatter ORAS5 vs GLORYS12 (annual means) ---
    ax = axes[2]
    names = list(products.keys())
    ts_a = products[names[0]]
    ts_b = products[names[1]]

    # Compute annual means
    def _annual(ts):
        idx = pd.DatetimeIndex(ts.time.values)
        df = pd.DataFrame({"value": ts.values.ravel(), "year": idx.year})
        return df.groupby("year")["value"].mean()

    ann_a = _annual(ts_a)
    ann_b = _annual(ts_b)
    common_years = ann_a.index.intersection(ann_b.index)

    if len(common_years) >= 3:
        va = ann_a.loc[common_years].values
        vb = ann_b.loc[common_years].values

        scale_a = 1e3 if np.nanmax(np.abs(va)) < 1 else 1.0
        scale_b = 1e3 if np.nanmax(np.abs(vb)) < 1 else 1.0
        va *= scale_a
        vb *= scale_b

        valid = np.isfinite(va) & np.isfinite(vb)
        ax.scatter(va[valid], vb[valid], s=12, color=COLORS["purple"],
                   edgecolors="none", alpha=0.7, zorder=3)

        r, p = stats.pearsonr(va[valid], vb[valid])
        bias = np.mean(vb[valid] - va[valid])

        # 1:1 line
        vmin = min(va[valid].min(), vb[valid].min()) - 5
        vmax = max(va[valid].max(), vb[valid].max()) + 5
        ax.plot([vmin, vmax], [vmin, vmax], color="0.6", linewidth=0.5,
                linestyle=":", zorder=1)

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")

        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.annotate(
            f"r = {r:.2f} ({p_str})\n"
            f"bias = {bias:+.1f} {unit}\n"
            f"n = {valid.sum()} yr",
            xy=(0.03, 0.95), xycoords="axes fraction",
            fontsize=5, va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "0.7", "alpha": 0.9},
        )

    ax.set_xlabel(f"{names[0]} $F_{{ovS}}$ [{unit}]")
    ax.set_ylabel(f"{names[1]} $F_{{ovS}}$ [{unit}]")
    add_panel_label(ax, "c")

    # --- Panel (d): Full vs overlap trend comparison (bar chart) ---
    ax = axes[3]
    bar_data = []
    for name, ts in products.items():
        years_full = _time_to_years(ts["time"])
        vals_full = ts.values * (1e3 if np.nanmax(np.abs(ts.values)) < 1 else 1.0)
        trend_full = _compute_trend(years_full, vals_full)

        subset = ts.sel(time=slice(f"{overlap_start}-01", f"{overlap_end}-12"))
        years_sub = _time_to_years(subset["time"])
        vals_sub = subset.values * (1e3 if np.nanmax(np.abs(subset.values)) < 1 else 1.0)
        trend_sub = _compute_trend(years_sub, vals_sub)

        bar_data.append({
            "product": name,
            "full_slope": trend_full["slope"],
            "full_p": trend_full["pvalue"],
            "overlap_slope": trend_sub["slope"],
            "overlap_p": trend_sub["pvalue"],
        })

    x = np.arange(len(bar_data))
    width = 0.35
    for i, d in enumerate(bar_data):
        color = product_colors.get(d["product"], COLORS["grey"])
        full_alpha = 1.0 if d["full_p"] < 0.05 else 0.4
        over_alpha = 1.0 if d["overlap_p"] < 0.05 else 0.4

        ax.bar(i - width / 2, d["full_slope"], width, color=color,
               alpha=full_alpha, label="Full record" if i == 0 else "")
        ax.bar(i + width / 2, d["overlap_slope"], width, color=color,
               alpha=over_alpha, edgecolor=color, linewidth=1, facecolor="white",
               label=f"{overlap_start}-{overlap_end}" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels([d["product"] for d in bar_data])
    ax.set_ylabel(f"$F_{{ovS}}$ trend [{unit}/yr]")
    ax.axhline(0, color="0.5", linewidth=0.3)
    ax.legend(loc="lower left", fontsize=5)
    add_panel_label(ax, "d")
    ax.annotate("Faded = not significant (p > 0.05)",
                xy=(0.97, 0.05), xycoords="axes fraction",
                fontsize=4, ha="right", color="0.5")

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save_publication_figure(fig, output_dir / "fovs_product_comparison")

    # Print summary
    print("\n" + "=" * 70)
    print("PRODUCT COMPARISON SUMMARY")
    print("=" * 70)
    for d in bar_data:
        fp = "p < 0.001" if d["full_p"] < 0.001 else f"p = {d['full_p']:.3f}"
        op = "p < 0.001" if d["overlap_p"] < 0.001 else f"p = {d['overlap_p']:.3f}"
        print(f"  {d['product']}:")
        print(f"    Full record:  {d['full_slope']:+.2f} {unit}/yr ({fp})")
        print(f"    {overlap_start}-{overlap_end}: {d['overlap_slope']:+.2f} {unit}/yr ({op})")


def _single_product_analysis(
    products: dict[str, xr.DataArray],
    output_dir: Path,
) -> None:
    """Generate analysis for a single product when only one is available."""
    name, ts = next(iter(products.items()))
    print(f"\nSingle-product analysis for {name}")

    years = _time_to_years(ts["time"])
    scale = 1e3 if np.nanmax(np.abs(ts.values)) < 1 else 1.0
    unit = "mSv" if scale == 1e3 else "Sv"
    vals = ts.values * scale

    trend = _compute_trend(years, vals)
    p_str = "p < 0.001" if trend["pvalue"] < 0.001 else f"p = {trend['pvalue']:.3f}"
    print(f"  Full trend: {trend['slope']:+.2f} {unit}/yr ({p_str})")
    print("  Only one product available — run compute_fingerprints.py for other products")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare F_ovS across reanalysis products (ORAS5 vs GLORYS12)."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="figures/assessment")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading F_ovS from available products...")
    products = load_fovs_products(results_dir)

    if not products:
        print("ERROR: No F_ovS data found in", results_dir)
        print("  Run: python scripts/compute_oras5_fovs.py")
        print("  Run: python scripts/compute_fingerprints.py --product glorys12")
        return

    print(f"\nFound {len(products)} product(s): {', '.join(products.keys())}")
    compare_products(products, output_dir)


if __name__ == "__main__":
    main()
