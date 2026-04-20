#!/usr/bin/env python3
"""Figure 1: Multi-product F_ovS at 34.5S observational bedrock.

Shows annual-mean F_ovS from up to 4 reanalyses:
  - ORAS5       (1958-2025, monthly)
  - GLORYS12V1  (1993-2025, monthly)
  - SODA 3.15.2 (1980-2022, annual)
  - ECCO-V4r4   (1992-2017, annual)

Plus direct hydrography anchors at SAMBA / hydrographic occupations.

Trends computed with three methods (Santer N_eff, GLS Prais-Winsten, naive
OLS) using the functions from scripts/plot_amoc_reanalysis_anomalies.py.

Reads:
  data/results/{oras5,glorys12,soda,ecco}_f_ovs.nc  (skips missing)

Outputs: figures/paper2/fig1_multiprod_fovs.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

# Import the three trend functions from the AMOC script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amoc_reanalysis_anomalies import (  # noqa: E402
    _linear_trend_gls,
    _linear_trend_ols,
    _linear_trend_santer,
)

PRODUCTS = [
    ("ORAS5",      "oras5_f_ovs.nc",    "#1f77b4"),
    ("GLORYS12V1", "glorys12_f_ovs.nc", "#2ca02c"),
    ("SODA3.15.2", "soda_f_ovs.nc",     "#e377c2"),
    ("ECCO-V4r4",  "ecco_f_ovs.nc",     "#d62728"),
]

# Published direct-hydrography F_ovS estimates at 34.5S
# (value, error_1sigma, citation_tag, year_center)
DIRECT_HYDRO = [
    (-0.10, 0.10, "Garzoli & Matano 2011", 2005),
    (-0.09, 0.05, "Meinen et al. 2018",    2015),
    (-0.17, 0.07, "Kersalé et al. 2020",   2015),
]


def _load_annual(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (years, F_ovS_annual) from monthly or annual NetCDF."""
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    var = "F_ovS" if "F_ovS" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]

    if "year" in da.dims:
        years = da["year"].values.astype(float)
        vals = da.values
    elif "time" in da.dims:
        t = pd.DatetimeIndex(da["time"].values)
        years_int = t.year.values
        vals_monthly = da.values
        uniq = np.unique(years_int)
        vals = np.array([np.nanmean(vals_monthly[years_int == y]) for y in uniq])
        years = uniq.astype(float)
    else:
        ds.close()
        return None
    ds.close()
    return years, vals


def _running_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(values) - half):
        chunk = values[i - half : i + half + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 0:
            out[i] = valid.mean()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/paper2/fig1_multiprod_fovs"),
    )
    args = parser.parse_args()

    apply_nature_style()

    loaded: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    trend_rows = []
    for label, fname, color in PRODUCTS:
        ts = _load_annual(args.results_dir / fname)
        if ts is None:
            print(f"  {label}: no file at {args.results_dir / fname}, skipping")
            continue
        yrs, vals = ts
        loaded.append((label, yrs, vals, color))
        print(
            f"  {label}: {int(yrs[0])}-{int(yrs[-1])}, mean={float(np.mean(vals)):+.4f} Sv, "
            f"n={len(yrs)}"
        )

        # Trends (Sv/decade)
        sl_ols, p_ols, _ = _linear_trend_ols(yrs, vals)
        sl_san, p_san, neff = _linear_trend_santer(yrs, vals)
        sl_gls, p_gls, _ = _linear_trend_gls(yrs, vals)
        trend_rows.append({
            "product": label,
            "n_years": len(yrs),
            "mean_Sv": float(np.mean(vals)),
            "ols_slope_Sv_dec": sl_ols,
            "ols_p": p_ols,
            "santer_slope_Sv_dec": sl_san,
            "santer_p": p_san,
            "santer_Neff": neff,
            "gls_slope_Sv_dec": sl_gls,
            "gls_p": p_gls,
        })

    if not loaded:
        print("No F_ovS products loaded. Run compute_*_fovs.py scripts first.")
        return

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7.6, 3.8))

    ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)
    # Light shading to flag bistable region F_ovS < 0
    ax.axhspan(-1.0, 0.0, color="#FFF3E0", alpha=0.5, zorder=0)
    ax.text(
        1958.5, -0.28, "bistable regime   (F$_{ovS}$ < 0)",
        fontsize=6.5, color="#CC5500", style="italic", va="bottom",
    )

    for label, yrs, vals, color in loaded:
        ax.plot(yrs, vals, color=color, alpha=0.25, lw=0.5, zorder=3)
        rm = _running_mean(vals, window=5)
        ax.plot(yrs, rm, color=color, lw=1.8, zorder=6, label=label)

    # Direct-hydrography anchors
    for val, err, tag, yr in DIRECT_HYDRO:
        ax.errorbar(yr, val, yerr=err, color="black", marker="D",
                    markersize=4, capsize=2, lw=1.0, zorder=10,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.annotate(tag, xy=(yr, val), xytext=(5, 2),
                    textcoords="offset points", fontsize=5.2,
                    color="0.3", zorder=11)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=4, frameon=False, fontsize=6.8, handlelength=1.8,
        columnspacing=1.4,
    )

    # Trend table
    def _stars(p: float) -> str:
        if not np.isfinite(p): return ""
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    tx, ty = 0.015, 0.30
    ax.text(
        tx, ty, "Linear trends (mSv/yr)   Santer / GLS",
        transform=ax.transAxes, fontsize=7, fontweight="bold",
        va="top", ha="left",
    )
    ax.text(
        tx, ty - 0.055, f"{'product':<11s}   N   santer   GLS",
        transform=ax.transAxes, fontsize=5.5, family="monospace",
        color="0.4", va="top", ha="left",
    )
    for i, row in enumerate(trend_rows):
        lab = row["product"]
        n = row["n_years"]
        san_str = f"{row['santer_slope_Sv_dec'] * 100:+.2f}{_stars(row['santer_p']):s}"
        gls_str = f"{row['gls_slope_Sv_dec'] * 100:+.2f}{_stars(row['gls_p']):s}"
        color = next(c for lbl, _, _, c in loaded if lbl == lab)
        ax.text(
            tx, ty - 0.09 - i * 0.048,
            f"{lab:<11s}  {n:>3d}  {san_str:>6s}  {gls_str:>6s}",
            transform=ax.transAxes, fontsize=6, color=color,
            family="monospace", va="top", ha="left",
        )
    ax.text(
        tx, ty - 0.09 - len(trend_rows) * 0.048 - 0.01,
        "** p<0.01  * p<0.05   (Santer N$_{eff}$, GLS Prais-Winsten AR1)",
        transform=ax.transAxes, fontsize=5.2, color="0.4",
        va="top", ha="left",
    )

    ax.set_xlim(1958, 2026)
    ax.set_ylim(-0.3, 0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\mathrm{F}_{ovS}$  at 34.5°S (Sv)")
    fig.suptitle(
        r"Overturning freshwater transport at 34.5°S: four reanalyses + direct hydrography",
        y=0.99, fontsize=9, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_publication_figure(fig, args.output)

    # Also save trend table as CSV for the supplementary
    pd.DataFrame(trend_rows).to_csv(
        args.results_dir / "fovs_multiprod_trends.csv", index=False,
    )
    print(f"Saved: {args.results_dir / 'fovs_multiprod_trends.csv'}")


if __name__ == "__main__":
    main()
