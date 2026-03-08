#!/usr/bin/env python3
"""Compute lead/lag cross-correlations between AMOC fingerprints.

Tests the hypothesis that fingerprints are physically related but operate
on different timescales — e.g., F_ovS may lead salinity pile-up by 5-10
years. If significant lagged correlations exist, that is a publishable
finding even when contemporaneous correlations are weak.

Outputs:
  - Lag cross-correlation functions (CCF) for all fingerprint pairs
  - Optimal lag and corresponding r for each pair
  - Summary CSV
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats


def load_fingerprints(results_dir: Path) -> dict[str, xr.DataArray]:
    """Load all available fingerprint time series."""
    files = {
        "F_ovS": ["oras5_f_ovs.nc", "f_ovs.nc"],
        "Salinity_pileup": ["salinity_pileup.nc"],
        "NAWH": ["nawh.nc"],
        "GS_destab": ["gulf_stream_destab.nc"],
    }

    fingerprints = {}
    for name, candidates in files.items():
        for fname in candidates:
            path = results_dir / fname
            if path.exists():
                fingerprints[name] = xr.open_dataarray(path)
                print(f"  Loaded {name}: {len(fingerprints[name])} timesteps from {fname}")
                break

    return fingerprints


def annual_mean(ts: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Convert monthly time series to annual means.

    Returns (years, values) arrays.
    """
    try:
        idx = pd.DatetimeIndex(ts.time.values)
        years_frac = np.array([t.year for t in idx])
    except Exception:
        years_frac = np.arange(len(ts))

    unique_years = np.unique(years_frac)
    annual_vals = np.array([
        float(np.nanmean(ts.values[years_frac == y]))
        for y in unique_years
    ])
    return unique_years.astype(float), annual_vals


def lag_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cross-correlation function at integer lags.

    Parameters
    ----------
    x, y : np.ndarray
        Annual mean time series (same length, aligned).
    max_lag : int
        Maximum lag in years.

    Returns
    -------
    lags : np.ndarray
        Lag values (negative = x leads y).
    ccf : np.ndarray
        Cross-correlation at each lag.
    pvalues : np.ndarray
        Two-sided p-value at each lag.
    """
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.full_like(lags, np.nan, dtype=float)
    pvalues = np.full_like(lags, np.nan, dtype=float)

    # Detrend both series
    x_dt = signal.detrend(x)
    y_dt = signal.detrend(y)

    for i, lag in enumerate(lags):
        if lag >= 0:
            x_seg = x_dt[:n - lag] if lag > 0 else x_dt
            y_seg = y_dt[lag:] if lag > 0 else y_dt
        else:
            x_seg = x_dt[-lag:]
            y_seg = y_dt[:n + lag]

        valid = np.isfinite(x_seg) & np.isfinite(y_seg)
        if valid.sum() >= 5:
            r, p = stats.pearsonr(x_seg[valid], y_seg[valid])
            ccf[i] = r
            pvalues[i] = p

    return lags, ccf, pvalues


def compute_all_lag_correlations(
    fingerprints: dict[str, xr.DataArray],
    max_lag: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Compute lag correlations for all fingerprint pairs.

    Returns summary DataFrame and dict of full CCF arrays.
    """
    # Convert all to annual means on common years
    annual = {}
    for name, ts in fingerprints.items():
        years, vals = annual_mean(ts)
        annual[name] = (years, vals)

    names = list(annual.keys())
    summary_rows = []
    ccf_data = {}

    for name_a, name_b in combinations(names, 2):
        years_a, vals_a = annual[name_a]
        years_b, vals_b = annual[name_b]

        # Find overlapping years
        common_years = np.intersect1d(years_a, years_b)
        if len(common_years) < 10:
            print(f"  {name_a} vs {name_b}: only {len(common_years)} overlapping years, skipping")
            continue

        # Extract aligned data
        a_aligned = np.array([vals_a[years_a == y][0] for y in common_years])
        b_aligned = np.array([vals_b[years_b == y][0] for y in common_years])

        # Effective max lag (can't exceed half the series)
        eff_max_lag = min(max_lag, len(common_years) // 3)

        lags, ccf, pvals = lag_cross_correlation(a_aligned, b_aligned, eff_max_lag)

        # Find optimal lag
        valid_ccf = np.where(np.isfinite(ccf), np.abs(ccf), 0)
        best_idx = np.argmax(valid_ccf)
        best_lag = int(lags[best_idx])
        best_r = float(ccf[best_idx])
        best_p = float(pvals[best_idx])

        # Contemporaneous correlation (lag=0)
        zero_idx = np.where(lags == 0)[0][0]
        r0 = float(ccf[zero_idx])
        p0 = float(pvals[zero_idx])

        pair_key = f"{name_a}_vs_{name_b}"
        ccf_data[pair_key] = {
            "lags": lags,
            "ccf": ccf,
            "pvalues": pvals,
            "years_overlap": common_years,
        }

        summary_rows.append({
            "pair": pair_key,
            "series_a": name_a,
            "series_b": name_b,
            "n_years": len(common_years),
            "year_range": f"{int(common_years[0])}-{int(common_years[-1])}",
            "r_lag0": r0,
            "p_lag0": p0,
            "best_lag": best_lag,
            "best_r": best_r,
            "best_p": best_p,
            "lag_interpretation": (
                f"{name_a} leads by {abs(best_lag)} yr" if best_lag < 0
                else f"{name_b} leads by {best_lag} yr" if best_lag > 0
                else "contemporaneous"
            ),
        })

        # Print result
        p0_str = "p < 0.001" if p0 < 0.001 else f"p = {p0:.3f}"
        bp_str = "p < 0.001" if best_p < 0.001 else f"p = {best_p:.3f}"
        print(f"  {name_a} vs {name_b}:")
        print(f"    Lag 0: r = {r0:+.3f} ({p0_str})")
        print(f"    Best:  r = {best_r:+.3f} at lag {best_lag:+d} yr ({bp_str})")

    return pd.DataFrame(summary_rows), ccf_data


def save_ccf_netcdf(ccf_data: dict, output_dir: Path) -> None:
    """Save cross-correlation functions as NetCDF."""
    for pair_key, data in ccf_data.items():
        ds = xr.Dataset({
            "ccf": xr.DataArray(
                data["ccf"], dims=("lag",),
                coords={"lag": data["lags"]},
                attrs={"long_name": "Cross-correlation coefficient",
                       "description": f"Detrended lag cross-correlation for {pair_key}"},
            ),
            "pvalue": xr.DataArray(
                data["pvalues"], dims=("lag",),
                coords={"lag": data["lags"]},
                attrs={"long_name": "Two-sided p-value"},
            ),
        })
        ds.attrs["pair"] = pair_key
        ds.attrs["n_overlap_years"] = len(data["years_overlap"])
        outpath = output_dir / f"lag_ccf_{pair_key}.nc"
        ds.to_netcdf(outpath)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute lead/lag cross-correlations between AMOC fingerprints."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="data/results/robustness")
    parser.add_argument(
        "--max-lag", type=int, default=20,
        help="Maximum lag in years (default: 20).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading fingerprints...")
    fingerprints = load_fingerprints(results_dir)

    if len(fingerprints) < 2:
        print(f"ERROR: Need at least 2 fingerprints, found {len(fingerprints)}")
        print("  Run: python scripts/compute_fingerprints.py --product glorys12")
        return

    print(f"\nComputing lag cross-correlations (max_lag={args.max_lag} yr)...")
    summary_df, ccf_data = compute_all_lag_correlations(
        fingerprints, max_lag=args.max_lag
    )

    if summary_df.empty:
        print("No valid pairs found.")
        return

    # Print summary
    print("\n" + "=" * 90)
    print("LAG CROSS-CORRELATION SUMMARY")
    print("=" * 90)
    for _, row in summary_df.iterrows():
        sig0 = "*" if row["p_lag0"] < 0.05 else " "
        sig_best = "*" if row["best_p"] < 0.05 else " "
        print(
            f"  {row['pair']:>35s}  "
            f"lag0: r={row['r_lag0']:+.3f} {sig0}  "
            f"best: r={row['best_r']:+.3f} at lag={row['best_lag']:+d} {sig_best}  "
            f"({row['lag_interpretation']})"
        )

    # Check for publishable findings
    sig_lagged = summary_df[
        (summary_df["best_p"] < 0.05) & (summary_df["best_lag"] != 0)
    ]
    if not sig_lagged.empty:
        print(f"\nPublishable finding: {len(sig_lagged)} pair(s) with significant lagged correlation!")
        for _, row in sig_lagged.iterrows():
            print(f"  {row['pair']}: r={row['best_r']:+.3f} at lag {row['best_lag']:+d} yr")
    else:
        print("\nNo significant lagged correlations found.")
        print("Fingerprints may genuinely operate on independent timescales.")

    # Save
    csv_path = output_dir / "lag_correlation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")

    save_ccf_netcdf(ccf_data, output_dir)
    print(f"Saved CCF NetCDFs to: {output_dir}")


if __name__ == "__main__":
    main()
