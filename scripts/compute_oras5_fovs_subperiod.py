#!/usr/bin/env python3
"""Compute F_ovS trend for sub-periods of the ORAS5 record.

Primary use case: compare 1993-2023 (GLORYS12 overlap period) against
the full 1958-2023 record. If the post-1993 trend disappears, the
pre-1993 period drives the full-record trend — a key finding for the
reanalysis assessment paper.

Reads the already-computed oras5_f_ovs.nc and slices by time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def compute_subperiod_trends(
    f_ovs: xr.DataArray,
    periods: list[tuple[int, int]],
) -> pd.DataFrame:
    """Compute linear trends for multiple sub-periods.

    Parameters
    ----------
    f_ovs : xr.DataArray
        Full F_ovS time series with time coordinate.
    periods : list of (start_year, end_year)
        Sub-periods to analyze.

    Returns
    -------
    pd.DataFrame
        One row per period with slope, p-value, n_months, etc.
    """
    rows = []
    for start, end in periods:
        subset = f_ovs.sel(time=slice(f"{start}-01", f"{end}-12"))
        values = subset.values.ravel()
        valid = np.isfinite(values)
        n = int(valid.sum())

        if n < 3:
            rows.append({
                "period": f"{start}-{end}",
                "start_year": start,
                "end_year": end,
                "n_months": n,
                "slope_sv_yr": np.nan,
                "slope_msv_yr": np.nan,
                "pvalue": np.nan,
                "stderr": np.nan,
                "r_squared": np.nan,
                "mean_sv": np.nan,
            })
            continue

        ts = pd.DatetimeIndex(subset.time.values)
        years = np.array([t.year + (t.month - 1) / 12.0 for t in ts])
        result = stats.linregress(years[valid], values[valid])

        rows.append({
            "period": f"{start}-{end}",
            "start_year": start,
            "end_year": end,
            "n_months": n,
            "slope_sv_yr": result.slope,
            "slope_msv_yr": result.slope * 1e3,
            "pvalue": result.pvalue,
            "stderr": result.stderr,
            "r_squared": result.rvalue ** 2,
            "mean_sv": float(np.nanmean(values[valid])),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ORAS5 F_ovS trends for sub-periods."
    )
    parser.add_argument(
        "--results-dir", default="data/results",
        help="Directory containing oras5_f_ovs.nc (default: data/results).",
    )
    parser.add_argument(
        "--output-dir", default="data/results/robustness",
        help="Output directory for sub-period analysis (default: data/results/robustness).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fovs_path = results_dir / "oras5_f_ovs.nc"
    if not fovs_path.exists():
        # Fall back to generic f_ovs.nc
        fovs_path = results_dir / "f_ovs.nc"
    if not fovs_path.exists():
        print(f"ERROR: Neither oras5_f_ovs.nc nor f_ovs.nc found in {results_dir}")
        print("  Run: python scripts/compute_oras5_fovs.py")
        return

    f_ovs = xr.open_dataarray(fovs_path)
    print(f"Loaded F_ovS: {len(f_ovs)} timesteps")

    ts = pd.DatetimeIndex(f_ovs.time.values)
    first_year = ts.year.min()
    last_year = ts.year.max()
    print(f"  Time span: {first_year}-{last_year}")

    # Define sub-periods for comparison
    periods = [
        (first_year, last_year),      # Full record
        (1993, last_year),            # GLORYS12 overlap
        (first_year, 1992),           # Pre-altimetric era
        (1993, 2010),                 # Early satellite era
        (2004, last_year),            # RAPID era
        (1958, 1990),                 # Pre-greenhouse acceleration
        (1990, last_year),            # Post-1990
    ]
    # Filter to periods within data range
    periods = [(s, e) for s, e in periods if s >= first_year and e <= last_year]

    print(f"\nComputing trends for {len(periods)} sub-periods...")
    df = compute_subperiod_trends(f_ovs, periods)

    # Print results
    print("\nSub-period trend analysis:")
    print("=" * 85)
    for _, row in df.iterrows():
        p_str = (
            "p < 0.001" if row["pvalue"] < 0.001
            else f"p = {row['pvalue']:.3f}" if row["pvalue"] < 0.01
            else f"p = {row['pvalue']:.2f}"
        )
        sig = "*" if row["pvalue"] < 0.05 else " "
        print(
            f"  {row['period']:>10s}  "
            f"slope = {row['slope_msv_yr']:+7.2f} mSv/yr  "
            f"({p_str}) {sig}  "
            f"n = {row['n_months']:>4.0f} mo  "
            f"mean = {row['mean_sv']:+.4f} Sv"
        )
    print("=" * 85)

    # Key comparison
    full = df[df["period"] == f"{first_year}-{last_year}"]
    glorys_overlap = df[df["start_year"] == 1993]
    if not full.empty and not glorys_overlap.empty:
        full_slope = full.iloc[0]["slope_msv_yr"]
        overlap_slope = glorys_overlap.iloc[0]["slope_msv_yr"]
        overlap_p = glorys_overlap.iloc[0]["pvalue"]
        print(f"\nKey comparison:")
        print(f"  Full record trend:    {full_slope:+.2f} mSv/yr")
        print(f"  1993-{last_year} trend:     {overlap_slope:+.2f} mSv/yr (p={overlap_p:.3f})")
        if abs(overlap_slope) < abs(full_slope) * 0.5:
            print("  >> Post-1993 trend is <50% of full-record trend")
            print("  >> Pre-1993 period likely drives the long-term decline")
        elif overlap_p > 0.05:
            print("  >> Post-1993 trend is NOT significant")
            print("  >> Consistent with GLORYS12 showing no F_ovS trend")

    # Save
    csv_path = output_dir / "fovs_subperiod_trends.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
