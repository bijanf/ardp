#!/usr/bin/env python3
"""Robustness and statistical validation of AMOC fingerprints.

Performs:
1. Multi-product trend comparison (GLORYS12, ORAS5, C-GLORS)
2. Trend significance testing (Mann-Kendall, bootstrapped confidence intervals)
3. Sensitivity to time window (sliding-window trend analysis)
4. Cross-correlation between fingerprints
5. Comparison with RAPID array observations (if available)

Results are saved to data/results/robustness/ as NetCDF and CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def mann_kendall_test(ts: np.ndarray) -> dict:
    """Perform Mann-Kendall trend test.

    Returns dict with: tau, p_value, trend_detected (at 95% confidence).
    """
    valid = np.isfinite(ts)
    x = ts[valid]
    n = len(x)
    if n < 4:
        return {"tau": np.nan, "p_value": np.nan, "trend_detected": False}

    # Compute S statistic
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # Variance of S
    unique, counts = np.unique(x, return_counts=True)
    n_ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    for t in n_ties:
        var_s -= t * (t - 1) * (2 * t + 5) / 18.0

    if var_s <= 0:
        return {"tau": 0.0, "p_value": 1.0, "trend_detected": False}

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2 * stats.norm.sf(abs(z))
    tau = 2 * s / (n * (n - 1))

    return {
        "tau": tau,
        "p_value": p_value,
        "trend_detected": p_value < 0.05,
    }


def bootstrap_trend_ci(
    years: np.ndarray,
    values: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Bootstrap confidence interval for linear trend slope."""
    valid = np.isfinite(values)
    y = years[valid]
    v = values[valid]
    n = len(v)

    if n < 3:
        return {"slope_median": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    rng = np.random.default_rng(42)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        result = stats.linregress(y[idx], v[idx])
        slopes[i] = result.slope

    alpha = (1 - ci) / 2
    return {
        "slope_median": np.median(slopes),
        "ci_lower": np.percentile(slopes, alpha * 100),
        "ci_upper": np.percentile(slopes, (1 - alpha) * 100),
    }


def sliding_window_trend(
    ts: xr.DataArray,
    window_years: int = 15,
) -> xr.DataArray:
    """Compute trend in sliding windows to assess sensitivity.

    Parameters
    ----------
    ts : xr.DataArray
        Time series.
    window_years : int
        Window size in years.

    Returns
    -------
    xr.DataArray
        Trend slope for each window center time.
    """
    time = ts["time"]
    try:
        timestamps = pd.DatetimeIndex(time.values)
        years = np.array([t.year + (t.month - 1) / 12.0 for t in timestamps])
    except Exception:
        years = np.arange(len(time))

    values = ts.values.ravel()
    window_size = int(window_years * 12)  # monthly data
    if len(values) < window_size:
        window_size = len(values) // 2

    n = len(values)
    slopes = np.full(n, np.nan)
    pvalues = np.full(n, np.nan)

    half = window_size // 2
    for i in range(half, n - half):
        start, end = i - half, i + half
        y_win = years[start:end]
        v_win = values[start:end]
        valid = np.isfinite(v_win)
        if valid.sum() >= 3:
            result = stats.linregress(y_win[valid], v_win[valid])
            slopes[i] = result.slope
            pvalues[i] = result.pvalue

    trend_ts = xr.DataArray(
        slopes,
        dims=("time",),
        coords={"time": time},
        attrs={"units": f"{ts.attrs.get('units', '')}/yr",
               "window_years": window_years},
    )
    return trend_ts


def cross_correlation_matrix(
    fingerprints: dict[str, xr.DataArray],
) -> pd.DataFrame:
    """Compute pairwise Pearson correlations between fingerprints."""
    names = list(fingerprints.keys())
    n = len(names)
    corr = np.eye(n)
    pvals = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ts_i = fingerprints[names[i]]
            ts_j = fingerprints[names[j]]
            # Align on common time
            common = xr.align(ts_i, ts_j, join="inner")
            vi = common[0].values.ravel()
            vj = common[1].values.ravel()
            valid = np.isfinite(vi) & np.isfinite(vj)
            if valid.sum() >= 3:
                r, p = stats.pearsonr(vi[valid], vj[valid])
                corr[i, j] = corr[j, i] = r
                pvals[i, j] = pvals[j, i] = p

    return pd.DataFrame(corr, index=names, columns=names)


def analyze_single_product(
    results_dir: Path,
    output_dir: Path,
    product: str,
) -> dict:
    """Run full robustness analysis on one product's results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "F_ovS": "f_ovs.nc",
        "Salinity_pileup": "salinity_pileup.nc",
        "NAWH": "nawh.nc",
        "GS_destab": "gulf_stream_destab.nc",
    }

    fingerprints = {}
    summary_rows = []

    for name, fname in files.items():
        path = results_dir / fname
        if not path.exists():
            continue
        ts = xr.open_dataarray(path)
        fingerprints[name] = ts

        try:
            time = ts["time"]
            timestamps = pd.DatetimeIndex(time.values)
            years = np.array([t.year + (t.month - 1) / 12.0 for t in timestamps])
        except Exception:
            years = np.arange(len(ts))

        values = ts.values.ravel()

        # Linear regression
        valid = np.isfinite(values)
        if valid.sum() >= 3:
            lr = stats.linregress(years[valid], values[valid])
        else:
            continue

        # Mann-Kendall
        mk = mann_kendall_test(values)

        # Bootstrap CI
        boot = bootstrap_trend_ci(years, values)

        summary_rows.append({
            "fingerprint": name,
            "product": product,
            "n_months": int(valid.sum()),
            "start_year": float(years[valid][0]),
            "end_year": float(years[valid][-1]),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "ols_slope": lr.slope,
            "ols_stderr": lr.stderr,
            "ols_pvalue": lr.pvalue,
            "ols_r_squared": lr.rvalue ** 2,
            "mk_tau": mk["tau"],
            "mk_pvalue": mk["p_value"],
            "mk_significant": mk["trend_detected"],
            "boot_slope_median": boot["slope_median"],
            "boot_ci_lower": boot["ci_lower"],
            "boot_ci_upper": boot["ci_upper"],
        })

        # Sliding window trend
        if len(ts) > 36:
            window_yrs = min(15, len(ts) // 24)
            if window_yrs >= 3:
                sw_trend = sliding_window_trend(ts, window_years=window_yrs)
                sw_trend.to_netcdf(
                    output_dir / f"sliding_trend_{name.lower()}_{product}.nc"
                )

    # Save summary
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(output_dir / f"trend_summary_{product}.csv", index=False)
        print(f"\nTrend summary for {product}:")
        print(df.to_string(index=False))

    # Cross-correlations
    if len(fingerprints) >= 2:
        corr_df = cross_correlation_matrix(fingerprints)
        corr_df.to_csv(output_dir / f"cross_correlations_{product}.csv")
        print(f"\nCross-correlations ({product}):")
        print(corr_df.to_string())

    return {"summary": summary_rows, "fingerprints": fingerprints}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical robustness analysis of AMOC fingerprints."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="data/results/robustness")
    parser.add_argument("--product", default="glorys12")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print(f"Running robustness analysis for {args.product}...")
    analyze_single_product(results_dir, output_dir, args.product)
    print("\nRobustness analysis complete.")


if __name__ == "__main__":
    main()
