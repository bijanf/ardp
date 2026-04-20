#!/usr/bin/env python3
"""CMIP6 lead-lag analysis: does F_ovS at 34.5S precede AMOC weakening at 26.5N?

For each CMIP6 model that has BOTH F_ovS (from data/results/cmip6/) and
AMOC26N (from data/results/yearly_amoc26n_cmip6.npz), compute the
cross-correlation between F_ovS(t) and AMOC(t+τ) over lags τ ∈
[-max_lag, +max_lag] years.

Convention (matching scripts/compute_lag_correlations.py):
  - Positive τ => F_ovS *leads* AMOC by τ years
  - Negative τ => AMOC *leads* F_ovS by |τ| years

We use the historical + SSP5-8.5 concatenation (1850-2100) for the
strongest signal, segregating models by whether AMOC weakens
substantially (>30% decline by 2081-2100 vs 1850-1900 baseline) vs.
stable. In collapsing models, F_ovS should lead AMOC by a positive lag
(early-warning signature).

Outputs
-------
- `data/results/cmip6_fovs_amoc_leadlag.nc` : per-model CCF arrays and
  metadata (peak lag, peak r, AMOC decline fraction, collapsing flag).
- `data/results/cmip6_fovs_amoc_leadlag.csv` : human-readable summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
CMIP6_DIR = RESULTS_DIR / "cmip6"
MAX_LAG = 50
COLLAPSE_THRESHOLD = 0.30  # 30% decline by end-of-century
BASELINE_YEARS = (1850, 1900)
END_CENTURY_YEARS = (2081, 2100)


# ═══════════════════════════════════════════════════════════════════════
# Cross-correlation (copied from compute_lag_correlations.py to avoid
# fragile script-level imports; should eventually move to ardp/stats/)
# ═══════════════════════════════════════════════════════════════════════

def lag_cross_correlation(
    x: np.ndarray, y: np.ndarray, max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation at integer lags of detrended series.

    Convention: positive lag => x leads y.

    Returns (lags, ccf, pvalues).
    """
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.full_like(lags, np.nan, dtype=float)
    pvals = np.full_like(lags, np.nan, dtype=float)

    x_dt = signal.detrend(x)
    y_dt = signal.detrend(y)

    for i, lag in enumerate(lags):
        if lag >= 0:
            x_seg, y_seg = x_dt[: n - lag] if lag > 0 else x_dt, \
                           y_dt[lag:] if lag > 0 else y_dt
        else:
            x_seg = x_dt[-lag:]
            y_seg = y_dt[: n + lag]
        valid = np.isfinite(x_seg) & np.isfinite(y_seg)
        if valid.sum() >= 5:
            r, p = stats.pearsonr(x_seg[valid], y_seg[valid])
            ccf[i] = r
            pvals[i] = p
    return lags, ccf, pvals


# ═══════════════════════════════════════════════════════════════════════
# Per-model annual F_ovS from monthly file
# ═══════════════════════════════════════════════════════════════════════

def _load_fovs_annual(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load monthly F_ovS and resample to annual means.

    Returns (years, fovs_annual).
    """
    ds = xr.open_dataset(path)
    # Heuristic: use 'F_ovS' or first data variable
    var_name = "F_ovS" if "F_ovS" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var_name]

    # time dim is months; convert to years
    t = da["time"].values
    try:
        years = np.array([int(str(x)[:4]) for x in t])
    except Exception:
        # Some datasets use float years
        years = np.array(t).astype(int)

    vals = da.values
    uniq_years = np.unique(years)
    annual = np.array(
        [np.nanmean(vals[years == y]) for y in uniq_years]
    )
    ds.close()
    return uniq_years.astype(float), annual


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag", type=int, default=MAX_LAG)
    parser.add_argument(
        "--scenario", choices=["hist_ssp585", "hist_ssp245", "historical"],
        default="hist_ssp585",
    )
    args = parser.parse_args()

    # Load AMOC26N
    amoc_data = np.load(RESULTS_DIR / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    amoc_models = [str(m) for m in amoc_data["models"]]
    log.info(f"AMOC26N ensemble: {len(amoc_models)} models")

    # Find matching F_ovS files
    matched = []
    for model in amoc_models:
        fovs_path = CMIP6_DIR / f"fovs_{model}_{args.scenario}.nc"
        if fovs_path.exists():
            matched.append((model, fovs_path))
        else:
            log.warning(f"  {model}: no F_ovS file for {args.scenario}")

    log.info(f"Matched: {len(matched)} models")

    rows = []
    all_lags = np.arange(-args.max_lag, args.max_lag + 1)
    ccf_matrix = np.full((len(matched), len(all_lags)), np.nan)
    pval_matrix = np.full((len(matched), len(all_lags)), np.nan)
    amoc_decline = np.full(len(matched), np.nan)
    peak_lag = np.full(len(matched), np.nan, dtype=int)
    peak_r = np.full(len(matched), np.nan)
    peak_p = np.full(len(matched), np.nan)

    for i, (model, fovs_path) in enumerate(matched):
        try:
            f_years, f_annual = _load_fovs_annual(fovs_path)
            a_years = amoc_data[f"{model}_years"].astype(float)
            a_annual = amoc_data[f"{model}_amoc"]

            # Align to common years
            common = np.intersect1d(f_years, a_years)
            if len(common) < 100:
                log.warning(f"  {model}: only {len(common)} common years, skipping")
                continue

            f_idx = np.isin(f_years, common)
            a_idx = np.isin(a_years, common)
            f_aligned = f_annual[f_idx]
            a_aligned = a_annual[a_idx]

            valid = np.isfinite(f_aligned) & np.isfinite(a_aligned)
            if valid.sum() < 100:
                continue

            lags, ccf, pvals = lag_cross_correlation(
                f_aligned[valid], a_aligned[valid], args.max_lag
            )
            ccf_matrix[i, :] = ccf
            pval_matrix[i, :] = pvals

            # Peak (most negative correlation — F_ovS declining correlates
            # with AMOC declining, so r < 0 at the peak lag for
            # weakening-precursor relation; take arg-min of r)
            # Actually both decline, so the correlation is POSITIVE
            # (both go down together). Peak = max |r|.
            with np.errstate(invalid="ignore"):
                i_peak = int(np.nanargmax(np.abs(ccf)))
            peak_lag[i] = int(lags[i_peak])
            peak_r[i] = float(ccf[i_peak])
            peak_p[i] = float(pvals[i_peak])

            # AMOC decline: end-of-century vs baseline
            base_mask = (a_years >= BASELINE_YEARS[0]) & (a_years <= BASELINE_YEARS[1])
            end_mask = (a_years >= END_CENTURY_YEARS[0]) & (a_years <= END_CENTURY_YEARS[1])
            if base_mask.sum() >= 10 and end_mask.sum() >= 10:
                base = float(np.nanmean(a_annual[base_mask]))
                end = float(np.nanmean(a_annual[end_mask]))
                amoc_decline[i] = (base - end) / base if base > 0 else np.nan
            else:
                amoc_decline[i] = np.nan

            collapsing = bool(np.isfinite(amoc_decline[i]) and amoc_decline[i] > COLLAPSE_THRESHOLD)

            log.info(
                f"  {model:20s}  lag*={peak_lag[i]:+3d}y  "
                f"r={peak_r[i]:+.2f}  p={peak_p[i]:.2e}  "
                f"ΔAMOC={amoc_decline[i] * 100 if np.isfinite(amoc_decline[i]) else np.nan:+.1f}%  "
                f"{'COLLAPSING' if collapsing else 'stable'}"
            )
            rows.append({
                "model": model,
                "peak_lag_yr": int(peak_lag[i]),
                "peak_r": float(peak_r[i]),
                "peak_p": float(peak_p[i]),
                "amoc_decline_frac": float(amoc_decline[i]) if np.isfinite(amoc_decline[i]) else None,
                "collapsing": collapsing,
                "n_years": int(valid.sum()),
            })
        except Exception as e:
            log.error(f"  {model}: {e}")
            continue

    # Save
    out_nc = RESULTS_DIR / "cmip6_fovs_amoc_leadlag.nc"
    out_csv = RESULTS_DIR / "cmip6_fovs_amoc_leadlag.csv"

    ds_out = xr.Dataset(
        data_vars={
            "ccf": (("model", "lag"), ccf_matrix),
            "pvalue": (("model", "lag"), pval_matrix),
            "peak_lag": ("model", peak_lag),
            "peak_r": ("model", peak_r),
            "peak_p": ("model", peak_p),
            "amoc_decline_frac": ("model", amoc_decline),
        },
        coords={
            "model": [m for m, _ in matched],
            "lag": all_lags,
        },
        attrs={
            "scenario": args.scenario,
            "max_lag_years": args.max_lag,
            "collapse_threshold_frac": COLLAPSE_THRESHOLD,
            "lag_convention": "positive_lag=FovS_leads_AMOC",
            "baseline_years": f"{BASELINE_YEARS[0]}-{BASELINE_YEARS[1]}",
            "end_century_years": f"{END_CENTURY_YEARS[0]}-{END_CENTURY_YEARS[1]}",
        },
    )
    ds_out.to_netcdf(out_nc)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    log.info(f"Saved: {out_nc}")
    log.info(f"Saved: {out_csv}")

    # Summary
    if rows:
        collapsing = [r for r in rows if r["collapsing"]]
        stable = [r for r in rows if not r["collapsing"]]
        log.info("")
        log.info(f"Collapsing models (n={len(collapsing)}):")
        if collapsing:
            lags = [r["peak_lag_yr"] for r in collapsing]
            log.info(f"  Mean peak lag: {np.mean(lags):+.1f} yr  (F_ovS leads)")
            log.info(f"  Range:         {np.min(lags):+d} to {np.max(lags):+d} yr")
        log.info(f"Stable models (n={len(stable)}):")
        if stable:
            lags = [r["peak_lag_yr"] for r in stable]
            log.info(f"  Mean peak lag: {np.mean(lags):+.1f} yr")


if __name__ == "__main__":
    main()
