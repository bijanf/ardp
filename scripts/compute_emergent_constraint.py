#!/usr/bin/env python3
"""Emergent constraint: CMIP6 F_ovS 2000-2024 -> projected RAPID AMOC 2030-2040.

Predictor X:  model-by-model F_ovS at 34.5S, averaged over 2000-2024.
Predictand Y: model-by-model AMOC at 26.5N change
              (mean 2030-2040) - (mean 2005-2022, RAPID baseline)
              [Sv relative to present].

Observational X is the mean of the 2-4 reanalysis F_ovS products over
2000-2024 with bootstrap uncertainty.

If Y = a*X + b with good R^2 across CMIP6, the observational X PDF is
propagated through the regression to produce a constrained PDF for Y.

The resulting envelope for ΔAMOC(2030-2040) is a falsifiable forecast:
if RAPID observations land in that envelope, F_ovS-based constraints
are validated; if not, something is structurally wrong with the
reanalysis-CMIP6 bridge.

Outputs:
  data/results/emergent_constraint_rapid_forecast.json
  data/results/emergent_constraint_rapid_forecast.nc
  figures/paper2/fig4_emergent_projection.{png,pdf}  (if --make-figure)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.viz.style import apply_nature_style, save_publication_figure

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
CMIP6_DIR = RESULTS_DIR / "cmip6"

# Periods
PRED_PERIOD = (2000, 2024)   # observational F_ovS predictor window
BASELINE_PERIOD = (2005, 2022)  # RAPID baseline (where AMOC 'now' is defined)
FORECAST_PERIOD = (2030, 2040)  # where we forecast AMOC

REANALYSIS_PRODUCTS = [
    ("ORAS5",      "oras5_f_ovs.nc"),
    ("GLORYS12V1", "glorys12_f_ovs.nc"),
    ("SODA3.15.2", "soda_f_ovs.nc"),
    ("ECCO-V4r4",  "ecco_f_ovs.nc"),
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_annual(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load annual-mean F_ovS, handling monthly/annual time axes and
    cftime calendars (CMIP6 often uses DatetimeNoLeap/Datetime360Day).
    """
    if not path.exists():
        return None
    ds = xr.open_dataset(path, use_cftime=True)
    var = "F_ovS" if "F_ovS" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]
    if "year" in da.dims:
        years = da["year"].values.astype(float)
        vals = da.values
    elif "time" in da.dims:
        # Extract year from any datetime-like time coord via xarray's
        # .dt accessor, which handles both np.datetime64 and cftime.
        years_int = da["time"].dt.year.values
        vals_m = da.values
        uniq = np.unique(years_int)
        vals = np.array([np.nanmean(vals_m[years_int == y]) for y in uniq])
        years = uniq.astype(float)
    else:
        ds.close()
        return None
    ds.close()
    return years, vals


def _period_mean(years: np.ndarray, vals: np.ndarray, period: tuple[int, int]) -> float:
    y0, y1 = period
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    return float(np.mean(vals[mask])) if mask.sum() >= 3 else np.nan


def _period_trend(years: np.ndarray, vals: np.ndarray, period: tuple[int, int]) -> float:
    """Linear slope of F_ovS vs year in Sv/yr (positive = increasing)."""
    y0, y1 = period
    mask = (years >= y0) & (years <= y1) & np.isfinite(vals)
    if mask.sum() < 5:
        return np.nan
    res = stats.linregress(years[mask], vals[mask])
    return float(res.slope)


def _load_cmip6_fovs_annual(model: str, scenario: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = CMIP6_DIR / f"fovs_{model}_{scenario}.nc"
    if not path.exists():
        return None
    return _load_annual(path)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-figure", action="store_true", default=True)
    parser.add_argument("--scenario", default="hist_ssp585")
    parser.add_argument(
        "--predictor", choices=["mean", "trend"], default="mean",
        help="Predictor: mean F_ovS or F_ovS trend over PRED_PERIOD",
    )
    parser.add_argument(
        "--forecast-end", type=int, default=2040,
        help="End year of forecast period (mean over forecast-end-10 to forecast-end)",
    )
    parser.add_argument(
        "--fractional", action="store_true",
        help="Use ΔAMOC/AMOC_baseline as predictand (dimensionless) to remove "
        "CMIP6 baseline-AMOC bias confounding.",
    )
    args = parser.parse_args()
    # Adjust forecast period based on --forecast-end
    global FORECAST_PERIOD
    FORECAST_PERIOD = (args.forecast_end - 10, args.forecast_end)
    suffix = f"{args.predictor}_to{args.forecast_end}"

    apply_nature_style()

    # ── Step 1: Observational predictor X_obs ──
    log.info("=" * 50)
    log.info(f"Observational F_ovS mean over {PRED_PERIOD[0]}-{PRED_PERIOD[1]}")
    log.info("=" * 50)

    predictor_func = _period_trend if args.predictor == "trend" else _period_mean
    predictor_label = "F_ovS trend (Sv/yr)" if args.predictor == "trend" else "F_ovS mean (Sv)"

    obs_means = []
    for label, fname in REANALYSIS_PRODUCTS:
        ts = _load_annual(RESULTS_DIR / fname)
        if ts is None:
            log.warning(f"  {label}: no data, skipping")
            continue
        yrs, vals = ts
        m = predictor_func(yrs, vals, PRED_PERIOD)
        if np.isfinite(m):
            obs_means.append((label, m))
            log.info(f"  {label}: {predictor_label} = {m:+.5f}")

    if len(obs_means) < 2:
        log.error("Need at least 2 reanalysis products for observational uncertainty.")
        return

    obs_values = np.array([m for _, m in obs_means])
    X_obs_mean = float(np.mean(obs_values))
    X_obs_sigma = float(np.std(obs_values, ddof=1))
    log.info(f"  Ensemble mean  X_obs = {X_obs_mean:+.4f} Sv")
    log.info(f"  Ensemble sigma σ_X   = {X_obs_sigma:+.4f} Sv")

    # ── Step 2: CMIP6 predictor X_i and predictand Y_i ──
    log.info("")
    log.info("CMIP6 model ensemble (predictor + predictand)")
    amoc_data = np.load(RESULTS_DIR / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    models = [str(m) for m in amoc_data["models"]]

    rows = []
    for model in models:
        fovs_ts = _load_cmip6_fovs_annual(model, args.scenario)
        if fovs_ts is None:
            continue
        f_years, f_vals = fovs_ts
        X_i = predictor_func(f_years, f_vals, PRED_PERIOD)

        a_years = amoc_data[f"{model}_years"].astype(float)
        a_vals = amoc_data[f"{model}_amoc"]
        A_base = _period_mean(a_years, a_vals, BASELINE_PERIOD)
        A_fore = _period_mean(a_years, a_vals, FORECAST_PERIOD)
        if not (np.isfinite(X_i) and np.isfinite(A_base) and np.isfinite(A_fore)):
            continue
        dY = A_fore - A_base
        Y_i = (dY / A_base) if args.fractional and A_base > 0 else dY
        rows.append({
            "model": model, "X_i": X_i,
            "A_baseline": A_base, "A_forecast": A_fore, "Y_i": Y_i,
        })
        log.info(
            f"  {model:20s}  X={X_i:+.4f} Sv  Y=ΔAMOC({FORECAST_PERIOD[0]}-{FORECAST_PERIOD[1]})={Y_i:+.2f} Sv"
        )

    if len(rows) < 5:
        log.error("Need at least 5 CMIP6 models with both X and Y.")
        return

    X = np.array([r["X_i"] for r in rows])
    Y = np.array([r["Y_i"] for r in rows])

    # ── Step 3: Fit emergent constraint Y = a*X + b ──
    res = stats.linregress(X, Y)
    a, b, r_val, p_val, se = res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
    log.info("")
    log.info(f"Emergent constraint fit: Y = {a:+.3f} * X + {b:+.3f}")
    log.info(f"  R^2 = {r_val ** 2:.3f},  p = {p_val:.3e},  SE(slope) = {se:.3f}")

    # Residuals
    Y_hat = a * X + b
    residuals = Y - Y_hat
    sigma_res = float(np.std(residuals, ddof=2))

    # ── Step 4: Propagate observational PDF through regression ──
    # Y_obs_pred PDF = regression_point(X_obs) +- sqrt( (a*σ_X)^2 + σ_res^2 )
    Y_point = a * X_obs_mean + b
    Y_sigma = float(np.sqrt((a * X_obs_sigma) ** 2 + sigma_res ** 2))

    # 95% forecast interval
    Y_low = Y_point - 1.96 * Y_sigma
    Y_high = Y_point + 1.96 * Y_sigma

    log.info("")
    log.info(
        f"Falsifiable forecast for ΔAMOC26N("
        f"{FORECAST_PERIOD[0]}-{FORECAST_PERIOD[1]} vs {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}):"
    )
    log.info(f"  Central estimate : {Y_point:+.2f} Sv")
    log.info(f"  95% CI           : [{Y_low:+.2f}, {Y_high:+.2f}] Sv")
    log.info(f"  Components       : a*σ_X = {a * X_obs_sigma:.3f} Sv, σ_res = {sigma_res:.3f} Sv")

    # Unconstrained CMIP6 range (for comparison)
    Y_uncon_mean = float(np.mean(Y))
    Y_uncon_low, Y_uncon_high = float(np.percentile(Y, 2.5)), float(np.percentile(Y, 97.5))
    log.info(
        f"  (For comparison, unconstrained CMIP6 Y: "
        f"{Y_uncon_mean:+.2f} Sv, 95% [{Y_uncon_low:+.2f}, {Y_uncon_high:+.2f}])"
    )

    # ── Step 5: Save ──
    out_json = {
        "predictor_period": list(PRED_PERIOD),
        "baseline_period": list(BASELINE_PERIOD),
        "forecast_period": list(FORECAST_PERIOD),
        "scenario": args.scenario,
        "n_obs_products": len(obs_means),
        "obs_products": [p for p, _ in obs_means],
        "X_obs_mean_Sv": X_obs_mean,
        "X_obs_sigma_Sv": X_obs_sigma,
        "n_cmip6": len(rows),
        "regression_slope": a,
        "regression_intercept": b,
        "regression_R2": r_val ** 2,
        "regression_pvalue": p_val,
        "regression_SE_slope": se,
        "sigma_residual_Sv": sigma_res,
        "Y_point_forecast_Sv": Y_point,
        "Y_95CI_low_Sv": Y_low,
        "Y_95CI_high_Sv": Y_high,
        "Y_unconstrained_mean_Sv": Y_uncon_mean,
        "Y_unconstrained_95CI": [Y_uncon_low, Y_uncon_high],
        "cmip6_models": rows,
    }
    out_path = RESULTS_DIR / "emergent_constraint_rapid_forecast.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    log.info(f"Saved: {out_path}")

    # ── Step 6: Optional figure ──
    if args.make_figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.7))

        # Left: scatter + regression
        ax1.scatter(X, Y, s=35, color="0.4", edgecolor="white", linewidth=0.5,
                    zorder=4, label=f"CMIP6 (n={len(rows)})")
        xx = np.linspace(min(X.min(), X_obs_mean - 3 * X_obs_sigma) - 0.02,
                         max(X.max(), X_obs_mean + 3 * X_obs_sigma) + 0.02, 200)
        ax1.plot(xx, a * xx + b, color="#CC3333", lw=1.5, zorder=5,
                 label=f"Y = {a:+.2f}·X {b:+.2f}  (R² = {r_val ** 2:.2f})")
        # Confidence band
        ax1.fill_between(xx, a * xx + b - 1.96 * sigma_res,
                         a * xx + b + 1.96 * sigma_res,
                         color="#CC3333", alpha=0.15, zorder=3)

        # Observational X as vertical PDF
        ax1.axvspan(X_obs_mean - X_obs_sigma, X_obs_mean + X_obs_sigma,
                    color="#3366AA", alpha=0.2, zorder=2)
        ax1.axvline(X_obs_mean, color="#3366AA", lw=1.5, zorder=6,
                    label=f"Obs F$_{{ovS}}$ (2000-2024): "
                          f"{X_obs_mean:+.3f} ± {X_obs_sigma:.3f} Sv")
        ax1.set_xlabel(r"F$_{ovS}$ at 34.5°S, 2000-2024 (Sv)")
        ax1.set_ylabel(
            rf"$\Delta$AMOC$_{{26°N}}$  "
            rf"(mean {FORECAST_PERIOD[0]}-{FORECAST_PERIOD[1]} "
            rf"− {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}), Sv"
        )
        # Title removed — LaTeX provides the (a) label.
        ax1.axhline(0, color="0.6", lw=0.4)
        ax1.legend(loc="best", fontsize=6, frameon=False)

        # Right: forecast range (bar + whisker)
        ax2.barh(
            ["Unconstrained\nCMIP6", "Constrained\nby obs"],
            [Y_uncon_high - Y_uncon_low, Y_high - Y_low],
            left=[Y_uncon_low, Y_low],
            color=["0.75", "#CC3333"],
            edgecolor="0.2", linewidth=0.5, zorder=3,
        )
        ax2.scatter([Y_uncon_mean, Y_point], [0, 1],
                    color="black", s=40, zorder=5)
        ax2.axvline(0, color="0.6", lw=0.6, zorder=1)
        ax2.set_xlabel(
            rf"Forecast $\Delta$AMOC "
            rf"({FORECAST_PERIOD[0]}-{FORECAST_PERIOD[1]} vs {BASELINE_PERIOD[0]}-{BASELINE_PERIOD[1]}), Sv"
        )
        # Title removed — LaTeX provides the (b) label.

        # Annotate the narrowing
        ax2.annotate(
            f"central:   {Y_point:+.2f} Sv\n"
            f"95% CI:  [{Y_low:+.2f}, {Y_high:+.2f}] Sv\n"
            f"falsifiable: RAPID mean\n"
            f"{FORECAST_PERIOD[0]}-{FORECAST_PERIOD[1]} should fall here",
            xy=(Y_point, 1), xytext=(Y_point - 1.0, 0.6),
            fontsize=5.8, color="#CC3333", ha="left",
        )

        fig.tight_layout()
        save_publication_figure(fig, Path("figures/paper2/fig4_emergent_projection"))


if __name__ == "__main__":
    main()
