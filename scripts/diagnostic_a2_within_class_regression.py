#!/usr/bin/env python3
"""Phase-A diagnostic A2 — within-class emergent regression R².

The pooled emergent constraint (Fig. 6b in v2) has R²=0.01: F_ovS at
2000-2024 has essentially zero predictive power for ΔAMOC over
2030-2040. The editorial critique: this proves the metric is useless,
not that the constraint is "mechanism-blind".

The fix: re-run the regression separately on the velocity-dominant
and salinity-dominant subsets. If within-class R² jumps to ≥0.4, the
mechanism partition mathematically recovers signal that pooling
destroys — that's the "killer figure" defence.

Reads:
  data/results/emergent_constraint_rapid_forecast.json
  data/results/fovs_decomposition_cmip6_summary.csv  (mechanism class)

Outputs:
  figures/paper2/diagA2_within_class.{png,pdf}
  data/results/diagA2_within_class.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ardp.viz.style import apply_nature_style, save_publication_figure


def _classify(row):
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def _fit(X, Y):
    if len(X) < 3:
        return None
    res = stats.linregress(X, Y)
    Y_hat = res.slope * np.asarray(X) + res.intercept
    sigma_res = float(np.std(Y - Y_hat, ddof=2)) if len(X) > 2 else float("nan")
    return {
        "n": int(len(X)),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "R2": float(res.rvalue ** 2),
        "pvalue": float(res.pvalue),
        "stderr": float(res.stderr),
        "sigma_residual_Sv": sigma_res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emergent", type=Path,
                        default=Path("data/results/emergent_constraint_rapid_forecast.json"))
    parser.add_argument("--summary", type=Path,
                        default=Path("data/results/fovs_decomposition_cmip6_summary.csv"))
    parser.add_argument("--out-fig", type=Path,
                        default=Path("figures/paper2/diagA2_within_class"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("data/results/diagA2_within_class.json"))
    args = parser.parse_args()

    apply_nature_style()

    with open(args.emergent) as f:
        ec = json.load(f)
    rows = pd.DataFrame(ec["cmip6_models"])

    summary = pd.read_csv(args.summary)
    summary["class"] = summary.apply(_classify, axis=1)
    rows = pd.merge(rows, summary[["model", "class"]], on="model", how="left")
    print(f"\nEmergent ensemble: n={len(rows)} models with X+Y")
    print(f"  Models without a class (= no decomposition): "
          f"{rows[rows['class'].isna()]['model'].tolist()}")
    print(f"  Class breakdown:\n{rows['class'].value_counts(dropna=False)}")

    X_obs_mean = ec["X_obs_mean_Sv"]
    X_obs_sigma = ec["X_obs_sigma_Sv"]

    fits = {}
    fits["pooled"] = _fit(rows["X_i"].values, rows["Y_i"].values)
    for cls in ("v-dominant", "s-dominant", "mixed"):
        sub = rows[rows["class"] == cls]
        fits[cls] = _fit(sub["X_i"].values, sub["Y_i"].values)

    print("\n--- Regression fits ---")
    for k, f in fits.items():
        if f is None:
            print(f"  {k:12s}  insufficient sample")
            continue
        print(f"  {k:12s}  n={f['n']:2d}  slope={f['slope']:+.2f}  "
              f"R²={f['R2']:.3f}  p={f['pvalue']:.3g}  "
              f"σ_res={f['sigma_residual_Sv']:.3f} Sv")

    # Constrained ΔAMOC under each class-conditional regression
    forecasts = {}
    for k, f in fits.items():
        if f is None:
            forecasts[k] = None
            continue
        Y_point = f["slope"] * X_obs_mean + f["intercept"]
        Y_sigma = float(np.sqrt((f["slope"] * X_obs_sigma) ** 2
                                + f["sigma_residual_Sv"] ** 2))
        forecasts[k] = {
            "Y_point": float(Y_point),
            "Y_sigma": Y_sigma,
            "Y_95CI": [Y_point - 1.96 * Y_sigma, Y_point + 1.96 * Y_sigma],
        }
    print("\n--- Constrained forecasts (X_obs = "
          f"{X_obs_mean:+.3f} ± {X_obs_sigma:.3f} Sv) ---")
    for k, fc in forecasts.items():
        if fc is None:
            continue
        print(f"  {k:12s}  ΔAMOC = {fc['Y_point']:+.2f} Sv  "
              f"(95% CI [{fc['Y_95CI'][0]:+.2f}, {fc['Y_95CI'][1]:+.2f}])")

    # Save JSON
    with open(args.out_json, "w") as f:
        json.dump({"fits": fits, "forecasts": forecasts,
                   "X_obs_mean_Sv": X_obs_mean, "X_obs_sigma_Sv": X_obs_sigma},
                  f, indent=2)
    print(f"\nSaved: {args.out_json}")

    # ── Plot ──
    cls_color = {"pooled": "0.4", "v-dominant": "#E69F00",
                 "s-dominant": "#56B4E9", "mixed": "#009E73",
                 "increasing": "0.5"}
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    # All CMIP6 points coloured by class
    for cls in rows["class"].dropna().unique():
        sub = rows[rows["class"] == cls]
        ax.scatter(sub["X_i"], sub["Y_i"], s=50, color=cls_color.get(cls, "0.5"),
                   edgecolor="0.2", linewidth=0.5, zorder=4,
                   label=f"{cls} (n={len(sub)})")
    # Regression lines
    xx = np.linspace(rows["X_i"].min() - 0.02, rows["X_i"].max() + 0.02, 100)
    for k, f in fits.items():
        if f is None or f["n"] < 3:
            continue
        ax.plot(xx, f["slope"] * xx + f["intercept"],
                color=cls_color.get(k, "0.5"),
                lw=1.2 if k == "pooled" else 2.0,
                ls="--" if k == "pooled" else "-",
                zorder=5,
                label=(f"{k}: slope={f['slope']:+.1f}, "
                       f"R²={f['R2']:.2f}"))
    # Obs band
    ax.axvspan(X_obs_mean - X_obs_sigma, X_obs_mean + X_obs_sigma,
               color="#3366AA", alpha=0.15, zorder=2)
    ax.axvline(X_obs_mean, color="#3366AA", lw=0.8, zorder=3)
    ax.set_xlabel(r"F$_{ovS}$ at 34.5°S, 2000-2024 (Sv)")
    ax.set_ylabel(rf"$\Delta$AMOC$_{{26°N}}$ "
                  rf"({ec['forecast_period'][0]}-{ec['forecast_period'][1]} "
                  rf"$-$ {ec['baseline_period'][0]}-{ec['baseline_period'][1]}, Sv)")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.legend(loc="best", fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    save_publication_figure(fig, args.out_fig)


if __name__ == "__main__":
    main()
