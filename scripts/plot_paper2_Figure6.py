#!/usr/bin/env python3
"""Paper 2 Figure 6: lead-lag relationships and emergent constraint.

Single combined PDF with two panels (the previous five-panel layout
was too crowded — peak-r and peak-lag panels are redundant views of
the lead-lag CCF, and the categorical forecast-bar panel was visually
confusing per user feedback):

  (a) Cross-correlation curves between F_ovS at 34.5°S and AMOC at
      26.5°N, per CMIP6 model, with collapsing- and stable-class
      ensemble means. Positive lag = F_ovS leads AMOC.
  (b) CMIP6 emergent regression of historical F_ovS (2000-2024)
      against ΔAMOC over the forecast window. The observational F_ovS
      ± 1σ band is propagated through the regression to give the
      constrained ΔAMOC central estimate (vertical blue band) and
      its 95\\% forecast interval (red band around the regression
      line).

Reads:
  data/results/cmip6_fovs_amoc_leadlag.nc
  data/results/emergent_constraint_rapid_forecast.json

Outputs: figures/paper2/Figure6.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

COLLAPSE_THRESHOLD = 0.30


def _panel_label(ax, label: str, x: float = 0.0, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leadlag", type=Path,
                        default=Path("data/results/cmip6_fovs_amoc_leadlag.nc"))
    parser.add_argument("--emergent", type=Path,
                        default=Path("data/results/emergent_constraint_rapid_forecast.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure6"))
    args = parser.parse_args()

    apply_nature_style()

    # ── Load lead-lag data ──
    ds = xr.open_dataset(args.leadlag)
    lags = ds["lag"].values
    ccf = ds["ccf"].values
    decline = ds["amoc_decline_frac"].values
    n_models = len(ds["model"].values)
    ds.close()
    collapsing = decline > COLLAPSE_THRESHOLD
    stable = ~collapsing

    # ── Load emergent constraint result ──
    with open(args.emergent) as f:
        ec = json.load(f)
    a = ec["regression_slope"]
    b = ec["regression_intercept"]
    r2 = ec["regression_R2"]
    sigma_res = ec["sigma_residual_Sv"]
    X_obs_mean = ec["X_obs_mean_Sv"]
    X_obs_sigma = ec["X_obs_sigma_Sv"]
    Y_point = ec["Y_point_forecast_Sv"]
    Y_low = ec["Y_95CI_low_Sv"]
    Y_high = ec["Y_95CI_high_Sv"]
    rows = ec["cmip6_models"]
    X_arr = np.array([r["X_i"] for r in rows])
    Y_arr = np.array([r["Y_i"] for r in rows])
    forecast_lo = ec["forecast_period"][0]
    forecast_hi = ec["forecast_period"][1]
    base_lo = ec["baseline_period"][0]
    base_hi = ec["baseline_period"][1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    # ── Panel (a): lead-lag CCF spaghetti + class means ──
    for i in range(n_models):
        c = "#CC3333" if collapsing[i] else "#3366AA"
        ax_a.plot(lags, ccf[i, :], color=c, alpha=0.25, lw=0.6, zorder=2)
    if collapsing.sum() > 1:
        mean_c = np.nanmean(ccf[collapsing, :], axis=0)
        std_c = np.nanstd(ccf[collapsing, :], axis=0)
        ax_a.fill_between(lags, mean_c - std_c, mean_c + std_c,
                          color="#CC3333", alpha=0.20, zorder=3)
        ax_a.plot(lags, mean_c, color="#CC3333", lw=2.2, zorder=5,
                  label=f"Collapsing  (n={int(collapsing.sum())})")
    if stable.sum() > 0:
        mean_s = np.nanmean(ccf[stable, :], axis=0)
        ax_a.plot(lags, mean_s, color="#3366AA", lw=1.8, zorder=5,
                  ls="--", label=f"Stable  (n={int(stable.sum())})")
    ax_a.axvline(0, color="0.6", lw=0.6, zorder=1)
    ax_a.axhline(0, color="0.6", lw=0.6, zorder=1)
    ax_a.set_xlabel(r"Lag $\tau$  (years)   "
                    r"— positive: F$_{ovS}$ leads AMOC")
    ax_a.set_ylabel(r"$r(\mathrm{F}_{ovS}(t),\ \mathrm{AMOC}(t+\tau))$")
    ax_a.set_xlim(lags[0], lags[-1])
    ax_a.set_ylim(-1, 1)
    ax_a.legend(loc="upper right", fontsize=8, frameon=False)
    _panel_label(ax_a, "(a)")

    # ── Panel (b): emergent regression with obs band ──
    # Regression details (slope, R², σ_res) intentionally omitted from
    # the figure per user feedback — reported in the figure caption +
    # Methods.
    _ = (a, b, r2, sigma_res, Y_low, Y_high)
    ax_b.scatter(X_arr, Y_arr, s=40, color="0.4", edgecolor="white",
                 linewidth=0.5, zorder=4, label=f"CMIP6 (n={len(rows)})")
    xx = np.linspace(min(X_arr.min(), X_obs_mean - 3 * X_obs_sigma) - 0.02,
                     max(X_arr.max(), X_obs_mean + 3 * X_obs_sigma) + 0.02,
                     200)
    ax_b.plot(xx, a * xx + b, color="#CC3333", lw=1.5, zorder=5,
              label="Emergent regression")
    ax_b.fill_between(xx, a * xx + b - 1.96 * sigma_res,
                      a * xx + b + 1.96 * sigma_res,
                      color="#CC3333", alpha=0.15, zorder=3,
                      label="95% forecast interval")
    ax_b.axvspan(X_obs_mean - X_obs_sigma, X_obs_mean + X_obs_sigma,
                 color="#3366AA", alpha=0.18, zorder=2)
    # Constrained forecast: a 2-D errorbar at (X_obs_mean, Y_point),
    # with horizontal whisker = ±1σ on the observational F_ovS and
    # vertical whisker = ±(Y_95high − Y_95low)/(2·1.96), i.e. ±1σ of
    # the propagated forecast uncertainty.
    sigma_y = (Y_high - Y_low) / (2 * 1.96)
    ax_b.errorbar([X_obs_mean], [Y_point],
                  xerr=[[X_obs_sigma], [X_obs_sigma]],
                  yerr=[[sigma_y], [sigma_y]],
                  fmt="o", color="#3366AA", ecolor="#3366AA",
                  elinewidth=1.5, capsize=4, capthick=1.5,
                  markersize=6, markeredgecolor="white",
                  markeredgewidth=0.8, zorder=8,
                  label=r"Constrained forecast (mean $\pm 1\sigma$)")
    ax_b.set_xlabel(r"F$_{ovS}$ at 34.5°S, 2000-2024 (Sv)")
    ax_b.set_ylabel(rf"$\Delta$AMOC$_{{26°N}}$ ({forecast_lo}-{forecast_hi} "
                    rf"$-$ {base_lo}-{base_hi}, Sv)")
    ax_b.axhline(0, color="0.6", lw=0.4)
    ax_b.legend(loc="upper left", fontsize=7.5, frameon=False)
    _panel_label(ax_b, "(b)")

    fig.tight_layout()
    save_publication_figure(fig, args.output)


if __name__ == "__main__":
    main()
