#!/usr/bin/env python3
"""Paper 2 Figure 6: lead-lag relationships and emergent constraint.

Two panels:

  (a) Cross-correlation curves between F_ovS at 34.5°S and AMOC at
      26.5°N, per CMIP6 model, with collapsing- and stable-class
      ensemble means. Positive lag = F_ovS leads AMOC.
  (b) CMIP6 emergent regression of historical F_ovS (2000-2024)
      against ΔAMOC over the forecast window; obs-mean propagated
      through the regression as a 2-D errorbar.

Outputs (default --mode both):
  figures/paper2/Figure6.{png,pdf}        single combined PDF
  figures/paper2/Figure6a.{png,pdf}       panel a only
  figures/paper2/Figure6b.{png,pdf}       panel b only

Reads:
  data/results/cmip6_fovs_amoc_leadlag.nc
  data/results/emergent_constraint_rapid_forecast.json
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


def _load_data(leadlag: Path, emergent: Path):
    ds = xr.open_dataset(leadlag)
    lags = ds["lag"].values
    ccf = ds["ccf"].values
    decline = ds["amoc_decline_frac"].values
    n_models = len(ds["model"].values)
    ds.close()
    collapsing = decline > COLLAPSE_THRESHOLD
    with open(emergent) as f:
        ec = json.load(f)
    return {
        "lags": lags, "ccf": ccf, "n_models": n_models,
        "collapsing": collapsing, "stable": ~collapsing,
        "ec": ec,
    }


def _draw_panel_a(ax, d):
    lags, ccf, n_models = d["lags"], d["ccf"], d["n_models"]
    collapsing, stable = d["collapsing"], d["stable"]
    for i in range(n_models):
        c = "#CC3333" if collapsing[i] else "#3366AA"
        ax.plot(lags, ccf[i, :], color=c, alpha=0.25, lw=0.6, zorder=2)
    if collapsing.sum() > 1:
        mean_c = np.nanmean(ccf[collapsing, :], axis=0)
        std_c = np.nanstd(ccf[collapsing, :], axis=0)
        ax.fill_between(lags, mean_c - std_c, mean_c + std_c,
                        color="#CC3333", alpha=0.20, zorder=3)
        ax.plot(lags, mean_c, color="#CC3333", lw=2.2, zorder=5,
                label=f"Collapsing  (n={int(collapsing.sum())})")
    if stable.sum() > 0:
        mean_s = np.nanmean(ccf[stable, :], axis=0)
        ax.plot(lags, mean_s, color="#3366AA", lw=1.8, zorder=5, ls="--",
                label=f"Stable  (n={int(stable.sum())})")
    ax.axvline(0, color="0.6", lw=0.6, zorder=1)
    ax.axhline(0, color="0.6", lw=0.6, zorder=1)
    ax.set_xlabel(r"Lag $\tau$  (years)   "
                  r"— positive: F$_{ovS}$ leads AMOC")
    ax.set_ylabel(r"$r(\mathrm{F}_{ovS}(t),\ \mathrm{AMOC}(t+\tau))$")
    ax.set_xlim(lags[0], lags[-1])
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def _draw_panel_b(ax, d):
    ec = d["ec"]
    a = ec["regression_slope"]
    b = ec["regression_intercept"]
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
    ax.scatter(X_arr, Y_arr, s=40, color="0.4", edgecolor="white",
               linewidth=0.5, zorder=4, label=f"CMIP6 (n={len(rows)})")
    xx = np.linspace(min(X_arr.min(), X_obs_mean - 3 * X_obs_sigma) - 0.02,
                     max(X_arr.max(), X_obs_mean + 3 * X_obs_sigma) + 0.02,
                     200)
    ax.plot(xx, a * xx + b, color="#CC3333", lw=1.5, zorder=5,
            label="Emergent regression")
    ax.fill_between(xx, a * xx + b - 1.96 * sigma_res,
                    a * xx + b + 1.96 * sigma_res,
                    color="#CC3333", alpha=0.15, zorder=3,
                    label="95% forecast interval")
    sigma_y = (Y_high - Y_low) / (2 * 1.96)
    ax.errorbar([X_obs_mean], [Y_point],
                xerr=[[X_obs_sigma], [X_obs_sigma]],
                yerr=[[sigma_y], [sigma_y]],
                fmt="o", color="#3366AA", ecolor="#3366AA",
                elinewidth=1.5, capsize=4, capthick=1.5,
                markersize=6, markeredgecolor="white",
                markeredgewidth=0.8, zorder=8,
                label=r"Constrained forecast (mean $\pm 1\sigma$)")
    ax.set_xlabel(r"F$_{ovS}$ at 34.5°S, 2000-2024 (Sv)")
    ax.set_ylabel(rf"$\Delta$AMOC$_{{26°N}}$ ({forecast_lo}-{forecast_hi} "
                  rf"$-$ {base_lo}-{base_hi}, Sv)")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)


def _render_combined(d, output: Path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    _draw_panel_a(ax_a, d)
    _panel_label(ax_a, "(a)")
    _draw_panel_b(ax_b, d)
    _panel_label(ax_b, "(b)")
    fig.tight_layout()
    save_publication_figure(fig, output)


def _render_split(d, output: Path):
    base = output.parent / output.name
    for letter, draw in [("a", _draw_panel_a), ("b", _draw_panel_b)]:
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        draw(ax, d)
        fig.tight_layout()
        save_publication_figure(fig, base.with_name(base.name + letter))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leadlag", type=Path,
                        default=Path("data/results/cmip6_fovs_amoc_leadlag.nc"))
    parser.add_argument("--emergent", type=Path,
                        default=Path("data/results/emergent_constraint_rapid_forecast.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure6"))
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    d = _load_data(args.leadlag, args.emergent)

    if args.mode in ("combined", "both"):
        _render_combined(d, args.output)
    if args.mode in ("split", "both"):
        _render_split(d, args.output)


if __name__ == "__main__":
    main()
