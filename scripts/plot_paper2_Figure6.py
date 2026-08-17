#!/usr/bin/env python3
"""Paper 2 Figure 6: combined diagnostics + SMILE robustness.

Four panels in a 2x2 layout:

  (a) Cross-correlation curves between F_ovS at 34.5 deg S and AMOC at
      26.5 deg N, per CMIP6 model, with collapsing- and stable-class
      ensemble means. Positive lag = F_ovS leads AMOC.
  (b) CMIP6 pooled emergent regression of historical F_ovS (2000-2024)
      against DeltaAMOC over the 2030-2040 forecast window; observed-mean
      propagated through the regression as a 2-D errorbar.
  (c) F_ovS velocity-share vs salinity-share for all 50 MPI-ESM1-2-LR
      Grand Ensemble members overlaid on the multi-model CMIP6 ensemble.
  (d) AMOC time series at 26.5 deg N (max of msftmz below 500 m), 1850-
      2100, hist+ssp585, for all 50 SMILE members; spaghetti, ensemble
      mean, +/-1 sigma envelope.

Outputs (default --mode both):
  figures/paper2/Figure6.{png,pdf}            single combined PDF
  figures/paper2/Figure6{a,b,c,d}.{png,pdf}   four standalone panels

Reads:
  data/results/cmip6_fovs_amoc_leadlag.nc
  data/results/emergent_constraint_rapid_forecast.json
  data/results/fovs_decomposition_smile_esgf.csv      (50 rows)
  data/results/fovs_decomposition_cmip6_summary.csv   (15 multi-model rows)
  data/results/smile_amoc26n_mpi_lr.npz               (50 trajectories)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure

COLLAPSE_THRESHOLD = 0.30

CLASS_COLORS = {
    "v-dominant": "#E69F00",
    "s-dominant": "#56B4E9",
    "mixed":      "#009E73",
    "increasing": "0.5",
}


def _classify(row):
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def _panel_label(ax, label: str, x: float = 0.0, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_diag(leadlag: Path, emergent: Path):
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


def _load_smile(results_dir: Path):
    smile = pd.read_csv(results_dir / "fovs_decomposition_smile_esgf.csv")
    smile["class"] = smile.apply(_classify, axis=1)
    cmip6 = pd.read_csv(results_dir / "fovs_decomposition_cmip6_summary.csv")
    cmip6["class"] = cmip6.apply(_classify, axis=1)
    npz = np.load(results_dir / "smile_amoc26n_mpi_lr.npz", allow_pickle=True)
    members = [str(m) for m in npz["members"]]
    common_years = np.arange(1850, 2101)
    matrix = np.full((len(members), len(common_years)), np.nan)
    for i, m in enumerate(members):
        y = npz[f"{m}_years"].astype(float)
        a = npz[f"{m}_amoc"].astype(float)
        for j, yr in enumerate(common_years):
            idx = np.where(y == yr)[0]
            if idx.size:
                matrix[i, j] = a[idx[0]]
    return smile, cmip6, members, common_years, matrix


# ---------------------------------------------------------------------------
# Panel renderers — each takes an ax and the relevant data dict / frames.
# ---------------------------------------------------------------------------

def _draw_leadlag(ax, d):
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
                  r"-- positive: F$_{ovS}$ leads AMOC")
    ax.set_ylabel(r"$r(\mathrm{F}_{ovS}(t),\ \mathrm{AMOC}(t+\tau))$")
    ax.set_xlim(lags[0], lags[-1])
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)


def _draw_emergent(ax, d):
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
    ax.set_xlabel(r"F$_{ovS}$ at 34.5$^\circ$S, 2000-2024 (Sv)")
    ax.set_ylabel(rf"$\Delta$AMOC$_{{26^\circ\mathrm{{N}}}}$ "
                  rf"({forecast_lo}-{forecast_hi} - "
                  rf"{base_lo}-{base_hi}, Sv)")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False)


def _draw_smile_class(ax, smile, cmip6):
    # Backdrop: multi-model CMIP6 ensemble.
    for cls in ("v-dominant", "s-dominant", "mixed", "increasing"):
        sub = cmip6[cmip6["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                   s=70, color=CLASS_COLORS[cls], alpha=0.30,
                   edgecolor="0.4", linewidth=0.4, zorder=3,
                   label=f"CMIP6 {cls} ({len(sub)})")
    # SMILE members.
    for cls in ("v-dominant", "s-dominant", "mixed", "increasing"):
        sub = smile[smile["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["velocity_share_pct"], sub["salinity_share_pct"],
                   s=40, marker="x", color=CLASS_COLORS[cls],
                   linewidth=1.5, zorder=5,
                   label=f"SMILE {cls} ({len(sub)})")
    if len(smile):
        ax.scatter([smile["velocity_share_pct"].mean()],
                   [smile["salinity_share_pct"].mean()],
                   s=70, marker="D", c="black", edgecolor="white",
                   linewidth=0.8, zorder=8, label="SMILE mean")
    ax.plot([0, 100], [100, 0], color="0.6", lw=0.6, ls="--", zorder=1)
    ax.axvline(60, color="#E69F00", lw=0.4, ls=":", alpha=0.6)
    ax.axhline(60, color="#56B4E9", lw=0.4, ls=":", alpha=0.6)
    ax.set_xlabel(r"Velocity share $f_v$  (%)")
    ax.set_ylabel(r"Salinity share $f_s$  (%)")
    ax.set_xlim(-80, 220)
    ax.set_ylim(-150, 220)
    ax.legend(loc="lower left", fontsize=6.0, frameon=False, ncol=2,
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.2,
              columnspacing=0.7)


def _draw_smile_amoc(ax, members, years, matrix):
    for i in range(len(members)):
        ax.plot(years, matrix[i], color="#E69F00", alpha=0.20, lw=0.5,
                zorder=2)
    mean_t = np.nanmean(matrix, axis=0)
    std_t = np.nanstd(matrix, axis=0)
    ax.fill_between(years, mean_t - std_t, mean_t + std_t,
                    color="#E69F00", alpha=0.30, zorder=4,
                    label=r"$\pm 1\sigma$ envelope")
    ax.plot(years, mean_t, color="#B45F00", lw=2.0, zorder=6,
            label="Ensemble mean")
    ax.axvline(2014, color="0.4", lw=0.8, ls="--", zorder=1,
               label=r"hist $\rightarrow$ ssp585")
    ax.axvspan(1950, 1980, color="0.85", alpha=0.4, zorder=0)
    ax.axvspan(2081, 2100, color="0.85", alpha=0.4, zorder=0)
    ax.set_xlim(1850, 2100)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"AMOC at 26.5$^\circ$N  (Sv)")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)


# ---------------------------------------------------------------------------
# Render combined + split
# ---------------------------------------------------------------------------

def _render_combined(diag, smile, cmip6, members, years, matrix, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6))
    _draw_leadlag(axes[0, 0], diag)
    _panel_label(axes[0, 0], "a")
    _draw_emergent(axes[0, 1], diag)
    _panel_label(axes[0, 1], "b")
    _draw_smile_class(axes[1, 0], smile, cmip6)
    _panel_label(axes[1, 0], "c")
    _draw_smile_amoc(axes[1, 1], members, years, matrix)
    _panel_label(axes[1, 1], "d")
    fig.tight_layout(h_pad=2.0, w_pad=2.5)
    save_publication_figure(fig, output)


def _render_split(diag, smile, cmip6, members, years, matrix, output: Path):
    base = output.parent / output.name
    panels = [
        ("a", lambda ax: _draw_leadlag(ax, diag),       (4.0, 3.4)),
        ("b", lambda ax: _draw_emergent(ax, diag),      (4.0, 3.4)),
        ("c", lambda ax: _draw_smile_class(ax, smile, cmip6), (4.6, 4.2)),
        ("d", lambda ax: _draw_smile_amoc(ax, members, years, matrix),
                                                         (5.2, 3.6)),
    ]
    for letter, draw, figsize in panels:
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax)
        fig.tight_layout()
        save_publication_figure(fig, base.with_name(base.name + letter))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leadlag", type=Path,
                        default=Path("data/results/cmip6_fovs_amoc_leadlag.nc"))
    parser.add_argument("--emergent", type=Path,
                        default=Path("data/results/emergent_constraint_rapid_forecast.json"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure6"))
    parser.add_argument("--mode", choices=["combined", "split", "both"],
                        default="both")
    args = parser.parse_args()

    apply_nature_style()
    diag = _load_diag(args.leadlag, args.emergent)
    smile, cmip6, members, years, matrix = _load_smile(args.results_dir)

    if args.mode in ("combined", "both"):
        _render_combined(diag, smile, cmip6, members, years, matrix,
                         args.output)
    if args.mode in ("split", "both"):
        _render_split(diag, smile, cmip6, members, years, matrix, args.output)


if __name__ == "__main__":
    main()
