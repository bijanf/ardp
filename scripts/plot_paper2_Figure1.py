#!/usr/bin/env python3
"""Paper 2 Figure 1: Atlantic bistability and opposing mechanisms.

Combined into a single PDF with three panels:

  (a) Long-term F_ovS time series at 34.5°S for ORAS5, GLORYS12V1,
      SODA 3.15.2, ECCO-V4r4, plus direct-hydrography anchors.
  (b) Mechanism decomposition (ΔF_v + ΔF_s + ΔF_cross) per reanalysis
      between 1993-2005 and 2013-2025. Hatched bar = ill-defined
      (|ΔF_total| < 10 mSv).
  (c) Vertical profiles of the per-depth ΔF_v and ΔF_s integrands.

Reads the same NetCDF files used by the atomic plot scripts in
``scripts/plot_paper2_fig1_multiprod_fovs.py`` and
``scripts/plot_paper2_fig2_decomposition.py``.

Outputs: figures/paper2/Figure1.{png,pdf}
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amoc_reanalysis_anomalies import (  # noqa: E402
    _linear_trend_gls,
    _linear_trend_ols,
    _linear_trend_santer,
)

PRODUCTS = [
    ("ORAS5",      "oras5",    "oras5_f_ovs.nc",    "#1f77b4"),
    ("GLORYS12V1", "glorys12", "glorys12_f_ovs.nc", "#2ca02c"),
    ("SODA3.15.2", "soda",     "soda_f_ovs.nc",     "#e377c2"),
    ("ECCO-V4r4",  "ecco",     "ecco_f_ovs.nc",     "#d62728"),
]

DIRECT_HYDRO = [
    (-0.10, 0.10, "Garzoli & Matano 2011", 2005),
    (-0.09, 0.05, "Meinen et al. 2018",    2015),
    (-0.17, 0.07, "Kersalé et al. 2020",   2015),
]

OUTLIER_THRESHOLD = 0.30


def _load_annual(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
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
    bad = np.abs(vals) > OUTLIER_THRESHOLD
    if bad.any():
        vals = vals.copy().astype(float)
        vals[bad] = np.nan
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


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _panel_label(ax, label: str, x: float = 0.02, y: float = 0.97) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/Figure1"))
    args = parser.parse_args()

    apply_nature_style()

    fig = plt.figure(figsize=(7.0, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.32, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ── Panel (a): F_ovS time series ──
    loaded: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    trend_rows = []
    for label, _key, fname, color in PRODUCTS:
        ts = _load_annual(args.results_dir / fname)
        if ts is None:
            continue
        yrs, vals = ts
        loaded.append((label, yrs, vals, color))
        sl_san, p_san, _ = _linear_trend_santer(yrs, vals)
        sl_gls, p_gls, _ = _linear_trend_gls(yrs, vals)
        sl_ols, p_ols, _ = _linear_trend_ols(yrs, vals)
        _ = (sl_ols, p_ols)
        trend_rows.append({
            "product": label, "n_years": len(yrs),
            "santer_slope_Sv_dec": sl_san, "santer_p": p_san,
            "gls_slope_Sv_dec": sl_gls, "gls_p": p_gls,
            "color": color,
        })

    ax_a.axhline(0.0, color="0.6", lw=0.6, zorder=1)
    ax_a.axhspan(-1.0, 0.0, color="#FFF3E0", alpha=0.5, zorder=0)
    ax_a.text(1958.5, -0.28,
              "bistable regime  (F$_{ovS}$ < 0)",
              fontsize=8, color="#CC5500", style="italic", va="bottom",
              zorder=2)
    for label, yrs, vals, color in loaded:
        ax_a.plot(yrs, vals, color=color, alpha=0.25, lw=0.5, zorder=3)
        rm = _running_mean(vals, window=5)
        ax_a.plot(yrs, rm, color=color, lw=1.8, zorder=6, label=label)
    for val, err, tag, yr in DIRECT_HYDRO:
        ax_a.errorbar(yr, val, yerr=err, color="black", marker="D",
                      markersize=4, capsize=2, lw=1.0, zorder=10,
                      markeredgecolor="white", markeredgewidth=0.5)
        ax_a.annotate(tag, xy=(yr, val), xytext=(5, 2),
                      textcoords="offset points", fontsize=7,
                      color="0.3", zorder=11)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14),
                ncol=4, frameon=False, fontsize=9, handlelength=1.8,
                columnspacing=1.8)
    # Trend table intentionally moved out of the figure — keeping the
    # trend numbers in the Main.tex prose instead, per user feedback
    # that the in-axes table overlapped with the bistable-regime
    # annotation in panel (a).
    _ = trend_rows
    ax_a.set_xlim(1958, 2026)
    ax_a.set_ylim(-0.30, 0.05)
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel(r"$\mathrm{F}_{ovS}$  at 34.5°S (Sv)")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    _panel_label(ax_a, "(a)")

    # ── Panel (b): stacked bar decomposition ──
    deco = {}
    for label, key, _, color in PRODUCTS:
        path = args.results_dir / f"fovs_decomposition_{key}.nc"
        if path.exists():
            deco[label] = (xr.open_dataset(path), color)
    labels = list(deco.keys())
    x = np.arange(len(labels))
    dv = np.array([float(deco[lab][0].attrs["delta_v_Sv"]) * 1000 for lab in labels])
    ds_ = np.array([float(deco[lab][0].attrs["delta_s_Sv"]) * 1000 for lab in labels])
    dc = np.array([float(deco[lab][0].attrs["delta_cross_Sv"]) * 1000 for lab in labels])
    dtot = dv + ds_ + dc
    has_trend = np.abs(dtot) >= 10.0
    width = 0.55
    ax_b.bar(x[has_trend], dv[has_trend], width=width, color="#E69F00",
             label=r"$\Delta F_v$  (velocity)")
    ax_b.bar(x[has_trend], ds_[has_trend], width=width,
             bottom=dv[has_trend], color="#56B4E9",
             label=r"$\Delta F_s$  (salinity)")
    ax_b.bar(x[has_trend], dc[has_trend], width=width,
             bottom=dv[has_trend] + ds_[has_trend], color="0.6",
             label=r"$\Delta F_\mathrm{cross}$")
    if (~has_trend).any():
        ax_b.bar(x[~has_trend], dtot[~has_trend], width=width,
                 color="0.85", edgecolor="0.4", hatch="///",
                 label=r"ill-defined ($|\Delta F| < 10$ mSv)")
    ax_b.scatter(x, dtot, color="black", s=30, marker="D", zorder=5,
                 label=r"$\Delta F_\mathrm{total}$")
    ax_b.axhline(0, color="0.6", lw=0.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax_b.set_ylabel(r"$\Delta\mathrm{F}_{ovS}$ (mSv, late − early)")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55),
                fontsize=8, frameon=False, ncol=2, handlelength=1.5,
                columnspacing=1.2)
    ax_b.set_ylim(-100, 28)
    # Annotate v/s percentage on bars
    for i, _lab in enumerate(labels):
        if abs(dtot[i]) < 10:
            # Move label well above zero so it does NOT overlap the
            # ill-defined hatched bar interior (user feedback).
            ax_b.text(i, 14, "no\ntrend", ha="center", va="bottom",
                      fontsize=8, color="0.3", style="italic",
                      fontweight="bold")
            continue
        v_pct = 100 * dv[i] / dtot[i]
        s_pct = 100 * ds_[i] / dtot[i]
        if dtot[i] < 0:
            yy, va = 4, "bottom"
        else:
            yy, va = -4, "top"
        ax_b.text(i, yy, f"v:{v_pct:+.0f}%\ns:{s_pct:+.0f}%",
                  ha="center", va=va, fontsize=8, color="0.2",
                  fontweight="bold")
    _panel_label(ax_b, "(b)")

    # ── Panel (c): depth profiles ──
    for lab in labels:
        ds_obj, color = deco[lab]
        depth = ds_obj["depth"].values
        v_prof = ds_obj["depth_Sv_v"].values * 1000
        s_prof = ds_obj["depth_Sv_s"].values * 1000
        ax_c.plot(v_prof, depth, color=color, lw=1.5, label=f"{lab} (v)")
        ax_c.plot(s_prof, depth, color=color, lw=1.5, ls="--",
                  label=f"{lab} (s)")
    ax_c.axvline(0, color="0.6", lw=0.6)
    ax_c.invert_yaxis()
    ax_c.set_xlabel(r"Per-depth $\Delta F$ (mSv)")
    ax_c.set_ylabel("Depth (m)")
    ax_c.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=7.5, frameon=False, ncol=1,
                handlelength=1.5)
    ax_c.set_ylim(5500, 0)
    _panel_label(ax_c, "(c)")

    fig.tight_layout()
    save_publication_figure(fig, args.output)

    for ds_obj, _ in deco.values():
        ds_obj.close()


if __name__ == "__main__":
    main()
