#!/usr/bin/env python3
"""Generate 4-figure set for JGR:Oceans / Ocean Science reanalysis assessment.

Reframed from "AMOC tipping evidence" to "How well do ocean reanalyses
capture AMOC freshwater transport?"

Figure set (max 4 for GRL format, adaptable to JGR:Oceans):
  Fig 1: Multi-product F_ovS comparison (ORAS5 full record + overlap period)
  Fig 2: RAPID validation at 26.5N (the success story, r=0.74)
  Fig 3: Salinity pile-up — the robust fingerprint (both products agree)
  Fig 4: Assessment summary — what reanalyses capture vs what they miss

Key differences from previous version:
  - Leads with product disagreement as a finding, not F_ovS trend as evidence
  - Honestly shows SAMBA validation failure alongside RAPID success
  - Salinity pile-up replaces GMT correlation as the robust result
  - Assessment framing instead of tipping narrative
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from ardp.viz.style import (
    COLORS,
    FINGERPRINT_COLORS,
    add_panel_label,
    add_trend_annotation,
    apply_nature_style,
    figure_grl_full,
    save_publication_figure,
)


def _time_to_years(time: xr.DataArray) -> np.ndarray:
    """Convert xarray time coordinate to fractional years."""
    try:
        ts = pd.DatetimeIndex(time.values)
        return np.array([t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
                         for t in ts])
    except Exception:
        if hasattr(time.values[0], "year"):
            return np.array([t.year + (t.month - 1) / 12.0
                             for t in time.values])
        return time.values.astype(float)


def _compute_trend(ts: xr.DataArray) -> dict:
    """Compute linear trend with confidence interval and p-value."""
    years = _time_to_years(ts["time"])
    values = ts.values.ravel()
    valid = np.isfinite(values)

    if valid.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "pvalue": np.nan,
                "stderr": np.nan, "trend_line": np.full_like(values, np.nan)}

    result = stats.linregress(years[valid], values[valid])
    trend_line = result.slope * years + result.intercept

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "pvalue": result.pvalue,
        "stderr": result.stderr,
        "trend_line": trend_line,
        "r_squared": result.rvalue ** 2,
    }


def _align_monthly(
    oras5: xr.DataArray, obs: xr.DataArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Align ORAS5 and observational data on common months.

    Returns (oras5_vals, obs_vals, common_times) or None if < 3 overlap.
    """
    oras5_months = pd.DatetimeIndex(oras5.time.values).to_period("M")
    obs_months = pd.DatetimeIndex(obs.time.values).to_period("M")
    common = oras5_months.intersection(obs_months)

    if len(common) < 3:
        return None

    o_mask = np.isin(oras5_months, common)
    s_mask = np.isin(obs_months, common)

    o_vals = oras5.values.ravel()[o_mask]
    s_vals = obs.values.ravel()[s_mask]
    times = oras5.time.values[o_mask]

    valid = np.isfinite(o_vals) & np.isfinite(s_vals)
    if valid.sum() < 3:
        return None

    return o_vals[valid], s_vals[valid], times[valid]


def _scatter_validation(
    ax: plt.Axes,
    obs_vals: np.ndarray,
    model_vals: np.ndarray,
    obs_label: str,
    model_label: str,
    color: str,
) -> None:
    """Plot validation scatter with statistics."""
    ax.scatter(obs_vals, model_vals, color=color, s=8,
               edgecolors="none", alpha=0.5, zorder=3)

    vmin = min(obs_vals.min(), model_vals.min()) - 1
    vmax = max(obs_vals.max(), model_vals.max()) + 1
    ax.plot([vmin, vmax], [vmin, vmax], color="0.6", linewidth=0.5,
            linestyle=":", zorder=1)

    reg = stats.linregress(obs_vals, model_vals)
    x_fit = np.linspace(vmin, vmax, 100)
    ax.plot(x_fit, reg.slope * x_fit + reg.intercept,
            color=COLORS["red"], linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{obs_label} [Sv]")
    ax.set_ylabel(f"{model_label} [Sv]")

    r, p = stats.pearsonr(obs_vals, model_vals)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(
        0.03, 0.97,
        f"r = {r:.2f}\nn = {len(obs_vals)}",
        transform=ax.transAxes,
        fontsize=5, va="top",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Multi-product F_ovS — the disagreement IS the finding
# ─────────────────────────────────────────────────────────────────────

def figure1_fovs_multiproduct(results_dir: Path, output_dir: Path) -> None:
    """Figure 1: (a) ORAS5 + GLORYS12 F_ovS time series, (b) MOC streamfunction.

    Two-panel figure: F_ovS with both products and trend lines on top, and
    the mean Atlantic overturning streamfunction Ψ(lat, depth) below.
    """
    import matplotlib.colors as mcolors

    fovs_path = results_dir / "oras5_f_ovs.nc"
    if not fovs_path.exists():
        fovs_path = results_dir / "f_ovs.nc"
    if not fovs_path.exists():
        print(f"Skipping Figure 1: no F_ovS data found in {results_dir}")
        return

    f_ovs = xr.open_dataarray(fovs_path)
    trend_full = _compute_trend(f_ovs)

    scale = 1e3 if np.nanmax(np.abs(f_ovs.values)) < 1 else 1.0
    unit = "mSv" if scale == 1e3 else "Sv"
    data = f_ovs * scale

    # Load GLORYS12 F_ovS if available
    glorys_fovs_path = results_dir / "glorys12_f_ovs.nc"
    has_glorys = glorys_fovs_path.exists()
    if has_glorys:
        glorys_fovs = xr.open_dataarray(glorys_fovs_path)
        glorys_data = glorys_fovs * scale

    # Check for MOC streamfunction cache
    psi_path = results_dir / "moc_streamfunction_2005_2024.npz"
    has_psi = psi_path.exists()

    apply_nature_style()

    if has_psi:
        fig = plt.figure(figsize=(6.73, 5.5))
        gs = fig.add_gridspec(
            2, 1, height_ratios=[1, 1.3], hspace=0.32,
            left=0.10, right=0.95, top=0.96, bottom=0.08,
        )
        ax = fig.add_subplot(gs[0])
    else:
        fig, ax = figure_grl_full(nrows=1, ncols=1, height_ratio=0.50)

    # --- Panel (a): F_ovS time series ---
    # ORAS5
    ax.plot(f_ovs.time.values, data.values, color=FINGERPRINT_COLORS["f_ovs"],
            linewidth=0.5, alpha=0.3)

    if len(data) > 24:
        rolling = data.rolling(time=12, center=True).mean()
        ax.plot(f_ovs.time.values, rolling.values,
                color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                label="ORAS5 12-month mean")

    slope_full = trend_full["slope"] * scale
    p_full = trend_full["pvalue"]
    p_str_full = "p < 0.001" if p_full < 0.001 else f"p = {p_full:.3f}"
    ax.plot(f_ovs.time.values, trend_full["trend_line"] * scale,
            color=COLORS["red"], linewidth=1.2, linestyle="--",
            label=f"ORAS5 1958\u20132025: {slope_full:+.2f} {unit}/yr ({p_str_full})")

    # GLORYS12 overlay
    if has_glorys:
        ax.plot(glorys_fovs.time.values, glorys_data.values,
                color=COLORS["green"], linewidth=0.5, alpha=0.3)

        if len(glorys_data) > 24:
            glorys_rolling = glorys_data.rolling(time=12, center=True).mean()
            ax.plot(glorys_fovs.time.values, glorys_rolling.values,
                    color=COLORS["green"], linewidth=1.5,
                    label="GLORYS12 12-month mean")

        glorys_trend = _compute_trend(glorys_fovs)
        glorys_slope = glorys_trend["slope"] * scale
        glorys_p = glorys_trend["pvalue"]
        glorys_p_str = "p < 0.001" if glorys_p < 0.001 else f"p = {glorys_p:.3f}"
        ax.plot(glorys_fovs.time.values, glorys_trend["trend_line"] * scale,
                color=COLORS["green"], linewidth=1.2, linestyle="--",
                alpha=0.8,
                label=f"GLORYS12 1993\u20132025: {glorys_slope:+.2f} {unit}/yr ({glorys_p_str})")
    else:
        # Fall back to ORAS5 sub-period trend if no GLORYS12
        f_ovs_sub = f_ovs.sel(time=slice("1993-01", None))
        if len(f_ovs_sub) > 3:
            trend_sub = _compute_trend(f_ovs_sub)
            slope_sub = trend_sub["slope"] * scale
            p_sub = trend_sub["pvalue"]
            p_str_sub = "p < 0.001" if p_sub < 0.001 else f"p = {p_sub:.3f}"
            ax.plot(f_ovs_sub.time.values, trend_sub["trend_line"] * scale,
                    color=COLORS["purple"], linewidth=1.2, linestyle="--",
                    label=f"1993\u20132025: {slope_sub:+.2f} {unit}/yr ({p_str_sub})")

    ax.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
    ax.legend(loc="lower left", fontsize=5)
    if has_psi:
        add_panel_label(ax, "a", x=-0.08, y=1.05)

    # --- Panel (b): MOC streamfunction cross-section ---
    if has_psi:
        cached = np.load(psi_path)
        psi = cached["psi"]
        lat_psi = cached["lat"]
        depth_psi = cached["depth"]

        ax_psi = fig.add_subplot(gs[1])

        # Limit to Atlantic latitudes
        lat_mask = (lat_psi >= -35) & (lat_psi <= 70)
        psi_plot = psi[:, lat_mask]
        lat_plot = lat_psi[lat_mask]

        # Discrete contour levels
        levels = np.arange(-6, 20, 2)
        cmap = plt.cm.RdBu_r.copy()
        norm = mcolors.BoundaryNorm(levels, cmap.N, extend="both")

        cf = ax_psi.contourf(
            lat_plot, depth_psi, psi_plot,
            levels=levels, cmap=cmap, norm=norm, extend="both",
        )
        # Contour lines
        ax_psi.contour(
            lat_plot, depth_psi, psi_plot,
            levels=levels, colors="0.4", linewidths=0.3,
        )
        # Zero contour (thicker)
        ax_psi.contour(
            lat_plot, depth_psi, psi_plot,
            levels=[0], colors="k", linewidths=0.8,
        )

        ax_psi.set_ylim(5500, 0)  # depth increases downward
        ax_psi.set_xlim(-35, 70)
        ax_psi.set_xlabel("Latitude (\u00b0N)", fontsize=7)
        ax_psi.set_ylabel("Depth (m)", fontsize=7)
        ax_psi.tick_params(labelsize=6)

        # Mark RAPID (26.5N) and SAMBA (34.5S) latitudes
        ax_psi.axvline(26.5, color="0.3", linewidth=0.5, linestyle=":",
                       zorder=5)
        ax_psi.axvline(-34.5, color="0.3", linewidth=0.5, linestyle=":",
                       zorder=5)
        ax_psi.text(27.5, 200, "RAPID", fontsize=5, color="0.3", rotation=90,
                    va="top")
        ax_psi.text(-33.5, 200, "34.5\u00b0S", fontsize=5, color="0.3",
                    rotation=90, va="top")

        # Colorbar
        cbar = fig.colorbar(cf, ax=ax_psi, orientation="vertical",
                            shrink=0.85, pad=0.02, aspect=25)
        cbar.set_label("$\\Psi$ (Sv)", fontsize=7)
        cbar.ax.tick_params(labelsize=5)

        ax_psi.set_title(
            "ORAS5 mean Atlantic overturning streamfunction (2005\u20132024)",
            fontsize=8, pad=6,
        )
        add_panel_label(ax_psi, "b", x=-0.08, y=1.05)

    save_publication_figure(fig, output_dir / "fig1_fovs_multiproduct")


# ─────────────────────────────────────────────────────────────────────
# Figure 2: RAPID validation — the success story
# ─────────────────────────────────────────────────────────────────────

def figure2_rapid_validation(results_dir: Path, output_dir: Path) -> None:
    """Figure 2: ORAS5 + GLORYS12 MOC validation at RAPID 26.5N.

    (a) Time series overlay: ORAS5 + GLORYS12 vs RAPID monthly MOC
    (b) Scatter: ORAS5 vs RAPID
    (c) Scatter: GLORYS12 vs RAPID
    """
    moc_oras5_path = results_dir / "oras5_moc_26N.nc"
    moc_glorys_path = results_dir / "glorys12_moc_26N.nc"
    rapid_path = Path("data/external/rapid_moc_monthly.nc")

    if not rapid_path.exists():
        print("Skipping Figure 2: run download_rapid.py")
        return

    rapid_ds = xr.open_dataset(rapid_path)
    rapid_moc = rapid_ds["moc_mar_hc10"]

    has_oras5 = moc_oras5_path.exists()
    has_glorys = moc_glorys_path.exists()

    if not has_oras5 and not has_glorys:
        print("Skipping Figure 2: no MOC data found")
        return

    apply_nature_style()
    fig = plt.figure(figsize=(6.73, 4.2))

    # Layout: wide time series on top, two square scatters below
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.2, 1], wspace=0.35, hspace=0.40,
        left=0.08, right=0.97, top=0.94, bottom=0.08,
    )

    ax_ts = fig.add_subplot(gs[0, :])  # top row, full width
    ax_sc1 = fig.add_subplot(gs[1, 0])
    ax_sc2 = fig.add_subplot(gs[1, 1])

    # --- Panel (a): Time series overlay ---
    # RAPID
    ax_ts.plot(rapid_moc.time.values, rapid_moc.values,
               color=COLORS["red"], linewidth=0.5, alpha=0.3)
    if len(rapid_moc) > 24:
        rolling_r = rapid_moc.rolling(time=12, center=True).mean()
        ax_ts.plot(rapid_moc.time.values, rolling_r.values,
                   color=COLORS["red"], linewidth=1.5, label="RAPID obs")

    # ORAS5
    if has_oras5:
        oras5_moc = xr.open_dataarray(moc_oras5_path)
        ax_ts.plot(oras5_moc.time.values, oras5_moc.values,
                   color=FINGERPRINT_COLORS["f_ovs"], linewidth=0.5, alpha=0.2)
        if len(oras5_moc) > 24:
            rolling_o = oras5_moc.rolling(time=12, center=True).mean()
            ax_ts.plot(oras5_moc.time.values, rolling_o.values,
                       color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                       label="ORAS5")

    # GLORYS12
    if has_glorys:
        glorys_moc = xr.open_dataarray(moc_glorys_path)
        ax_ts.plot(glorys_moc.time.values, glorys_moc.values,
                   color=COLORS["green"], linewidth=0.5, alpha=0.2)
        if len(glorys_moc) > 24:
            rolling_g = glorys_moc.rolling(time=12, center=True).mean()
            ax_ts.plot(glorys_moc.time.values, rolling_g.values,
                       color=COLORS["green"], linewidth=1.5,
                       label="GLORYS12")

    ax_ts.set_ylabel("MOC at 26.5\u00b0N [Sv]")
    ax_ts.legend(loc="lower left", fontsize=5)
    add_panel_label(ax_ts, "a", x=-0.06)

    # --- Panel (b): ORAS5 scatter ---
    if has_oras5:
        aligned = _align_monthly(oras5_moc, rapid_moc)
        if aligned is not None:
            o_v, r_v, _ = aligned
            _scatter_validation(ax_sc1, r_v, o_v, "RAPID [Sv]", "ORAS5 [Sv]",
                                FINGERPRINT_COLORS["f_ovs"])
    add_panel_label(ax_sc1, "b", x=-0.15)

    # --- Panel (c): GLORYS12 scatter ---
    if has_glorys:
        aligned_g = _align_monthly(glorys_moc, rapid_moc)
        if aligned_g is not None:
            g_v, r_v_g, _ = aligned_g
            _scatter_validation(ax_sc2, r_v_g, g_v, "RAPID [Sv]", "GLORYS12 [Sv]",
                                COLORS["green"])
    add_panel_label(ax_sc2, "c", x=-0.15)

    save_publication_figure(fig, output_dir / "fig2_rapid_validation")


# ─────────────────────────────────────────────────────────────────────
# Figure 3: Salinity pile-up — the robust fingerprint
# ─────────────────────────────────────────────────────────────────────

def figure3_salinity_pileup(results_dir: Path, output_dir: Path) -> None:
    """Figure 3: Salinity pile-up — robust across products.

    (a) Salinity pile-up time series with trend (ORAS5/GLORYS12)
    (b) F_ovS vs salinity pile-up scatter (physical relationship test)
    The ONE fingerprint that replicates: +0.01 PSU/yr (ORAS5), +0.006 (GLORYS12).
    """
    pileup_path = results_dir / "salinity_pileup.nc"
    fovs_path = results_dir / "oras5_f_ovs.nc"
    if not fovs_path.exists():
        fovs_path = results_dir / "f_ovs.nc"

    if not pileup_path.exists():
        print(f"Skipping Figure 3: {pileup_path} not found")
        return

    pileup = xr.open_dataarray(pileup_path)
    trend = _compute_trend(pileup)

    fig, (ax1, ax2) = figure_grl_full(nrows=1, ncols=2, height_ratio=0.45)

    # --- Panel (a): Salinity pile-up time series ---
    color = FINGERPRINT_COLORS["salinity_pileup"]
    ax1.plot(pileup.time.values, pileup.values, color=color,
             linewidth=0.5, alpha=0.4)
    ax1.plot(pileup.time.values, trend["trend_line"],
             color="0.3", linewidth=1.0, linestyle="--",
             label="Linear trend")

    if len(pileup) > 24:
        rolling = pileup.rolling(time=12, center=True).mean()
        ax1.plot(pileup.time.values, rolling.values, color=color,
                 linewidth=1.5, label="12-month mean")

    ax1.set_ylabel("Salinity pile-up [PSU]")
    ax1.legend(loc="upper left", fontsize=5)
    ax1.set_title("Salinity pile-up index", fontsize=7)
    add_panel_label(ax1, "a")
    add_trend_annotation(ax1, trend["slope"], "PSU", trend["pvalue"],
                         position="lower right")

    # --- Panel (b): F_ovS vs salinity pile-up ---
    if fovs_path.exists():
        f_ovs = xr.open_dataarray(fovs_path)

        # Annual means for cleaner scatter
        def _annual(ts):
            idx = pd.DatetimeIndex(ts.time.values)
            df = pd.DataFrame({"value": ts.values.ravel(), "year": idx.year})
            return df.groupby("year")["value"].mean()

        ann_fovs = _annual(f_ovs)
        ann_pileup = _annual(pileup)
        common_years = ann_fovs.index.intersection(ann_pileup.index)

        if len(common_years) >= 5:
            fv = ann_fovs.loc[common_years].values
            pv = ann_pileup.loc[common_years].values

            # Scale F_ovS
            scale = 1e3 if np.nanmax(np.abs(fv)) < 1 else 1.0
            fv *= scale
            f_unit = "mSv" if scale == 1e3 else "Sv"

            valid = np.isfinite(fv) & np.isfinite(pv)
            ax2.scatter(pv[valid], fv[valid], s=12, color=COLORS["purple"],
                        edgecolors="none", alpha=0.7, zorder=3)

            r, p = stats.pearsonr(pv[valid], fv[valid])
            reg = stats.linregress(pv[valid], fv[valid])
            x_range = np.linspace(pv[valid].min(), pv[valid].max(), 100)
            ax2.plot(x_range, reg.slope * x_range + reg.intercept,
                     color=COLORS["red"], linewidth=1.0, linestyle="--")

            ax2.text(
                0.03, 0.97,
                f"r = {r:.2f}\nn = {valid.sum()} yr",
                transform=ax2.transAxes,
                fontsize=5, va="top",
            )

            ax2.set_xlabel("Salinity pile-up [PSU]")
            ax2.set_ylabel(f"$F_{{ovS}}$ annual mean [{f_unit}]")
        else:
            ax2.text(0.5, 0.5, "Insufficient overlap",
                     transform=ax2.transAxes, ha="center")
    else:
        ax2.text(0.5, 0.5, "No F_ovS data",
                 transform=ax2.transAxes, ha="center", fontsize=6)

    ax2.set_title("$F_{ovS}$ vs salinity pile-up", fontsize=7)
    add_panel_label(ax2, "b")

    fig.tight_layout(w_pad=1.5)
    save_publication_figure(fig, output_dir / "fig3_salinity_pileup")


# ─────────────────────────────────────────────────────────────────────
# Figure 4: CMIP6 comparison — observed vs model F_ovS trends
# ─────────────────────────────────────────────────────────────────────

def figure4_cmip6_comparison(results_dir: Path, output_dir: Path) -> None:
    """Figure 4: Observed vs CMIP6 F_ovS trends and piControl means.

    Two-panel figure:
      (a) F_ovS historical trends: reanalyses (diamonds) vs CMIP6 (circles)
      (b) piControl mean F_ovS: where each model sits on the bistability axis

    All CMIP6 values are from Weijer et al. (2019) and van Westen et al. (2024).
    """
    cmip6_dir = results_dir / "cmip6"
    ref_path = cmip6_dir / "cmip6_fovs_reference.nc"

    if not ref_path.exists():
        print("Skipping Figure 4: run compute_cmip6_fovs.py first")
        return

    fovs_oras5_path = results_dir / "oras5_f_ovs.nc"
    fovs_glorys_path = results_dir / "glorys12_f_ovs.nc"

    ref_ds = xr.open_dataset(ref_path)

    apply_nature_style()
    fig, (ax_trend, ax_mean) = plt.subplots(
        1, 2, figsize=(6.73, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.45},
    )

    # ── Panel (a): Historical F_ovS trends ──
    labels = []
    trends_msv = []
    colors_list = []
    markers = []

    # Reanalysis trends
    if fovs_oras5_path.exists():
        fovs = xr.open_dataarray(fovs_oras5_path)
        trend = _compute_trend(fovs)
        labels.append("ORAS5 (1958\u20132025)")
        trends_msv.append(trend["slope"] * 1e3)
        colors_list.append(FINGERPRINT_COLORS["f_ovs"])
        markers.append("D")

        fovs_sub = fovs.sel(time=slice("1993-01", None))
        if len(fovs_sub) > 12:
            trend_sub = _compute_trend(fovs_sub)
            labels.append("ORAS5 (1993\u20132025)")
            trends_msv.append(trend_sub["slope"] * 1e3)
            colors_list.append(FINGERPRINT_COLORS["f_ovs"])
            markers.append("D")

    if fovs_glorys_path.exists():
        glorys_fovs = xr.open_dataarray(fovs_glorys_path)
        glorys_trend = _compute_trend(glorys_fovs)
        labels.append("GLORYS12 (1993\u20132025)")
        trends_msv.append(glorys_trend["slope"] * 1e3)
        colors_list.append(COLORS["green"])
        markers.append("D")

    # CMIP6 historical trends (published values)
    for model in ref_ds["model"].values:
        model_str = str(model)
        trend_val = float(ref_ds["historical_trend"].sel(model=model))
        labels.append(model_str)
        trends_msv.append(trend_val)
        colors_list.append(COLORS["grey"])
        markers.append("o")

    if len(labels) == 0:
        print("Skipping Figure 4: no F_ovS data found")
        plt.close(fig)
        return

    y_pos = np.arange(len(labels))
    ax_trend.axvline(0, color="0.7", linewidth=0.5, linestyle=":", zorder=1)

    for i, (label, trend_val, color, marker) in enumerate(
        zip(labels, trends_msv, colors_list, markers)
    ):
        size = 40 if marker == "D" else 25
        edgecolor = "k" if marker == "D" else "0.4"
        ax_trend.scatter(
            trend_val, i, s=size, color=color, marker=marker,
            edgecolors=edgecolor, linewidths=0.5, zorder=5,
        )

    ax_trend.set_yticks(y_pos)
    ax_trend.set_yticklabels(labels, fontsize=5.5)
    ax_trend.set_xlabel("F$_{ovS}$ trend [mSv yr$^{-1}$]")
    ax_trend.set_ylim(-0.5, len(labels) - 0.5)
    ax_trend.invert_yaxis()
    add_panel_label(ax_trend, "a", x=-0.30, y=1.05)

    # ── Panel (b): piControl mean F_ovS (bistability diagnostic) ──
    model_names = [str(m) for m in ref_ds["model"].values]
    pi_means = ref_ds["picontrol_mean"].values

    y_pos_models = np.arange(len(model_names))

    # Color by regime
    regime_colors = []
    for val in pi_means:
        if val < -0.01:
            regime_colors.append(COLORS["red"])     # bistable
        elif val > 0.01:
            regime_colors.append(COLORS["blue"])    # monostable
        else:
            regime_colors.append(COLORS["yellow"])  # near-zero

    ax_mean.axvline(0, color="0.3", linewidth=0.8, linestyle="-", zorder=1)

    for i, (name, val, color) in enumerate(
        zip(model_names, pi_means, regime_colors)
    ):
        ax_mean.scatter(
            val * 1e3, i, s=30, color=color,
            edgecolors="0.3", linewidths=0.5, zorder=5,
        )

    # Mark observed ORAS5 mean
    if fovs_oras5_path.exists():
        fovs = xr.open_dataarray(fovs_oras5_path)
        obs_mean = float(fovs.mean()) * 1e3
        ax_mean.axvline(
            obs_mean, color=FINGERPRINT_COLORS["f_ovs"],
            linewidth=1.2, linestyle="--", zorder=3,
            label=f"ORAS5 mean: {obs_mean:.0f} mSv",
        )

    # Shade bistable region
    xlim = ax_mean.get_xlim()
    ax_mean.axvspan(min(xlim[0], -200), 0, color=COLORS["red"],
                    alpha=0.06, zorder=0)
    ax_mean.text(
        -5, len(model_names) - 0.3, "bistable",
        fontsize=5, color=COLORS["red"], ha="right", style="italic",
    )
    ax_mean.text(
        5, len(model_names) - 0.3, "monostable",
        fontsize=5, color=COLORS["blue"], ha="left", style="italic",
    )

    ax_mean.set_yticks(y_pos_models)
    ax_mean.set_yticklabels(model_names, fontsize=5.5)
    ax_mean.set_xlabel("piControl mean F$_{ovS}$ [mSv]")
    ax_mean.set_ylim(-0.5, len(model_names) - 0.5)
    ax_mean.invert_yaxis()
    ax_mean.legend(loc="lower left", fontsize=5)
    add_panel_label(ax_mean, "b", x=-0.20, y=1.05)

    fig.tight_layout()
    save_publication_figure(fig, output_dir / "fig4_cmip6_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reanalysis assessment figures (4-figure set)."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="figures/grl")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reanalysis assessment figures (4-figure set)...")
    print("  Framing: 'How well do reanalyses capture AMOC freshwater transport?'")
    print()

    figure1_fovs_multiproduct(results_dir, output_dir)
    figure2_rapid_validation(results_dir, output_dir)
    figure3_salinity_pileup(results_dir, output_dir)
    figure4_cmip6_comparison(results_dir, output_dir)

    print("\nDone. Figures saved to:", output_dir)
    print("\nFigure summary:")
    print("  Fig 1: F_ovS full vs satellite-era trends (product-dependence)")
    print("  Fig 2: RAPID validation at 26.5N (success story)")
    print("  Fig 3: Salinity pile-up (robust across products)")
    print("  Fig 4: CMIP6 F_ovS comparison + piControl null distribution")


if __name__ == "__main__":
    main()
