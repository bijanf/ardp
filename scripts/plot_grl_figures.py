#!/usr/bin/env python3
"""Generate 4-figure set for GRL (Geophysical Research Letters).

GRL format: max 4 figures.
  - Single column: 8.4 cm (3.31 in)
  - Full width: 17.1 cm (6.73 in)

Produces:
  Figure 1: F_ovS trend (headline) + ORAS5 MOC validated at RAPID 26.5N
  Figure 2: 2x2 AMOC fingerprints (physical consistency)
  Figure 3: F_ovS vs Global Mean Temperature (thermodynamic driver)
  Figure 4: Multi-panel validation: RAPID scatter, SAMBA scatter, ORAS5 MOC trends

Validation strategy:
  1. RAPID at 26.5N: 20 years of gold-standard MOC observations (2004-2023)
  2. SAMBA at 34.5S: 4 years at the critical F_ovS latitude (2013-2017)
  3. Both use ORAS5 MOC (same variable, same units) for apples-to-apples comparison
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


def _annual_june_means(obs_ds: xr.Dataset | xr.DataArray, var: str | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Extract June means from observational data for comparison with ORAS5 June values."""
    if var is not None:
        da = obs_ds[var]
    else:
        da = obs_ds

    times = pd.DatetimeIndex(da.time.values)
    june_mask = times.month == 6
    if june_mask.sum() == 0:
        # Fall back to annual means
        years = np.unique(times.year)
        vals = np.array([float(da.sel(time=da.time.dt.year == y).mean())
                         for y in years])
        return years, vals

    june_da = da.isel(time=june_mask)
    years = pd.DatetimeIndex(june_da.time.values).year.values
    return years, june_da.values.ravel()


def figure1_fovs_rapid(results_dir: Path, output_dir: Path) -> None:
    """Figure 1: F_ovS trend + RAPID MOC validation at 26.5N.

    (a) ORAS5 F_ovS 1958-2023 with trend — the headline tipping indicator
    (b) ORAS5 MOC vs RAPID observations at 26.5N — establishes reanalysis credibility
    """
    fovs_path = results_dir / "f_ovs.nc"
    moc_26n_path = results_dir / "oras5_moc_26N.nc"
    rapid_path = Path("data/external/rapid_moc_monthly.nc")

    if not fovs_path.exists():
        print(f"Skipping Figure 1: {fovs_path} not found")
        return

    f_ovs = xr.open_dataarray(fovs_path)
    trend = _compute_trend(f_ovs)

    scale = 1e3 if np.nanmax(np.abs(f_ovs.values)) < 1 else 1.0
    unit = "mSv" if scale == 1e3 else "Sv"
    data = f_ovs * scale

    fig, (ax1, ax2) = figure_grl_full(nrows=1, ncols=2, height_ratio=0.45)

    # --- Panel (a): F_ovS time series ---
    ax1.plot(f_ovs.time.values, data.values, color=FINGERPRINT_COLORS["f_ovs"],
             linewidth=0.5, alpha=0.5)
    ax1.plot(f_ovs.time.values, trend["trend_line"] * scale,
             color=COLORS["red"], linewidth=1.0, linestyle="--",
             label="Linear trend")

    if len(data) > 12:
        rolling = data.rolling(time=5, center=True).mean()
        ax1.plot(f_ovs.time.values, rolling.values,
                 color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                 label="$F_{ovS}$ 5-pt mean")

    ax1.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
    ax1.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax1.legend(loc="lower left", fontsize=5)
    add_panel_label(ax1, "a")
    add_trend_annotation(ax1, trend["slope"] * scale, unit, trend["pvalue"])

    # --- Panel (b): ORAS5 vs RAPID MOC at 26.5N ---
    has_rapid = moc_26n_path.exists() and rapid_path.exists()
    if has_rapid:
        oras5_moc = xr.open_dataarray(moc_26n_path)
        rapid_ds = xr.open_dataset(rapid_path)

        # Get ORAS5 years and values
        oras5_years = _time_to_years(oras5_moc["time"])
        oras5_yrs = np.floor(oras5_years).astype(int)
        oras5_vals = oras5_moc.values.ravel()

        # Get RAPID June means for matching years
        rapid_yrs, rapid_vals = _annual_june_means(rapid_ds, "moc_mar_hc10")

        # Find common years
        common = np.intersect1d(oras5_yrs, rapid_yrs)
        if len(common) >= 3:
            o_vals = np.array([oras5_vals[oras5_yrs == y][0] for y in common])
            r_vals = np.array([rapid_vals[rapid_yrs == y][0] for y in common])

            valid = np.isfinite(o_vals) & np.isfinite(r_vals)
            o_v, r_v, yrs = o_vals[valid], r_vals[valid], common[valid]

            # Scatter
            ax2.scatter(r_v, o_v, color=FINGERPRINT_COLORS["f_ovs"], s=20,
                        edgecolors="none", alpha=0.8, zorder=3)

            # 1:1 line
            vmin = min(o_v.min(), r_v.min()) - 1
            vmax = max(o_v.max(), r_v.max()) + 1
            ax2.plot([vmin, vmax], [vmin, vmax], color="0.6", linewidth=0.5,
                     linestyle=":", zorder=1)

            # Regression
            reg = stats.linregress(r_v, o_v)
            x_fit = np.linspace(vmin, vmax, 100)
            ax2.plot(x_fit, reg.slope * x_fit + reg.intercept,
                     color=COLORS["red"], linewidth=1.0, linestyle="--", zorder=2)

            ax2.set_xlim(vmin, vmax)
            ax2.set_ylim(vmin, vmax)
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_xlabel("RAPID MOC [Sv]")
            ax2.set_ylabel("ORAS5 MOC [Sv]")

            r, p = stats.pearsonr(r_v, o_v)
            bias = np.mean(o_v - r_v)
            rmse = np.sqrt(np.mean((o_v - r_v) ** 2))
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax2.annotate(
                f"r = {r:.2f} ({p_str})\n"
                f"bias = {bias:+.1f} Sv\n"
                f"RMSE = {rmse:.1f} Sv\n"
                f"n = {len(o_v)} years",
                xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=5, va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "0.7", "alpha": 0.9},
            )
        else:
            ax2.text(0.5, 0.5, "Insufficient overlap",
                     transform=ax2.transAxes, ha="center")
    else:
        missing = []
        if not moc_26n_path.exists():
            missing.append("compute_oras5_moc.py")
        if not rapid_path.exists():
            missing.append("download_rapid.py")
        ax2.text(0.5, 0.5, f"Run: {', '.join(missing)}",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=6)

    add_panel_label(ax2, "b")

    fig.tight_layout(w_pad=1.5)
    save_publication_figure(fig, output_dir / "fig1_fovs_rapid")


def figure2_fingerprints(results_dir: Path, output_dir: Path) -> None:
    """Figure 2: 2x2 panel of AMOC fingerprints."""
    files = {
        "f_ovs": ("f_ovs.nc", "$F_{ovS}$", "Sv", FINGERPRINT_COLORS["f_ovs"]),
        "pileup": ("salinity_pileup.nc", "Salinity pile-up", "PSU",
                    FINGERPRINT_COLORS["salinity_pileup"]),
        "nawh": ("nawh.nc", "NAWH index", r"$\degree$C",
                 FINGERPRINT_COLORS["nawh"]),
        "gs": ("gulf_stream_destab.nc", "GS destabilization",
               r"$\degree$lon", FINGERPRINT_COLORS["gulf_stream"]),
    }

    data = {}
    for key, (fname, label, unit, color) in files.items():
        path = results_dir / fname
        if path.exists():
            data[key] = (xr.open_dataarray(path), label, unit, color)

    if not data:
        print("Skipping Figure 2: no fingerprint results found")
        return

    n = len(data)
    nrows = 2 if n > 2 else 1
    ncols = 2 if n > 1 else 1

    fig, axes = figure_grl_full(nrows=nrows, ncols=ncols, height_ratio=0.45)
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    panels = "abcdefgh"
    for i, (key, (ts, label, unit, color)) in enumerate(data.items()):
        ax = axes[i]
        trend = _compute_trend(ts)

        ax.plot(ts.time.values, ts.values, color=color, linewidth=0.7)
        ax.plot(ts.time.values, trend["trend_line"],
                color="0.3", linewidth=0.8, linestyle="--")

        if len(ts) > 24:
            rolling = ts.rolling(time=12, center=True).mean()
            ax.plot(ts.time.values, rolling.values, color=color,
                    linewidth=1.3, alpha=0.9)

        ax.set_ylabel(f"{label} [{unit}]")
        add_panel_label(ax, panels[i])
        add_trend_annotation(ax, trend["slope"], unit, trend["pvalue"],
                             position="upper left")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save_publication_figure(fig, output_dir / "fig2_fingerprints")


def figure3_gmt_correlation(results_dir: Path, output_dir: Path) -> None:
    """Figure 3: F_ovS decline correlated with global warming."""
    fovs_path = results_dir / "f_ovs.nc"
    gmt_path = Path("data/external/gmt_gistemp.nc")

    if not fovs_path.exists():
        print(f"Skipping Figure 3: {fovs_path} not found")
        return
    if not gmt_path.exists():
        print(f"Skipping Figure 3: {gmt_path} not found")
        print("  Run: python scripts/download_gmt.py")
        return

    f_ovs = xr.open_dataarray(fovs_path)
    gmt = xr.open_dataarray(gmt_path)

    fovs_years = _time_to_years(f_ovs["time"])
    fovs_annual_year = np.floor(fovs_years).astype(int)
    unique_years = np.unique(fovs_annual_year)
    fovs_ann_vals = np.array([
        float(f_ovs.values[fovs_annual_year == y].mean())
        for y in unique_years
    ])

    scale = 1e3 if np.nanmax(np.abs(fovs_ann_vals)) < 1 else 1.0
    unit = "mSv" if scale == 1e3 else "Sv"
    fovs_ann_vals *= scale

    gmt_years = _time_to_years(gmt["time"]).astype(int)
    overlap = np.intersect1d(unique_years, gmt_years)
    if len(overlap) < 5:
        print("Skipping Figure 3: insufficient year overlap")
        return

    fovs_overlap = np.array([fovs_ann_vals[unique_years == y][0] for y in overlap])
    gmt_overlap = np.array([float(gmt.values[gmt_years == y][0]) for y in overlap])

    valid = np.isfinite(fovs_overlap) & np.isfinite(gmt_overlap)
    fovs_overlap = fovs_overlap[valid]
    gmt_overlap = gmt_overlap[valid]
    overlap = overlap[valid]

    fovs_trend = stats.linregress(overlap, fovs_overlap)
    gmt_trend = stats.linregress(overlap, gmt_overlap)
    r, p_corr = stats.pearsonr(gmt_overlap, fovs_overlap)

    fig, (ax1, ax2) = figure_grl_full(nrows=1, ncols=2, height_ratio=0.45)

    color_fovs = FINGERPRINT_COLORS["f_ovs"]
    color_gmt = COLORS["red"]

    ax1.plot(overlap, fovs_overlap, color=color_fovs, linewidth=0.8,
             marker="o", markersize=2, label=f"$F_{{ovS}}$ [{unit}]")
    ax1.plot(overlap, fovs_trend.slope * overlap + fovs_trend.intercept,
             color=color_fovs, linewidth=1.0, linestyle="--", alpha=0.7)
    ax1.set_ylabel(f"$F_{{ovS}}$ [{unit}]", color=color_fovs)
    ax1.tick_params(axis="y", colors=color_fovs)

    ax1_r = ax1.twinx()
    ax1_r.spines["right"].set_visible(True)
    ax1_r.plot(overlap, gmt_overlap, color=color_gmt, linewidth=0.8,
               marker="s", markersize=2, label="GMT anomaly")
    ax1_r.plot(overlap, gmt_trend.slope * overlap + gmt_trend.intercept,
               color=color_gmt, linewidth=1.0, linestyle="--", alpha=0.7)
    ax1_r.set_ylabel("GMT anomaly [\u00b0C]", color=color_gmt)
    ax1_r.tick_params(axis="y", colors=color_gmt)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=5)
    add_panel_label(ax1, "a")

    p_str = "p < 0.001" if fovs_trend.pvalue < 0.001 else f"p = {fovs_trend.pvalue:.3f}"
    ax1.annotate(
        f"$F_{{ovS}}$ trend: {fovs_trend.slope:+.2f} {unit}/yr ({p_str})",
        xy=(0.03, 0.12), xycoords="axes fraction", fontsize=5,
        color=color_fovs,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.9},
    )

    ax2.scatter(gmt_overlap, fovs_overlap, color=color_fovs, s=12,
                edgecolors="none", alpha=0.7, zorder=3)
    reg = stats.linregress(gmt_overlap, fovs_overlap)
    gmt_range = np.linspace(gmt_overlap.min(), gmt_overlap.max(), 100)
    ax2.plot(gmt_range, reg.slope * gmt_range + reg.intercept,
             color=COLORS["red"], linewidth=1.0, linestyle="--")
    ax2.set_xlabel("GMT anomaly [\u00b0C]")
    ax2.set_ylabel(f"$F_{{ovS}}$ annual mean [{unit}]")

    p_corr_str = "p < 0.001" if p_corr < 0.001 else (
        f"p = {p_corr:.3f}" if p_corr < 0.01 else f"p = {p_corr:.2f}")
    ax2.annotate(
        f"r = {r:.2f} ({p_corr_str})",
        xy=(0.03, 0.95), xycoords="axes fraction", fontsize=6,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.9},
    )
    add_panel_label(ax2, "b")

    fig.tight_layout(w_pad=2.5)
    save_publication_figure(fig, output_dir / "fig3_gmt_correlation")


def figure4_validation_summary(results_dir: Path, output_dir: Path) -> None:
    """Figure 4: Triple validation — RAPID scatter, SAMBA scatter, MOC trends.

    (a) ORAS5 vs RAPID MOC scatter at 26.5N (20 years)
    (b) ORAS5 vs SAMBA MOC scatter at 34.5S (4 years)
    (c) ORAS5 MOC time series at 26.5N and 34.5S with trends
    """
    moc_26n_path = results_dir / "oras5_moc_26N.nc"
    moc_34s_path = results_dir / "oras5_moc_34S.nc"
    rapid_path = Path("data/external/rapid_moc_monthly.nc")
    samba_path = Path("data/external/samba_moc_monthly.nc")

    if not moc_26n_path.exists() or not moc_34s_path.exists():
        print("Skipping Figure 4: run compute_oras5_moc.py first")
        return

    oras5_26n = xr.open_dataarray(moc_26n_path)
    oras5_34s = xr.open_dataarray(moc_34s_path)

    fig = plt.figure(figsize=(6.73, 4.5))
    from ardp.viz.style import apply_nature_style
    apply_nature_style()

    # Layout: top row = two scatters, bottom = full-width time series
    ax_rapid = fig.add_axes([0.08, 0.55, 0.38, 0.40])
    ax_samba = fig.add_axes([0.58, 0.55, 0.38, 0.40])
    ax_ts = fig.add_axes([0.08, 0.08, 0.88, 0.38])

    # --- Panel (a): RAPID scatter ---
    if rapid_path.exists():
        rapid_ds = xr.open_dataset(rapid_path)
        rapid_yrs, rapid_vals = _annual_june_means(rapid_ds, "moc_mar_hc10")

        oras5_yrs = np.floor(_time_to_years(oras5_26n["time"])).astype(int)
        oras5_vals = oras5_26n.values.ravel()

        common = np.intersect1d(oras5_yrs, rapid_yrs)
        if len(common) >= 3:
            o_v = np.array([oras5_vals[oras5_yrs == y][0] for y in common])
            r_v = np.array([rapid_vals[rapid_yrs == y][0] for y in common])
            valid = np.isfinite(o_v) & np.isfinite(r_v)
            o_v, r_v = o_v[valid], r_v[valid]

            ax_rapid.scatter(r_v, o_v, color=FINGERPRINT_COLORS["f_ovs"],
                             s=18, edgecolors="none", alpha=0.8, zorder=3)

            vmin = min(o_v.min(), r_v.min()) - 1
            vmax = max(o_v.max(), r_v.max()) + 1
            ax_rapid.plot([vmin, vmax], [vmin, vmax], color="0.6",
                          linewidth=0.5, linestyle=":", zorder=1)
            reg = stats.linregress(r_v, o_v)
            x_fit = np.linspace(vmin, vmax, 100)
            ax_rapid.plot(x_fit, reg.slope * x_fit + reg.intercept,
                          color=COLORS["red"], linewidth=1.0, linestyle="--")
            ax_rapid.set_xlim(vmin, vmax)
            ax_rapid.set_ylim(vmin, vmax)
            ax_rapid.set_aspect("equal", adjustable="box")

            r, p = stats.pearsonr(r_v, o_v)
            bias = np.mean(o_v - r_v)
            rmse = np.sqrt(np.mean((o_v - r_v) ** 2))
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax_rapid.annotate(
                f"r = {r:.2f} ({p_str})\n"
                f"bias = {bias:+.1f} Sv\n"
                f"RMSE = {rmse:.1f} Sv\n"
                f"n = {len(o_v)}",
                xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=5, va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "0.7", "alpha": 0.9},
            )

    ax_rapid.set_xlabel("RAPID MOC [Sv]")
    ax_rapid.set_ylabel("ORAS5 MOC [Sv]")
    ax_rapid.set_title("26.5\u00b0N validation", fontsize=7)
    add_panel_label(ax_rapid, "a", x=-0.15)

    # --- Panel (b): SAMBA scatter ---
    if samba_path.exists():
        samba_ds = xr.open_dataset(samba_path)
        samba_yrs, samba_vals = _annual_june_means(samba_ds, "upper_cell")

        oras5_yrs_34 = np.floor(_time_to_years(oras5_34s["time"])).astype(int)
        oras5_vals_34 = oras5_34s.values.ravel()

        common = np.intersect1d(oras5_yrs_34, samba_yrs)
        if len(common) >= 3:
            o_v = np.array([oras5_vals_34[oras5_yrs_34 == y][0] for y in common])
            s_v = np.array([samba_vals[samba_yrs == y][0] for y in common])
            valid = np.isfinite(o_v) & np.isfinite(s_v)
            o_v, s_v = o_v[valid], s_v[valid]

            ax_samba.scatter(s_v, o_v, color=COLORS["green"],
                             s=18, edgecolors="none", alpha=0.8, zorder=3)

            vmin = min(o_v.min(), s_v.min()) - 2
            vmax = max(o_v.max(), s_v.max()) + 2
            ax_samba.plot([vmin, vmax], [vmin, vmax], color="0.6",
                          linewidth=0.5, linestyle=":", zorder=1)

            if len(o_v) >= 3:
                reg = stats.linregress(s_v, o_v)
                x_fit = np.linspace(vmin, vmax, 100)
                ax_samba.plot(x_fit, reg.slope * x_fit + reg.intercept,
                              color=COLORS["red"], linewidth=1.0, linestyle="--")

                r, p = stats.pearsonr(s_v, o_v)
                bias = np.mean(o_v - s_v)
                p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
                ax_samba.annotate(
                    f"r = {r:.2f} ({p_str})\n"
                    f"bias = {bias:+.1f} Sv\n"
                    f"n = {len(o_v)}",
                    xy=(0.03, 0.95), xycoords="axes fraction",
                    fontsize=5, va="top",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                          "edgecolor": "0.7", "alpha": 0.9},
                )
            ax_samba.set_xlim(vmin, vmax)
            ax_samba.set_ylim(vmin, vmax)
            ax_samba.set_aspect("equal", adjustable="box")
        else:
            ax_samba.text(0.5, 0.5, f"Overlap: {len(common)} years",
                          transform=ax_samba.transAxes, ha="center")
    else:
        ax_samba.text(0.5, 0.5, "Run: download_samba.py",
                      transform=ax_samba.transAxes, ha="center", fontsize=6)

    ax_samba.set_xlabel("SAMBA MOC [Sv]")
    ax_samba.set_ylabel("ORAS5 MOC [Sv]")
    ax_samba.set_title("34.5\u00b0S validation", fontsize=7)
    add_panel_label(ax_samba, "b", x=-0.15)

    # --- Panel (c): ORAS5 MOC time series at both latitudes ---
    years_26n = _time_to_years(oras5_26n["time"])
    years_34s = _time_to_years(oras5_34s["time"])

    ax_ts.plot(years_26n, oras5_26n.values, color=FINGERPRINT_COLORS["f_ovs"],
               linewidth=0.8, marker="o", markersize=2,
               label="ORAS5 MOC 26.5\u00b0N")
    ax_ts.plot(years_34s, oras5_34s.values, color=COLORS["green"],
               linewidth=0.8, marker="s", markersize=2,
               label="ORAS5 MOC 34.5\u00b0S")

    # Trends
    trend_26n = _compute_trend(oras5_26n)
    trend_34s = _compute_trend(oras5_34s)
    ax_ts.plot(years_26n, trend_26n["trend_line"], color=FINGERPRINT_COLORS["f_ovs"],
               linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ts.plot(years_34s, trend_34s["trend_line"], color=COLORS["green"],
               linewidth=1.0, linestyle="--", alpha=0.7)

    # RAPID mean
    if rapid_path.exists():
        rapid_ds = xr.open_dataset(rapid_path)
        rapid_mean = float(rapid_ds["moc_mar_hc10"].mean())
        ax_ts.axhline(rapid_mean, color="0.5", linewidth=0.5, linestyle=":",
                       label=f"RAPID mean ({rapid_mean:.1f} Sv)")

    ax_ts.set_ylabel("MOC upper-cell transport [Sv]")
    ax_ts.set_xlabel("Year")
    ax_ts.legend(loc="upper right", fontsize=5)

    p26_str = "p < 0.001" if trend_26n["pvalue"] < 0.001 else f"p = {trend_26n['pvalue']:.3f}"
    p34_str = "p < 0.001" if trend_34s["pvalue"] < 0.001 else f"p = {trend_34s['pvalue']:.3f}"
    ax_ts.annotate(
        f"26.5\u00b0N trend: {trend_26n['slope']:+.3f} Sv/yr ({p26_str})\n"
        f"34.5\u00b0S trend: {trend_34s['slope']:+.3f} Sv/yr ({p34_str})",
        xy=(0.03, 0.05), xycoords="axes fraction",
        fontsize=5, va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.9},
    )
    add_panel_label(ax_ts, "c", x=-0.06)

    save_publication_figure(fig, output_dir / "fig4_validation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GRL publication figures (max 4)."
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="figures/grl")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating GRL figures (4-figure set)...")
    figure1_fovs_rapid(results_dir, output_dir)
    figure2_fingerprints(results_dir, output_dir)
    figure3_gmt_correlation(results_dir, output_dir)
    figure4_validation_summary(results_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
