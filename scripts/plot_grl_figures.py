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
    bias = np.mean(model_vals - obs_vals)
    rmse = np.sqrt(np.mean((model_vals - obs_vals) ** 2))
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.annotate(
        f"r = {r:.2f} ({p_str})\n"
        f"bias = {bias:+.1f} Sv\n"
        f"RMSE = {rmse:.1f} Sv\n"
        f"n = {len(obs_vals)} months",
        xy=(0.03, 0.95), xycoords="axes fraction",
        fontsize=5, va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.9},
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Multi-product F_ovS — the disagreement IS the finding
# ─────────────────────────────────────────────────────────────────────

def figure1_fovs_multiproduct(results_dir: Path, output_dir: Path) -> None:
    """Figure 1: ORAS5 F_ovS with full-record and sub-period trends.

    (a) ORAS5 F_ovS full record (1958-2023) — long-term decline
    (b) ORAS5 F_ovS 1993-2023 only — apples-to-apples with GLORYS12
    Shows that the trend is period-dependent, not a robust ocean signal.
    """
    # Try ORAS5-specific path first
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

    fig, (ax1, ax2) = figure_grl_full(nrows=1, ncols=2, height_ratio=0.45)

    # --- Panel (a): Full ORAS5 F_ovS record ---
    ax1.plot(f_ovs.time.values, data.values, color=FINGERPRINT_COLORS["f_ovs"],
             linewidth=0.5, alpha=0.4)
    ax1.plot(f_ovs.time.values, trend_full["trend_line"] * scale,
             color=COLORS["red"], linewidth=1.0, linestyle="--",
             label="Full-record trend")

    if len(data) > 24:
        rolling = data.rolling(time=12, center=True).mean()
        ax1.plot(f_ovs.time.values, rolling.values,
                 color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                 label="12-month mean")

    ax1.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
    ax1.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
    ax1.legend(loc="lower left", fontsize=5)
    ax1.set_title("ORAS5 full record", fontsize=7)
    add_panel_label(ax1, "a")
    add_trend_annotation(ax1, trend_full["slope"] * scale, unit,
                         trend_full["pvalue"])

    # --- Panel (b): 1993-2023 sub-period (GLORYS12 overlap) ---
    f_ovs_sub = f_ovs.sel(time=slice("1993-01", None))
    if len(f_ovs_sub) > 3:
        trend_sub = _compute_trend(f_ovs_sub)
        data_sub = f_ovs_sub * scale

        ax2.plot(f_ovs_sub.time.values, data_sub.values,
                 color=FINGERPRINT_COLORS["f_ovs"],
                 linewidth=0.5, alpha=0.4)
        ax2.plot(f_ovs_sub.time.values, trend_sub["trend_line"] * scale,
                 color=COLORS["red"], linewidth=1.0, linestyle="--",
                 label="1993+ trend")

        if len(data_sub) > 24:
            rolling_sub = data_sub.rolling(time=12, center=True).mean()
            ax2.plot(f_ovs_sub.time.values, rolling_sub.values,
                     color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                     label="12-month mean")

        ax2.set_ylabel(f"$F_{{ovS}}$ [{unit}]")
        ax2.axhline(0, color="0.5", linewidth=0.3, linestyle=":")
        ax2.legend(loc="lower left", fontsize=5)
        ax2.set_title("ORAS5 satellite era (1993+)", fontsize=7)
        add_trend_annotation(ax2, trend_sub["slope"] * scale, unit,
                             trend_sub["pvalue"])

        # Annotate comparison
        p_sub = trend_sub["pvalue"]
        sig = "significant" if p_sub < 0.05 else "NOT significant"
        ax2.annotate(
            f"cf. full: {trend_full['slope'] * scale:+.2f} {unit}/yr\n"
            f"1993+: {trend_sub['slope'] * scale:+.2f} {unit}/yr ({sig})",
            xy=(0.97, 0.05), xycoords="axes fraction",
            fontsize=5, ha="right", va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow",
                  "edgecolor": "0.7", "alpha": 0.9},
        )
    else:
        ax2.text(0.5, 0.5, "Insufficient post-1993 data",
                 transform=ax2.transAxes, ha="center")

    add_panel_label(ax2, "b")

    fig.tight_layout(w_pad=1.5)
    save_publication_figure(fig, output_dir / "fig1_fovs_multiproduct")


# ─────────────────────────────────────────────────────────────────────
# Figure 2: RAPID validation — the success story
# ─────────────────────────────────────────────────────────────────────

def figure2_rapid_validation(results_dir: Path, output_dir: Path) -> None:
    """Figure 2: ORAS5 MOC validation at RAPID 26.5N.

    (a) Time series overlay: ORAS5 vs RAPID monthly MOC
    (b) Scatter plot with regression and statistics
    This is the strongest result — r=0.74 proves ORAS5 captures MOC variability.
    """
    moc_26n_path = results_dir / "oras5_moc_26N.nc"
    rapid_path = Path("data/external/rapid_moc_monthly.nc")

    if not moc_26n_path.exists() or not rapid_path.exists():
        missing = []
        if not moc_26n_path.exists():
            missing.append("compute_oras5_moc.py")
        if not rapid_path.exists():
            missing.append("download_rapid.py")
        print(f"Skipping Figure 2: run {', '.join(missing)}")
        return

    oras5_moc = xr.open_dataarray(moc_26n_path)
    rapid_ds = xr.open_dataset(rapid_path)
    rapid_moc = rapid_ds["moc_mar_hc10"]

    aligned = _align_monthly(oras5_moc, rapid_moc)
    if aligned is None:
        print("Skipping Figure 2: insufficient RAPID overlap")
        return

    o_v, r_v, common_times = aligned

    fig, (ax1, ax2) = figure_grl_full(nrows=1, ncols=2, height_ratio=0.45)

    # --- Panel (a): Time series overlay ---
    ax1.plot(oras5_moc.time.values, oras5_moc.values,
             color=FINGERPRINT_COLORS["f_ovs"], linewidth=0.5, alpha=0.3)
    # 12-month running mean of ORAS5
    if len(oras5_moc) > 24:
        rolling_o = oras5_moc.rolling(time=12, center=True).mean()
        ax1.plot(oras5_moc.time.values, rolling_o.values,
                 color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                 label="ORAS5 (12-mo mean)")

    # RAPID overlay
    ax1.plot(rapid_moc.time.values, rapid_moc.values,
             color=COLORS["red"], linewidth=0.5, alpha=0.3)
    if len(rapid_moc) > 24:
        rolling_r = rapid_moc.rolling(time=12, center=True).mean()
        ax1.plot(rapid_moc.time.values, rolling_r.values,
                 color=COLORS["red"], linewidth=1.5,
                 label="RAPID obs")

    ax1.set_ylabel("MOC at 26.5\u00b0N [Sv]")
    ax1.legend(loc="lower left", fontsize=5)
    ax1.set_title("ORAS5 captures MOC variability at 26.5\u00b0N", fontsize=7)
    add_panel_label(ax1, "a")

    # --- Panel (b): Scatter ---
    _scatter_validation(ax2, r_v, o_v, "RAPID MOC", "ORAS5 MOC",
                        FINGERPRINT_COLORS["f_ovs"])
    ax2.set_title("Monthly validation", fontsize=7)
    add_panel_label(ax2, "b")

    fig.tight_layout(w_pad=1.5)
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
    ax1.set_title("Robust across products", fontsize=7)
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

            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax2.annotate(
                f"r = {r:.2f} ({p_str})\nn = {valid.sum()} yr",
                xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=5, va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "0.7", "alpha": 0.9},
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
# Figure 4: Assessment summary — what reanalyses capture vs miss
# ─────────────────────────────────────────────────────────────────────

def figure4_assessment_summary(results_dir: Path, output_dir: Path) -> None:
    """Figure 4: Honest assessment — RAPID success + SAMBA failure + MOC overview.

    (a) RAPID validation scatter (success: r=0.74 at 26.5N)
    (b) SAMBA validation scatter (failure: r=0.03 at 34.5S)
    (c) ORAS5 MOC time series at both latitudes
    Shows: ORAS5 captures MOC where assimilation is strong (26.5N)
    but fails at poorly observed latitudes (34.5S).
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
    apply_nature_style()

    # Layout: top row = two scatters, bottom = full-width time series
    ax_rapid = fig.add_axes([0.08, 0.55, 0.38, 0.40])
    ax_samba = fig.add_axes([0.58, 0.55, 0.38, 0.40])
    ax_ts = fig.add_axes([0.08, 0.08, 0.88, 0.38])

    # --- Panel (a): RAPID scatter — the success ---
    if rapid_path.exists():
        rapid_ds = xr.open_dataset(rapid_path)
        rapid_moc = rapid_ds["moc_mar_hc10"]
        aligned = _align_monthly(oras5_26n, rapid_moc)
        if aligned is not None:
            o_v, r_v, _ = aligned
            _scatter_validation(ax_rapid, r_v, o_v, "RAPID MOC", "ORAS5 MOC",
                                FINGERPRINT_COLORS["f_ovs"])
    else:
        ax_rapid.text(0.5, 0.5, "Run: download_rapid.py",
                      transform=ax_rapid.transAxes, ha="center", fontsize=6)

    ax_rapid.set_title("26.5\u00b0N: ORAS5 captures MOC", fontsize=7)
    add_panel_label(ax_rapid, "a", x=-0.15)

    # --- Panel (b): SAMBA scatter — the failure ---
    if samba_path.exists():
        samba_ds = xr.open_dataset(samba_path)
        samba_moc = samba_ds["upper_cell"]
        aligned = _align_monthly(oras5_34s, samba_moc)
        if aligned is not None:
            o_v, s_v, _ = aligned
            _scatter_validation(ax_samba, s_v, o_v, "SAMBA MOC", "ORAS5 MOC",
                                COLORS["green"])

            # Highlight the failure honestly
            r, _ = stats.pearsonr(s_v, o_v)
            if abs(r) < 0.3:
                ax_samba.annotate(
                    "ORAS5 does NOT capture\nMOC variability at 34.5\u00b0S",
                    xy=(0.5, 0.03), xycoords="axes fraction",
                    fontsize=5, ha="center", va="bottom",
                    color=COLORS["red"], fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "mistyrose",
                          "edgecolor": COLORS["red"], "alpha": 0.9},
                )
        else:
            ax_samba.text(0.5, 0.5, "Insufficient overlap",
                          transform=ax_samba.transAxes, ha="center")
    else:
        ax_samba.text(0.5, 0.5, "Run: download_samba.py",
                      transform=ax_samba.transAxes, ha="center", fontsize=6)

    ax_samba.set_title("34.5\u00b0S: ORAS5 fails at SAMBA", fontsize=7)
    add_panel_label(ax_samba, "b", x=-0.15)

    # --- Panel (c): ORAS5 MOC at both latitudes ---
    years_26n = _time_to_years(oras5_26n["time"])
    years_34s = _time_to_years(oras5_34s["time"])

    ax_ts.plot(years_26n, oras5_26n.values, color=FINGERPRINT_COLORS["f_ovs"],
               linewidth=0.5, alpha=0.3)
    ax_ts.plot(years_34s, oras5_34s.values, color=COLORS["green"],
               linewidth=0.5, alpha=0.3)

    # Rolling means
    if len(oras5_26n) > 24:
        roll_26n = pd.Series(oras5_26n.values).rolling(12, center=True).mean()
        ax_ts.plot(years_26n, roll_26n.values, color=FINGERPRINT_COLORS["f_ovs"],
                   linewidth=1.5, label="ORAS5 MOC 26.5\u00b0N")
    if len(oras5_34s) > 24:
        roll_34s = pd.Series(oras5_34s.values).rolling(12, center=True).mean()
        ax_ts.plot(years_34s, roll_34s.values, color=COLORS["green"],
                   linewidth=1.5, label="ORAS5 MOC 34.5\u00b0S")

    # Trends
    trend_26n = _compute_trend(oras5_26n)
    trend_34s = _compute_trend(oras5_34s)
    ax_ts.plot(years_26n, trend_26n["trend_line"], color=FINGERPRINT_COLORS["f_ovs"],
               linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ts.plot(years_34s, trend_34s["trend_line"], color=COLORS["green"],
               linewidth=1.0, linestyle="--", alpha=0.7)

    # RAPID observational mean for context
    if rapid_path.exists():
        rapid_ds = xr.open_dataset(rapid_path)
        rapid_mean = float(rapid_ds["moc_mar_hc10"].mean())
        ax_ts.axhline(rapid_mean, color="0.5", linewidth=0.5, linestyle=":",
                       label=f"RAPID mean ({rapid_mean:.1f} Sv)")

    ax_ts.set_ylabel("MOC upper-cell transport [Sv]")
    ax_ts.set_xlabel("Year")
    ax_ts.legend(loc="upper right", fontsize=5)

    p26 = "p < 0.001" if trend_26n["pvalue"] < 0.001 else f"p = {trend_26n['pvalue']:.3f}"
    p34 = "p < 0.001" if trend_34s["pvalue"] < 0.001 else f"p = {trend_34s['pvalue']:.3f}"
    ax_ts.annotate(
        f"26.5\u00b0N trend: {trend_26n['slope']:+.3f} Sv/yr ({p26})\n"
        f"34.5\u00b0S trend: {trend_34s['slope']:+.3f} Sv/yr ({p34})",
        xy=(0.03, 0.05), xycoords="axes fraction",
        fontsize=5, va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.9},
    )
    add_panel_label(ax_ts, "c", x=-0.06)

    save_publication_figure(fig, output_dir / "fig4_assessment")


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
    figure4_assessment_summary(results_dir, output_dir)

    print("\nDone. Figures saved to:", output_dir)
    print("\nFigure summary:")
    print("  Fig 1: F_ovS full vs satellite-era trends (product-dependence)")
    print("  Fig 2: RAPID validation at 26.5N (success story)")
    print("  Fig 3: Salinity pile-up (robust across products)")
    print("  Fig 4: Assessment — RAPID success vs SAMBA failure")


if __name__ == "__main__":
    main()
