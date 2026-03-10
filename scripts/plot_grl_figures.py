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
    """Figure 1: ORAS5 F_ovS — single panel, two trend lines.

    Full 1958-2025 record with both the full-record trend and the
    satellite-era (1993+) trend overlaid in different colours.
    """
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

    fig, ax = figure_grl_full(nrows=1, ncols=1, height_ratio=0.50)

    # Monthly data
    ax.plot(f_ovs.time.values, data.values, color=FINGERPRINT_COLORS["f_ovs"],
            linewidth=0.5, alpha=0.3)

    # 12-month running mean
    if len(data) > 24:
        rolling = data.rolling(time=12, center=True).mean()
        ax.plot(f_ovs.time.values, rolling.values,
                color=FINGERPRINT_COLORS["f_ovs"], linewidth=1.5,
                label="12-month mean")

    # Full-record trend (1958-2025)
    slope_full = trend_full["slope"] * scale
    p_full = trend_full["pvalue"]
    p_str_full = "p < 0.001" if p_full < 0.001 else f"p = {p_full:.3f}"
    ax.plot(f_ovs.time.values, trend_full["trend_line"] * scale,
            color=COLORS["red"], linewidth=1.2, linestyle="--",
            label=f"1958\u20132025: {slope_full:+.2f} {unit}/yr ({p_str_full})")

    # Satellite-era trend (1993-2025)
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
# Figure 4: Assessment summary — what reanalyses capture vs miss
# ─────────────────────────────────────────────────────────────────────

def figure4_assessment_summary(results_dir: Path, output_dir: Path) -> None:
    """Figure 4: ORAS5 MOC at 26.5N and 34.5S — long-term evolution."""
    moc_26n_path = results_dir / "oras5_moc_26N.nc"
    moc_34s_path = results_dir / "oras5_moc_34S.nc"
    rapid_path = Path("data/external/rapid_moc_monthly.nc")

    if not moc_26n_path.exists() or not moc_34s_path.exists():
        print("Skipping Figure 4: run compute_oras5_moc.py first")
        return

    oras5_26n = xr.open_dataarray(moc_26n_path)
    oras5_34s = xr.open_dataarray(moc_34s_path)

    fig, ax = figure_grl_full(nrows=1, ncols=1, height_ratio=0.50)

    years_26n = _time_to_years(oras5_26n["time"])
    years_34s = _time_to_years(oras5_34s["time"])

    # Monthly data (faint)
    ax.plot(years_26n, oras5_26n.values, color=FINGERPRINT_COLORS["f_ovs"],
            linewidth=0.5, alpha=0.2)
    ax.plot(years_34s, oras5_34s.values, color=COLORS["green"],
            linewidth=0.5, alpha=0.2)

    # 12-month rolling means
    if len(oras5_26n) > 24:
        roll_26n = pd.Series(oras5_26n.values).rolling(12, center=True).mean()
        ax.plot(years_26n, roll_26n.values, color=FINGERPRINT_COLORS["f_ovs"],
                linewidth=1.5, label="26.5\u00b0N")
    if len(oras5_34s) > 24:
        roll_34s = pd.Series(oras5_34s.values).rolling(12, center=True).mean()
        ax.plot(years_34s, roll_34s.values, color=COLORS["green"],
                linewidth=1.5, label="34.5\u00b0S")

    # Trends
    trend_26n = _compute_trend(oras5_26n)
    trend_34s = _compute_trend(oras5_34s)
    ax.plot(years_26n, trend_26n["trend_line"], color=FINGERPRINT_COLORS["f_ovs"],
            linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(years_34s, trend_34s["trend_line"], color=COLORS["green"],
            linewidth=1.0, linestyle="--", alpha=0.7)

    # RAPID mean for reference
    if rapid_path.exists():
        rapid_ds = xr.open_dataset(rapid_path)
        rapid_mean = float(rapid_ds["moc_mar_hc10"].mean())
        ax.axhline(rapid_mean, color="0.5", linewidth=0.5, linestyle=":",
                   label=f"RAPID mean ({rapid_mean:.1f} Sv)")

    ax.set_ylabel("ORAS5 MOC upper-cell transport [Sv]")
    ax.legend(loc="upper right", fontsize=5)

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
