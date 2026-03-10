#!/usr/bin/env python3
"""SSS trend decomposition: why Atlantic trends are NOT just water cycle.

Shows that the "salty gets saltier" amplification pattern (Held & Soden 2006)
fails in the Atlantic — ocean dynamics (AMOC, ITCZ, Arctic exchange) dominate.

Panel (a): Zonal-mean scatter of climatological SSS vs SSS trend, with
           labeled deviations identifying distinct physical processes.
Panel (b): Map of the deviation from the amplification model, highlighting
           where ocean dynamics dominate.

The failure of amplification is the argument: Atlantic SSS trends require
active ocean circulation changes, not just passive E-P response.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter

from ardp.viz.style import (
    add_panel_label,
    apply_nature_style,
    save_publication_figure,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_oras5_sss(data_dir: Path) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Load ORAS5 2D surface salinity."""
    files = sorted(data_dir.glob("sosaline_control_monthly_highres_2D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ORAS5 SSS files in {data_dir}")
    print(f"  Found {len(files)} monthly files")

    ds0 = xr.open_dataset(files[0])
    nav_lon = ds0["nav_lon"].values
    nav_lat = ds0["nav_lat"].values
    ds0.close()

    chunks: list[np.ndarray] = []
    times: list[np.datetime64] = []
    for f in files:
        ds = xr.open_dataset(f)
        chunks.append(ds["sosaline"].values[0])
        times.append(ds["time_counter"].values[0])
        ds.close()

    sss = xr.DataArray(
        np.stack(chunks, axis=0),
        dims=("time", "y", "x"),
        coords={"time": np.array(times)},
    )
    return sss, nav_lon, nav_lat


def load_glorys12_sss(data_dir: Path) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Load GLORYS12 surface salinity."""
    sys.path.insert(0, str(Path(__file__).parent))
    from plot_sss_trend_map import load_glorys12_sss as _load

    sss = _load(data_dir)
    lon1d = sss["x"].values
    lat1d = sss["y"].values
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    return sss, lon2d, lat2d


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

def compute_trend_field(
    sss: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel OLS trend on deseasonalized anomalies."""
    times = sss["time"].values
    time_objs = times.astype("datetime64[us]").astype("object")
    years = np.array([
        t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
        for t in time_objs
    ])
    months = np.array([t.month for t in time_objs])

    data = sss.values
    nt, ny, nx = data.shape
    data_2d = data.reshape(nt, -1)

    print("  Removing seasonal cycle...")
    anom_2d = data_2d.copy()
    for m in range(1, 13):
        mask = months == m
        clim = np.nanmean(data_2d[mask, :], axis=0)
        anom_2d[mask, :] -= clim

    valid_mask = np.isfinite(anom_2d)
    n_valid = valid_mask.sum(axis=0).reshape(ny, nx)

    trend = np.full(ny * nx, np.nan)
    pvalue = np.full(ny * nx, np.nan)

    all_valid = valid_mask.all(axis=0)
    if all_valid.any():
        Y = anom_2d[:, all_valid]
        t_mean = years.mean()
        y_mean = Y.mean(axis=0)
        t_anom = years - t_mean
        ss_tt = np.sum(t_anom ** 2)
        slopes = t_anom @ (Y - y_mean) / ss_tt
        residuals = Y - (np.outer(t_anom, slopes) + y_mean)
        dof = nt - 2
        mse = np.sum(residuals ** 2, axis=0) / dof
        se_slope = np.sqrt(mse / ss_tt)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = slopes / se_slope
        pvals = 2.0 * sp_stats.t.sf(np.abs(t_stat), dof)
        trend[all_valid] = slopes * 10.0
        pvalue[all_valid] = pvals

    partial = (~all_valid) & (n_valid.ravel() >= 24)
    idx_partial = np.where(partial)[0]
    if len(idx_partial) > 0:
        print(f"  Processing {len(idx_partial)} partial pixels...")
        for j in idx_partial:
            mask_j = valid_mask[:, j]
            slope, _, _, p, _ = sp_stats.linregress(years[mask_j], anom_2d[mask_j, j])
            trend[j] = slope * 10.0
            pvalue[j] = p

    return trend.reshape(ny, nx), pvalue.reshape(ny, nx), n_valid


# ---------------------------------------------------------------------------
# Atlantic masking
# ---------------------------------------------------------------------------

def build_atlantic_mask(
    lon2d: np.ndarray, lat2d: np.ndarray, trend: np.ndarray
) -> np.ndarray:
    """Build Atlantic basin mask excluding marginal seas."""
    mask = np.ones_like(trend, dtype=bool)
    mask &= (lon2d >= -80) & (lon2d <= 20)
    mask &= ~((lon2d > -6) & (lat2d > 30) & (lat2d < 46))
    mask &= ~((lon2d > 10) & (lat2d > 54))
    mask &= ~((lon2d < -75) & (lat2d > 50) & (lat2d < 66))
    mask &= ~((lon2d < -82) & (lat2d > 18) & (lat2d < 31))
    mask &= np.isfinite(trend)
    return mask


# ---------------------------------------------------------------------------
# Zonal-mean analysis
# ---------------------------------------------------------------------------

def compute_zonal_means(
    mean_sss: np.ndarray,
    trend: np.ndarray,
    lat2d: np.ndarray,
    atlantic_mask: np.ndarray,
    band_width: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute zonal-mean SSS and trend in latitude bands.

    Returns lat_centers, zonal_mean_sss, zonal_mean_trend.
    """
    lat_edges = np.arange(-55, 70 + band_width, band_width)
    n = len(lat_edges) - 1
    lat_c = np.full(n, np.nan)
    zm_sss = np.full(n, np.nan)
    zm_trend = np.full(n, np.nan)

    valid = atlantic_mask & np.isfinite(mean_sss)

    for i in range(n):
        band = valid & (lat2d >= lat_edges[i]) & (lat2d < lat_edges[i + 1])
        if band.sum() > 100:
            lat_c[i] = 0.5 * (lat_edges[i] + lat_edges[i + 1])
            zm_sss[i] = np.nanmean(mean_sss[band])
            zm_trend[i] = np.nanmean(trend[band])

    ok = np.isfinite(lat_c)
    return lat_c[ok], zm_sss[ok], zm_trend[ok]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_decomposition_figure(
    mean_sss: np.ndarray,
    trend: np.ndarray,
    atlantic_mask: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    time_label: str,
    outpath: Path,
) -> None:
    """Two-panel figure: zonal-mean scatter + deviation map."""
    apply_nature_style()

    fig = plt.figure(figsize=(6.73, 4.5))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, 1.3], wspace=0.15,
        left=0.09, right=0.95, top=0.90, bottom=0.13,
    )

    proj = ccrs.PlateCarree()

    # --- Compute zonal means ---
    lat_c, zm_sss, zm_trend = compute_zonal_means(
        mean_sss, trend, lat2d, atlantic_mask, band_width=5.0
    )

    # Regression on zonal means
    slope, intercept, r, p, se = sp_stats.linregress(zm_sss, zm_trend)
    r_sq = r**2

    # Predicted and residual at each zonal band
    zm_predicted = intercept + slope * zm_sss
    zm_residual = zm_trend - zm_predicted

    # Also compute pixel-level predicted and residual for map
    valid = atlantic_mask & np.isfinite(mean_sss)
    trend_predicted = np.full_like(trend, np.nan)
    trend_predicted[valid] = intercept + slope * mean_sss[valid]
    trend_residual = np.full_like(trend, np.nan)
    trend_residual[valid] = trend[valid] - trend_predicted[valid]

    # --- Panel (a): Zonal-mean scatter ---
    ax = fig.add_subplot(gs[0])

    # Color points by latitude (discrete colorbar)
    import matplotlib.colors as mcolors
    lat_bounds = np.arange(-60, 80, 10)
    lat_norm = mcolors.BoundaryNorm(lat_bounds, plt.cm.coolwarm.N)
    sc = ax.scatter(
        zm_sss, zm_trend,
        c=lat_c, cmap="coolwarm", norm=lat_norm, s=50, zorder=10,
        edgecolors="0.3", linewidths=0.5,
    )

    # Regression line
    x_range = np.array([zm_sss.min() - 0.3, zm_sss.max() + 0.3])
    ax.plot(
        x_range, intercept + slope * x_range,
        color="0.5", linewidth=1.0, linestyle="--", zorder=5,
    )

    # Zero line
    ax.axhline(0, color="0.7", linewidth=0.4, linestyle="-", zorder=0)

    # Concise stats: R² and p only
    p_str = f"p = {p:.2f}" if p >= 0.01 else "p < 0.01"
    ax.text(
        0.04, 0.97,
        f"R\u00b2 = {r_sq:.2f}, {p_str}\nn = {len(zm_sss)} bands",
        transform=ax.transAxes,
        fontsize=5, va="top", ha="left",
    )

    # Colorbar
    cbar_sc = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar_sc.set_label("Latitude (\u00b0N)", fontsize=5)
    cbar_sc.ax.tick_params(labelsize=4.5)

    ax.set_xlabel("Zonal-mean climatological SSS (PSU)", fontsize=6)
    ax.set_ylabel("Zonal-mean SSS trend (PSU decade$^{-1}$)", fontsize=6)
    ax.tick_params(labelsize=5)

    add_panel_label(ax, "a", x=-0.15, y=1.05)

    # --- Panel (b): Deviation map ---
    ax_map = fig.add_subplot(gs[1], projection=proj)
    ax_map.set_extent([-80, 30, -55, 70], crs=proj)

    ax_map.add_feature(cfeature.LAND, facecolor="#a09e99", edgecolor="none", zorder=2)
    ax_map.add_feature(cfeature.OCEAN, facecolor="#f7f9fc", edgecolor="none", zorder=0)
    ax_map.coastlines(linewidth=0.3, color="0.45", zorder=3)

    gl = ax_map.gridlines(
        draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5,
        linestyle=":", zorder=1,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 5, "color": "0.4"}
    gl.ylabel_style = {"size": 5, "color": "0.4"}

    vmax = np.nanpercentile(np.abs(trend_residual), 98)
    vmax = np.ceil(vmax * 20) / 20

    # Discrete colorbar: define level boundaries
    import matplotlib.colors as mcolors
    n_levels = 10
    bounds = np.linspace(-vmax, vmax, n_levels + 1)
    cmap_base = plt.cm.RdBu_r.copy()
    cmap_base.set_bad(color="none")  # transparent so ocean background shows
    norm = mcolors.BoundaryNorm(bounds, cmap_base.N)

    im = ax_map.pcolormesh(
        lon2d, lat2d, trend_residual,
        transform=proj,
        cmap=cmap_base, norm=norm,
        shading="auto", zorder=1,
    )

    # Smoothed contours
    resid_smooth = trend_residual.copy()
    resid_smooth[np.isnan(resid_smooth)] = 0
    resid_smooth = gaussian_filter(resid_smooth, sigma=8)
    resid_smooth[np.isnan(trend_residual)] = np.nan

    ax_map.contour(
        lon2d, lat2d, resid_smooth,
        levels=[0.04],
        colors=["#c0392b"], linewidths=[1.0], linestyles=["solid"],
        transform=proj, zorder=4,
    )
    ax_map.contour(
        lon2d, lat2d, resid_smooth,
        levels=[-0.04],
        colors=["#2471a3"], linewidths=[1.0], linestyles=["solid"],
        transform=proj, zorder=4,
    )
    ax_map.contour(
        lon2d, lat2d, resid_smooth,
        levels=[0],
        colors=["0.4"], linewidths=[0.6], linestyles=["--"],
        transform=proj, zorder=3,
    )

    cax = fig.add_axes([0.52, 0.04, 0.38, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label(
        "SSS trend beyond water cycle (PSU decade$^{-1}$)",
        fontsize=5.5,
    )
    cbar.ax.tick_params(labelsize=4.5)

    ax_map.set_title(
        f"Residual SSS trend ({time_label})",
        fontsize=7, pad=8,
    )
    add_panel_label(ax_map, "b", x=0.01, y=1.03)

    save_publication_figure(fig, outpath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSS trend decomposition: amplification vs ocean dynamics"
    )
    parser.add_argument(
        "--product",
        choices=["oras5", "glorys12"],
        default="glorys12",
        help="Reanalysis product (default: glorys12)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/grl/fig_sss_decomposition"),
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Start year for trend computation (e.g. 1993)",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="End year for trend computation (e.g. 2023)",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path("data/oras5") if args.product == "oras5" \
            else Path("data/glorys12")

    if args.product == "oras5":
        print("Loading ORAS5 SSS data...")
        sss, lon2d, lat2d = load_oras5_sss(args.data_dir)
    else:
        print("Loading GLORYS12 SSS data...")
        sss, lon2d, lat2d = load_glorys12_sss(args.data_dir)

    # Filter by year range if requested
    if args.start_year is not None or args.end_year is not None:
        time_objs = sss["time"].values.astype("datetime64[us]").astype("object")
        year_arr = np.array([t.year for t in time_objs])
        keep = np.ones(len(year_arr), dtype=bool)
        if args.start_year is not None:
            keep &= year_arr >= args.start_year
        if args.end_year is not None:
            keep &= year_arr <= args.end_year
        sss = sss.isel(time=keep)
        print(f"  Filtered to {args.start_year or 'start'}-{args.end_year or 'end'}: {sss.shape[0]} months")

    t0 = sss["time"].values[0]
    t1 = sss["time"].values[-1]
    t0_obj = np.datetime64(t0, "us").astype("object")
    t1_obj = np.datetime64(t1, "us").astype("object")
    time_label = f"{args.product.upper()}  {t0_obj.year}\u2013{t1_obj.year}"
    print(f"  Shape: {sss.shape}, time: {t0} to {t1}")

    print("Computing climatological mean SSS...")
    mean_sss = np.nanmean(sss.values, axis=0)

    print("Computing per-pixel trends...")
    trend, pvalue, n_valid = compute_trend_field(sss)

    atlantic_mask = build_atlantic_mask(lon2d, lat2d, trend)
    print(f"  Atlantic ocean pixels: {atlantic_mask.sum():,}")

    # Zonal-mean analysis
    lat_c, zm_sss, zm_trend = compute_zonal_means(
        mean_sss, trend, lat2d, atlantic_mask
    )
    slope, intercept, r, p, se = sp_stats.linregress(zm_sss, zm_trend)
    print(f"\n  Zonal-mean amplification regression:")
    print(f"    beta = {slope:.4f}, alpha = {intercept:.4f}")
    print(f"    R\u00b2 = {r**2:.4f}, p = {p:.4f}")

    # Diagnostics
    stsa_mask = (
        atlantic_mask
        & (lat2d >= -35) & (lat2d <= -15)
        & (lon2d >= -60) & (lon2d <= 20)
    )
    if stsa_mask.any():
        valid = atlantic_mask & np.isfinite(mean_sss)
        trend_pred = np.full_like(trend, np.nan)
        trend_pred[valid] = intercept + slope * mean_sss[valid]
        stsa_obs = np.nanmean(trend[stsa_mask])
        stsa_pred = np.nanmean(trend_pred[stsa_mask])
        stsa_resid = stsa_obs - stsa_pred
        print(f"\n  Subtropical South Atlantic:")
        print(f"    Observed:      {stsa_obs:+.4f} PSU/dec")
        print(f"    Amplification: {stsa_pred:+.4f} PSU/dec")
        print(f"    Deviation:     {stsa_resid:+.4f} PSU/dec ({stsa_resid/stsa_obs*100:.0f}% unexplained)")

    print("\nPlotting...")
    plot_decomposition_figure(
        mean_sss, trend, atlantic_mask, lon2d, lat2d,
        time_label, args.output,
    )
    print("Done.")


if __name__ == "__main__":
    main()
