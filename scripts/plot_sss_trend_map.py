#!/usr/bin/env python3
"""Plot geographic SSS trend map with significance stippling.

Loads GLORYS12 SSS data, computes per-pixel linear trends (PSU/decade),
and produces a two-panel GRL figure:
  (a) SSS trend map with significance stippling and salinity pile-up region boxes
  (b) Zonal-mean SSS trend profile (latitude vs trend)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter

from ardp.constants import SUBTROPICAL_SOUTH_ATLANTIC
from ardp.viz.style import (
    add_panel_label,
    apply_nature_style,
    save_publication_figure,
)


def load_glorys12_sss(data_dir: Path) -> xr.DataArray:
    """Load GLORYS12 surface salinity (SSS).

    Returns a DataArray with dims (time, y, x) and 1D lat/lon coords.
    """
    # Only open salinity variable, select surface depth to minimize memory
    files = sorted(data_dir.glob("*.nc"))
    chunks: list[xr.DataArray] = []
    for f in files:
        ds = xr.open_dataset(f)
        sss_chunk = ds["so"].isel(depth=0).load()
        chunks.append(sss_chunk)
        ds.close()

    sss = xr.concat(chunks, dim="time")

    # Rename dims
    if "latitude" in sss.dims:
        sss = sss.rename({"latitude": "y"})
    if "longitude" in sss.dims:
        sss = sss.rename({"longitude": "x"})

    return sss


def load_oras5_sss(data_dir: Path) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Load ORAS5 2D surface salinity.

    Returns (sss DataArray, nav_lon 2D, nav_lat 2D).
    ORAS5 uses a curvilinear NEMO grid with 2D coordinates.
    """
    files = sorted(data_dir.glob("sosaline_control_monthly_highres_2D_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ORAS5 SSS files in {data_dir}")
    print(f"  Found {len(files)} monthly files")

    ds0 = xr.open_dataset(files[0])
    nav_lon = ds0["nav_lon"].values
    nav_lat = ds0["nav_lat"].values
    ds0.close()

    chunks_list: list[np.ndarray] = []
    times: list[np.datetime64] = []
    for f in files:
        ds = xr.open_dataset(f)
        chunks_list.append(ds["sosaline"].values[0])
        times.append(ds["time_counter"].values[0])
        ds.close()

    sss = xr.DataArray(
        np.stack(chunks_list, axis=0),
        dims=("time", "y", "x"),
        coords={"time": np.array(times)},
    )
    return sss, nav_lon, nav_lat


def _deseasonalize(data_2d: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Remove monthly climatology from each pixel (in-place safe).

    Parameters
    ----------
    data_2d : ndarray (T, N)
        Raw values, may contain NaN.
    months : ndarray (T,)
        Month index 1–12 for each timestep.

    Returns
    -------
    anomaly : ndarray (T, N)
        Deseasonalized anomalies.
    """
    anomaly = data_2d.copy()
    for m in range(1, 13):
        mask = months == m
        clim = np.nanmean(data_2d[mask, :], axis=0)  # (N,)
        anomaly[mask, :] -= clim
    return anomaly


def compute_trend_field(
    sss: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel OLS trend and p-value on deseasonalized anomalies.

    Methodology:
      1. Remove monthly climatology at each pixel (mean Jan, mean Feb, ...),
         producing anomalies that isolate interannual variability.
      2. Fit OLS: anomaly(t) = a + b*t + ε, where t is fractional year.
      3. The slope b is identical whether computed on raw or anomaly data
         (seasonal cycle is periodic ⊥ linear trend over full years), but
         removing the seasonal cycle reduces residual variance, yielding
         smaller standard errors and more powerful significance tests.

    Parameters
    ----------
    sss : xr.DataArray
        Surface salinity with dims (time, y, x).

    Returns
    -------
    trend : ndarray (y, x)
        Linear trend in PSU/decade.
    pvalue : ndarray (y, x)
        Two-sided p-value for the slope (on deseasonalized residuals).
    n_valid : ndarray (y, x)
        Number of valid (non-NaN) timesteps per pixel.
    """
    from scipy import stats as sp_stats

    # Convert time to fractional years and extract month indices
    times = sss["time"].values
    time_objs = times.astype("datetime64[us]").astype("object")
    years = np.array([
        t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
        for t in time_objs
    ])
    months = np.array([t.month for t in time_objs])

    data = sss.values  # (T, Y, X)
    nt, ny, nx = data.shape

    # Reshape to (T, N) where N = ny * nx
    data_2d = data.reshape(nt, -1)

    # Deseasonalize: subtract monthly climatology at each pixel
    print("  Removing seasonal cycle (monthly climatology)...")
    anom_2d = _deseasonalize(data_2d, months)

    # Count valid observations per pixel
    valid_mask = np.isfinite(anom_2d)  # (T, N)
    n_valid = valid_mask.sum(axis=0).reshape(ny, nx)

    # Vectorized OLS on anomalies via normal equations:
    # anom = a + b*t  =>  slope is the interannual trend
    trend = np.full(ny * nx, np.nan)
    pvalue = np.full(ny * nx, np.nan)

    # --- Fast path: pixels valid at all timesteps ---
    all_valid = valid_mask.all(axis=0)  # (N,)
    if all_valid.any():
        Y = anom_2d[:, all_valid]  # (T, M)
        t = years
        t_mean = t.mean()
        y_mean = Y.mean(axis=0)  # (M,)

        # Slope = sum((t - t_mean)*(y - y_mean)) / sum((t - t_mean)^2)
        t_anom = t - t_mean  # (T,)
        ss_tt = np.sum(t_anom ** 2)  # scalar
        ss_ty = t_anom @ (Y - y_mean)  # (M,)
        slopes = ss_ty / ss_tt  # (M,)

        # Residuals and standard error of slope
        predicted = np.outer(t_anom, slopes) + y_mean  # (T, M)
        residuals = Y - predicted
        dof = nt - 2
        mse = np.sum(residuals ** 2, axis=0) / dof  # (M,)
        se_slope = np.sqrt(mse / ss_tt)  # (M,)

        # t-statistic and p-value
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = slopes / se_slope
        pvals = 2.0 * sp_stats.t.sf(np.abs(t_stat), dof)

        trend[all_valid] = slopes * 10.0  # per decade
        pvalue[all_valid] = pvals

    # --- Slow path: pixels with some NaN ---
    partial = (~all_valid) & (n_valid.ravel() >= 24)  # need >= 2 years
    idx_partial = np.where(partial)[0]
    if len(idx_partial) > 0:
        for j in idx_partial:
            mask_j = valid_mask[:, j]
            t_j = years[mask_j]
            y_j = anom_2d[mask_j, j]
            slope, intercept, r, p, se = sp_stats.linregress(t_j, y_j)
            trend[j] = slope * 10.0
            pvalue[j] = p

    trend = trend.reshape(ny, nx)
    pvalue = pvalue.reshape(ny, nx)

    return trend, pvalue, n_valid


def plot_sss_trend_figure(
    trend: np.ndarray,
    pvalue: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    outpath: Path,
    title: str = "GLORYS12 SSS trend  1993\u20132025",
    is_2d_coords: bool = False,
    pileup_path: Path | None = None,
) -> None:
    """Create GRL figure: trend map + zonal-mean profile + optional pile-up.

    If pileup_path is provided, adds a bottom panel (c) with the salinity
    pile-up time series, creating a 3-panel figure. Otherwise 2-panel.

    Parameters
    ----------
    lon, lat : 1D arrays (regular grid) or 2D arrays (curvilinear grid).
    is_2d_coords : if True, lon/lat are already 2D (e.g. ORAS5 NEMO grid).
    pileup_path : path to salinity_pileup.nc (optional).
    """
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    from scipy import stats as sp_stats

    apply_nature_style()

    has_pileup = pileup_path is not None and pileup_path.exists()

    if has_pileup:
        fig = plt.figure(figsize=(6.73, 6.5))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[2.5, 1], height_ratios=[2.2, 1],
            wspace=0.08, hspace=0.30,
            left=0.02, right=0.95, top=0.95, bottom=0.07,
        )
    else:
        fig = plt.figure(figsize=(6.73, 5.0))
        gs = fig.add_gridspec(
            1, 2, width_ratios=[2.5, 1], wspace=0.08,
            left=0.02, right=0.95, top=0.92, bottom=0.12,
        )

    proj = ccrs.PlateCarree()

    # Build 2D coordinate arrays
    if is_2d_coords:
        lon2d = lon
        lat2d = lat
    else:
        lon2d, lat2d = np.meshgrid(lon, lat)

    # --- Panel (a): SSS trend map ---
    ax_map = fig.add_subplot(gs[0, 0] if has_pileup else gs[0], projection=proj)
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
    gl.xlabel_style = {"size": 6, "color": "0.4"}
    gl.ylabel_style = {"size": 6, "color": "0.4"}

    vmax = np.nanpercentile(np.abs(trend), 98)
    vmax = np.ceil(vmax * 20) / 20

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="none")
    n_levels = 12
    bounds = np.linspace(-vmax, vmax, n_levels + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax_map.pcolormesh(
        lon2d, lat2d, trend,
        transform=proj, cmap=cmap, norm=norm,
        shading="auto", zorder=1,
    )

    # Significance stippling
    nonsig_mask = ((pvalue >= 0.05) & np.isfinite(trend)).astype(float)
    nonsig_smooth = gaussian_filter(nonsig_mask, sigma=1.5)
    ax_map.contourf(
        lon2d, lat2d, nonsig_smooth,
        levels=[0.5, 1.5], hatches=["xxx"], colors="none",
        transform=proj, zorder=2, alpha=0.0,
    )
    for collection in ax_map.collections:
        collection.set_linewidth(0.0)
        collection.set_edgecolor("0.5")

    # Data-driven hotspot contours
    trend_smooth = trend.copy()
    trend_smooth[np.isnan(trend_smooth)] = 0
    trend_smooth = gaussian_filter(trend_smooth, sigma=8)
    trend_smooth[np.isnan(trend)] = np.nan

    ax_map.contour(
        lon2d, lat2d, trend_smooth, levels=[0.08],
        colors=["#c0392b"], linewidths=[1.0], linestyles=["solid"],
        transform=proj, zorder=4,
    )
    ax_map.contour(
        lon2d, lat2d, trend_smooth, levels=[-0.08],
        colors=["#2471a3"], linewidths=[1.0], linestyles=["solid"],
        transform=proj, zorder=4,
    )
    ax_map.contour(
        lon2d, lat2d, trend_smooth, levels=[0],
        colors=["0.4"], linewidths=[0.6], linestyles=["--"],
        transform=proj, zorder=3,
    )

    # Colorbar below map
    if has_pileup:
        cax = fig.add_axes([0.05, 0.38, 0.52, 0.015])
    else:
        cax = fig.add_axes([0.05, 0.01, 0.52, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("SSS trend (PSU decade$^{-1}$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_map.set_title(title, fontsize=9, pad=8)
    add_panel_label(ax_map, "a", x=0.01, y=1.03)

    # --- Panel (b): Zonal-mean SSS trend profile ---
    ax_zonal = fig.add_subplot(gs[0, 1] if has_pileup else gs[1])

    atlantic_mask = np.ones_like(trend, dtype=bool)
    atlantic_mask &= (lon2d >= -80) & (lon2d <= 20)
    atlantic_mask &= ~((lon2d > -6) & (lat2d > 30) & (lat2d < 46))
    atlantic_mask &= ~((lon2d > 10) & (lat2d > 54))
    atlantic_mask &= ~((lon2d < -75) & (lat2d > 50) & (lat2d < 66))
    atlantic_mask &= ~((lon2d < -82) & (lat2d > 18) & (lat2d < 31))
    atlantic_mask &= np.isfinite(trend)

    trend_atlantic_masked = np.where(atlantic_mask, trend, np.nan)

    lat_edges = np.arange(-55, 71, 0.25)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    n_bins = len(lat_centers)

    zonal_mean = np.full(n_bins, np.nan)
    ci_lo = np.full(n_bins, np.nan)
    ci_hi = np.full(n_bins, np.nan)

    flat_trend = trend_atlantic_masked.ravel()
    flat_lat = lat2d.ravel()

    for j in range(n_bins):
        in_band = (flat_lat >= lat_edges[j]) & (flat_lat < lat_edges[j + 1])
        vals = flat_trend[in_band]
        vals = vals[np.isfinite(vals)]
        n_ocean = len(vals)
        if n_ocean < 5:
            continue

        zonal_mean[j] = np.mean(vals)
        sigma = np.std(vals, ddof=1)

        if sigma < 1e-12:
            ci_lo[j] = zonal_mean[j]
            ci_hi[j] = zonal_mean[j]
            continue

        anomaly = vals - zonal_mean[j]
        if len(anomaly) > 2:
            r1 = np.corrcoef(anomaly[:-1], anomaly[1:])[0, 1]
            r1 = np.clip(r1, 0.0, 0.99)
            n_eff = max(2, n_ocean * (1 - r1) / (1 + r1))
        else:
            n_eff = n_ocean

        se = sigma / np.sqrt(n_eff)
        ci_lo[j] = zonal_mean[j] - 1.96 * se
        ci_hi[j] = zonal_mean[j] + 1.96 * se

    lat_plot = lat_centers

    ax_zonal.fill_betweenx(
        lat_plot, ci_lo, ci_hi,
        alpha=0.2, color="#4477AA", linewidth=0,
        label="95% CI ($N_{eff}$-adj.)",
    )
    sig_lat = (ci_lo > 0) | (ci_hi < 0)
    zonal_mean_sig = np.where(sig_lat, zonal_mean, np.nan)
    zonal_mean_nonsig = np.where(~sig_lat, zonal_mean, np.nan)

    ax_zonal.plot(
        zonal_mean_sig, lat_plot,
        color="#4477AA", linewidth=1.5, solid_capstyle="round",
        zorder=5, label="significant",
    )
    ax_zonal.plot(
        zonal_mean_nonsig, lat_plot,
        color="#4477AA", linewidth=0.8, linestyle="--",
        zorder=4, alpha=0.6, label="not significant",
    )
    ax_zonal.axvline(0, color="0.6", linewidth=0.5, linestyle="--", zorder=0)

    ax_zonal.set_ylim(-55, 70)
    ax_zonal.set_xlabel("SSS trend\n(PSU decade$^{-1}$)", fontsize=7)
    ax_zonal.set_ylabel("Latitude (\u00b0N)", fontsize=7)
    ax_zonal.set_title("Zonal mean\n(Atlantic)", fontsize=8)
    ax_zonal.legend(fontsize=5, loc="upper left", framealpha=0.7)
    ax_zonal.tick_params(labelsize=6)

    ax_zonal.fill_betweenx(
        lat_plot,
        np.where(zonal_mean > 0, 0, zonal_mean),
        np.where(zonal_mean > 0, zonal_mean, 0),
        where=(zonal_mean > 0),
        alpha=0.06, color="#c0392b", linewidth=0,
    )
    ax_zonal.fill_betweenx(
        lat_plot,
        np.where(zonal_mean < 0, zonal_mean, 0),
        np.where(zonal_mean < 0, 0, zonal_mean),
        where=(zonal_mean < 0),
        alpha=0.06, color="#2471a3", linewidth=0,
    )

    ax_zonal.spines["top"].set_visible(False)
    ax_zonal.spines["right"].set_visible(False)

    add_panel_label(ax_zonal, "b", x=-0.3, y=1.03)

    # --- Panel (c): Salinity pile-up time series (deseasonalized) ---
    if has_pileup:
        import xarray as xr
        pileup = xr.open_dataarray(pileup_path)

        # Deseasonalize: remove monthly climatology
        import pandas as pd
        try:
            ts = pd.DatetimeIndex(pileup.time.values)
            months_pu = np.array([t.month for t in ts])
            yrs = np.array([t.year + (t.month - 1) / 12 for t in ts])
        except Exception:
            months_pu = np.tile(np.arange(1, 13), len(pileup) // 12 + 1)[:len(pileup)]
            yrs = np.arange(len(pileup))
        pu_vals = pileup.values.ravel().copy()
        for m in range(1, 13):
            mask_m = months_pu == m
            pu_vals[mask_m] -= np.nanmean(pu_vals[mask_m])
        # Add back the grand mean so the y-axis shows absolute PSU
        pu_vals += np.nanmean(pileup.values)

        ax_pu = fig.add_subplot(gs[1, :])

        color = "#EE6677"
        ax_pu.plot(pileup.time.values, pu_vals, color=color,
                   linewidth=0.5, alpha=0.4)

        # Trend on deseasonalized values
        reg = sp_stats.linregress(yrs, pu_vals)
        trend_line = reg.slope * yrs + reg.intercept
        p_str = "p < 0.001" if reg.pvalue < 0.001 else f"p = {reg.pvalue:.3f}"
        ax_pu.plot(pileup.time.values, trend_line,
                   color="0.3", linewidth=1.0, linestyle="--",
                   label=f"Trend: {reg.slope:+.4f} PSU/yr ({p_str})")

        # 12-month rolling mean
        if len(pu_vals) > 24:
            kernel = np.ones(12) / 12
            rolling = np.convolve(pu_vals, kernel, mode="same")
            # Mask edges
            rolling[:6] = np.nan
            rolling[-6:] = np.nan
            ax_pu.plot(pileup.time.values, rolling, color=color,
                       linewidth=1.5, label="12-month mean")

        ax_pu.set_ylabel("Salinity pile-up [PSU]", fontsize=7)
        ax_pu.legend(loc="upper left", fontsize=6)
        ax_pu.tick_params(labelsize=6)
        ax_pu.spines["top"].set_visible(False)
        ax_pu.spines["right"].set_visible(False)

        add_panel_label(ax_pu, "c", x=-0.04, y=1.10)

    save_publication_figure(fig, outpath)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SSS trend map from reanalysis data"
    )
    parser.add_argument(
        "--product",
        choices=["glorys12", "oras5"],
        default="glorys12",
        help="Reanalysis product (default: glorys12)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing NetCDF files (auto-detected from product)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (without extension)",
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Start year for trend computation (e.g. 1993)",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="End year for trend computation (e.g. 2023)",
    )
    parser.add_argument(
        "--pileup", type=Path, default=None,
        help="Path to salinity_pileup.nc to add time series panel (c)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/results"),
        help="Directory for caching trend fields",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path("data/glorys12") if args.product == "glorys12" \
            else Path("data/oras5")
    if args.output is None:
        args.output = Path(f"figures/grl/fig_sss_trend_map{'_oras5' if args.product == 'oras5' else ''}")

    is_2d = False
    suffix = "_oras5" if args.product == "oras5" else "_glorys12"
    cache_file = args.cache_dir / f"sss_trend_cache{suffix}.npz"

    if cache_file.exists():
        print(f"Loading cached trend field from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        trend = cached["trend"]
        pvalue = cached["pvalue"]
        lon = cached["lon"]
        lat = cached["lat"]
        is_2d = bool(cached["is_2d"])
        time_label = str(cached["time_label"])
        print(f"  Shape: {trend.shape}, label: {time_label}")
    else:
        if args.product == "oras5":
            print("Loading ORAS5 SSS data...")
            sss, lon, lat = load_oras5_sss(args.data_dir)
            is_2d = True
        else:
            print("Loading GLORYS12 SSS data...")
            sss = load_glorys12_sss(args.data_dir)
            lon = sss["x"].values
            lat = sss["y"].values

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
        time_label = f"{args.product.upper()} SSS trend  {t0_obj.year}\u2013{t1_obj.year}"
        print(f"  Shape: {sss.shape}, time range: {t0} to {t1}")

        print("Computing per-pixel trends (vectorized OLS)...")
        trend, pvalue, n_valid = compute_trend_field(sss)

        # Cache to disk
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file, trend=trend, pvalue=pvalue,
            lon=lon, lat=lat, is_2d=is_2d, time_label=time_label,
        )
        print(f"  Cached trend field to {cache_file}")

    # Summary statistics
    land_frac = np.isnan(trend).mean() * 100
    sig_frac = (pvalue < 0.05).mean() * 100
    print(f"  Land fraction: {land_frac:.1f}%")
    print(f"  Significant pixels (p<0.05): {sig_frac:.1f}%")

    # Check South Atlantic trend sign
    if is_2d:
        stsa_mask = (lat >= -35) & (lat <= -15) & (lon >= -60) & (lon <= 20)
        stsa_trend = np.nanmean(trend[stsa_mask])
    else:
        lat_mask = (lat >= -35) & (lat <= -15)
        lon_mask = (lon >= -60) & (lon <= 20)
        stsa_trend = np.nanmean(trend[np.ix_(lat_mask, lon_mask)])
    print(f"  STSA mean trend: {stsa_trend:+.4f} PSU/decade")

    # Auto-detect pile-up file if not specified
    pileup_path = args.pileup
    if pileup_path is None:
        default_pileup = Path("data/results/salinity_pileup.nc")
        if default_pileup.exists():
            pileup_path = default_pileup
            print(f"  Auto-detected pile-up data: {pileup_path}")

    print("Plotting...")
    plot_sss_trend_figure(trend, pvalue, lon, lat, args.output,
                          title=time_label, is_2d_coords=is_2d,
                          pileup_path=pileup_path)
    print("Done.")


if __name__ == "__main__":
    main()
