#!/usr/bin/env python3
"""PAPER_3 round-2 WP2: SSS trend maps with honest (N_eff-adjusted) significance.

Reuses the loaders of scripts/plot_sss_trend_map.py (no re-derivation of the
data path) but replaces the naive per-pixel OLS p-value with a Santer et al.
(2000) N_eff-adjusted p-value, because the input is a 396-month deseasonalized
monthly series whose residuals are strongly autocorrelated: the naive p-value
is badly overconfident.

For each product it writes

  revision/rev_papaer3_02/figures/WP2_sss_trend_map_<product>.pdf   (+ .png)
  revision/rev_papaer3_02/results/wp2_cache/sss_trend_cache_<product>.npz
      (naive p, keys compatible with plot_sss_trend_map.py --cache-dir)
  revision/rev_papaer3_02/results/wp2_cache/WP2_fields_<product>.npz
      (trend, p_naive, p_neff, lag1, n_eff, lon, lat)
  revision/rev_papaer3_02/results/WP2_bands_<product>.json
      (band statistics for 60-80 N, 40-60 N, 10-35 S)

Band significance is computed the defensible way: an area-weighted band-mean
SSS time series is built first, then a single OLS trend with a Santer N_eff
p-value is fitted to it. Averaging per-pixel trends and testing the spatial
spread (what panel b of the original figure does) is reported alongside, but
the temporal test is the one quoted.

Usage:
    python scripts/analysis_paper3_r2_sss_trend_map.py --product oras5
    python scripts/analysis_paper3_r2_sss_trend_map.py --product glorys12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from plot_sss_trend_map import load_glorys12_sss, load_oras5_sss  # noqa: E402

OUT_FIG = REPO / "revision" / "rev_papaer3_02" / "figures"
OUT_RES = REPO / "revision" / "rev_papaer3_02" / "results"
CACHE = OUT_RES / "wp2_cache"

START_YEAR, END_YEAR = 1993, 2025

# (label, lat_min, lat_max)
BANDS: list[tuple[str, float, float]] = [
    ("60-80N", 60.0, 80.0),
    ("40-60N", 40.0, 60.0),
    ("10-35S", -35.0, -10.0),
]


# ══════════════════════════════════════════════════════════════════════
# Atlantic mask (same exclusions as plot_sss_trend_map.py panel b)
# ══════════════════════════════════════════════════════════════════════


def atlantic_mask(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Atlantic basin, excluding Mediterranean, Baltic, Hudson Bay, Gulf of Mexico."""
    m = (lon2d >= -80) & (lon2d <= 20)
    m &= ~((lon2d > -6) & (lat2d > 30) & (lat2d < 46))  # Mediterranean
    m &= ~((lon2d > 10) & (lat2d > 54))  # Baltic
    m &= ~((lon2d < -75) & (lat2d > 50) & (lat2d < 66))  # Hudson Bay
    m &= ~((lon2d < -82) & (lat2d > 18) & (lat2d < 31))  # Gulf of Mexico
    return m


# ══════════════════════════════════════════════════════════════════════
# Trend statistics
# ══════════════════════════════════════════════════════════════════════


def ols_santer_1d(t: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope with naive and Santer N_eff-adjusted p-values (1D series)."""
    good = np.isfinite(y)
    t, y = t[good], y[good]
    n = len(t)
    if n < 10:
        return {"n": n, "slope": np.nan, "p_naive": np.nan, "p_neff": np.nan}
    res = sp_stats.linregress(t, y)
    resid = y - (res.intercept + res.slope * t)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    r1 = min(max(r1, -0.99), 0.99)
    neff = max(n * (1.0 - r1) / (1.0 + r1), 3.0)
    se_adj = res.stderr * np.sqrt((n - 2) / (neff - 2))
    p_adj = 2.0 * sp_stats.t.sf(abs(res.slope / se_adj), df=neff - 2)
    return {
        "n": int(n),
        "slope_per_decade": float(res.slope) * 10.0,
        "p_naive": float(res.pvalue),
        "lag1_autocorr": r1,
        "n_eff": float(neff),
        "p_neff": float(p_adj),
    }


def trend_field_neff(
    data: np.ndarray, years: np.ndarray, months: np.ndarray, block: int = 200_000
) -> dict:
    """Per-pixel OLS trend with naive and Santer N_eff p-values.

    data : (T, Y, X). Deseasonalization (monthly climatology removal) happens
    here, in place on a float64 working copy, block by block over pixels so
    peak memory stays bounded.
    """
    nt, ny, nx = data.shape
    npix = ny * nx
    flat = data.reshape(nt, npix)

    trend = np.full(npix, np.nan)
    p_naive = np.full(npix, np.nan)
    p_neff = np.full(npix, np.nan)
    lag1 = np.full(npix, np.nan)
    n_eff = np.full(npix, np.nan)

    t = years.astype(np.float64)
    t_anom = t - t.mean()
    ss_tt = float(np.sum(t_anom**2))
    dof = nt - 2

    month_idx = [months == m for m in range(1, 13)]

    for lo in range(0, npix, block):
        hi = min(lo + block, npix)
        chunk = flat[:, lo:hi].astype(np.float64)

        # Deseasonalize in place
        for mask_m in month_idx:
            chunk[mask_m, :] -= np.nanmean(chunk[mask_m, :], axis=0)

        # Only fully valid pixels get the vectorized treatment; the rest stay
        # NaN (land, or partially masked -> excluded rather than half-fitted).
        ok = np.isfinite(chunk).all(axis=0)
        if not ok.any():
            continue
        y = chunk[:, ok]

        y_mean = y.mean(axis=0)
        slopes = (t_anom @ (y - y_mean)) / ss_tt
        resid = y - (np.outer(t_anom, slopes) + y_mean)
        mse = np.sum(resid**2, axis=0) / dof
        se = np.sqrt(mse / ss_tt)

        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = slopes / se
        pn = 2.0 * sp_stats.t.sf(np.abs(tstat), dof)

        # Santer N_eff from lag-1 autocorrelation of the residuals
        r0 = resid[:-1] - resid[:-1].mean(axis=0)
        r1_ = resid[1:] - resid[1:].mean(axis=0)
        denom = np.sqrt(np.sum(r0**2, axis=0) * np.sum(r1_**2, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = np.where(denom > 0, np.sum(r0 * r1_, axis=0) / denom, 0.0)
        r1 = np.clip(r1, -0.99, 0.99)
        neff = np.maximum(nt * (1.0 - r1) / (1.0 + r1), 3.0)
        se_adj = se * np.sqrt((nt - 2) / (neff - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_adj = slopes / se_adj
        pa = 2.0 * sp_stats.t.sf(np.abs(t_adj), neff - 2)

        idx = np.where(ok)[0] + lo
        trend[idx] = slopes * 10.0  # PSU per decade
        p_naive[idx] = pn
        p_neff[idx] = pa
        lag1[idx] = r1
        n_eff[idx] = neff

    shape = (ny, nx)
    return {
        "trend": trend.reshape(shape),
        "p_naive": p_naive.reshape(shape),
        "p_neff": p_neff.reshape(shape),
        "lag1": lag1.reshape(shape),
        "n_eff": n_eff.reshape(shape),
    }


# ══════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════


def regrid_for_contour(
    lon2d: np.ndarray, lat2d: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a curvilinear field onto a regular lat/lon grid.

    Contouring a NEMO-type curvilinear grid directly produces spurious bands
    that run right across the map, because the quadrilaterals fold at the
    tripolar seam and at the longitude wrap. plot_sss_trend_map.py regrids for
    the same reason; the same recipe is used here (linear griddata, then a
    KD-tree distance mask so land stays empty).
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    reg_lon = np.arange(-80, 30.25, 0.25)
    reg_lat = np.arange(-55, 70.25, 0.25)
    rlon2d, rlat2d = np.meshgrid(reg_lon, reg_lat)

    src_lon, src_lat = lon2d.ravel(), lat2d.ravel()
    in_ext = (src_lon >= -85) & (src_lon <= 35) & (src_lat >= -60) & (src_lat <= 75)
    pts = np.column_stack([src_lon[in_ext], src_lat[in_ext]])
    vals = field.ravel()[in_ext]
    ok = np.isfinite(vals)

    out = griddata(pts[ok], vals[ok], (rlon2d, rlat2d), method="linear")
    tree = cKDTree(pts[ok])
    dist, _ = tree.query(np.column_stack([rlon2d.ravel(), rlat2d.ravel()]))
    out[dist.reshape(rlon2d.shape) > 1.0] = np.nan
    return rlon2d, rlat2d, out


def make_figure(
    trend: np.ndarray,
    p_neff: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    outbase: Path,
    title: str,
    is_2d: bool = False,
) -> dict:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.25,
        }
    )

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.09, 3.9))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.6, 1],
        wspace=0.22,
        left=0.03,
        right=0.97,
        top=0.90,
        bottom=0.16,
    )

    # --- (a) map ---
    ax = fig.add_subplot(gs[0], projection=proj)
    ax.set_extent([-80, 30, -55, 70], crs=proj)
    # Land sits above the hatch layer so hatching never bleeds over continents.
    ax.add_feature(cfeature.LAND, facecolor="#4d4d4d", edgecolor="none", zorder=4)
    ax.coastlines(linewidth=0.25, color="0.3", zorder=5)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        color="0.75",
        alpha=0.6,
        linestyle=":",
        zorder=1,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6, "color": "0.35"}
    gl.ylabel_style = {"size": 6, "color": "0.35"}

    vmax = float(np.ceil(np.nanpercentile(np.abs(trend), 98) * 20) / 20)
    bounds = np.linspace(-vmax, vmax, 13)  # zero-centered, discrete
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="none")
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        trend,
        transform=proj,
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=2,
        rasterized=True,
    )

    # Hatch where the N_eff-adjusted trend is NOT significant at 5 %.
    # (Same polarity as the original script: hatching marks untrustworthy area.)
    nonsig = np.where(np.isfinite(trend), (p_neff >= 0.05).astype(float), np.nan)
    if is_2d:
        hlon, hlat, nonsig = regrid_for_contour(lon2d, lat2d, nonsig)
    else:
        hlon, hlat = lon2d, lat2d
    hatch_cs = ax.contourf(
        hlon,
        hlat,
        np.nan_to_num(nonsig, nan=0.0),
        levels=[0.5, 1.5],
        hatches=["////"],
        colors="none",
        transform=proj,
        zorder=3,
    )
    # Do NOT zero the linewidth here: that is what made the original stipple
    # layer invisible. Style the hatch explicitly instead.
    hatch_cs.set_edgecolor("0.25")
    hatch_cs.set_facecolor("none")

    cax = fig.add_axes([0.06, 0.09, 0.44, 0.022])
    cbar = fig.colorbar(
        im, cax=cax, orientation="horizontal", extend="both", ticks=bounds[::2]
    )
    cbar.set_label("SSS trend (PSU decade$^{-1}$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6, width=0.4, length=2)
    cbar.outline.set_linewidth(0.4)

    ax.set_title(title, fontsize=7, pad=4)
    ax.text(
        0.005,
        1.02,
        "a",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    # --- (b) zonal mean over the Atlantic ---
    axz = fig.add_subplot(gs[1])
    amask = atlantic_mask(lon2d, lat2d) & np.isfinite(trend)
    tr_at = np.where(amask, trend, np.nan)
    p_at = np.where(amask, p_neff, np.nan)

    edges = np.arange(-55, 70.5, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    zm = np.full(len(centers), np.nan)
    frac_sig = np.full(len(centers), np.nan)
    flat_tr, flat_lat, flat_p = tr_at.ravel(), lat2d.ravel(), p_at.ravel()
    for j in range(len(centers)):
        sel = (flat_lat >= edges[j]) & (flat_lat < edges[j + 1])
        v = flat_tr[sel]
        pv = flat_p[sel]
        v_ok = np.isfinite(v)
        if v_ok.sum() < 5:
            continue
        zm[j] = float(np.mean(v[v_ok]))
        frac_sig[j] = float(np.mean(pv[v_ok] < 0.05))

    # Solid where the majority of the band's pixels are significant, dashed
    # otherwise (rule 6 line convention).
    sig_band = frac_sig >= 0.5
    axz.plot(
        np.where(sig_band, zm, np.nan),
        centers,
        color="#20558a",
        linewidth=1.2,
        zorder=5,
        label="majority significant",
    )
    axz.plot(
        np.where(~sig_band, zm, np.nan),
        centers,
        color="#20558a",
        linewidth=0.9,
        linestyle="--",
        zorder=4,
        alpha=0.75,
        label="not significant",
    )
    axz.axvline(0, color="0.6", linewidth=0.4, linestyle="-", zorder=0)

    axz.set_ylim(-55, 70)
    axz.set_xlabel("SSS trend (PSU decade$^{-1}$)", fontsize=7)
    axz.set_ylabel("Latitude (°N)", fontsize=7)
    axz.tick_params(labelsize=6, width=0.4, length=2)
    axz.legend(fontsize=5, loc="lower left", frameon=False)
    axz.spines["top"].set_visible(False)
    axz.spines["right"].set_visible(False)
    axz.text(
        -0.28,
        1.02,
        "b",
        transform=axz.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    plt.close(fig)

    return {
        "hatched_pixels": int(np.nansum(nonsig)),
        "vmax_PSU_per_decade": vmax,
        "zonal_bins": len(centers),
    }


# ══════════════════════════════════════════════════════════════════════


def replot(product: str) -> None:
    """Redraw the figure from the saved fields, skipping the trend computation."""
    f = np.load(CACHE / f"WP2_fields_{product}.npz")
    trend, p_neff = f["trend"], f["p_neff"]
    lon2d, lat2d = f["lon2d"], f["lat2d"]
    is_2d = product == "oras5"
    title = f"{product.upper()} SSS trend {START_YEAR}–{END_YEAR}"
    meta = make_figure(
        trend,
        p_neff,
        lon2d,
        lat2d,
        OUT_FIG / f"WP2_sss_trend_map_{product}",
        title,
        is_2d=is_2d,
    )
    # Keep the stored figure metadata in step with the figure just drawn.
    jpath = OUT_RES / f"WP2_bands_{product}.json"
    if jpath.exists():
        payload = json.loads(jpath.read_text())
        payload["figure_meta"] = meta
        jpath.write_text(json.dumps(payload, indent=2))
    print(f"replotted {product}: {meta}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", choices=["oras5", "glorys12"], required=True)
    ap.add_argument(
        "--replot",
        action="store_true",
        help="Redraw from WP2_fields_<product>.npz instead of recomputing trends.",
    )
    args = ap.parse_args()
    product = args.product

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    if args.replot:
        replot(product)
        return

    print(f"Loading {product.upper()} SSS ...")
    if product == "oras5":
        sss, lon2d, lat2d = load_oras5_sss(Path("data/oras5"))
        is_2d = True
    else:
        sss = load_glorys12_sss(Path("data/glorys12"))
        lon1d = sss["x"].values
        lat1d = sss["y"].values
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        is_2d = False

    tobj = sss["time"].values.astype("datetime64[us]").astype("object")
    yr = np.array([t.year for t in tobj])
    keep = (yr >= START_YEAR) & (yr <= END_YEAR)
    sss = sss.isel(time=keep)
    tobj = [t for t, k in zip(tobj, keep, strict=True) if k]
    years = np.array([t.year + (t.month - 1) / 12.0 for t in tobj])
    months = np.array([t.month for t in tobj])
    print(f"  {len(years)} months, {years[0]:.2f} to {years[-1]:.2f}")

    data = sss.values
    del sss

    print("Computing per-pixel trends with Santer N_eff ...")
    f = trend_field_neff(data, years, months)
    trend, p_naive, p_neff = f["trend"], f["p_naive"], f["p_neff"]

    ocean = np.isfinite(trend)
    print(f"  ocean pixels: {ocean.sum()} ({100 * ocean.mean():.1f} %)")
    print(f"  significant naive : {100 * np.nanmean(p_naive[ocean] < 0.05):.1f} %")
    print(f"  significant N_eff : {100 * np.nanmean(p_neff[ocean] < 0.05):.1f} %")
    print(f"  median lag-1 resid autocorr: {np.nanmedian(f['lag1'][ocean]):.3f}")
    print(f"  median N_eff: {np.nanmedian(f['n_eff'][ocean]):.1f} of {len(years)}")

    # --- band statistics: area-weighted band-mean series, then OLS+Santer ---
    amask = atlantic_mask(lon2d, lat2d)
    w = np.cos(np.deg2rad(lat2d))
    bands = {}
    for label, lo, hi in BANDS:
        sel = amask & (lat2d >= lo) & (lat2d < hi) & ocean
        n_pix = int(sel.sum())
        if n_pix == 0:
            bands[label] = {"n_pixels": 0, "note": "no ocean pixels in domain"}
            continue
        ww = w[sel]
        series = np.einsum("tp,p->t", data[:, sel], ww) / ww.sum()
        stat = ols_santer_1d(years, series)
        stat["n_pixels"] = n_pix
        stat["lat_min_covered"] = float(lat2d[sel].min())
        stat["lat_max_covered"] = float(lat2d[sel].max())
        stat["mean_of_pixel_trends_per_decade"] = float(np.mean(trend[sel]))
        stat["frac_pixels_sig_neff"] = float(np.mean(p_neff[sel] < 0.05))
        stat["frac_pixels_sig_naive"] = float(np.mean(p_naive[sel] < 0.05))
        bands[label] = stat
        print(
            f"  {label}: {stat['slope_per_decade']:+.4f} PSU/dec, "
            f"p_neff={stat['p_neff']:.3f} (n_pix={n_pix})"
        )

    title = f"{product.upper()} SSS trend {START_YEAR}–{END_YEAR}"
    meta = make_figure(
        trend,
        p_neff,
        lon2d,
        lat2d,
        OUT_FIG / f"WP2_sss_trend_map_{product}",
        title,
        is_2d=is_2d,
    )
    print(f"  hatched (non-significant) pixels drawn: {meta['hatched_pixels']}")

    # Cache compatible with plot_sss_trend_map.py --cache-dir (naive p, as it
    # would itself have produced), so the original script can render from it.
    np.savez_compressed(
        CACHE / f"sss_trend_cache_{product}.npz",
        trend=trend,
        pvalue=p_naive,
        lon=(lon2d if is_2d else lon2d[0, :]),
        lat=(lat2d if is_2d else lat2d[:, 0]),
        is_2d=is_2d,
        time_label=title,
    )
    np.savez_compressed(
        CACHE / f"WP2_fields_{product}.npz",
        trend=trend,
        p_naive=p_naive,
        p_neff=p_neff,
        lag1=f["lag1"],
        n_eff=f["n_eff"],
        lon2d=lon2d,
        lat2d=lat2d,
    )

    summary = {
        "product": product,
        "window": f"{START_YEAR}-{END_YEAR}",
        "n_months": int(len(years)),
        "domain_lat": [float(np.nanmin(lat2d)), float(np.nanmax(lat2d))],
        "domain_lon": [float(np.nanmin(lon2d)), float(np.nanmax(lon2d))],
        "ocean_pixels": int(ocean.sum()),
        "frac_sig_naive": float(np.nanmean(p_naive[ocean] < 0.05)),
        "frac_sig_neff": float(np.nanmean(p_neff[ocean] < 0.05)),
        "median_lag1_resid": float(np.nanmedian(f["lag1"][ocean])),
        "median_n_eff": float(np.nanmedian(f["n_eff"][ocean])),
        "figure_meta": meta,
        "bands": bands,
    }
    (OUT_RES / f"WP2_bands_{product}.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {OUT_RES / f'WP2_bands_{product}.json'}")


if __name__ == "__main__":
    main()
