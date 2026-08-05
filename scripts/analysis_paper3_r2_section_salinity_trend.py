#!/usr/bin/env python3
"""PAPER_3 round-2 WP5: subsurface salinity-trend agreement at the 34.5 S section.

The two products disagree on SURFACE salinity trends over the South Atlantic
(GLORYS12 about +0.08, ORAS5 about -0.01 PSU/decade for 1993-2025). This script
asks whether the F_ovS-relevant quantity, the zonal-mean salinity profile
s_bar(z) at the section row actually used by
scripts/compute_fovs_decomposition.py, agrees better at depth.

Grid logic is taken from that script's `_oras5_period_mean` and
`_glorys12_period_mean`: same SAMBA_LAT row selection, same Atlantic longitude
mask, same e1t zonal weights, NaN-safe zonal mean.

MEMORY RULE: the single j-row (and only the salinity variable) is selected
before any values are materialised. GLORYS12 yearly files are about 10 GB each
and must never be loaded whole.

Per depth level the annual s_bar(z, year) series is fitted with OLS and the
Santer N_eff p-value from scripts/analysis_paper3_r2_gate_checks.py.

Outputs:
    revision/rev_papaer3_02/results/WP5_section_salinity_trends.nc
    revision/rev_papaer3_02/figures/WP5_section_salinity_trend_profile.pdf (+ .png)
    revision/rev_papaer3_02/results/WP5_layer_means.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, SAMBA_LAT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# matplotlib's font subsetter logs hundreds of INFO lines per saved PDF.
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

OUT_RES = REPO / "revision" / "rev_papaer3_02" / "results"
OUT_FIG = REPO / "revision" / "rev_papaer3_02" / "figures"

START_YEAR, END_YEAR = 1993, 2025
LAYERS = [("0-300 m", 0.0, 300.0), ("300-1200 m", 300.0, 1200.0)]
PLOT_ZMAX = 1500.0


def zonal_mean_profile(s_row: np.ndarray, e1t_atl: np.ndarray) -> np.ndarray:
    """e1t-weighted, NaN-safe zonal mean of a (nz, n_atl) salinity section."""
    ocean = np.isfinite(s_row)
    w = np.where(ocean, e1t_atl[None, :], 0.0)
    num = (np.where(ocean, s_row, 0.0) * w).sum(axis=1)
    den = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ══════════════════════════════════════════════════════════════════════
# ORAS5
# ══════════════════════════════════════════════════════════════════════


def oras5_annual_profiles(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (years, s_bar[year, z], grid_info) for ORAS5."""
    first = sorted(data_dir.glob("vosaline_*_3D_199301_*.nc"))
    if not first:
        raise RuntimeError("No ORAS5 vosaline file for 1993-01")
    ds = xr.open_dataset(first[0])
    nav_lat = ds["nav_lat"].values
    nav_lon = ds["nav_lon"].values
    lat_1d = np.nanmean(nav_lat, axis=1)
    j_idx = int(np.abs(lat_1d - SAMBA_LAT).argmin())
    actual_lat = float(lat_1d[j_idx])
    lon_row = nav_lon[j_idx, :]
    atl = (lon_row >= ATLANTIC_LON_MIN) & (lon_row <= ATLANTIC_LON_MAX)
    lat_row = nav_lat[j_idx, :]
    dlon = np.diff(lon_row)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dlon = np.append(dlon, dlon[-1])
    e1t = np.clip(np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(lat_row)), 1.0, None)
    # vosaline files carry the t-grid depth axis (`deptht`); `_oras5_period_mean`
    # reads `depthv` because it opens the vomecrty file instead. Both are 75
    # levels; the t-grid axis is the right one for a salinity profile.
    depth_name = "deptht" if "deptht" in ds.coords else "depthv"
    depth = ds[depth_name].values.astype(float)
    ds.close()
    e1t_atl = e1t[atl]

    years, profiles = [], []
    for year in range(START_YEAR, END_YEAR + 1):
        monthly = []
        for m in range(1, 13):
            cands = list(data_dir.glob(f"vosaline_*_3D_{year}{m:02d}_*.nc"))
            if not cands:
                continue
            d = xr.open_dataset(cands[0])
            # j-row and surface-to-bottom column only: never the full 3D field
            s_row = d["vosaline"].isel(time_counter=0, y=j_idx).values[:, atl]
            d.close()
            monthly.append(zonal_mean_profile(s_row, e1t_atl))
        if not monthly:
            log.warning(f"ORAS5 {year}: no files, skipped")
            continue
        years.append(year)
        profiles.append(np.nanmean(np.stack(monthly), axis=0))
        log.info(f"ORAS5 {year}: {len(monthly)} months")

    return (
        np.array(years, dtype=float),
        np.stack(profiles),
        {
            "depth": depth,
            "actual_lat": actual_lat,
            "n_atl": int(atl.sum()),
            "e3t": np.diff(depth, prepend=0.0),
        },
    )


# ══════════════════════════════════════════════════════════════════════
# GLORYS12
# ══════════════════════════════════════════════════════════════════════


def glorys12_annual_profiles(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (years, s_bar[year, z], grid_info) for GLORYS12."""
    files = sorted(data_dir.glob("glorys12_*.nc"))
    files = [f for f in files if START_YEAR <= int(f.stem.split("_")[1]) <= END_YEAR]
    if not files:
        raise RuntimeError("No GLORYS12 files in window")

    ds = xr.open_dataset(files[0])
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    depth = ds["depth"].values.astype(float)
    j_idx = int(np.abs(lat - SAMBA_LAT).argmin())
    actual_lat = float(lat[j_idx])
    atl = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    dlon = np.append(np.diff(lon), np.diff(lon)[-1])
    e1t = np.clip(np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(actual_lat)), 1.0, None)
    ds.close()
    e1t_atl = e1t[atl]

    years, profiles = [], []
    for f in files:
        year = int(f.stem.split("_")[1])
        d = xr.open_dataset(f)
        # Select the j-row lazily, then materialise only that slab.
        s_year = d["so"].isel(latitude=j_idx).values[:, :, atl]  # (time, z, x)
        d.close()
        monthly = [
            zonal_mean_profile(s_year[t], e1t_atl) for t in range(s_year.shape[0])
        ]
        years.append(year)
        profiles.append(np.nanmean(np.stack(monthly), axis=0))
        log.info(f"GLORYS12 {year}: {s_year.shape[0]} months")
        del s_year

    return (
        np.array(years, dtype=float),
        np.stack(profiles),
        {
            "depth": depth,
            "actual_lat": actual_lat,
            "n_atl": int(atl.sum()),
            "e3t": np.diff(depth, prepend=0.0),
        },
    )


# ══════════════════════════════════════════════════════════════════════


def trends_per_level(years: np.ndarray, prof: np.ndarray) -> dict:
    """OLS trend + Santer N_eff p-value per depth level. prof is (n_year, nz)."""
    nz = prof.shape[1]
    out = {
        k: np.full(nz, np.nan)
        for k in ("trend", "p_ols", "p_santer", "lag1", "n_eff", "n_years", "mean")
    }
    for k in range(nz):
        y = prof[:, k]
        good = np.isfinite(y)
        if good.sum() < 10:
            continue
        r = ols_santer(years[good], y[good])
        out["trend"][k] = r["slope"] * 10.0  # PSU per decade
        out["p_ols"][k] = r["p_ols"]
        out["p_santer"][k] = r["p_santer"]
        out["lag1"][k] = r["lag1_autocorr"]
        out["n_eff"][k] = r["n_eff"]
        out["n_years"][k] = r["n_years"]
        out["mean"][k] = float(np.nanmean(y))
    return out


def layer_mean(
    trend: np.ndarray, depth: np.ndarray, e3t: np.ndarray, z0: float, z1: float
) -> dict:
    """Thickness-weighted mean of the per-level trend over [z0, z1)."""
    sel = (depth >= z0) & (depth < z1) & np.isfinite(trend)
    if not sel.any():
        return {"n_levels": 0, "trend_PSU_per_decade": None}
    w = e3t[sel]
    return {
        "n_levels": int(sel.sum()),
        "depth_range_m": [float(depth[sel].min()), float(depth[sel].max())],
        "trend_PSU_per_decade": float(np.sum(trend[sel] * w) / np.sum(w)),
    }


def make_figure(res: dict, outbase: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
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
        }
    )

    colors = {"oras5": "#20558a", "glorys12": "#b3541e"}
    fig, ax = plt.subplots(figsize=(3.46, 4.2))

    for prod, r in res.items():
        z = r["depth"]
        t = r["trends"]["trend"]
        p = r["trends"]["p_santer"]
        m = z <= PLOT_ZMAX
        # The dashed line runs the full profile and the solid significant
        # segments are drawn over it, so the two styles join without gaps at
        # every significance boundary.
        ax.plot(
            np.where(m, t, np.nan),
            z,
            color=colors[prod],
            linewidth=0.9,
            linestyle="--",
            alpha=0.85,
            zorder=4,
            label=f"{prod.upper()} (n.s.)",
        )
        sig = m & (p < 0.05)
        # Extend each run by one level on both sides so the solid overlay meets
        # the dashed line instead of stopping short of it.
        grow = sig | np.roll(sig, 1) | np.roll(sig, -1)
        grow[0] = sig[0]
        grow[-1] = sig[-1]
        ax.plot(
            np.where(grow & m, t, np.nan),
            z,
            color=colors[prod],
            linewidth=1.4,
            zorder=5,
            label=f"{prod.upper()} (p < 0.05)",
        )

    ax.axvline(0, color="0.55", linewidth=0.5, zorder=0)
    ax.set_ylim(PLOT_ZMAX, 0)
    ax.set_xlabel("Zonal-mean salinity trend (PSU decade$^{-1}$)")
    ax.set_ylabel("Depth (m)")
    ax.legend(frameon=False, loc="lower right", fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {outbase}.pdf / .png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        nargs="+",
        default=["oras5", "glorys12"],
        choices=["oras5", "glorys12"],
    )
    ap.add_argument(
        "--replot",
        action="store_true",
        help="Redraw the figure from WP5_section_salinity_trends.nc, no reread.",
    )
    args = ap.parse_args()

    OUT_RES.mkdir(parents=True, exist_ok=True)

    if args.replot:
        ds = xr.open_dataset(OUT_RES / "WP5_section_salinity_trends.nc")
        cached = {
            prod: {
                "depth": ds[f"depth_{prod}"].values,
                "trends": {
                    "trend": ds[f"trend_{prod}"].values,
                    "p_santer": ds[f"p_santer_{prod}"].values,
                },
            }
            for prod in args.products
        }
        ds.close()
        make_figure(cached, OUT_FIG / "WP5_section_salinity_trend_profile")
        return
    res: dict = {}
    for prod in args.products:
        log.info(f"=== {prod.upper()} ===")
        if prod == "oras5":
            yrs, prof, grid = oras5_annual_profiles(Path("data/oras5"))
        else:
            yrs, prof, grid = glorys12_annual_profiles(Path("data/glorys12"))
        log.info(
            f"{prod}: section at {grid['actual_lat']:.2f} deg, "
            f"{grid['n_atl']} Atlantic points, {len(grid['depth'])} levels, "
            f"{len(yrs)} years"
        )
        res[prod] = {
            "years": yrs,
            "profiles": prof,
            "depth": grid["depth"],
            "e3t": grid["e3t"],
            "actual_lat": grid["actual_lat"],
            "n_atl": grid["n_atl"],
            "trends": trends_per_level(yrs, prof),
        }

    # NetCDF: one depth dimension per product (75 vs 50 levels)
    dvars, coords, attrs = {}, {}, {"window": f"{START_YEAR}-{END_YEAR}"}
    for prod, r in res.items():
        d = f"depth_{prod}"
        coords[d] = r["depth"]
        coords[f"year_{prod}"] = r["years"].astype(int)
        for key in ("trend", "p_ols", "p_santer", "lag1", "n_eff", "mean"):
            dvars[f"{key}_{prod}"] = ((d,), r["trends"][key])
        dvars[f"s_bar_{prod}"] = ((f"year_{prod}", d), r["profiles"])
        attrs[f"section_latitude_{prod}"] = r["actual_lat"]
        attrs[f"n_atlantic_points_{prod}"] = r["n_atl"]
    ds = xr.Dataset(dvars, coords=coords, attrs=attrs)
    nc = OUT_RES / "WP5_section_salinity_trends.nc"
    ds.to_netcdf(nc)
    log.info(f"Saved {nc}")

    make_figure(res, OUT_FIG / "WP5_section_salinity_trend_profile")

    # Layer means and sign agreement
    summary: dict = {
        "window": f"{START_YEAR}-{END_YEAR}",
        "layers": {},
        "per_product": {},
    }
    for prod, r in res.items():
        summary["per_product"][prod] = {
            "section_latitude": r["actual_lat"],
            "n_atlantic_points": r["n_atl"],
            "n_levels": int(len(r["depth"])),
            "n_years": int(len(r["years"])),
            "layer_means": {
                name: layer_mean(r["trends"]["trend"], r["depth"], r["e3t"], z0, z1)
                for name, z0, z1 in LAYERS
            },
            "surface_level_trend": float(r["trends"]["trend"][0]),
            "surface_level_depth_m": float(r["depth"][0]),
            "surface_level_p_santer": float(r["trends"]["p_santer"][0]),
        }
        for name, _z0, _z1 in LAYERS:
            lm = summary["per_product"][prod]["layer_means"][name]
            log.info(
                f"{prod} {name}: {lm['trend_PSU_per_decade']:+.5f} PSU/dec "
                f"({lm['n_levels']} levels)"
            )

    # Sign agreement on a common 5 m depth axis up to 1500 m
    if len(res) == 2:
        zc = np.arange(2.5, PLOT_ZMAX + 2.5, 5.0)
        interp = {}
        for prod, r in res.items():
            z, t, p = r["depth"], r["trends"]["trend"], r["trends"]["p_santer"]
            ok = np.isfinite(t)
            interp[prod] = (
                np.interp(zc, z[ok], t[ok], left=np.nan, right=np.nan),
                np.interp(zc, z[ok], p[ok], left=np.nan, right=np.nan),
            )
        (t1, p1), (t2, p2) = interp["oras5"], interp["glorys12"]
        same = np.sign(t1) == np.sign(t2)
        both_sig = (p1 < 0.05) & (p2 < 0.05)
        valid = np.isfinite(t1) & np.isfinite(t2)
        summary["sign_agreement"] = {
            "common_axis": "2.5 to 1500 m, 5 m steps, linear interpolation",
            "frac_same_sign": float(np.mean(same[valid])),
            "frac_same_sign_and_both_significant": float(
                np.mean((same & both_sig)[valid])
            ),
            "depth_ranges_same_sign": _ranges(zc, same & valid),
            "depth_ranges_same_sign_both_sig": _ranges(zc, same & both_sig & valid),
            "depth_ranges_opposite_sign": _ranges(zc, (~same) & valid),
        }
        sa = summary["sign_agreement"]
        log.info(
            f"same sign over {100 * sa['frac_same_sign']:.1f} % of 0-1500 m; "
            f"same sign AND both significant over "
            f"{100 * sa['frac_same_sign_and_both_significant']:.1f} %"
        )

    (OUT_RES / "WP5_layer_means.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Saved {OUT_RES / 'WP5_layer_means.json'}")


def _ranges(z: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """Contiguous depth ranges where mask is True."""
    out: list[list[float]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = z[i]
        elif not m and start is not None:
            out.append([float(start), float(z[i - 1])])
            start = None
    if start is not None:
        out.append([float(start), float(z[-1])])
    return out


if __name__ == "__main__":
    main()
