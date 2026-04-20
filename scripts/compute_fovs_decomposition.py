#!/usr/bin/env python3
"""Mechanism decomposition of F_ovS trend for each reanalysis.

Computes the ΔF_v + ΔF_s + ΔF_cross split for the trend from an early
stable period (1993-2005) to a recent period (2013-2025). Reuses the raw
monthly files for ORAS5 and GLORYS12 — no re-download needed.

For each product, outputs a small NetCDF containing:
  - scalars: delta_total, delta_v, delta_s, delta_cross, residual
  - profiles: per-depth contributions of each term [Sv]

The key physical question the decomposition answers: is the observed
F_ovS decline driven by a change in ocean **circulation structure** (Δv,
which could be wind) or by **salinity redistribution** (Δs, the
salt-advection-feedback signature)?

Usage:
    python scripts/compute_fovs_decomposition.py --product oras5
    python scripts/compute_fovs_decomposition.py --product glorys12
    python scripts/compute_fovs_decomposition.py --product all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0, SAMBA_LAT
from ardp.physics.fovs_decomposition import decompose_fovs_trend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")

# Period definitions (inclusive years)
EARLY = (1993, 2005)
LATE = (2013, 2025)


# ═══════════════════════════════════════════════════════════════════════
# ORAS5 period-mean section
# ═══════════════════════════════════════════════════════════════════════

def _oras5_period_mean(data_dir: Path, period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (v_mean, s_mean, grid_info) for a period, averaged across all months."""
    y0, y1 = period
    v_files = []
    s_files = []
    for year in range(y0, y1 + 1):
        for m in range(1, 13):
            yyyymm = f"{year}{m:02d}"
            v_cands = list(data_dir.glob(f"vomecrty_*_3D_{yyyymm}_*.nc"))
            s_cands = list(data_dir.glob(f"vosaline_*_3D_{yyyymm}_*.nc"))
            if v_cands and s_cands:
                v_files.append(v_cands[0])
                s_files.append(s_cands[0])
    if not v_files:
        raise RuntimeError(f"No ORAS5 files for period {y0}-{y1}")
    log.info(f"ORAS5 {y0}-{y1}: {len(v_files)} months")

    # Grid info from first file
    ds = xr.open_dataset(v_files[0])
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
    cos_lat = np.cos(np.deg2rad(lat_row))
    e1t = np.abs(dlon) * 111000.0 * cos_lat
    e1t = np.clip(e1t, 1.0, None)
    depth = ds["depthv"].values
    e3t = np.diff(depth, prepend=0.0)
    ds.close()

    nz = len(depth)
    nx_atl = int(atl.sum())
    v_sum = np.zeros((nz, nx_atl))
    s_sum = np.zeros((nz, nx_atl))
    s_count = np.zeros((nz, nx_atl))  # for proper NaN averaging

    for vf, sf in zip(v_files, s_files):
        ds_v = xr.open_dataset(vf)
        v = ds_v["vomecrty"].isel(time_counter=0, y=j_idx).values[:, atl]
        ds_v.close()
        v_sum += np.where(np.isnan(v), 0.0, v)

        ds_s = xr.open_dataset(sf)
        s = ds_s["vosaline"].isel(time_counter=0, y=j_idx).values[:, atl]
        ds_s.close()
        ocean = ~np.isnan(s)
        s_sum += np.where(ocean, s, 0.0)
        s_count += ocean.astype(float)

    v_mean = v_sum / len(v_files)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_mean = np.where(s_count > 0, s_sum / s_count, np.nan)

    return v_mean, s_mean, {
        "e1t_atl": e1t[atl],
        "e3t": e3t,
        "depth": depth,
        "actual_lat": actual_lat,
        "atl_count": nx_atl,
    }


# ═══════════════════════════════════════════════════════════════════════
# GLORYS12 period-mean section
# ═══════════════════════════════════════════════════════════════════════

def _glorys12_period_mean(data_dir: Path, period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    y0, y1 = period
    files = sorted(data_dir.glob("glorys12_*.nc"))
    files = [f for f in files if y0 <= int(f.stem.split("_")[1]) <= y1]
    if not files:
        raise RuntimeError(f"No GLORYS12 files for period {y0}-{y1}")
    log.info(f"GLORYS12 {y0}-{y1}: {len(files)} yearly files")

    # Grid info and mean across all months of all years
    ds = xr.open_dataset(files[0])
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    depth = ds["depth"].values
    j_idx = int(np.abs(lat - SAMBA_LAT).argmin())
    actual_lat = float(lat[j_idx])
    atl = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    e1t = np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(actual_lat))
    e1t = np.clip(e1t, 1.0, None)
    e3t = np.diff(depth, prepend=0.0)
    ds.close()

    # Open each year, take j-row mean across time, average across years
    v_list, s_list = [], []
    for f in files:
        ds = xr.open_dataset(f)
        v_year = ds["vo"].isel(latitude=j_idx).mean(dim="time").values[:, atl]
        s_year = ds["so"].isel(latitude=j_idx).mean(dim="time").values[:, atl]
        ds.close()
        v_list.append(v_year)
        s_list.append(s_year)

    v_mean = np.mean(np.stack(v_list), axis=0)
    # NaN-safe mean for salinity (land cells stay NaN); suppress warnings
    # for all-NaN slices below topography (harmless).
    s_stack = np.stack(s_list)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            s_mean = np.nanmean(s_stack, axis=0)

    return v_mean, s_mean, {
        "e1t_atl": e1t[atl],
        "e3t": e3t,
        "depth": depth,
        "actual_lat": actual_lat,
        "atl_count": int(atl.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
# ECCO-V4r4 period-mean section (reuses downloaded monthly granules)
# ═══════════════════════════════════════════════════════════════════════

def _ecco_period_mean(ecco_cache: Path, period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read local ECCO monthly granules for a period, compute (v, s) section means."""
    y0, y1 = period
    vel_files = []
    sal_files = []
    for year in range(y0, y1 + 1):
        v_dir = ecco_cache / "vel" / str(year)
        s_dir = ecco_cache / "sal" / str(year)
        if v_dir.exists():
            vel_files.extend(sorted(v_dir.glob("*.nc")))
        if s_dir.exists():
            sal_files.extend(sorted(s_dir.glob("*.nc")))
    if not vel_files or not sal_files:
        raise RuntimeError(f"ECCO {y0}-{y1}: no local files (run compute_ecco_fovs.py first)")
    log.info(f"ECCO {y0}-{y1}: {len(vel_files)} vel files, {len(sal_files)} sal files")

    # Open multi-file datasets
    ds_v = xr.open_mfdataset(vel_files, combine="by_coords", parallel=False)
    ds_s = xr.open_mfdataset(sal_files, combine="by_coords", parallel=False)

    lat = ds_v["latitude"].values
    lon = ds_v["longitude"].values
    z = np.abs(ds_v["Z"].values.astype(float))
    j_idx = int(np.abs(lat - SAMBA_LAT).argmin())
    actual_lat = float(lat[j_idx])
    atl = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    e1t = np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(actual_lat))
    e1t = np.clip(e1t, 1.0, None)
    order = np.argsort(z)
    z_sorted = z[order]
    e3t_sorted = np.diff(z_sorted, prepend=0.0)
    e3t = np.empty_like(e3t_sorted)
    e3t[order] = e3t_sorted

    # Time-mean section at j-row, Atlantic x-points
    v_mean = ds_v["NVEL"].mean(dim="time").isel(latitude=j_idx).load().values[:, atl]
    s_mean = ds_s["SALT"].mean(dim="time").isel(latitude=j_idx).load().values[:, atl]
    s_mean = np.where(s_mean > 0, s_mean, np.nan)  # ECCO land uses 0
    ds_v.close(); ds_s.close()

    return v_mean, s_mean, {
        "e1t_atl": e1t[atl], "e3t": e3t, "depth": z, "actual_lat": actual_lat,
        "atl_count": int(atl.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
# SODA 3.15.2 period-mean section (downloads quarterly snapshots)
# ═══════════════════════════════════════════════════════════════════════

def _soda_period_mean(period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Download SODA quarterly 5-day files for a period; compute (v, s) section mean.

    Reuses the helpers in compute_soda_fovs.py so we stay consistent with
    the F_ovS time-series pipeline.
    """
    from compute_soda_fovs import (
        build_section_grid,
        download_snapshot,
        ensure_ref_file,
    )
    y0, y1 = period
    ref = ensure_ref_file()
    grid = build_section_grid(ref)

    # Accumulate v and salt sections (Atlantic-masked) at the v j-row
    # (using t-grid salt aligned by length, matching the F_ovS kernel).
    n_atl = min(grid["n_atl_v"], grid["n_atl_t"])
    v_sum = np.zeros((len(grid["depth"]), n_atl))
    s_sum = np.zeros((len(grid["depth"]), n_atl))
    s_count = np.zeros((len(grid["depth"]), n_atl))
    n_samples = 0
    tmp = Path("data/soda/_tmp_decomp.nc")

    for year in range(y0, y1 + 1):
        for month in (1, 4, 7, 10):
            if not download_snapshot(year, month, tmp):
                continue
            try:
                ds = xr.open_dataset(tmp, decode_timedelta=False)
                v = ds["v"].isel(time=0, yu_ocean=grid["j_v"]).values
                s = ds["salt"].isel(time=0, yt_ocean=grid["j_t"]).values
                ds.close()
                v_a = v[:, grid["atl_v"]][:, :n_atl]
                s_a = s[:, grid["atl_t"]][:, :n_atl]
                ocean = ~np.isnan(s_a)
                v_sum += np.where(np.isnan(v_a), 0.0, v_a)
                s_sum += np.where(ocean, s_a, 0.0)
                s_count += ocean.astype(float)
                n_samples += 1
            except Exception as e:
                log.warning(f"  SODA {year}-{month:02d}: {e}")
            finally:
                tmp.unlink(missing_ok=True)

    if n_samples == 0:
        raise RuntimeError(f"No SODA snapshots acquired for {y0}-{y1}")

    v_mean = v_sum / n_samples
    with np.errstate(invalid="ignore", divide="ignore"):
        s_mean = np.where(s_count > 0, s_sum / s_count, np.nan)

    log.info(f"SODA {y0}-{y1}: {n_samples} quarterly snapshots")
    return v_mean, s_mean, {
        "e1t_atl": grid["e1t_v_atl"][:n_atl],
        "e3t": grid["e3t"],
        "depth": grid["depth"],
        "actual_lat": grid["lat_v"],
        "atl_count": n_atl,
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-product decomposition
# ═══════════════════════════════════════════════════════════════════════

def process_product(
    product: str,
    early_override: tuple[int, int] | None = None,
    late_override: tuple[int, int] | None = None,
    tag: str = "",
) -> None:
    log.info(f"=== {product.upper()} F_ovS decomposition{' [' + tag + ']' if tag else ''} ===")

    # Per-product window clipping: ECCO ends 2017, SODA ends 2022.
    product_max = {"oras5": 2025, "glorys12": 2025, "soda": 2022, "ecco": 2017}[product]

    def _clip(window: tuple[int, int]) -> tuple[int, int]:
        y0, y1 = window
        return (y0, min(y1, product_max))

    early = _clip(early_override if early_override is not None else EARLY)
    late = _clip(late_override if late_override is not None else
                 (2013, 2017) if product == "ecco" else
                 (2013, 2020) if product == "soda" else
                 LATE)

    if product == "oras5":
        v1, s1, grid = _oras5_period_mean(Path("data/oras5"), early)
        v2, s2, _ = _oras5_period_mean(Path("data/oras5"), late)
    elif product == "glorys12":
        v1, s1, grid = _glorys12_period_mean(Path("data/glorys12"), early)
        v2, s2, _ = _glorys12_period_mean(Path("data/glorys12"), late)
    elif product == "ecco":
        v1, s1, grid = _ecco_period_mean(Path("data/ecco"), early)
        v2, s2, _ = _ecco_period_mean(Path("data/ecco"), late)
    elif product == "soda":
        v1, s1, grid = _soda_period_mean(early)
        v2, s2, _ = _soda_period_mean(late)
    else:
        raise ValueError(f"Unknown product: {product}")

    log.info(
        f"Section at lat {grid['actual_lat']:.2f}, "
        f"{grid['atl_count']} Atlantic points, "
        f"{len(grid['depth'])} depth levels"
    )

    result = decompose_fovs_trend(
        v1, s1, v2, s2,
        e1t_atl=grid["e1t_atl"], e3t=grid["e3t"], s0=S0,
    )

    log.info(f"F_ov({early[0]}-{early[1]}) = {result['F_ov_1']:+.4f} Sv")
    log.info(f"F_ov({late[0]}-{late[1]})  = {result['F_ov_2']:+.4f} Sv")
    log.info(f"ΔF_total   = {result['delta_total']:+.4f} Sv")
    log.info(f"  ΔF_v     = {result['delta_v']:+.4f} Sv  (velocity-driven)")
    log.info(f"  ΔF_s     = {result['delta_s']:+.4f} Sv  (salinity-driven)")
    log.info(f"  ΔF_cross = {result['delta_cross']:+.4f} Sv  (covariance)")
    log.info(f"  residual = {result['residual']:.2e} Sv  (~0 = decomposition exact)")
    log.info(f"  v_bar    = {result['v_bar_1']:+.5f} / {result['v_bar_2']:+.5f} m/s")
    log.info(f"  V_net    = {result['V_net_1_Sv']:+.3f} / {result['V_net_2_Sv']:+.3f} Sv")

    # Which mechanism dominates?
    v_frac = 100 * result['delta_v'] / result['delta_total'] if result['delta_total'] != 0 else 0
    s_frac = 100 * result['delta_s'] / result['delta_total'] if result['delta_total'] != 0 else 0
    log.info(f"Velocity share: {v_frac:+.1f}%, Salinity share: {s_frac:+.1f}%")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out = RESULTS_DIR / f"fovs_decomposition_{product}{suffix}.nc"

    ds = xr.Dataset(
        data_vars={
            "depth_Sv_v": ("depth", result["depth_Sv_v"]),
            "depth_Sv_s": ("depth", result["depth_Sv_s"]),
            "depth_Sv_cross": ("depth", result["depth_Sv_cross"]),
        },
        coords={"depth": grid["depth"]},
        attrs={
            "product": product,
            "window_tag": tag,
            "early_period": f"{early[0]}-{early[1]}",
            "late_period": f"{late[0]}-{late[1]}",
            "section_latitude": grid["actual_lat"],
            "F_ov_early_Sv": result["F_ov_1"],
            "F_ov_late_Sv": result["F_ov_2"],
            "delta_total_Sv": result["delta_total"],
            "delta_v_Sv": result["delta_v"],
            "delta_s_Sv": result["delta_s"],
            "delta_cross_Sv": result["delta_cross"],
            "residual_Sv": result["residual"],
            "v_bar_early_ms": result["v_bar_1"],
            "v_bar_late_ms": result["v_bar_2"],
            "V_net_early_Sv": result["V_net_1_Sv"],
            "V_net_late_Sv": result["V_net_2_Sv"],
            "reference_salinity_PSU": S0,
        },
    )
    ds.to_netcdf(out)
    log.info(f"Saved: {out}")


def _parse_window(s: str | None) -> tuple[int, int] | None:
    if s is None:
        return None
    y0, y1 = s.split("-")
    return (int(y0), int(y1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product", choices=["oras5", "glorys12", "ecco", "soda", "all"], default="all",
    )
    parser.add_argument(
        "--early", default=None,
        help="Override early period, e.g. '2006-2012'. Per-product end-year clipping still applies.",
    )
    parser.add_argument(
        "--late", default=None,
        help="Override late period, e.g. '2018-2024'. Per-product end-year clipping still applies.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Suffix for output files (e.g. 'postargo'). Leave blank for the default pre-registered window.",
    )
    args = parser.parse_args()

    early = _parse_window(args.early)
    late = _parse_window(args.late)

    products = ["oras5", "glorys12", "ecco", "soda"] if args.product == "all" else [args.product]
    for p in products:
        try:
            process_product(p, early_override=early, late_override=late, tag=args.tag)
        except Exception as e:
            log.error(f"{p}: {e}")


if __name__ == "__main__":
    main()
