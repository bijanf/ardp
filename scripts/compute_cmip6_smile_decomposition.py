#!/usr/bin/env python3
"""SMILE F_ovS decomposition for MPI-ESM1-2-LR Grand Ensemble.

Phase-C robustness test for the velocity-vs-salinity mechanism class:
if the MPI-ESM1-2-LR Grand Ensemble (10 members on Pangeo, all
r{1..10}i1p1f1) clusters in one mechanism class, the binary
classification is structural and the 10-percentage-point AMOC
weakening gap survives the small-N criticism. If the members scatter
across the velocity-share / salinity-share plane, the manuscript
needs to acknowledge that mechanism class is partly aliased onto
internal variability.

For each member, this script streams the 34.5°S section directly
from the Pangeo cloud-hosted zarr stores (no local download), runs
the same decomposition as the multi-model ensemble (1950-1980 vs
2080-2100), and writes a per-member CSV.

Reads:  Pangeo CMIP6 catalog (intake-esm)
Writes: data/results/fovs_decomposition_smile_mpi_esm1_2_lr.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import intake
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.constants import (  # noqa: E402
    ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, PANGEO_CATALOG_URL, S0,
)
from ardp.physics.fovs_decomposition import decompose_fovs_trend  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
TARGET_LAT = -34.5
SOURCE = "MPI-ESM1-2-LR"
DEFAULT_BASELINE = (1950, 1980)
DEFAULT_FORCED = (2080, 2100)


def _open_section(cat, member_id: str, experiment_id: str,
                  variable_id: str) -> xr.DataArray | None:
    """Open the Pangeo zarr store and slice to the 34.5S latitude row."""
    res = cat.search(source_id=SOURCE, experiment_id=experiment_id,
                     variable_id=variable_id, table_id="Omon",
                     member_id=member_id, grid_label="gn")
    if len(res.df) == 0:
        return None
    key = res.df.iloc[0]["zstore"]
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_zarr(key, consolidated=True, decode_times=time_coder)
    da = ds[variable_id]

    # Find latitude coordinate
    lat_name = next((n for n in ("lat", "latitude", "nav_lat") if n in ds.coords), None)
    if lat_name is None:
        ds.close()
        return None
    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 1:
        j_idx = int(np.abs(lat_vals - TARGET_LAT).argmin())
        j_dim = lat_name
    elif lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_idx = int(np.abs(lat_1d - TARGET_LAT).argmin())
        j_dim = ds[lat_name].dims[0]
    else:
        ds.close()
        return None

    section = da.isel({j_dim: j_idx})
    return section


def _section_lon(da: xr.DataArray) -> np.ndarray:
    """Return the 1-D longitude coordinate at the section row."""
    for cand in ("longitude", "lon", "nav_lon"):
        if cand in da.coords:
            v = da[cand].values
            return v if v.ndim == 1 else v[0]
    raise RuntimeError("No longitude coord on section")


def _period_mean(da: xr.DataArray, years: tuple[int, int]) -> np.ndarray:
    y0, y1 = years
    t_year = da["time"].dt.year.values
    mask = (t_year >= y0) & (t_year <= y1)
    if mask.sum() == 0:
        raise RuntimeError(f"No timesteps in {y0}-{y1}")
    return da.isel(time=np.where(mask)[0]).mean(dim="time", skipna=True).values


def _grid_metrics(lon_180: np.ndarray, depth: np.ndarray, lat: float):
    atl = (lon_180 >= ATLANTIC_LON_MIN) & (lon_180 <= ATLANTIC_LON_MAX)
    dlon = np.diff(lon_180)
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dlon = np.append(dlon, dlon[-1])
    e1t = np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(lat))
    e1t = np.clip(e1t, 1.0, None)
    order = np.argsort(depth)
    z_sorted = depth[order]
    e3t_sorted = np.diff(z_sorted, prepend=0.0)
    e3t = np.empty_like(e3t_sorted)
    e3t[order] = e3t_sorted
    return atl, e1t, e3t


def process_member(cat, member_id: str,
                   baseline: tuple[int, int],
                   forced: tuple[int, int]) -> dict | None:
    log.info(f"  {member_id}: opening zarr stores...")
    try:
        vo_hist = _open_section(cat, member_id, "historical", "vo")
        so_hist = _open_section(cat, member_id, "historical", "so")
        vo_ssp = _open_section(cat, member_id, "ssp585", "vo")
        so_ssp = _open_section(cat, member_id, "ssp585", "so")
        if any(x is None for x in (vo_hist, so_hist, vo_ssp, so_ssp)):
            log.error(f"    {member_id}: missing fields")
            return None

        v1 = _period_mean(vo_hist, baseline)
        s1 = _period_mean(so_hist, baseline)
        v2 = _period_mean(vo_ssp, forced)
        s2 = _period_mean(so_ssp, forced)

        lon = _section_lon(vo_hist)
        lon_180 = np.where(lon > 180, lon - 360, lon)
        depth = vo_hist["lev"].values.astype(float) if "lev" in vo_hist.coords \
            else vo_hist[next(d for d in vo_hist.dims if d not in ("time", "i", "x"))].values.astype(float)

        atl, e1t, e3t = _grid_metrics(lon_180, depth, TARGET_LAT)

        v1_atl = v1[:, atl]
        s1_atl = s1[:, atl]
        v2_atl = v2[:, atl]
        s2_atl = s2[:, atl]
        e1t_atl = e1t[atl]

        for arr in (s1_atl, s2_atl):
            arr[arr <= 0] = np.nan
            arr[arr > 100] = np.nan
        for arr in (v1_atl, v2_atl):
            np.nan_to_num(arr, copy=False, nan=0.0)

        result = decompose_fovs_trend(v1_atl, s1_atl, v2_atl, s2_atl,
                                       e1t_atl, e3t, s0=S0)
        dt = result["delta_total"]
        v_frac = 100 * result["delta_v"] / dt if abs(dt) > 1e-6 else np.nan
        s_frac = 100 * result["delta_s"] / dt if abs(dt) > 1e-6 else np.nan
        log.info(
            f"    {member_id}: F1={result['F_ov_1']:+.3f}  F2={result['F_ov_2']:+.3f}"
            f"  ΔF={dt:+.3f} Sv  v:{v_frac:+.0f}%  s:{s_frac:+.0f}%"
        )
        return {
            "model": SOURCE,
            "member_id": member_id,
            "F_ov_baseline": result["F_ov_1"],
            "F_ov_forced": result["F_ov_2"],
            "delta_total": dt,
            "delta_v": result["delta_v"],
            "delta_s": result["delta_s"],
            "delta_cross": result["delta_cross"],
            "velocity_share_pct": v_frac,
            "salinity_share_pct": s_frac,
        }
    except Exception as e:
        log.error(f"    {member_id}: FAILED ({e})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-years", nargs=2, type=int,
                        default=list(DEFAULT_BASELINE), metavar=("Y0", "Y1"))
    parser.add_argument("--forced-years", nargs=2, type=int,
                        default=list(DEFAULT_FORCED), metavar=("Y0", "Y1"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/results/fovs_decomposition_smile_mpi_esm1_2_lr.csv"))
    parser.add_argument("--members", nargs="+", default=None,
                        help="Subset (default: all 10 r{1..10}i1p1f1)")
    args = parser.parse_args()

    baseline = tuple(args.baseline_years)
    forced = tuple(args.forced_years)
    log.info(f"SMILE decomposition for {SOURCE}: baseline {baseline} → forced {forced}")
    log.info("Opening Pangeo CMIP6 catalog...")
    cat = intake.open_esm_datastore(PANGEO_CATALOG_URL)

    if args.members is not None:
        members = args.members
    else:
        # Find all members with all four (vo, so) × (hist, ssp585) on gn grid
        all_sets = []
        for var in ("vo", "so"):
            for exp in ("historical", "ssp585"):
                df = cat.search(source_id=SOURCE, experiment_id=exp,
                                variable_id=var, table_id="Omon",
                                grid_label="gn").df
                all_sets.append(set(df["member_id"]))
        members = sorted(set.intersection(*all_sets))
    log.info(f"Members to process: {members}")

    rows = []
    for m in members:
        row = process_member(cat, m, baseline, forced)
        if row is not None:
            rows.append(row)

    if not rows:
        log.error("No members processed successfully.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    log.info(f"Saved: {args.output}")
    log.info("")
    log.info(f"--- SMILE summary across {len(rows)} members ---")
    log.info(f"  ΔF_total mean:     {df['delta_total'].mean()*1000:+.1f} mSv")
    log.info(f"  ΔF_total spread:   ±{df['delta_total'].std()*1000:.1f} mSv (1σ)")
    log.info(f"  velocity_share:    mean={df['velocity_share_pct'].mean():+.1f}%  "
             f"sd={df['velocity_share_pct'].std():.1f}%")
    log.info(f"  salinity_share:    mean={df['salinity_share_pct'].mean():+.1f}%  "
             f"sd={df['salinity_share_pct'].std():.1f}%")
    weakening = df[df["delta_total"] < -0.01]
    v_dom = weakening[weakening["velocity_share_pct"] > 60]
    s_dom = weakening[weakening["salinity_share_pct"] > 60]
    log.info(f"  Weakening members: {len(weakening)}/{len(df)}")
    log.info(f"  v-dominant: {len(v_dom)}/{len(weakening)}")
    log.info(f"  s-dominant: {len(s_dom)}/{len(weakening)}")


if __name__ == "__main__":
    main()
