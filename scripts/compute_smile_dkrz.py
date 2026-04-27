#!/usr/bin/env python3
"""SMILE F_ovS decomposition — DKRZ Levante version.

Designed to run on Levante (or any system with the CMIP6 / MPI-GE
archive available as a mounted POSIX filesystem). For each ensemble
member it loads the 34.5°S section from the local NetCDF files,
computes the velocity-vs-salinity F_ovS decomposition between two
windows, and writes one CSV row per member.

This script is dependency-free except for numpy + xarray + pandas;
it does NOT need intake-esm. Run on Levante with:

    module load python3
    python scripts/compute_smile_dkrz.py \\
        --archive /work/ik1017/CMIP6/data/CMIP6 \\
        --output  fovs_decomposition_smile_mpi_lr_dkrz.csv

Then `scp` the CSV back to your workstation and feed it into
`scripts/plot_paper2_FigureSMILE.py` for the manuscript figure.

Two archives are supported:

  --archive-style cmip6   (default): the ESGF/CMIP6 directory tree
      <archive>/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp585/r{N}i1p1f1/
                  Omon/vo/gn/v*/...nc
      gives ~30 members of the standard CMIP6 contribution.

  --archive-style mpi-ge  : the dedicated MPI Grand Ensemble layout
      <archive>/{historical,ssp585}/{vo,so}_{member}_*.nc
      can give 100 members at MPI-ESM-LR resolution. Adjust the
      glob pattern via --filename-pattern if your local layout
      differs.

The output schema matches `fovs_decomposition_cmip6_summary.csv` so
the SMILE CSV can be concatenated with the multi-model ensemble for
unified plotting.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.constants import (  # noqa: E402
    ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0,
)
from ardp.physics.fovs_decomposition import decompose_fovs_trend  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_LAT = -34.5
DEFAULT_BASELINE = (1950, 1980)
DEFAULT_FORCED = (2080, 2100)


# ────────────────────────────────────────────────────────────────────────
# File discovery
# ────────────────────────────────────────────────────────────────────────

def _discover_cmip6(archive: Path, model: str = "MPI-ESM1-2-LR",
                    grid: str = "gn") -> dict[str, dict]:
    """Return {member: {var, exp -> [paths]}} for a CMIP6 ESGF tree."""
    out: dict[str, dict] = {}
    for activity in ("CMIP", "ScenarioMIP"):
        # historical lives under CMIP/, ssp585 under ScenarioMIP/
        if activity == "CMIP":
            exps = ("historical",)
        else:
            exps = ("ssp585",)
        for exp in exps:
            base = archive / activity / "MPI-M" / model / exp
            if not base.exists():
                continue
            for member_dir in sorted(base.iterdir()):
                m = re.match(r"^r\d+i\d+p\d+f\d+$", member_dir.name)
                if m is None:
                    continue
                for var in ("vo", "so"):
                    var_dir = member_dir / "Omon" / var / grid
                    if not var_dir.exists():
                        continue
                    # Pick the latest version directory
                    versions = sorted(var_dir.glob("v*"))
                    if not versions:
                        continue
                    latest = versions[-1]
                    files = sorted(latest.glob("*.nc"))
                    if files:
                        out.setdefault(member_dir.name, {}).setdefault(var, {})[exp] = files
    return out


def _discover_mpi_ge(archive: Path, pattern: str) -> dict[str, dict]:
    """Return {member: {var: {exp: [paths]}}} for an MPI-GE flat tree.

    The expected pattern uses {var}, {member}, {exp}; e.g. on /pool the
    layout is roughly historical/{var}/*_{member}_*.nc."""
    out: dict[str, dict] = {}
    for path in archive.rglob(pattern.replace("{var}", "*").replace(
        "{member}", "*").replace("{exp}", "*")):
        # Try to extract member id like r{N}i1p1f1 or just rN from name
        name = path.stem
        m = re.search(r"r(\d+)(i\d+p\d+f\d+)?", name)
        if m is None:
            continue
        member = f"r{m.group(1)}i1p1f1"
        var_match = re.search(r"\b(vo|so)\b", name)
        if var_match is None:
            continue
        var = var_match.group(1)
        exp_match = re.search(r"(historical|ssp585|hist)", name)
        if exp_match is None:
            continue
        exp = "historical" if exp_match.group(1) in ("historical", "hist") else "ssp585"
        out.setdefault(member, {}).setdefault(var, {}).setdefault(exp, []).append(path)
    return out


# ────────────────────────────────────────────────────────────────────────
# Section extraction & decomposition (mirrors compute_cmip6_fovs_decomposition)
# ────────────────────────────────────────────────────────────────────────

def _open_concat(paths: list[Path], var: str) -> xr.DataArray:
    ds = xr.open_mfdataset(paths, combine="by_coords", parallel=False,
                           decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    return ds[var]


def _slice_to_section(da: xr.DataArray) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Slice a (time, lev, j, i) global field down to the 34.5°S row."""
    lat_name = next((n for n in ("latitude", "lat", "nav_lat")
                     if n in da.coords), None)
    if lat_name is None:
        raise RuntimeError(f"no latitude coord; coords: {list(da.coords)}")
    lat_vals = da[lat_name].values
    if lat_vals.ndim == 1:
        j_idx = int(np.abs(lat_vals - TARGET_LAT).argmin())
        j_dim = lat_name
    elif lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_idx = int(np.abs(lat_1d - TARGET_LAT).argmin())
        j_dim = da[lat_name].dims[0]
    else:
        raise RuntimeError(f"unexpected lat shape {lat_vals.shape}")
    section = da.isel({j_dim: j_idx})

    # Longitude row
    lon_name = next((n for n in ("longitude", "lon", "nav_lon")
                     if n in section.coords or n in da.coords), None)
    if lon_name in section.coords:
        lon_vals = section[lon_name].values
    else:
        lon_vals = da[lon_name].isel({j_dim: j_idx}).values
    if lon_vals.ndim != 1:
        lon_vals = lon_vals[0] if lon_vals.shape[0] == 1 else lon_vals
    lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

    # Depth coord
    lev_name = next((n for n in ("lev", "depth", "olevel", "z_t")
                     if n in section.coords), None)
    depth = section[lev_name].values.astype(float) if lev_name else np.asarray([])
    return section, lon_180, depth


def _period_mean(da: xr.DataArray, years: tuple[int, int]) -> np.ndarray:
    y0, y1 = years
    t_year = da["time"].dt.year.values
    mask = (t_year >= y0) & (t_year <= y1)
    if mask.sum() == 0:
        raise RuntimeError(f"no timesteps in {y0}-{y1}")
    return da.isel(time=np.where(mask)[0]).mean(dim="time", skipna=True).values


def _grid_metrics(lon_180, depth, lat):
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


def process_member(member: str, sources: dict,
                   baseline: tuple[int, int],
                   forced: tuple[int, int]) -> dict | None:
    try:
        for var in ("vo", "so"):
            for exp in ("historical", "ssp585"):
                if exp not in sources.get(var, {}):
                    log.warning(f"  {member}: missing {var}/{exp}, skipping")
                    return None
        vo_hist = _open_concat(sources["vo"]["historical"], "vo")
        so_hist = _open_concat(sources["so"]["historical"], "so")
        vo_ssp = _open_concat(sources["vo"]["ssp585"], "vo")
        so_ssp = _open_concat(sources["so"]["ssp585"], "so")

        v1_full, lon_180, depth = _slice_to_section(vo_hist)
        s1_full, *_ = _slice_to_section(so_hist)
        v2_full, *_ = _slice_to_section(vo_ssp)
        s2_full, *_ = _slice_to_section(so_ssp)

        v1 = _period_mean(v1_full, baseline)
        s1 = _period_mean(s1_full, baseline)
        v2 = _period_mean(v2_full, forced)
        s2 = _period_mean(s2_full, forced)

        atl, e1t, e3t = _grid_metrics(lon_180, depth, TARGET_LAT)
        v1a, s1a = v1[:, atl], s1[:, atl]
        v2a, s2a = v2[:, atl], s2[:, atl]
        e1t_atl = e1t[atl]

        for arr in (s1a, s2a):
            arr[arr <= 0] = np.nan
            arr[arr > 100] = np.nan
        for arr in (v1a, v2a):
            np.nan_to_num(arr, copy=False, nan=0.0)

        result = decompose_fovs_trend(v1a, s1a, v2a, s2a, e1t_atl, e3t, s0=S0)
        dt = result["delta_total"]
        v_frac = 100 * result["delta_v"] / dt if abs(dt) > 1e-6 else np.nan
        s_frac = 100 * result["delta_s"] / dt if abs(dt) > 1e-6 else np.nan
        log.info(
            f"  {member}  F1={result['F_ov_1']:+.3f}  F2={result['F_ov_2']:+.3f}"
            f"  ΔF={dt:+.3f} Sv  v:{v_frac:+.0f}%  s:{s_frac:+.0f}%"
        )
        return {
            "member_id": member,
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
        log.error(f"  {member}: FAILED ({e})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True,
                        help="Root of the CMIP6 / MPI-GE archive (Levante: "
                             "/work/ik1017/CMIP6/data/CMIP6 for CMIP6 ESGF tree)")
    parser.add_argument("--archive-style", choices=("cmip6", "mpi-ge"),
                        default="cmip6")
    parser.add_argument("--filename-pattern", type=str,
                        default="*_{exp}_*_{member}_*.nc",
                        help="(mpi-ge mode) glob pattern; placeholders are "
                             "{var}, {member}, {exp}")
    parser.add_argument("--model", type=str, default="MPI-ESM1-2-LR",
                        help="(cmip6 mode) source_id to scan")
    parser.add_argument("--grid", type=str, default="gn",
                        help="(cmip6 mode) grid_label, default gn")
    parser.add_argument("--baseline-years", nargs=2, type=int,
                        default=list(DEFAULT_BASELINE), metavar=("Y0", "Y1"))
    parser.add_argument("--forced-years", nargs=2, type=int,
                        default=list(DEFAULT_FORCED), metavar=("Y0", "Y1"))
    parser.add_argument("--output", type=Path,
                        default=Path("fovs_decomposition_smile_dkrz.csv"))
    parser.add_argument("--members", nargs="+", default=None,
                        help="Subset (default: all discovered)")
    args = parser.parse_args()

    baseline = tuple(args.baseline_years)
    forced = tuple(args.forced_years)
    log.info(f"SMILE decomposition. Baseline {baseline} → forced {forced}")
    log.info(f"Archive: {args.archive}  (style: {args.archive_style})")

    if args.archive_style == "cmip6":
        members = _discover_cmip6(args.archive, args.model, args.grid)
    else:
        members = _discover_mpi_ge(args.archive, args.filename_pattern)

    log.info(f"Discovered {len(members)} members: {sorted(members.keys())}")

    if args.members is not None:
        members = {m: members[m] for m in args.members if m in members}
        log.info(f"Filtered to: {sorted(members.keys())}")

    rows = []
    for m in sorted(members):
        # Reshape to {var: {exp: paths}}
        sources = members[m]
        row = process_member(m, sources, baseline, forced)
        if row is not None:
            rows.append(row)

    if not rows:
        log.error("No members processed successfully.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    log.info(f"\nSaved: {args.output}  ({len(rows)} members)")
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
