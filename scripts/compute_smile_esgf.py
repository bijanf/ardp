#!/usr/bin/env python3
"""SMILE F_ovS decomposition — ESGF download/process/delete pipeline.

Streams the MPI-ESM1-2-LR Grand Ensemble (~50 members) one at a time
from ESGF replicas (CEDA preferred for HTTP robustness; DKRZ as
fallback). For each member:

  1. Search ESGF for the (member, var, exp) datasets via pyesgf.
  2. Figure out which file chunks cover the requested years.
  3. HTTP-download only those chunks to a /tmp scratch dir.
  4. Open with xarray, slice to the 34.5°S row, period-mean, decompose.
  5. Delete the scratch files before moving to the next member.

Per-member peak disk: ~5-10 GB (only the year ranges that intersect
the baseline + forced windows are downloaded).

Run with the default 50-member-discovery and centennial windows:

    python scripts/compute_smile_esgf.py

Outputs:
    data/results/fovs_decomposition_smile_esgf.csv

Resume-friendly: members already in the CSV are skipped on re-run.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

warnings.filterwarnings("ignore")

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
SOURCE = "MPI-ESM1-2-LR"
DEFAULT_BASELINE = (1950, 1980)
DEFAULT_FORCED = (2080, 2100)
DEFAULT_OUTPUT = Path("data/results/fovs_decomposition_smile_esgf.csv")
SCRATCH_BASE = Path("/tmp/smile_esgf")

# Replicas in priority order (some hosts return 404 for retracted versions).
REPLICA_HOSTS = ["esgf.ceda.ac.uk", "esgf3.dkrz.de",
                 "esgf-data1.llnl.gov", "esgf-data2.llnl.gov",
                 "esgf-data3.ceda.ac.uk", "esgf-data.dkrz.de"]


def _esgf_search(source_id: str, experiment_id: str, variable: str,
                 grid_label: str = "gn") -> dict[str, list[dict]]:
    """Return {member_id: [{filename, urls}, ...]} for one (model, exp, var).

    Uses a direct Solr query at type=File rather than pyesgf's
    per-dataset file_context loop — ~50× faster (1 HTTP round-trip
    instead of 50). DKRZ ESGF index, distrib=false.
    """
    base = "https://esgf-data.dkrz.de/esg-search/search"
    params = {
        "format": "application/solr+json",
        "type": "File",
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable": variable,
        "table_id": "Omon",
        "grid_label": grid_label,
        "distrib": "false",
        "limit": 10000,
    }
    t0 = __import__("time").time()
    r = requests.get(base, params=params, timeout=120)
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    log.info(f"  Solr {variable}/{experiment_id}: {len(docs)} files in "
             f"{__import__('time').time()-t0:.1f}s")
    members: dict[str, list[dict]] = {}
    seen_files: dict[str, set[str]] = {}
    for d in docs:
        member = d.get("variant_label")
        if isinstance(member, list):
            member = member[0]
        urls = d.get("url", [])
        download_url = None
        for u in urls:
            # Each entry is "URL|mime|service" — pick HTTPServer
            parts = u.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                download_url = parts[0]
                break
        if download_url is None:
            continue
        fname = d.get("title") or download_url.split("/")[-1]
        if member not in seen_files:
            seen_files[member] = set()
        if fname in seen_files[member]:
            continue
        seen_files[member].add(fname)
        members.setdefault(member, []).append({
            "filename": fname,
            "download_url": download_url,
        })
    return members


def _years_in_filename(fname: str) -> tuple[int, int] | None:
    """Parse the YYYYMM-YYYYMM range from a CMIP6 filename."""
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", fname)
    if m is None:
        return None
    y0 = int(m.group(1)[:4])
    y1 = int(m.group(2)[:4])
    return y0, y1


def _files_covering(files: list[dict], years: tuple[int, int]) -> list[dict]:
    """Subset files whose [y0, y1] range intersects the requested window."""
    y_lo, y_hi = years
    out = []
    for f in files:
        rng = _years_in_filename(f["filename"])
        if rng is None:
            out.append(f)  # keep ambiguous files
            continue
        f_lo, f_hi = rng
        if f_hi < y_lo or f_lo > y_hi:
            continue
        out.append(f)
    return out


def _download(url: str, dest: Path, timeout: int = 600) -> bool:
    """HTTP-download with replica fallback + progress logging."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"    [cache] {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return True
    candidates = [url]
    for host in REPLICA_HOSTS:
        for src_host in REPLICA_HOSTS:
            if f"//{src_host}/" in url and src_host != host:
                candidates.append(url.replace(f"//{src_host}/", f"//{host}/"))
    import time as _t
    for u in candidates:
        try:
            t0 = _t.time()
            with requests.get(u, stream=True, timeout=timeout,
                              allow_redirects=True) as r:
                if r.status_code != 200:
                    log.info(f"    [skip {r.status_code}] {u.split('/')[2]}")
                    continue
                size = int(r.headers.get("content-length", 0))
                size_mb = size / 1e6 if size else float("nan")
                host = u.split("/")[2]
                log.info(f"    [GET {host}] {dest.name}  ({size_mb:.0f} MB)")
                last_log = t0
                done = 0
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        now = _t.time()
                        # Progress every 10s
                        if now - last_log >= 10.0:
                            mb = done / 1e6
                            mbps = mb / (now - t0) if now > t0 else 0
                            pct = (100 * done / size) if size else 0
                            log.info(f"      ... {mb:.0f}/{size_mb:.0f} MB "
                                     f"({pct:.0f}%, {mbps:.1f} MB/s)")
                            last_log = now
                elapsed = _t.time() - t0
                final_mb = done / 1e6
                avg_mbps = final_mb / elapsed if elapsed > 0 else 0
                log.info(f"    [done] {dest.name}  "
                         f"{final_mb:.0f} MB in {elapsed:.0f}s ({avg_mbps:.1f} MB/s)")
                return True
        except Exception as exc:  # noqa: BLE001
            log.warning(f"    {u.split('/')[2]} -> {exc}")
            continue
    return False


def _open_concat(paths: list[Path], var: str) -> xr.DataArray:
    ds = xr.open_mfdataset(
        sorted(paths), combine="by_coords", parallel=False,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        engine="netcdf4",
    )
    return ds[var]


def _slice_to_section(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (period-mean section [lev,x], lon_180, depth)."""
    raise NotImplementedError  # supplied per call below


def _section_period_mean(da: xr.DataArray, years: tuple[int, int]):
    lat_name = next((n for n in ("latitude", "lat", "nav_lat")
                     if n in da.coords), None)
    if lat_name is None:
        raise RuntimeError(f"no latitude coord; coords: {list(da.coords)}")
    lat_vals = da[lat_name].values
    if lat_vals.ndim == 1:
        j_idx = int(np.abs(lat_vals - TARGET_LAT).argmin())
        j_dim = lat_name
    else:  # 2-D
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_idx = int(np.abs(lat_1d - TARGET_LAT).argmin())
        j_dim = da[lat_name].dims[0]

    section_full = da.isel({j_dim: j_idx})
    t_year = section_full["time"].dt.year.values
    mask = (t_year >= years[0]) & (t_year <= years[1])
    if mask.sum() == 0:
        raise RuntimeError(f"no timesteps in {years}")
    section_period = section_full.isel(time=np.where(mask)[0])
    arr = section_period.mean(dim="time", skipna=True).values

    lon_name = next((n for n in ("longitude", "lon", "nav_lon")
                     if n in section_full.coords or n in da.coords), None)
    if lon_name in section_full.coords:
        lon_vals = section_full[lon_name].values
    else:
        lon_vals = da[lon_name].isel({j_dim: j_idx}).values
    if lon_vals.ndim != 1:
        lon_vals = lon_vals[0] if lon_vals.shape[0] == 1 else lon_vals
    lon_180 = np.where(lon_vals > 180, lon_vals - 360, lon_vals)

    lev_name = next((n for n in ("lev", "depth", "olevel", "z_t")
                     if n in section_full.coords), None)
    depth = section_full[lev_name].values.astype(float)
    return arr, lon_180, depth


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


def process_member(member: str,
                   files_by_var_exp: dict[str, dict[str, list[dict]]],
                   baseline: tuple[int, int],
                   forced: tuple[int, int],
                   scratch: Path) -> dict | None:
    """Download what we need, decompose, clean up."""
    member_dir = scratch / member
    member_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded: dict[str, dict[str, list[Path]]] = {}
        for var in ("vo", "so"):
            for exp, years in [("historical", baseline), ("ssp585", forced)]:
                files = files_by_var_exp.get(var, {}).get(exp)
                if not files:
                    log.warning(f"  {member}: no {var}/{exp} files")
                    return None
                needed = _files_covering(files, years)
                if not needed:
                    log.warning(f"  {member}: no {var}/{exp} chunks cover {years}")
                    return None
                paths: list[Path] = []
                for f in needed:
                    dest = member_dir / f["filename"]
                    if not _download(f["download_url"], dest):
                        log.error(f"  {member}: failed to download {f['filename']}")
                        return None
                    paths.append(dest)
                downloaded.setdefault(var, {})[exp] = paths

        v1_arr, lon_180, depth = _section_period_mean(
            _open_concat(downloaded["vo"]["historical"], "vo"), baseline)
        s1_arr, *_ = _section_period_mean(
            _open_concat(downloaded["so"]["historical"], "so"), baseline)
        v2_arr, *_ = _section_period_mean(
            _open_concat(downloaded["vo"]["ssp585"], "vo"), forced)
        s2_arr, *_ = _section_period_mean(
            _open_concat(downloaded["so"]["ssp585"], "so"), forced)

        atl, e1t, e3t = _grid_metrics(lon_180, depth, TARGET_LAT)
        v1a, s1a = v1_arr[:, atl], s1_arr[:, atl]
        v2a, s2a = v2_arr[:, atl], s2_arr[:, atl]
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
            f"  {member}: F1={result['F_ov_1']:+.3f}  F2={result['F_ov_2']:+.3f}"
            f"  ΔF={dt:+.3f} Sv  v:{v_frac:+.0f}%  s:{s_frac:+.0f}%"
        )
        return {
            "model": SOURCE,
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
    except Exception as exc:  # noqa: BLE001
        log.error(f"  {member}: FAILED ({exc})")
        return None
    finally:
        shutil.rmtree(member_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-years", nargs=2, type=int,
                        default=list(DEFAULT_BASELINE), metavar=("Y0", "Y1"))
    parser.add_argument("--forced-years", nargs=2, type=int,
                        default=list(DEFAULT_FORCED), metavar=("Y0", "Y1"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--members", nargs="+", default=None,
                        help="Subset (default: all discovered)")
    parser.add_argument("--scratch", type=Path, default=SCRATCH_BASE)
    args = parser.parse_args()

    baseline = tuple(args.baseline_years)
    forced = tuple(args.forced_years)
    log.info(f"SMILE/ESGF for {SOURCE}: baseline {baseline} → forced {forced}")
    log.info("Querying ESGF (DKRZ index, federated)...")

    # Build {member: {var: {exp: [files]}}}
    catalog: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for var in ("vo", "so"):
        for exp in ("historical", "ssp585"):
            members = _esgf_search(SOURCE, exp, var)
            log.info(f"  {var}/{exp}: {len(members)} members")
            for m, files in members.items():
                catalog.setdefault(m, {}).setdefault(var, {})[exp] = files
    common = sorted(m for m, d in catalog.items()
                    if all(d.get(v, {}).get(e) for v in ("vo", "so")
                           for e in ("historical", "ssp585")))
    log.info(f"Members with all four (vo, so) × (hist, ssp585): {len(common)}")
    log.info(f"  {common}")

    if args.members is not None:
        common = [m for m in common if m in args.members]
        log.info(f"Filtered to: {common}")

    # Resume: skip members already in the CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.output.exists():
        existing = set(pd.read_csv(args.output)["member_id"].astype(str))
        log.info(f"Resuming: {len(existing)} members already in {args.output}")
    todo = [m for m in common if m not in existing]
    log.info(f"To process: {len(todo)}")

    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in todo:
        row = process_member(m, catalog[m], baseline, forced, args.scratch)
        if row is None:
            continue
        rows.append(row)
        # Append-write so partial results survive interruption
        pd.DataFrame([row]).to_csv(
            args.output, mode="a",
            header=not args.output.exists() or args.output.stat().st_size == 0,
            index=False,
        )

    if not rows:
        log.warning("No new members processed.")
        return
    df_all = pd.read_csv(args.output)
    log.info("")
    log.info(f"--- SMILE/ESGF summary across {len(df_all)} members ---")
    log.info(f"  ΔF_total mean:     {df_all['delta_total'].mean()*1000:+.1f} mSv")
    log.info(f"  ΔF_total spread:   ±{df_all['delta_total'].std()*1000:.1f} mSv (1σ)")
    log.info(f"  velocity_share:    mean={df_all['velocity_share_pct'].mean():+.1f}%  "
             f"sd={df_all['velocity_share_pct'].std():.1f}%")
    weakening = df_all[df_all["delta_total"] < -0.01]
    v_dom = weakening[weakening["velocity_share_pct"] > 60]
    s_dom = weakening[weakening["salinity_share_pct"] > 60]
    log.info(f"  Weakening members: {len(weakening)}/{len(df_all)}")
    log.info(f"  v-dominant: {len(v_dom)}/{len(weakening)}")
    log.info(f"  s-dominant: {len(s_dom)}/{len(weakening)}")


if __name__ == "__main__":
    main()
