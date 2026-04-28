#!/usr/bin/env python3
"""Per-model CMIP6 expansion downloader.

Generalizes the SMILE pipeline to a per-model loop. For each new
CMIP6 model in NEW_MODELS that we don't already have on disk, download
the historical+ssp585 vo+so files from federated ESGF, extract the
34.5 deg S section, save to data/cmip6_sections/{model}_{exp}_{var}.nc
in the same format as the existing 15-model ensemble, and delete the
raw downloads.

Run with no args; resume-friendly (skips files already on disk).

Outputs:
    data/cmip6_sections/{model}_{historical,ssp585}_{vo,so}.nc
    (4 files per model)

After this completes, re-run compute_cmip6_fovs_decomposition.py to
update the headline ensemble summary CSV.
"""
from __future__ import annotations

import logging
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import requests
import xarray as xr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
SECTIONS_DIR = REPO / "data" / "cmip6_sections"
SCRATCH_BASE = Path("/tmp/cmip6_expansion")

TARGET_LAT = -34.5
TIMEOUT = 600

NEW_MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CESM2-WACCM",
    "CIESM",
    "CMCC-ESM2",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MRI-ESM2-0",
    "TaiESM1",
]

EXP_YEARS = {
    "historical": (1950, 2014),
    "ssp585":     (2015, 2100),
}

REPLICA_HOSTS = ["esgf-data.dkrz.de", "esgf3.dkrz.de",
                 "esgf.ceda.ac.uk", "esgf-data3.ceda.ac.uk",
                 "esgf-data1.llnl.gov", "esgf-data2.llnl.gov",
                 "esgf-node.ipsl.upmc.fr", "vesg.ipsl.upmc.fr"]


def _solr_search(source_id: str, experiment_id: str, variable: str,
                 grid_label: str = "gn") -> list[dict]:
    """ESGF Solr file search for one (model, exp, var, grid). distrib=true."""
    base = "https://esgf-data.dkrz.de/esg-search/search"
    params = {
        "format": "application/solr+json",
        "type": "File",
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable": variable,
        "table_id": "Omon",
        "variant_label": "r1i1p1f1",
        "grid_label": grid_label,
        "distrib": "true",
        "limit": 10000,
    }
    r = requests.get(base, params=params, timeout=120)
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    seen: set[str] = set()
    files: list[dict] = []
    for d in docs:
        urls = d.get("url", [])
        download_url = None
        for u in urls:
            parts = u.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                download_url = parts[0]
                break
        if download_url is None:
            continue
        fname = d.get("title") or download_url.split("/")[-1]
        if fname in seen:
            continue
        seen.add(fname)
        files.append({"filename": fname, "download_url": download_url})
    return files


def _years_in_filename(fname: str) -> tuple[int, int] | None:
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", fname)
    if m is None:
        return None
    return int(m.group(1)[:4]), int(m.group(2)[:4])


def _files_covering(files: list[dict], years: tuple[int, int]) -> list[dict]:
    y_lo, y_hi = years
    out = []
    for f in files:
        rng = _years_in_filename(f["filename"])
        if rng is None:
            out.append(f)  # ambiguous name -> keep
            continue
        f_lo, f_hi = rng
        if f_hi < y_lo or f_lo > y_hi:
            continue
        out.append(f)
    return out


def _download(url: str, dest: Path) -> bool:
    """HTTP-download with replica fallback."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"    [cache] {dest.name}")
        return True
    candidates = [url]
    for src_host in REPLICA_HOSTS:
        if f"//{src_host}/" in url:
            for host in REPLICA_HOSTS:
                if host != src_host:
                    candidates.append(url.replace(f"//{src_host}/",
                                                   f"//{host}/"))
            break
    import time as _t
    for u in candidates:
        host = u.split("/")[2]
        try:
            t0 = _t.time()
            with requests.get(u, stream=True, timeout=TIMEOUT,
                              allow_redirects=True) as r:
                if r.status_code != 200:
                    log.info(f"    [skip {r.status_code}] {host}")
                    continue
                size = int(r.headers.get("content-length", 0))
                size_mb = size / 1e6 if size else float("nan")
                log.info(f"    [GET {host}] {dest.name} ({size_mb:.0f} MB)")
                done = 0
                last = t0
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        now = _t.time()
                        if now - last >= 15.0:
                            mb = done / 1e6
                            mbps = mb / (now - t0) if now > t0 else 0
                            pct = (100 * done / size) if size else 0
                            log.info(f"      ... {mb:.0f} MB "
                                     f"({pct:.0f}%, {mbps:.1f} MB/s)")
                            last = now
                elapsed = _t.time() - t0
                log.info(f"    [done] {done/1e6:.0f} MB in {elapsed:.0f}s")
                return True
        except Exception as exc:
            log.warning(f"    {host} -> {exc}")
            continue
    return False


def _find_j(ds: xr.Dataset) -> tuple[int, str, str, float]:
    """Locate j-index nearest TARGET_LAT, returning (j, j_dim, lat_name, lat)."""
    lat_name = None
    for name in ["lat", "latitude", "nav_lat"]:
        if name in ds.coords or name in ds.data_vars:
            lat_name = name
            break
    if lat_name is None:
        for name in list(ds.coords) + list(ds.data_vars):
            if "lat" in str(name).lower():
                lat_name = name
                break
    if lat_name is None:
        raise ValueError(f"No lat in {list(ds.coords)}")
    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 1:
        j = int(np.abs(lat_vals - TARGET_LAT).argmin())
        return j, lat_name, lat_name, float(lat_vals[j])
    elif lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j = int(np.abs(lat_1d - TARGET_LAT).argmin())
        return j, ds[lat_name].dims[0], lat_name, float(lat_1d[j])
    raise ValueError(f"Bad lat shape {lat_vals.shape}")


def _process_model_exp_var(model: str, exp: str, var: str,
                            scratch: Path) -> bool:
    """Download files covering EXP_YEARS[exp], extract section, save."""
    out_path = SECTIONS_DIR / f"{model}_{exp}_{var}.nc"
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info(f"  [cache] {out_path.name}")
        return True

    files = _solr_search(model, exp, var)
    if not files:
        log.warning(f"  no files for {model}/{exp}/{var}")
        return False
    files = _files_covering(files, EXP_YEARS[exp])
    if not files:
        log.warning(f"  no files cover {EXP_YEARS[exp]} for "
                    f"{model}/{exp}/{var}")
        return False
    log.info(f"  {len(files)} files cover {EXP_YEARS[exp]} window")

    scratch.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for f in files:
        dest = scratch / f["filename"]
        ok = _download(f["download_url"], dest)
        if not ok:
            log.error(f"  FAILED {f['filename']}")
            return False
        local_paths.append(dest)

    log.info(f"  opening {len(local_paths)} files with xarray ...")
    try:
        ds = xr.open_mfdataset(local_paths, combine="by_coords",
                                decode_times=True,
                                data_vars="minimal", coords="minimal",
                                compat="override")
    except Exception as exc:
        log.error(f"  open_mfdataset failed: {exc}")
        return False

    # Slice to year window
    y_lo, y_hi = EXP_YEARS[exp]
    ds = ds.sel(time=slice(f"{y_lo}-01-01", f"{y_hi}-12-31"))

    # j-index extraction
    try:
        j_idx, j_dim, lat_name, actual_lat = _find_j(ds)
    except ValueError as exc:
        log.error(f"  cannot find j: {exc}")
        return False
    log.info(f"  section at j={j_idx} (lat={actual_lat:.2f}), "
             f"j_dim='{j_dim}'")

    section = ds.isel({j_dim: j_idx})
    if var not in section.data_vars:
        log.error(f"  var {var} missing from section: "
                  f"{list(section.data_vars)}")
        return False
    out = section[[var]]
    out[var].attrs.update({
        "source_id": model, "experiment_id": exp,
        "grid_label": "gn", "member_id": "r1i1p1f1",
        "section_latitude": actual_lat, "section_j_index": int(j_idx),
    })

    log.info(f"  loading {var} into memory + saving ...")
    out.load()
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path, format="NETCDF4")
    log.info(f"  wrote {out_path.name} "
             f"({out_path.stat().st_size/1e6:.0f} MB)")

    out.close()
    ds.close()
    return True


def main() -> None:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_BASE.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures: list[str] = []
    for i, model in enumerate(NEW_MODELS, 1):
        log.info(f"\n=== [{i}/{len(NEW_MODELS)}] {model} ===")
        scratch = SCRATCH_BASE / model
        for exp in ("historical", "ssp585"):
            for var in ("vo", "so"):
                log.info(f"  -- {model}/{exp}/{var} --")
                ok = _process_model_exp_var(model, exp, var, scratch)
                if ok:
                    successes += 1
                else:
                    failures.append(f"{model}/{exp}/{var}")
        # Cleanup scratch for this model
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
            log.info(f"  cleaned scratch for {model}")

    log.info(f"\nDone. {successes}/{len(NEW_MODELS)*4} (model,exp,var) "
             f"combos completed.")
    if failures:
        log.warning(f"FAILURES ({len(failures)}):")
        for f in failures:
            log.warning(f"  - {f}")


if __name__ == "__main__":
    main()
