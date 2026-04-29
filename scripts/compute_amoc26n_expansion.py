#!/usr/bin/env python3
"""Extract AMOC at 26.5 deg N for 5 expansion models.

Of the 9 newly-added CMIP6 models (used to build the 25-model F_ovS
mechanism scatter), only 5 publish a meridional overturning
streamfunction usable for an AMOC-weakening calculation at 26.5 deg N
on the r1i1p1f1, gn grid:

    msftmz, hist+ssp585: ACCESS-ESM1-5, CESM2-WACCM, FGOALS-f3-L,
                         MRI-ESM2-0
    msftyz, hist+ssp585: CMCC-ESM2

The remaining 4 (BCC-CSM2-MR, CIESM, EC-Earth3, TaiESM1) do not have
ssp585 streamfunction on ESGF and stay AMOC-blind; they remain in the
mechanism scatter but not in the weakening / lead-lag / emergent
constraint regressions.

For each (model, exp) the script downloads the (depth x lat) (or
(basin x depth x lat)) streamfunction, slices to 26.5 deg N, takes
max over depth >= 500 m, and annual-means.  Hist + ssp585 are then
spliced.  The resulting (years, amoc) pair is appended to
``data/results/yearly_amoc26n_cmip6.npz``.

Resume-friendly: skips models already in the npz unless --force.
"""
from __future__ import annotations

import argparse
import logging
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
NPZ_OUT = REPO / "data" / "results" / "yearly_amoc26n_cmip6.npz"
SCRATCH = Path("/tmp/amoc26n_expansion")

RAPID_LAT = 26.5
DEPTH_MIN = 500.0

REPLICA_HOSTS = ["esgf-data.dkrz.de", "esgf3.dkrz.de",
                 "esgf.ceda.ac.uk", "esgf-data3.ceda.ac.uk",
                 "esgf-data1.llnl.gov", "esgf-data2.llnl.gov",
                 "esgf-node.ipsl.upmc.fr", "vesg.ipsl.upmc.fr"]

# (model, variable, fovs_estimate_for_metadata)
TARGETS = {
    "ACCESS-ESM1-5": ("msftmz", None),
    "CESM2-WACCM":   ("msftmz", None),
    "FGOALS-f3-L":   ("msftmz", None),
    "MRI-ESM2-0":    ("msftmz", None),
    "CMCC-ESM2":     ("msftyz", None),
}


def _solr_search(model: str, exp: str, var: str) -> list[dict]:
    """Search ESGF for streamfunction files, accepting gn / gr / gr2z grids
    (some models publish only a regridded zonal-only grid like gr2z)."""
    base = "https://esgf-data.dkrz.de/esg-search/search"
    params = {
        "format": "application/solr+json", "type": "File", "project": "CMIP6",
        "source_id": model, "experiment_id": exp, "variable": var,
        "table_id": "Omon", "variant_label": "r1i1p1f1",
        "distrib": "true", "limit": 10000,
    }
    r = requests.get(base, params=params, timeout=120)
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    # Pick the first available grid; prefer gn, then gr, then any
    grids_present = set()
    for d in docs:
        g = d.get("grid_label")
        if isinstance(g, list):
            g = g[0] if g else None
        if g:
            grids_present.add(g)
    if not grids_present:
        return []
    grid = next((g for g in ["gn", "gr", "gr1", "gr2", "gr2z"]
                 if g in grids_present), next(iter(grids_present)))
    docs = [d for d in docs
            if (d.get("grid_label", [None])[0]
                if isinstance(d.get("grid_label"), list)
                else d.get("grid_label")) == grid]
    seen: set[str] = set()
    files: list[dict] = []
    for d in docs:
        urls = d.get("url", [])
        url = None
        for u in urls:
            parts = u.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                url = parts[0]
                break
        if url is None:
            continue
        fname = d.get("title") or url.split("/")[-1]
        if fname in seen:
            continue
        seen.add(fname)
        files.append({"filename": fname, "url": url})
    return files


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        # Validate vs Content-Length where possible
        for u in [url] + [
            url.replace(f"//{src}/", f"//{h}/")
            for src in REPLICA_HOSTS for h in REPLICA_HOSTS
            if src != h and f"//{src}/" in url
        ]:
            try:
                r = requests.head(u, timeout=30, allow_redirects=True)
            except Exception:
                continue
            if r.status_code != 200:
                continue
            remote = int(r.headers.get("content-length", 0))
            local = dest.stat().st_size
            if remote > 0 and local < int(0.99 * remote):
                log.info(f"    [partial] {dest.name} "
                         f"{local/1e6:.0f}/{remote/1e6:.0f} MB -> redo")
                dest.unlink()
                break
            log.info(f"    [cache] {dest.name}")
            return True
        if dest.exists():
            return True
    candidates = [url]
    for src in REPLICA_HOSTS:
        if f"//{src}/" in url:
            for h in REPLICA_HOSTS:
                if h != src:
                    candidates.append(url.replace(f"//{src}/", f"//{h}/"))
            break
    import time
    for u in candidates:
        host = u.split("/")[2]
        t0 = time.time()
        try:
            with requests.get(u, stream=True, timeout=600,
                              allow_redirects=True) as r:
                if r.status_code != 200:
                    log.info(f"    [skip {r.status_code}] {host}")
                    continue
                size = int(r.headers.get("content-length", 0))
                log.info(f"    [GET {host}] {dest.name} "
                         f"({size/1e6:.0f} MB)")
                done = 0
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            done += len(chunk)
                # Reject obvious error-page replies and partial transfers.
                # msftmz files are >=10 MB; <1 MB is always garbage.
                bad = done < 1_000_000 or (size > 0 and done < int(0.99 * size))
                if bad:
                    log.warning(f"    [bad {done/1e3:.0f}kB / "
                                f"expected {size/1e6:.0f} MB] {host} "
                                f"-> next replica")
                    if dest.exists():
                        dest.unlink()
                    continue
                log.info(f"    [done] {done/1e6:.0f} MB in "
                         f"{time.time()-t0:.0f}s")
                return True
        except Exception as exc:  # noqa: BLE001
            log.warning(f"    {host} -> {exc}")
            continue
    return False


def _extract_amoc(ds: xr.Dataset, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, amoc_max_below500_at_26.5N) in Sv."""
    da = ds[var]
    lat_name = next((n for n in ("lat", "latitude", "rlat") if n in ds.coords),
                    None)
    lev_name = next((n for n in ("lev", "depth", "olevel") if n in ds.coords),
                    None)
    if lat_name is None or lev_name is None:
        raise RuntimeError(
            f"{var}: missing lat/lev in coords {list(ds.coords)}")
    lat_vals = ds[lat_name].values
    j_idx = int(np.abs(lat_vals - RAPID_LAT).argmin())
    sec = da.isel({lat_name: j_idx})
    depth = ds[lev_name].values.astype(float)
    deep_idx = np.where(depth >= DEPTH_MIN)[0]
    sec_deep = sec.isel({lev_name: deep_idx})
    if "basin" in sec_deep.dims:
        # Some models stack Atl/Pac/Indo basins; pick Atlantic-Arctic (0).
        # CMIP6 convention: basin=0 atlantic_arctic_ocean.
        sec_deep = sec_deep.isel(basin=0)
    psi_max = sec_deep.max(dim=lev_name)
    psi_annual = psi_max.groupby("time.year").mean(dim="time")
    years = psi_annual["year"].values.astype(float)
    vals = psi_annual.values.astype(float)
    if np.nanmean(np.abs(vals)) > 1e6:  # kg/s -> Sv
        vals = vals / 1e9
    return years, vals


def process_model(model: str, var: str, scratch: Path
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    paths_by_exp: dict[str, list[Path]] = {}
    for exp in ("historical", "ssp585"):
        files = _solr_search(model, exp, var)
        if not files:
            log.error(f"  {model}/{exp}: no {var} files on ESGF")
            return None
        log.info(f"  {model}/{exp}: {len(files)} {var} files")
        local = []
        for f in files:
            dest = scratch / f["filename"]
            if not _download(f["url"], dest):
                log.error(f"  {model}/{exp}: download failed for "
                          f"{f['filename']}")
                return None
            local.append(dest)
        paths_by_exp[exp] = local
    try:
        ds_h = xr.open_mfdataset(
            sorted(paths_by_exp["historical"]),
            combine="by_coords", parallel=False,
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            engine="netcdf4",
        )
        years_h, amoc_h = _extract_amoc(ds_h, var)
        ds_h.close()
        ds_s = xr.open_mfdataset(
            sorted(paths_by_exp["ssp585"]),
            combine="by_coords", parallel=False,
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            engine="netcdf4",
        )
        years_s, amoc_s = _extract_amoc(ds_s, var)
        ds_s.close()
    except Exception as exc:  # noqa: BLE001
        log.error(f"  {model}: open/extract failed: {exc}")
        return None

    combined_y = np.concatenate([years_h, years_s])
    combined_a = np.concatenate([amoc_h, amoc_s])
    order = np.argsort(combined_y)
    combined_y = combined_y[order]
    combined_a = combined_a[order]
    _, unique_idx = np.unique(combined_y, return_index=True)
    years = combined_y[sorted(unique_idx)]
    amoc = combined_a[sorted(unique_idx)]
    base = float(np.nanmean(amoc[(years >= 1950) & (years <= 1980)]))
    end = float(np.nanmean(amoc[(years >= 2080) & (years <= 2100)]))
    pct = 100 * (base - end) / base if base > 0 else float("nan")
    log.info(f"  {model}: {len(years)} years; base={base:.1f} Sv, "
             f"end={end:.1f} Sv, weakening={pct:+.0f}%")
    return years, amoc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if model is already in npz.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated subset of TARGETS")
    args = parser.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, np.ndarray] = {}
    if NPZ_OUT.exists():
        with np.load(NPZ_OUT, allow_pickle=True) as d:
            existing = {k: d[k].copy() for k in d.files}
    have_models = list(existing.get("models", np.array([], dtype=object)))
    log.info(f"Existing npz has {len(have_models)} models")

    if args.models:
        targets = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        targets = list(TARGETS.keys())

    new_models: list[str] = []
    failures: list[str] = []
    for m in targets:
        if m in have_models and not args.force:
            log.info(f"=== {m}: already in npz, skip ===")
            continue
        log.info(f"\n=== {m} ({TARGETS[m][0]}) ===")
        scratch = SCRATCH / m
        scratch.mkdir(parents=True, exist_ok=True)
        result = process_model(m, TARGETS[m][0], scratch)
        if result is None:
            failures.append(m)
            shutil.rmtree(scratch, ignore_errors=True)
            continue
        years, amoc = result
        existing[f"{m}_years"] = years
        existing[f"{m}_amoc"] = amoc
        existing[f"{m}_fovs"] = np.array([np.nan])
        new_models.append(m)
        shutil.rmtree(scratch, ignore_errors=True)

    if new_models:
        merged = list(have_models) + [m for m in new_models
                                      if m not in have_models]
        existing["models"] = np.array(merged, dtype=object)
        np.savez_compressed(NPZ_OUT, **existing)
        log.info(f"\nWrote {NPZ_OUT}  ({len(merged)} models total, "
                 f"+{len(new_models)} new)")
    else:
        log.info("Nothing to write.")

    if failures:
        log.warning(f"FAILURES ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
