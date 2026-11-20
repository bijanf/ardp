#!/usr/bin/env python3
"""AMOC strength at 26.5N from the published Atlantic overturning streamfunction (msftmz or msftyz).

For each CMIP6 model of the original 16-model ensemble the streamfunction is taken, for the historical and SSP5-8.5
experiments, from the member used for the F_ovS decomposition where it is published, otherwise from the
lowest-numbered member that has both experiments (CESM2 r4i1p1f1 from the Pangeo catalogue; GISS-E2-1-G r1i1p1f2).
Files are searched on the CEDA and DKRZ ESGF indexes and every replica is tried. The AMOC is the annual mean of the
maximum below 500 m at the latitude row closest to 26.5N in the Atlantic basin, which is chosen by its label
(amoc26n_basin.py; MPI-ESM1-2 stores the global ocean first). CNRM-CM6-1 stores the
streamfunction on a row index; its row latitude is the mean latitude of the row over 80W-15W in the model's vo grid.

The five models added later already use the published streamfunction (compute_amoc26n_expansion.py). FIO-ESM-2-0 and
NESM3 publish no streamfunction; for them the series integrated from the meridional velocity is kept.

Writes data/results/amoc26n_published_streamfunction_check.npz and rebuilds data/results/yearly_amoc26n_cmip6.npz
(the previous file is kept as yearly_amoc26n_cmip6_velocity_based.npz).

Run from the repository root:
    python scripts/compute_amoc26n_published.py --out data/results
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import sys

import numpy as np
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amoc26n_basin import atlantic_basin  # noqa: E402

INDEXES = ["https://esgf.ceda.ac.uk/esg-search/search", "https://esgf-data.dkrz.de/esg-search/search"]
ESGF = {  # model: (variable, member)
    "ACCESS-CM2": ("msftmz", "r1i1p1f1"), "CanESM5": ("msftmz", "r1i1p1f1"), "FGOALS-g3": ("msftmz", "r1i1p1f1"),
    "MIROC6": ("msftmz", "r1i1p1f1"), "MPI-ESM1-2-HR": ("msftmz", "r1i1p1f1"), "MPI-ESM1-2-LR": ("msftmz", "r1i1p1f1"),
    "CMCC-CM2-SR5": ("msftyz", "r1i1p1f1"), "IPSL-CM6A-LR": ("msftyz", "r1i1p1f1"), "GFDL-CM4": ("msftyz", "r1i1p1f1"),
    "UKESM1-0-LL": ("msftyz", "r1i1p1f2"), "HadGEM3-GC31-LL": ("msftyz", "r1i1p1f3"), "CNRM-CM6-1": ("msftyz", "r1i1p1f2"),
    "GISS-E2-1-G": ("msftmz", "r1i1p1f2"),
}
PANGEO = {"CESM2": ("msftmz", "r4i1p1f1",
                    "gs://cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r4i1p1f1/Omon/msftmz/gn/v20190919/",
                    "gs://cmip6/CMIP6/ScenarioMIP/NCAR/CESM2/ssp585/r4i1p1f1/Omon/msftmz/gn/v20200528/")}
CNRM_GRID = "gs://cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Omon/vo/gn/v20180917/"


def file_urls(model, exp_id, var, member):
    urls = {}
    for base in INDEXES:
        p = {"format": "application/solr+json", "type": "File", "project": "CMIP6", "source_id": model,
             "experiment_id": exp_id, "variable_id": var, "member_id": member, "table_id": "Omon",
             "distrib": "true", "limit": 1000}
        try:
            docs = requests.get(base, params=p, timeout=120).json()["response"]["docs"]
        except Exception:
            continue
        for d in docs:
            for u in d.get("url", []):
                if "HTTPServer" in u:
                    urls.setdefault(d["title"], set()).add(u.split("|")[0])
    out = {}
    for fn, us in urls.items():
        m = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", fn)
        if m and int(m.group(1)) > 2100:
            continue
        out[fn] = sorted(us)
    return out


def fetch(urls, dest):
    for u in urls:
        try:
            with requests.get(u, stream=True, timeout=600) as r:
                if r.status_code != 200:
                    continue
                size = int(r.headers.get("Content-Length", 0))
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if size == 0 or dest.stat().st_size >= size:
                return True
        except Exception:
            continue
    return False


def _row_latitude_cnrm():
    import gcsfs
    g = xr.open_zarr(gcsfs.GCSFileSystem(token="anon").get_mapper(CNRM_GRID), consolidated=True)
    lat, lon = np.asarray(g["lat"].values), np.asarray(g["lon"].values)
    lon = np.where(lon > 180, lon - 360, lon)
    atl = (lon > -80) & (lon < -15)
    return np.array([np.nanmean(np.where(atl[j], lat[j], np.nan)) if atl[j].any() else np.nan for j in range(lat.shape[0])])


def amoc_series(ds, var, row_lat=None):
    da, bidx, blabel = atlantic_basin(ds, ds[var])
    zdim = next(d for d in da.dims if d in ("lev", "olevel", "depth") or "lev" in d)
    depth = np.asarray(ds[zdim].values, float)
    if np.nanmax(depth) > 1e4:  # centimetres
        depth = depth / 100.0
    for d in [d for d in da.dims if d not in ("time", zdim)]:
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
    ydim = next(d for d in da.dims if d not in ("time", zdim))
    if row_lat is None:
        cand = next(c for c in ("lat", "latitude", "rlat", "nav_lat", "y") if c in ds.coords or c in ds)
        row_lat = np.asarray(ds[cand].values, float)
        if row_lat.ndim == 2:
            row_lat = np.nanmean(row_lat, axis=1) if row_lat.shape[1] < row_lat.shape[0] else np.nanmean(row_lat, axis=0)
    j = int(np.nanargmin(np.abs(row_lat - 26.5)))
    psi = da.isel({ydim: j}).isel({zdim: np.where(depth >= 500.0)[0]}).max(dim=zdim)
    ann = psi.groupby("time.year").mean("time")
    v = ann.values.astype(float)
    if np.nanmean(np.abs(v)) > 1e6:  # kg s-1 to Sv
        v = v / 1e9
    return ann["year"].values.astype(int), v, float(row_lat[j]), bidx, blabel


def _splice(parts):
    y = np.concatenate([p[0] for p in parts])
    v = np.concatenate([p[1] for p in parts])
    o = np.argsort(y)
    y, v = y[o], v[o]
    _, first = np.unique(y, return_index=True)
    return y[sorted(first)], v[sorted(first)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/results")
    a = ap.parse_args()
    res_dir = Path(a.out)
    out_npz = res_dir / "amoc26n_published_streamfunction_check.npz"
    meta_file = res_dir / "amoc26n_published_streamfunction_members.json"
    res = dict(np.load(out_npz, allow_pickle=True)) if out_npz.exists() else {}
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    scratch = res_dir / "tmp_amoc26n_published"
    for model, (var, member) in ESGF.items():
        if f"{model}_years" in res:
            continue
        work = scratch / model
        work.mkdir(parents=True, exist_ok=True)
        row_lat = _row_latitude_cnrm() if model == "CNRM-CM6-1" else None
        parts, missing, latj = [], [], None
        for e in ("historical", "ssp585"):
            paths = []
            for fn, us in sorted(file_urls(model, e, var, member).items()):
                p = work / fn
                (paths.append(p) if fetch(us, p) else missing.append(fn))
            ds = xr.open_mfdataset(sorted(paths), combine="by_coords", use_cftime=True)
            y, v, latj, bidx, blabel = amoc_series(ds, var, row_lat)
            parts.append((y, v))
            ds.close()
        res[f"{model}_years"], res[f"{model}_amoc"] = _splice(parts)
        meta[model] = {"variable": var, "member": member, "latitude_used": latj, "basin_index": bidx,
                       "basin_label": blabel, "missing_files": missing}
        shutil.rmtree(work, ignore_errors=True)
        np.savez_compressed(out_npz, **res)  # saved after each model, so an interrupted run resumes
        meta_file.write_text(json.dumps(meta, indent=1))
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    for model, (var, member, zh, zs) in PANGEO.items():
        if f"{model}_years" in res:
            continue
        parts = []
        for z in (zh, zs):
            y, v, latj, bidx, blabel = amoc_series(xr.open_zarr(fs.get_mapper(z), consolidated=True, use_cftime=True), var)
            parts.append((y, v))
        res[f"{model}_years"], res[f"{model}_amoc"] = _splice(parts)
        meta[model] = {"variable": var, "member": member, "latitude_used": latj, "basin_index": bidx,
                       "basin_label": blabel, "missing_files": [], "source": "Pangeo"}
    np.savez_compressed(out_npz, **res)
    meta_file.write_text(json.dumps(meta, indent=1))
    # rebuild the AMOC file of the analysis
    cur = res_dir / "yearly_amoc26n_cmip6.npz"
    vel = res_dir / "yearly_amoc26n_cmip6_velocity_based.npz"
    if not vel.exists():
        shutil.copy2(cur, vel)
    old = np.load(vel, allow_pickle=True)
    new = {k: old[k] for k in old.files}
    for m in [str(x) for x in old["models"]]:
        if f"{m}_years" in res:
            new[f"{m}_years"], new[f"{m}_amoc"] = res[f"{m}_years"], res[f"{m}_amoc"]
    np.savez_compressed(cur, **new)


if __name__ == "__main__":
    main()
