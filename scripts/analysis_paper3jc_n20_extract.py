#!/usr/bin/env python3
"""Extract the 34.5 S section of one MPI-ESM1-2-LR ensemble member.

The vo, so and thkcello stores of one member and experiment are opened lazily
from the Pangeo CMIP6 archive. The section row is chosen by the rule of the
CMIP6 section download (the row whose mean latitude is nearest 34.5 S), and
only that row is read, for 1950-2014 (historical) or 2015-2100 (ssp585). The
monthly section diagnostics are computed by the functions of the CMIP6 r1
pipeline (contiguous Atlantic run, bounds-based cell thickness, removal of the
net section transport, limb factorisation F_ovS = -(1/S0) T dS), so that the
ensemble members and the r1 run are processed identically.

The model's own thkcello is read for the first month only, as a check of the
bounds-based thickness on the section row; it does not enter the diagnostics.

Writes ``data/results/paper3jc_smile_mpi_lr/parts/<member>_<experiment>.nc``
with the annual means of F_ov, T, dS, S_north, S_south, F_az and psi34 (the
maximum of the streamfunction below 500 m, the upper-cell transport).

Usage: python scripts/analysis_paper3jc_n20_extract.py --member r2i1p1f1 \
           --experiment historical
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import dask  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analysis_paper3r1_cmip6 as pipeline  # noqa: E402
from download_cmip6_fovs_sections import _find_lat_and_j  # noqa: E402

MODEL = "MPI-ESM1-2-LR"
GRID = "gn"
VERSION = "v20190710"
ACTIVITY = {"historical": "CMIP", "ssp585": "ScenarioMIP"}
YEARS = {"historical": (1950, 2014), "ssp585": (2015, 2100)}
OUT_DIR = REPO / "data" / "results" / "paper3jc_smile_mpi_lr" / "parts"
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def store_url(member: str, experiment: str, variable: str) -> str:
    return (
        f"gs://cmip6/CMIP6/{ACTIVITY[experiment]}/MPI-M/{MODEL}/{experiment}/"
        f"{member}/Omon/{variable}/{GRID}/{VERSION}/"
    )


def open_store(member: str, experiment: str, variable: str) -> xr.Dataset:
    url = store_url(member, experiment, variable)
    ds = xr.open_zarr(url, consolidated=True, decode_times=TIME_CODER)
    label = ds.attrs.get("variant_label")
    if label != member or ds.attrs.get("experiment_id") != experiment:
        raise RuntimeError(f"store {url} is {label}/{ds.attrs.get('experiment_id')}")
    return ds


def time_slice(time: xr.DataArray, window: tuple[int, int]) -> slice:
    """Contiguous index range of the years in the window."""
    years = time.dt.year.values
    y0, y1 = window
    idx = np.flatnonzero((years >= y0) & (years <= y1))
    if idx.size == 0:
        raise RuntimeError("no months in the requested years")
    if idx.size != idx[-1] - idx[0] + 1:
        raise RuntimeError("requested years are not contiguous in the store")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def section_inputs(dsv, dss, dsk, window: tuple[int, int]) -> dict:
    """Section row of the three stores, in the layout of the r1 pipeline."""
    lat_name, j, lat_mean, _ = _find_lat_and_j(dsv, "vo")
    tsl = time_slice(dsv["time"], window)
    # Each zarr chunk holds all rows, so the row is read once per time chunk
    # and kept in memory (about 50 MB per variable for a 25-year window).
    v = dsv["vo"].isel(j=j, time=tsl).load()
    s = dss["so"].isel(j=j, time=tsl).load()
    lon = dsv["longitude"].isel(j=j).values.astype(float)
    lat = dsv["latitude"].isel(j=j).values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon).astype(float)
    return {
        "d": {
            "v": v.transpose("time", "lev", "i"),
            "s": s.transpose("time", "lev", "i"),
            "lon": lon,
            "lat": lat,
            "depth": dsv["lev"].values.astype(float),
            "years": v["time"].dt.year.values,
            "months": v["time"].dt.month.values,
        },
        "j": int(j),
        "lat_mean": float(lat_mean),
        "tsl": tsl,
        "thk": dsk["thkcello"].isel(j=j, time=tsl.start),
    }


def thickness_check(thk_da, depth: np.ndarray, s_first: np.ndarray) -> float:
    """Median ratio of the model thkcello to the bounds thickness, wet cells."""
    e3 = pipeline.bounds_thickness(depth)
    thk = thk_da.transpose("lev", "i").values.astype(float)
    bounds = np.broadcast_to(e3[:, None], thk.shape)
    wet = np.isfinite(s_first) & np.isfinite(thk) & (thk > 0)
    return float(np.median(thk[wet] / bounds[wet]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--member", required=True)
    parser.add_argument("--experiment", required=True, choices=sorted(YEARS))
    parser.add_argument("--start", type=int, help="first year (default: experiment)")
    parser.add_argument("--end", type=int, help="last year (default: experiment)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    default = YEARS[args.experiment]
    window = (args.start or default[0], args.end or default[1])
    tag = "" if window == default else f"_{window[0]}-{window[1]}"
    out = OUT_DIR / f"{args.member}_{args.experiment}{tag}.nc"
    if out.exists() and not args.overwrite:
        print(f"exists, skipping: {out}")
        return
    dask.config.set(scheduler="threads", num_workers=args.threads)
    t0 = time.time()

    dsv = open_store(args.member, args.experiment, "vo")
    dss = open_store(args.member, args.experiment, "so")
    dsk = open_store(args.member, args.experiment, "thkcello")
    print(f"opened stores in {time.time() - t0:.0f} s", flush=True)

    inp = section_inputs(dsv, dss, dsk, window)
    d = inp["d"]
    print(
        f"row j={inp['j']} mean lat {inp['lat_mean']:.3f}, "
        f"{d['v'].sizes['time']} months",
        flush=True,
    )

    s_first = d["s"].isel(time=0).values.astype(float)
    s_first = np.where((s_first > 0) & (s_first < 100), s_first, np.nan)
    thk_ratio = thickness_check(inp["thk"], d["depth"], s_first)

    df, depth_sorted, meta = pipeline.monthly_series(d)
    print(f"monthly series done in {time.time() - t0:.0f} s", flush=True)
    ann = pipeline.annual(df, depth_sorted)
    ann = ann.loc[(ann.index >= window[0]) & (ann.index <= window[1])]

    ds_out = ann.to_xarray()
    ds_out.attrs.update(
        {
            "member": args.member,
            "experiment": args.experiment,
            "window": list(window),
            "model": MODEL,
            "section_j_index": inp["j"],
            "section_mean_latitude": inp["lat_mean"],
            "n_points": meta["n_points"],
            "lon_range": meta["lon_range"],
            "lat_range": meta["lat_range"],
            "thkcello_to_bounds_thickness_median_ratio": thk_ratio,
            "source": store_url(args.member, args.experiment, "vo"),
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out)
    print(
        f"wrote {out} ({ds_out.sizes['year']} years) in {time.time() - t0:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
