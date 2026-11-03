#!/usr/bin/env python3
"""Archive the full longitude-depth fields of the 34.5 S section.

The zonally integrated profiles in ``data/results/paper3v2_section_<product>.nc``
are enough for the limb factorisation, but three questions need the fields
themselves:

* where along the section, and at which depths, the northward limb gains salt
  (the longitude-depth structure of the salinity trend);
* how sensitive the net-transport removal is to the representation of the
  bottom topography, which requires knowing which cells are the deepest wet
  cell of their water column;
* which water bodies a fixed longitude box admits at this latitude.

For each product this script stores, on the same contiguous Atlantic section
used everywhere else (the ocean run containing 25 W on the grid row nearest
34.5 S), the monthly meridional velocity and salinity, the zonal cell widths,
the vertical cell thicknesses used at compute time and the level of the deepest
wet cell in every column. It also stores the surface salinity along the whole
grid row so that the water bodies outside the contiguous section can be
identified.

Writes ``data/results/paper3r1_section3d_<product>.nc``.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compute_paper3v2_section_profiles import (  # noqa: E402
    TARGET_LAT,
    atlantic_segment,
)

RESULTS = REPO / "data" / "results"


def bottom_level(s_static: np.ndarray) -> np.ndarray:
    """Index of the deepest wet level in each column (-1 where the column is dry)."""
    wet = ~np.isnan(s_static)
    nz = wet.shape[0]
    idx = np.where(wet, np.arange(nz)[:, None], -1)
    return idx.max(axis=0)


def to_dataset(
    product: str,
    times: list[pd.Timestamp],
    v: np.ndarray,
    s: np.ndarray,
    depth: np.ndarray,
    e3t: np.ndarray,
    lon: np.ndarray,
    lat: float,
    e1t: np.ndarray,
    row_lon: np.ndarray,
    row_sss: np.ndarray,
) -> xr.Dataset:
    order = np.argsort(times)
    ds = xr.Dataset(
        {
            "v": (("time", "depth", "x"), v[order].astype("float32")),
            "s": (("time", "depth", "x"), s[order].astype("float32")),
            "e1t": ("x", e1t),
            "lon": ("x", lon),
            "e3t": ("depth", e3t),
            "kbot": ("x", bottom_level(s[order][0])),
            "row_lon": ("row", row_lon),
            "row_sss_mean": ("row", row_sss),
        },
        coords={"time": pd.DatetimeIndex(np.array(times)[order]), "depth": depth},
    )
    ds["v"].attrs["units"] = "m s-1"
    ds["s"].attrs["units"] = "PSU"
    ds["e1t"].attrs["units"] = "m"
    ds["e3t"].attrs.update(
        units="m", note="vertical cell thickness as used at compute time"
    )
    ds["kbot"].attrs["long_name"] = "index of the deepest wet level in each column"
    ds["row_sss_mean"].attrs["long_name"] = (
        "first-time-step surface salinity along the whole grid row"
    )
    ds.attrs.update(product=product, section_latitude=lat, n_atlantic_points=len(lon))
    return ds


# ── GLORYS12V1 ────────────────────────────────────────────────────────


def glorys_year(path: Path):
    ds = xr.open_dataset(path)
    lat, lon = ds["latitude"].values, ds["longitude"].values
    j = int(np.abs(lat - TARGET_LAT).argmin())
    sub = ds[["vo", "so"]].isel(latitude=j).load()
    ds.close()
    so0 = sub["so"].isel(time=0, depth=0).values
    sel, e1t = atlantic_segment(lon, np.full(lon.shape, float(lat[j])), ~np.isnan(so0))
    times = [pd.Timestamp(t) for t in sub["time"].values]
    return (
        times,
        sub["vo"].values[:, :, sel],
        sub["so"].values[:, :, sel],
        lon[sel],
        e1t,
        float(lat[j]),
        lon,
        so0,
    )


def run_glorys(workers: int) -> xr.Dataset:
    files = sorted((REPO / "data" / "glorys12").glob("glorys12_*.nc"))
    with xr.open_dataset(files[0]) as d0:
        depth = d0["depth"].values
    e3t = np.diff(depth, prepend=0.0)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(glorys_year, files))
    times = [t for p in parts for t in p[0]]
    v = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    p0 = parts[0]
    return to_dataset(
        "GLORYS12V1", times, v, s, depth, e3t, p0[3], p0[5], p0[4], p0[6], p0[7]
    )


# ── ORAS5 ─────────────────────────────────────────────────────────────


def _yyyymm(p: Path) -> str:
    m = re.search(r"_3D_(\d{6})_", p.name)
    if not m:
        raise ValueError(p.name)
    return m.group(1)


def oras5_month(args):
    v_path, s_path, tag, j = args
    dv = xr.open_dataset(v_path)
    dsal = xr.open_dataset(s_path)
    lon_row = dv["nav_lon"].values[j]
    lat_row = dv["nav_lat"].values[j]
    v = dv["vomecrty"].isel(y=j).values[0]
    s = dsal["vosaline"].isel(y=j).values[0]
    dv.close()
    dsal.close()
    s = np.where(s == 0.0, np.nan, s)
    sel, e1t = atlantic_segment(lon_row, lat_row, ~np.isnan(s[0]))
    return (
        pd.Timestamp(f"{tag[:4]}-{tag[4:]}-15"),
        v[:, sel],
        s[:, sel],
        lon_row[sel],
        e1t,
        float(np.nanmean(lat_row)),
        lon_row,
        s[0],
    )


def run_oras5(workers: int) -> xr.Dataset:
    d = REPO / "data" / "oras5"
    vf = {_yyyymm(f): f for f in sorted(d.glob("vomecrty_*_3D_*.nc"))}
    sf = {_yyyymm(f): f for f in sorted(d.glob("vosaline_*_3D_*.nc"))}
    tags = sorted(set(vf) & set(sf))
    with xr.open_dataset(vf[tags[0]]) as d0:
        depth = d0["depthv"].values
        lat_1d = np.nanmean(d0["nav_lat"].values, axis=1)
    j = int(np.abs(lat_1d - TARGET_LAT).argmin())
    e3t = np.diff(depth, prepend=0.0)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(
            ex.map(oras5_month, [(vf[t], sf[t], t, j) for t in tags], chunksize=8)
        )
    p0 = parts[0]
    return to_dataset(
        "ORAS5",
        [p[0] for p in parts],
        np.stack([p[1] for p in parts]),
        np.stack([p[2] for p in parts]),
        depth,
        e3t,
        p0[3],
        p0[5],
        p0[4],
        p0[6],
        p0[7],
    )


# ── ECCO-V4r4 ─────────────────────────────────────────────────────────


def ecco_month(args):
    v_path, s_path = args
    dv = xr.open_dataset(v_path)
    dsal = xr.open_dataset(s_path)
    lat = dsal["latitude"].values
    lon = dsal["longitude"].values
    j = int(np.abs(lat - TARGET_LAT).argmin())
    s = dsal["SALT"].isel(latitude=j).values[0]
    v = dv["NVEL"].isel(latitude=j).values[0]
    time = pd.Timestamp(dsal["time"].values[0])
    dv.close()
    dsal.close()
    s = np.where(s == 0.0, np.nan, s)
    sel, e1t = atlantic_segment(lon, np.full(lon.shape, float(lat[j])), ~np.isnan(s[0]))
    return time, v[:, sel], s[:, sel], lon[sel], e1t, float(lat[j]), lon, s[0]


def run_ecco(workers: int) -> xr.Dataset:
    sal = {
        p.name.split("_mon_mean_")[1][:7]: p
        for p in (REPO / "data" / "ecco" / "sal").rglob("*.nc")
    }
    vel = {
        p.name.split("_mon_mean_")[1][:7]: p
        for p in (REPO / "data" / "ecco" / "vel").rglob("*.nc")
    }
    tags = sorted(set(sal) & set(vel))
    with xr.open_dataset(sal[tags[0]]) as d0:
        depth = np.abs(d0["Z"].values)
        zb = d0["Z_bnds"].values
    e3t = np.abs(zb[:, 1] - zb[:, 0])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(ecco_month, [(vel[t], sal[t]) for t in tags], chunksize=8))
    p0 = parts[0]
    return to_dataset(
        "ECCO-V4r4",
        [p[0] for p in parts],
        np.stack([p[1] for p in parts]),
        np.stack([p[2] for p in parts]),
        depth,
        e3t,
        p0[3],
        p0[5],
        p0[4],
        p0[6],
        p0[7],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="ecco,glorys12,oras5")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    runners = {"ecco": run_ecco, "glorys12": run_glorys, "oras5": run_oras5}
    for name in args.products.split(","):
        ds = runners[name](args.workers)
        out = RESULTS / f"paper3r1_section3d_{name}.nc"
        enc = {k: {"zlib": True, "complevel": 4} for k in ("v", "s")}
        ds.to_netcdf(out, encoding=enc)
        print(f"wrote {out}  {dict(ds.sizes)}", flush=True)


if __name__ == "__main__":
    main()
