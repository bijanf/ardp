#!/usr/bin/env python3
"""Archive the 34.5 S section profiles, and factorise F_ovS without a residual.

The previous factorisation, F_ovS = -Psi DvS / S0 with DvS defined as
-S0 F_ovS / Psi, is not a decomposition at all: the second factor is the first
one divided out, so it absorbs every change in the vertical structure of the
velocity as well as every change in salinity, and the "identity" closes to
machine precision by construction. Referees are right to reject an attribution
built on it.

There is an exact factorisation into two independently measured quantities.
Because the barotropic correction forces

    int V_bc(z) dz = 0,

the northward and southward limbs carry equal and opposite volume transport T.
Splitting the depth integral at the sign changes of V_bc and writing S_N and
S_S for the transport-weighted mean salinity of the northward and southward
limbs,

    F_ovS = -(1/S0) * T * (S_N - S_S)          exactly, with no residual,

where

    T   = int_{V_bc > 0} V_bc(z) dz            the overturning exchange transport
    S_N = int_{V_bc > 0} V_bc S_bar dz / T     transport-weighted upper-limb salinity
    S_S = int_{V_bc < 0} (-V_bc) S_bar dz / T  transport-weighted lower-limb salinity

Neither factor is defined in terms of the other, neither is a residual, T is
linear in the velocity field (no maximum operator and its upward sampling
bias), and S_N - S_S is a genuine transport-weighted salinity contrast that can
be plotted and compared against observations. The reference salinity cancels
from the limb split because int V_bc dz = 0.

This script writes, for ORAS5 and GLORYS12V1 at the 34.5 S section, monthly
values of every quantity needed downstream:

* the profiles themselves, V_int (raw), V_bc, A_xy and S_bar, so that no later
  analysis has to re-read the reanalysis archives;
* the exact factorisation T, S_N, S_S and their product;
* F_ov with and without the barotropic correction, and the net section
  transport, so the size of that correction is on the record;
* the azonal (gyre and eddy) freshwater transport F_az, which the overturning
  diagnostic excludes by construction;
* three definitions of the overturning strength, so the attribution can be
  shown not to depend on which one is used: the depth-space streamfunction
  maximum from the surface, the same with the integration restarted at 250 m
  (which removes the wind-driven surface cell), and the exchange transport T;
* the surface-layer (0 to 100 m) contribution to F_ov, where velocity and
  salinity covary most strongly.

Writes ``data/results/paper3v2_section_<product>.nc``.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS = REPO / "data" / "results"
S0 = 35.0
TARGET_LAT = -34.5
ANCHOR_LON = -25.0
SURFACE_LAYER_M = 100.0


def atlantic_segment(lon_row: np.ndarray, lat_row: np.ndarray, ocean: np.ndarray):
    """Contiguous ocean run containing ANCHOR_LON, in order of longitude."""
    order = np.argsort(lon_row)
    lon_s, ocean_s = lon_row[order], ocean[order]
    anchor = int(np.abs(lon_s - ANCHOR_LON).argmin())
    if not ocean_s[anchor]:
        raise RuntimeError("anchor longitude is not ocean at this row")
    lo = anchor
    while lo - 1 >= 0 and ocean_s[lo - 1]:
        lo -= 1
    hi = anchor
    while hi + 1 < len(lon_s) and ocean_s[hi + 1]:
        hi += 1
    sel = order[lo : hi + 1]
    lon_sel = lon_row[sel]
    dlon = np.diff(lon_sel)
    dlon = np.append(dlon, dlon[-1] if len(dlon) else 0.25)
    e1t = np.clip(np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(lat_row[sel])), 1.0, None)
    return sel, e1t


def section_quantities(
    v_sec: np.ndarray, s_sec: np.ndarray, e1t: np.ndarray, e3t: np.ndarray
) -> dict:
    """All scalar and profile diagnostics for one section snapshot."""
    nz = v_sec.shape[0]
    v_int = np.zeros(nz)
    a_xy = np.zeros(nz)
    s_bar = np.zeros(nz)
    f_az = 0.0
    for k in range(nz):
        ocean = ~np.isnan(s_sec[k]) & ~np.isnan(v_sec[k])
        if not ocean.any():
            continue
        w = np.where(ocean, e1t, 0.0)
        wsum = w.sum()
        if wsum <= 0:
            continue
        v_k = np.nan_to_num(v_sec[k], nan=0.0)
        s_k = np.nan_to_num(s_sec[k], nan=0.0)
        v_int[k] = float((v_k * w).sum())
        a_xy[k] = float(wsum)
        s_bar[k] = float((s_k * w).sum() / wsum)
        # Azonal component: covariance of the departures from the zonal mean.
        vp = np.where(ocean, v_k - v_int[k] / wsum, 0.0)
        sp = np.where(ocean, s_k - s_bar[k], 0.0)
        f_az += float((vp * sp * w).sum()) * e3t[k]
    f_az = -(1.0 / S0) * f_az / 1e6

    v_net = float((v_int * e3t).sum())
    a_tot = float((a_xy * e3t).sum())
    v_bar = v_net / a_tot if a_tot > 0 else 0.0
    v_bc = v_int - v_bar * a_xy

    f_ov = -(1.0 / S0) * float((v_bc * (s_bar - S0) * e3t).sum()) / 1e6
    f_ov_raw = -(1.0 / S0) * float((v_int * (s_bar - S0) * e3t).sum()) / 1e6

    # Exact limb factorisation. T is the northward-limb transport; because the
    # barotropic correction makes the column integral vanish, the southward
    # limb carries exactly -T, and F_ov = -(1/S0) T (S_N - S_S) with no residual.
    trans = v_bc * e3t
    north = trans > 0
    t_north = float(trans[north].sum())
    t_south = float(-trans[~north].sum())
    if t_north > 0:
        s_n = float((trans[north] * s_bar[north]).sum() / t_north)
        s_s = float((-trans[~north] * s_bar[~north]).sum() / t_south)
    else:
        s_n = s_s = np.nan

    # Overturning strength, three definitions.
    psi = np.cumsum(v_int * e3t) / 1e6
    psi_max_surface = float(np.nanmax(psi))
    depth_edges = np.cumsum(e3t)
    below = depth_edges >= 250.0
    if below.any():
        psi250 = np.cumsum(np.where(below, v_int * e3t, 0.0)) / 1e6
        psi_max_250 = float(np.nanmax(psi250))
    else:
        psi_max_250 = np.nan

    surf = depth_edges <= SURFACE_LAYER_M
    f_ov_surface = (
        -(1.0 / S0) * float((v_bc[surf] * (s_bar[surf] - S0) * e3t[surf]).sum()) / 1e6
    )

    return {
        "V_int": v_int,
        "V_bc": v_bc,
        "A_xy": a_xy,
        "S_bar": s_bar,
        "F_ov": f_ov,
        "F_ov_raw": f_ov_raw,
        "F_az": f_az,
        "V_net_Sv": v_net / 1e6,
        "T_limb_Sv": t_north / 1e6,
        "S_north": s_n,
        "S_south": s_s,
        "dS_limb": s_n - s_s,
        "psi_max_surface_Sv": psi_max_surface,
        "psi_max_250m_Sv": psi_max_250,
        "F_ov_surface_layer": f_ov_surface,
    }


# ── GLORYS12V1 ────────────────────────────────────────────────────────


def glorys_year(path: Path) -> list[dict]:
    ds = xr.open_dataset(path)
    lat, lon = ds["latitude"].values, ds["longitude"].values
    depth = ds["depth"].values
    e3t = np.diff(depth, prepend=0.0)
    j = int(np.abs(lat - TARGET_LAT).argmin())
    sub = ds[["vo", "so"]].isel(latitude=j).load()
    times = sub["time"].values
    ds.close()

    lat_row = np.full(lon.shape, float(lat[j]))
    ocean = ~np.isnan(sub["so"].isel(time=0, depth=0).values)
    sel, e1t = atlantic_segment(lon, lat_row, ocean)

    out = []
    for t in range(len(times)):
        q = section_quantities(
            sub["vo"].isel(time=t).values[:, sel],
            sub["so"].isel(time=t).values[:, sel],
            e1t,
            e3t,
        )
        q["time"] = pd.Timestamp(times[t])
        q["lat"] = float(lat[j])
        q["n_points"] = len(sel)
        out.append(q)
    return out


# ── ECCO-V4r4 ─────────────────────────────────────────────────────────
#
# ECCO is distributed as monthly files, so it can be treated exactly like the
# eddy-permitting products. The published version of this analysis used annual
# means, which is not equivalent: F_ov is bilinear in velocity and salinity, so
# forming it from annual means discards the within-year covariance. Computing
# ECCO both ways measures that bias directly.


def ecco_month(v_path: Path, s_path: Path) -> dict:
    dv = xr.open_dataset(v_path)
    dsal = xr.open_dataset(s_path)
    lat = dsal["latitude"].values
    lon = dsal["longitude"].values
    zb = dsal["Z_bnds"].values
    e3t = np.abs(zb[:, 1] - zb[:, 0])
    j = int(np.abs(lat - TARGET_LAT).argmin())

    s = dsal["SALT"].isel(latitude=j).values[0]
    v = dv["NVEL"].isel(latitude=j).values[0]
    time = pd.Timestamp(dsal["time"].values[0])
    dv.close()
    dsal.close()
    s = np.where(s == 0.0, np.nan, s)

    lat_row = np.full(lon.shape, float(lat[j]))
    sel, e1t = atlantic_segment(lon, lat_row, ~np.isnan(s[0]))
    q = section_quantities(v[:, sel], s[:, sel], e1t, e3t)
    q["time"] = time
    q["lat"] = float(lat[j])
    q["n_points"] = len(sel)
    return q


def run_ecco(workers: int) -> tuple[list[dict], np.ndarray]:
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
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(ecco_month, vel[t], sal[t]) for t in tags]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows, depth


# ── ORAS5 ─────────────────────────────────────────────────────────────


def _yyyymm(p: Path) -> str:
    m = re.search(r"_3D_(\d{6})_", p.name)
    if not m:
        raise ValueError(p.name)
    return m.group(1)


def oras5_month(v_path: Path, s_path: Path, tag: str, j: int) -> dict:
    dv = xr.open_dataset(v_path)
    dsal = xr.open_dataset(s_path)
    e3t = np.diff(dv["depthv"].values, prepend=0.0)
    lon_row = dv["nav_lon"].values[j]
    lat_row = dv["nav_lat"].values[j]
    v = dv["vomecrty"].isel(y=j).values[0]
    s = dsal["vosaline"].isel(y=j).values[0]
    dv.close()
    dsal.close()
    s = np.where(s == 0.0, np.nan, s)
    sel, e1t = atlantic_segment(lon_row, lat_row, ~np.isnan(s[0]))
    q = section_quantities(v[:, sel], s[:, sel], e1t, e3t)
    q["time"] = pd.Timestamp(f"{tag[:4]}-{tag[4:]}-15")
    q["lat"] = float(np.nanmean(lat_row))
    q["n_points"] = len(sel)
    return q


# ── assembly ──────────────────────────────────────────────────────────

PROFILE_VARS = ("V_int", "V_bc", "A_xy", "S_bar")


def to_dataset(rows: list[dict], product: str, depth: np.ndarray) -> xr.Dataset:
    rows = sorted(rows, key=lambda r: r["time"])
    times = pd.DatetimeIndex([r["time"] for r in rows])
    scalars = [
        k for k in rows[0] if k not in PROFILE_VARS + ("time", "lat", "n_points")
    ]
    ds = xr.Dataset(
        {k: ("time", np.array([r[k] for r in rows], dtype=float)) for k in scalars},
        coords={"time": times, "depth": depth},
    )
    for k in PROFILE_VARS:
        ds[k] = (("time", "depth"), np.array([r[k] for r in rows], dtype=float))
    ds.attrs.update(
        product=product,
        section_latitude=rows[0]["lat"],
        n_atlantic_points=rows[0]["n_points"],
        reference_salinity_PSU=S0,
        factorisation="F_ov = -(1/S0) * T_limb * dS_limb, exact, no residual",
        note=(
            "T_limb is the northward-limb transport of the barotropic-corrected "
            "velocity; dS_limb = S_north - S_south are transport-weighted limb "
            "salinities. Both are computed directly from the profile; neither is "
            "defined as a residual of the other."
        ),
    )
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="glorys12,oras5")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    wanted = set(args.products.split(","))

    if "glorys12" in wanted:
        files = sorted((REPO / "data" / "glorys12").glob("glorys12_*.nc"))
        with xr.open_dataset(files[0]) as d0:
            depth = d0["depth"].values
        rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(glorys_year, f) for f in files]):
                rows.extend(fut.result())
        ds = to_dataset(rows, "GLORYS12V1", depth)
        out = RESULTS / "paper3v2_section_glorys12.nc"
        ds.to_netcdf(out)
        print(f"wrote {out}  ({ds.sizes['time']} months)")

    if "ecco" in wanted:
        rows, depth = run_ecco(args.workers)
        ds = to_dataset(rows, "ECCO-V4r4", depth)
        out = RESULTS / "paper3v2_section_ecco.nc"
        ds.to_netcdf(out)
        print(f"wrote {out}  ({ds.sizes['time']} months)")

    if "oras5" in wanted:
        d = REPO / "data" / "oras5"
        vf = {_yyyymm(f): f for f in sorted(d.glob("vomecrty_*_3D_*.nc"))}
        sf = {_yyyymm(f): f for f in sorted(d.glob("vosaline_*_3D_*.nc"))}
        tags = sorted(set(vf) & set(sf))
        with xr.open_dataset(vf[tags[0]]) as d0:
            depth = d0["depthv"].values
            lat_1d = np.nanmean(d0["nav_lat"].values, axis=1)
        j = int(np.abs(lat_1d - TARGET_LAT).argmin())
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(oras5_month, vf[t], sf[t], t, j) for t in tags]
            for fut in as_completed(futs):
                rows.append(fut.result())
        ds = to_dataset(rows, "ORAS5", depth)
        out = RESULTS / "paper3v2_section_oras5.nc"
        ds.to_netcdf(out)
        print(f"wrote {out}  ({ds.sizes['time']} months)")


if __name__ == "__main__":
    main()
