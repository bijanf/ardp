#!/usr/bin/env python3
"""Vertical structure and magnitude of the section salinity trends: reanalyses
against the EN4 objective analysis.

For EN4.2.2 (g10) on its 34 S row and for each reanalysis on its own 34.5 S
section, the width-weighted zonal-mean salinity S(z) is formed monthly,
averaged to annual means and fitted with a least-squares trend at every level
over the years EN4 covers (2005 to 2024). The reanalysis trend profiles are
interpolated onto the EN4 levels and compared with the EN4 profile by layer
and by pattern correlation, and the EN4 salinity observation weight is averaged
along the section at each level, so that the comparison can be restricted to
the levels where EN4 is constrained by profiles rather than relaxed towards
climatology.

Writes ``PAPER_3_v2/analysis/r1_salinity_profiles.json`` and
``data/results/paper3r1_salinity_trend_profiles.nc``.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from analysis_paper3v2_en4 import EN4_DIR, TARGET_LAT, atlantic_segment  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_salinity_profiles.json"
WINDOW = (2005, 2024)
LAYERS = [(0, 100), (100, 500), (500, 1300), (1300, 2000), (2000, 4000), (4000, 6500)]
PRODUCTS = {"oras5": "ORAS5", "glorys12": "GLORYS12V1"}


def en4_profile(path: Path) -> dict | None:
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        depth = ds["depth"].values
        bnds = ds["depth_bnds"].values
        j = int(np.abs(lat - TARGET_LAT).argmin())
        sal = ds["salinity"].isel(time=0, lat=j).values
        wobs = ds["salinity_observation_weights"].isel(time=0, lat=j).values
        time = pd.Timestamp(ds["time"].values[0])
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    ocean = np.isfinite(sal[0])
    sel, e1t = atlantic_segment(lon180, np.full(lon.shape, float(lat[j])), ocean)
    s = sal[:, sel]
    w = np.where(np.isfinite(s), e1t, 0.0)
    wsum = w.sum(axis=1)
    sbar = np.where(
        wsum > 0,
        (np.nan_to_num(s) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1),
        np.nan,
    )
    ow = np.where(
        wsum > 0,
        (np.nan_to_num(wobs[:, sel]) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1),
        np.nan,
    )
    return {
        "time": time,
        "S_bar": sbar,
        "obs_weight": ow,
        "depth": depth,
        "e3t": np.abs(bnds[:, 1] - bnds[:, 0]),
    }


def trend_profile(years: np.ndarray, prof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = prof.shape[1]
    tr, pv = np.full(nz, np.nan), np.full(nz, np.nan)
    for k in range(nz):
        y = prof[:, k]
        if np.isfinite(y).all() and np.ptp(y) > 0:
            fit = ols_santer(years.astype(float), y)
            tr[k], pv[k] = fit["slope"] * 10.0, fit["p_santer"]
    return tr, pv


def layer_mean(
    depth: np.ndarray, e3t: np.ndarray, prof: np.ndarray, z0: float, z1: float
) -> float:
    m = (depth >= z0) & (depth < z1) & np.isfinite(prof)
    if not m.any():
        return float("nan")
    return float((prof[m] * e3t[m]).sum() / e3t[m].sum())


def main() -> None:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = [r for r in ex.map(en4_profile, files, chunksize=8) if r is not None]
    rows.sort(key=lambda r: r["time"])
    depth, e3t = rows[0]["depth"], rows[0]["e3t"]
    t = pd.DatetimeIndex([r["time"] for r in rows])
    en = xr.Dataset(
        {
            "S_bar": (("time", "depth"), np.array([r["S_bar"] for r in rows])),
            "obs_weight": (
                ("time", "depth"),
                np.array([r["obs_weight"] for r in rows]),
            ),
        },
        coords={"time": t, "depth": depth},
    )
    ea = en.groupby("time.year").mean()
    yrs = ea["year"].values
    m = (yrs >= WINDOW[0]) & (yrs <= WINDOW[1])
    en_tr, en_p = trend_profile(yrs[m], ea["S_bar"].values[m])
    obs_w = en["obs_weight"].mean("time").values

    out: dict = {
        "window": list(WINDOW),
        "en4_row_latitude": float(TARGET_LAT),
        "levels": depth.tolist(),
        "en4": {
            "trend_PSU_per_decade": en_tr.tolist(),
            "p": en_p.tolist(),
            "obs_weight_mean": obs_w.tolist(),
            "mean_profile": ea["S_bar"].values[m].mean(axis=0).tolist(),
        },
        "layers": {},
    }
    profiles = {"en4": en_tr}
    means = {"en4": ea["S_bar"].values[m].mean(axis=0)}
    pvals = {"en4": en_p}
    for key, label in PRODUCTS.items():
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
        a = ds["S_bar"].groupby("time.year").mean()
        yy = a["year"].values
        mm = (yy >= WINDOW[0]) & (yy <= WINDOW[1])
        tr, pv = trend_profile(yy[mm], a.values[mm])
        zr = ds["depth"].values
        ok = np.isfinite(tr)
        tr_i = np.interp(depth, zr[ok], tr[ok], left=np.nan, right=np.nan)
        mean_i = np.interp(
            depth, zr, a.values[mm].mean(axis=0), left=np.nan, right=np.nan
        )
        profiles[key], means[key] = tr_i, mean_i
        pvals[key] = np.interp(depth, zr[ok], pv[ok], left=np.nan, right=np.nan)
        out[key] = {
            "label": label,
            "trend_PSU_per_decade_native": tr.tolist(),
            "p_native": pv.tolist(),
            "depth_native": zr.tolist(),
            "trend_on_en4_levels": tr_i.tolist(),
            "mean_profile_on_en4_levels": mean_i.tolist(),
        }
        for name, zmax in (("upper_2000m", 2000.0), ("full_depth", 6500.0)):
            sel = (depth <= zmax) & np.isfinite(tr_i) & np.isfinite(en_tr)
            out[key][f"pattern_corr_{name}"] = float(
                np.corrcoef(tr_i[sel], en_tr[sel])[0, 1]
            )
            out[key][f"rms_diff_{name}"] = float(
                np.sqrt(np.mean((tr_i[sel] - en_tr[sel]) ** 2))
            )
            out[key][f"mean_state_rms_diff_{name}"] = float(
                np.sqrt(np.nanmean((mean_i[sel] - means["en4"][sel]) ** 2))
            )
    for z0, z1 in LAYERS:
        name = f"{z0}-{z1} m"
        out["layers"][name] = {
            "obs_weight_en4": layer_mean(depth, e3t, obs_w, z0, z1),
            **{k: layer_mean(depth, e3t, v, z0, z1) for k, v in profiles.items()},
        }
    OUT.write_text(json.dumps(out, indent=1))
    xr.Dataset(
        {f"trend_{k}": ("depth", v) for k, v in profiles.items()}
        | {f"p_{k}": ("depth", v) for k, v in pvals.items()}
        | {f"mean_{k}": ("depth", v) for k, v in means.items()}
        | {"obs_weight_en4": ("depth", obs_w)},
        coords={"depth": depth},
        attrs={"window": f"{WINDOW[0]}-{WINDOW[1]}", "units": "PSU per decade"},
    ).to_netcdf(RESULTS / "paper3r1_salinity_trend_profiles.nc")
    for k in PRODUCTS:
        print(
            k,
            {
                kk: round(vv, 3)
                for kk, vv in out[k].items()
                if kk.startswith(("pattern", "rms", "mean_state"))
            },
        )
    for name, d in out["layers"].items():
        print(f"{name:12s}", {k: round(v, 4) for k, v in d.items()})


if __name__ == "__main__":
    main()
