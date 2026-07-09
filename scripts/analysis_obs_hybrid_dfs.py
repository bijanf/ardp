#!/usr/bin/env python3
"""Observation-constrained hybrid dF_s: EN4/RG09 dS(z) x reanalysis V1(z).

The Argo record constrains the salinity change at 34.5S but not the
velocity field, so on its own it cannot say that the salinity term
dominates dF_ovS. This analysis closes half of that gap: it evaluates
the salinity-driven component

    dF_s^hyb = -(1/S0) * integral V1_bc(z) * dS_obs(z) dz

using the OBSERVED zonal-mean salinity change dS_obs(z) (EN4.2.2 and
Roemmich-Gilson, Argo-era windows) against each reanalysis's own
early-window baroclinic overturning profile V1_bc(z). If dF_s^hyb is
of one sign and similar magnitude for every choice of V1, then the
salinity contribution to the F_ovS change is observationally grounded,
independent of which product's salinity field one trusts.

The observed dS(z) is evaluated over each product's own decomposition
windows so that the hybrid value is directly comparable to that
product's own delta_s (ECCO's record ends in 2017, so its windows are
shorter than the 2006-2012 / 2018-2024 pair used for ORAS5/GLORYS12).

Reads:  data/en4/EN.4.2.2.f.analysis.g10.*.nc
        data/argo_rg09/RG_ArgoClim_*.nc
        data/{oras5,glorys12,ecco} via compute_fovs_decomposition helpers
Writes: revision/results/obs_hybrid_dFs.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import compute_fovs_decomposition as cfd  # noqa: E402

from ardp.constants import S0  # noqa: E402
from ardp.physics.fovs_decomposition import (  # noqa: E402
    _barotropic_correct,
    _section_profiles,
    decompose_fovs_trend,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

EN4_DIR = ROOT / "data" / "en4"
RG_DIR = ROOT / "data" / "argo_rg09"
OUT = ROOT / "revision" / "results" / "obs_hybrid_dFs.json"

LAT_BAND = (-36.0, -33.0)
LON_RANGE = (-70.0, 20.0)
EARLY = (2006, 2012)
LATE = (2018, 2024)

PRODUCTS = {
    "ORAS5": {
        "early": (2006, 2012),
        "late": (2018, 2024),
        "loader": lambda p: cfd._oras5_period_mean(ROOT / "data/oras5", p),
    },
    "GLORYS12V1": {
        "early": (2006, 2012),
        "late": (2018, 2024),
        "loader": lambda p: cfd._glorys12_period_mean(ROOT / "data/glorys12", p),
    },
    "ECCO-V4r4": {
        "early": (2006, 2011),
        "late": (2012, 2017),
        "loader": lambda p: cfd._ecco_period_mean(ROOT / "data/ecco", p),
    },
}


def _to_lon180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def en4_zonal_mean_profile(period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(depth, S_bar(z)) band/zonal mean over the 34.5S Atlantic section."""
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    files = [
        f for f in files if period[0] <= int(f.stem.split(".")[-1][:4]) <= period[1]
    ]
    if not files:
        raise RuntimeError(f"no EN4 files in {period}")
    acc = None
    n = 0
    depth = None
    for f in files:
        ds = xr.open_dataset(f)
        lat = ds["lat"].values
        lon180 = _to_lon180(ds["lon"].values)
        jj = np.where((lat >= LAT_BAND[0]) & (lat <= LAT_BAND[1]))[0]
        ii = np.where((lon180 >= LON_RANGE[0]) & (lon180 <= LON_RANGE[1]))[0]
        s = ds["salinity"].isel(time=0, lat=jj, lon=ii).values  # (z, jb, ib)
        if depth is None:
            depth = ds["depth"].values.astype(float)
        ds.close()
        if acc is None:
            acc = np.zeros_like(s, dtype=float)
            cnt = np.zeros_like(s, dtype=float)
        ok = np.isfinite(s)
        acc += np.where(ok, s, 0.0)
        cnt += ok
        n += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean3d = np.where(cnt > 0, acc / cnt, np.nan)
    prof = np.nanmean(mean3d, axis=(1, 2))  # equal-weight wet-cell mean
    log.info(f"EN4 {period}: {n} months, {np.isfinite(prof).sum()} wet levels")
    return depth, prof


def rg09_zonal_mean_profile(period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(pressure~depth, S_bar(z)) from RG09 mean + monthly anomalies."""
    core = xr.open_dataset(RG_DIR / "RG_ArgoClim_Salinity_2019.nc", decode_times=False)
    lat = core["LATITUDE"].values
    lon180 = _to_lon180(core["LONGITUDE"].values)
    jj = np.where((lat >= LAT_BAND[0]) & (lat <= LAT_BAND[1]))[0]
    ii = np.where((lon180 >= LON_RANGE[0]) & (lon180 <= LON_RANGE[1]))[0]
    pres = core["PRESSURE"].values.astype(float)

    mean = core["ARGO_SALINITY_MEAN"].isel(LATITUDE=jj, LONGITUDE=ii).values

    def _month_year(t: float) -> int:
        # t in months since 2004-01; 0.5 -> Jan 2004
        return 2004 + int(np.floor(t)) // 12

    anoms = []
    core_t = core["TIME"].values
    core_anom = core["ARGO_SALINITY_ANOMALY"].isel(LATITUDE=jj, LONGITUDE=ii).values
    for k, t in enumerate(core_t):
        if period[0] <= _month_year(float(t)) <= period[1]:
            anoms.append(core_anom[k])
    core.close()
    for f in sorted(RG_DIR.glob("RG_ArgoClim_2*.nc")):
        yyyymm = f.stem.split("_")[2]
        yr = int(yyyymm[:4])
        if period[0] <= yr <= period[1]:
            ext = xr.open_dataset(f, decode_times=False)
            anoms.append(
                ext["ARGO_SALINITY_ANOMALY"]
                .isel(TIME=0, LATITUDE=jj, LONGITUDE=ii)
                .values
            )
            ext.close()
    if not anoms:
        raise RuntimeError(f"no RG09 months in {period}")
    anom_mean = np.nanmean(np.stack(anoms), axis=0)
    s3d = mean + anom_mean  # (z, jb, ib) absolute salinity field for window
    prof = np.nanmean(s3d, axis=(1, 2))
    log.info(f"RG09 {period}: {len(anoms)} months")
    return pres, prof


_EN4_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_RG_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def en4_cached(period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if period not in _EN4_CACHE:
        _EN4_CACHE[period] = en4_zonal_mean_profile(period)
    return _EN4_CACHE[period]


def rg09_cached(period: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if period not in _RG_CACHE:
        _RG_CACHE[period] = rg09_zonal_mean_profile(period)
    return _RG_CACHE[period]


def hybrid_dfs(
    v_bc_1: np.ndarray,
    e3t: np.ndarray,
    depth_prod: np.ndarray,
    z_obs: np.ndarray,
    ds_obs: np.ndarray,
    zmax: float | None = None,
) -> float:
    """-(1/S0) * sum V1_bc(z) * dS_obs->prod(z) * e3t, in Sv."""
    ok = np.isfinite(ds_obs)
    dsz = np.interp(depth_prod, z_obs[ok], ds_obs[ok], left=ds_obs[ok][0], right=0.0)
    if zmax is not None:
        dsz = np.where(depth_prod <= zmax, dsz, 0.0)
    return float(-(1.0 / S0) * np.sum(v_bc_1 * dsz * e3t) / 1e6)


def main() -> None:
    results: dict = {
        "reference_windows": {"early": EARLY, "late": LATE},
        "lat_band": LAT_BAND,
        "lon_range": LON_RANGE,
        "obs": {},
        "products": {},
    }

    # ---- observed dS(z) over the reference (ORAS5/GLORYS12) windows ----
    z_en4, s1_en4 = en4_cached(EARLY)
    _, s2_en4 = en4_cached(LATE)
    results["obs"]["EN4"] = {
        "depth_m": z_en4.tolist(),
        "dS_PSU": (s2_en4 - s1_en4).tolist(),
        "windows": [EARLY, LATE],
        "note": "late minus early window mean, band/zonal wet-cell mean",
    }
    log.info(
        f"EN4 dS upper 300 m (ref windows): "
        f"{np.nanmean((s2_en4 - s1_en4)[z_en4 <= 300]):+.4f} PSU"
    )

    rg_ok = True
    try:
        z_rg, s1_rg = rg09_cached(EARLY)
        _, s2_rg = rg09_cached(LATE)
        results["obs"]["RG09"] = {
            "depth_m": z_rg.tolist(),
            "dS_PSU": (s2_rg - s1_rg).tolist(),
            "windows": [EARLY, LATE],
            "note": "pressure (dbar) treated as depth (m); Argo 0-2000 dbar",
        }
        log.info(
            f"RG09 dS upper 300 m (ref windows): "
            f"{np.nanmean((s2_rg - s1_rg)[z_rg <= 300]):+.4f} PSU"
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"RG09 profiles failed, continuing with EN4 only: {e}")
        rg_ok = False

    # ---- per-product V1(z) and hybrid dF_s over the product's windows ----
    for name, spec in PRODUCTS.items():
        w_early, w_late = spec["early"], spec["late"]
        log.info(f"--- {name}: loading period means {w_early} / {w_late} ...")
        v1, s1, grid = spec["loader"](w_early)
        v2, s2, _ = spec["loader"](w_late)
        res = decompose_fovs_trend(
            v1, s1, v2, s2, e1t_atl=grid["e1t_atl"], e3t=grid["e3t"], s0=S0
        )
        v_int_1, a_xy_1, _s_mean_1 = _section_profiles(v1, s1, grid["e1t_atl"])
        v_bc_1, _v_bar, _v_net = _barotropic_correct(v_int_1, a_xy_1, grid["e3t"])
        depth = np.asarray(grid["depth"], dtype=float)
        e3t = np.asarray(grid["e3t"], dtype=float)

        z_en4_w, s_en4_e = en4_cached(w_early)
        _, s_en4_l = en4_cached(w_late)
        ds_en4_w = s_en4_l - s_en4_e

        entry = {
            "windows": [w_early, w_late],
            "own_delta_s_Sv": res["delta_s"],
            "own_delta_v_Sv": res["delta_v"],
            "own_delta_total_Sv": res["delta_total"],
            "hybrid_dFs_Sv": {
                "EN4_full": hybrid_dfs(v_bc_1, e3t, depth, z_en4_w, ds_en4_w),
                "EN4_le2000m": hybrid_dfs(
                    v_bc_1, e3t, depth, z_en4_w, ds_en4_w, zmax=2000.0
                ),
            },
            "v_bc_1_m2s": v_bc_1.tolist(),
            "depth_m": depth.tolist(),
        }
        if rg_ok:
            z_rg_w, s_rg_e = rg09_cached(w_early)
            _, s_rg_l = rg09_cached(w_late)
            entry["hybrid_dFs_Sv"]["RG09"] = hybrid_dfs(
                v_bc_1, e3t, depth, z_rg_w, s_rg_l - s_rg_e
            )
        # Reference-window hybrids: same V1 against the 2006-2012 vs
        # 2018-2024 observed dS, so the velocity-profile invariance test
        # covers all three circulations on identical windows.
        entry["hybrid_dFs_ref_windows_Sv"] = {
            "EN4_le2000m": hybrid_dfs(
                v_bc_1, e3t, depth, z_en4, s2_en4 - s1_en4, zmax=2000.0
            ),
        }
        if rg_ok:
            entry["hybrid_dFs_ref_windows_Sv"]["RG09"] = hybrid_dfs(
                v_bc_1, e3t, depth, z_rg, s2_rg - s1_rg
            )
        results["products"][name] = entry
        msg = (
            f"{name:12s} own dF_s={res['delta_s'] * 1000:+7.1f} mSv | "
            f"hyb EN4(<=2000m)="
            f"{entry['hybrid_dFs_Sv']['EN4_le2000m'] * 1000:+7.1f} | "
            f"hyb EN4(full)={entry['hybrid_dFs_Sv']['EN4_full'] * 1000:+7.1f}"
        )
        if rg_ok:
            msg += f" | hyb RG09={entry['hybrid_dFs_Sv']['RG09'] * 1000:+7.1f}"
        log.info(msg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    log.info(f"saved {OUT}")


if __name__ == "__main__":
    main()
