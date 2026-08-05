#!/usr/bin/env python3
"""Is the sign of F_ovS a property of the basin or of one grid row?

The paper evaluates F_ovS at a single latitude, 34.5 S, because that is where
the SAMBA array sits and where the published diagnostic is defined. A referee is
entitled to ask whether the negative sign survives a change of section, since
F_ov is known to be sensitive to latitude. This script recomputes the whole
diagnostic on a ladder of latitudes in both eddy-permitting products and reports

* the record-mean F_ov at each latitude with a circular block-bootstrap interval,
* the trend at each latitude,
* the azonal (gyre and eddy) freshwater component F_az at each latitude, which
  the overturning diagnostic omits by construction, and
* the uncorrected F_ov, so that the size of the barotropic correction is on the
  record as a number rather than as an assertion.

Latitudes are restricted to the band in which the African coast closes the
eastern boundary. South of about 34.8 S the section passes below the Cape and
the Atlantic is no longer a closed basin at that row, so 34.5 S is the
southernmost usable section, which is itself worth stating.

The Atlantic segment at each row is found as the contiguous run of ocean points
containing 25 W, rather than by a fixed longitude box, so that no Pacific water
leaks in at the northern end of the ladder where South America is narrow.

Writes ``PAPER_3_v2/analysis/latitude_sensitivity.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

from ardp.physics.fovs import compute_fovs_from_section  # noqa: E402

OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
S0 = 35.0
ANCHOR_LON = -25.0  # mid-Atlantic, used to pick the right ocean segment
LATITUDES = (-25.0, -28.0, -30.0, -32.0, -34.5)

RNG = np.random.default_rng(20260805)
N_BOOT = 10_000
BLOCK_YEARS = 5


# ──────────────────────────────────────────────────────────────────────
# Section geometry
# ──────────────────────────────────────────────────────────────────────


def atlantic_segment(
    lon_row: np.ndarray, lat_row: np.ndarray, ocean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the contiguous ocean run containing ANCHOR_LON, and their widths.

    Returns the indices in order of increasing longitude, together with the
    zonal cell width in metres at each of them.
    """
    order = np.argsort(lon_row)
    lon_s = lon_row[order]
    ocean_s = ocean[order]

    anchor = int(np.abs(lon_s - ANCHOR_LON).argmin())
    if not ocean_s[anchor]:
        raise RuntimeError("anchor longitude is not an ocean point at this row")

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
    cos_lat = np.cos(np.deg2rad(lat_row[sel]))
    e1t = np.clip(np.abs(dlon) * 111000.0 * cos_lat, 1.0, None)
    return sel, e1t


def azonal_component(
    v_sec: np.ndarray, s_sec: np.ndarray, e1t: np.ndarray, e3t: np.ndarray
) -> float:
    """Freshwater transport carried by the departures from the zonal mean.

    F_az = -(1/S0) * int int v'(x,z) S'(x,z) dx dz, with primes the deviation
    from the width-weighted zonal mean at each depth. This is the gyre and eddy
    component that the overturning diagnostic excludes by construction.
    """
    nz = v_sec.shape[0]
    total = 0.0
    for k in range(nz):
        ocean = ~np.isnan(s_sec[k]) & ~np.isnan(v_sec[k])
        if ocean.sum() == 0:
            continue
        w = np.where(ocean, e1t, 0.0)
        wsum = w.sum()
        if wsum <= 0:
            continue
        v_k = np.nan_to_num(v_sec[k], nan=0.0)
        s_k = np.nan_to_num(s_sec[k], nan=0.0)
        v_mean = float((v_k * w).sum() / wsum)
        s_mean = float((s_k * w).sum() / wsum)
        vp = np.where(ocean, v_k - v_mean, 0.0)
        sp = np.where(ocean, s_k - s_mean, 0.0)
        total += float((vp * sp * w).sum()) * e3t[k]
    return -(1.0 / S0) * total / 1e6


# ──────────────────────────────────────────────────────────────────────
# GLORYS12V1
# ──────────────────────────────────────────────────────────────────────


def glorys_year(path: Path, lats: tuple[float, ...]) -> dict[float, list[dict]]:
    ds = xr.open_dataset(path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    depth = ds["depth"].values
    e3t = np.diff(depth, prepend=0.0)

    js = [int(np.abs(lat - t).argmin()) for t in lats]
    sub = ds[["vo", "so"]].isel(latitude=js).load()
    ds.close()

    out: dict[float, list[dict]] = {}
    for n, (target, j) in enumerate(zip(lats, js, strict=False)):
        lat_row = np.full(lon.shape, float(lat[j]))
        s0_field = sub["so"].isel(latitude=n, time=0, depth=0).values
        ocean = ~np.isnan(s0_field)
        sel, e1t = atlantic_segment(lon, lat_row, ocean)

        rows = []
        for t in range(sub.sizes["time"]):
            v = sub["vo"].isel(latitude=n, time=t).values[:, sel]
            s = sub["so"].isel(latitude=n, time=t).values[:, sel]
            f_ov, diag = compute_fovs_from_section(
                v, s, e1t, e3t, s0=S0, return_diagnostics=True
            )
            rows.append(
                {
                    "time": str(sub["time"].values[t])[:7],
                    "F_ov": f_ov,
                    "F_ov_raw": diag["F_ov_raw_Sv"],
                    "V_net_Sv": diag["V_net_Sv"],
                    "F_az": azonal_component(v, s, e1t, e3t),
                    "n_points": int(len(sel)),
                    "lat": float(lat[j]),
                    "lon_min": float(lon[sel].min()),
                    "lon_max": float(lon[sel].max()),
                }
            )
        out[target] = rows
    return out


def run_glorys(lats: tuple[float, ...], workers: int) -> dict:
    files = sorted((REPO / "data" / "glorys12").glob("glorys12_*.nc"))
    merged: dict[float, list[dict]] = {t: [] for t in lats}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(glorys_year, f, lats): f for f in files}
        for fut in as_completed(futs):
            for t, rows in fut.result().items():
                merged[t].extend(rows)
    for t in lats:
        merged[t].sort(key=lambda r: r["time"])
    return merged


# ──────────────────────────────────────────────────────────────────────
# ORAS5
# ──────────────────────────────────────────────────────────────────────


def _yyyymm(path: Path) -> str:
    m = re.search(r"_3D_(\d{6})_", path.name)
    if not m:
        raise ValueError(path.name)
    return m.group(1)


def oras5_month(
    v_path: Path, s_path: Path, tag: str, js: list[int], lats: tuple[float, ...]
) -> dict[float, dict]:
    dv = xr.open_dataset(v_path)
    dsal = xr.open_dataset(s_path)
    e3t = np.diff(dv["depthv"].values, prepend=0.0)
    lon2d = dv["nav_lon"].values
    lat2d = dv["nav_lat"].values

    v_all = dv["vomecrty"].isel(y=js).values[0]  # (z, nlat, x)
    s_all = dsal["vosaline"].isel(y=js).values[0]
    dv.close()
    dsal.close()

    out = {}
    for n, (target, j) in enumerate(zip(lats, js, strict=False)):
        lon_row = lon2d[j]
        lat_row = lat2d[j]
        v = v_all[:, n, :]
        s = s_all[:, n, :]
        s = np.where(s == 0.0, np.nan, s)
        ocean = ~np.isnan(s[0])
        sel, e1t = atlantic_segment(lon_row, lat_row, ocean)
        vv = v[:, sel]
        ss = s[:, sel]
        f_ov, diag = compute_fovs_from_section(
            vv, ss, e1t, e3t, s0=S0, return_diagnostics=True
        )
        out[target] = {
            "time": f"{tag[:4]}-{tag[4:]}",
            "F_ov": f_ov,
            "F_ov_raw": diag["F_ov_raw_Sv"],
            "V_net_Sv": diag["V_net_Sv"],
            "F_az": azonal_component(vv, ss, e1t, e3t),
            "n_points": int(len(sel)),
            "lat": float(np.nanmean(lat_row)),
            "lon_min": float(lon_row[sel].min()),
            "lon_max": float(lon_row[sel].max()),
        }
    return out


def run_oras5(lats: tuple[float, ...], workers: int, start_year: int) -> dict:
    d = REPO / "data" / "oras5"
    v_files = {_yyyymm(f): f for f in sorted(d.glob("vomecrty_*_3D_*.nc"))}
    s_files = {_yyyymm(f): f for f in sorted(d.glob("vosaline_*_3D_*.nc"))}
    tags = sorted(k for k in (set(v_files) & set(s_files)) if int(k[:4]) >= start_year)

    with xr.open_dataset(v_files[tags[0]]) as ds:
        lat_1d = np.nanmean(ds["nav_lat"].values, axis=1)
    js = [int(np.abs(lat_1d - t).argmin()) for t in lats]

    merged: dict[float, list[dict]] = {t: [] for t in lats}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(oras5_month, v_files[t], s_files[t], t, js, lats): t for t in tags
        }
        for fut in as_completed(futs):
            for t, row in fut.result().items():
                merged[t].append(row)
    for t in lats:
        merged[t].sort(key=lambda r: r["time"])
    return merged


# ──────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────


def annual_from_rows(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    years = np.array([int(r["time"][:4]) for r in rows])
    vals = np.array([r[key] for r in rows], dtype=float)
    uniq = np.unique(years)
    return uniq, np.array([vals[years == y].mean() for y in uniq])


def block_mean_ci(values: np.ndarray) -> list[float]:
    n = len(values)
    boot = np.empty(N_BOOT)
    n_blocks = int(np.ceil(n / BLOCK_YEARS))
    for i in range(N_BOOT):
        starts = RNG.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(BLOCK_YEARS)) % n for s in starts])[:n]
        boot[i] = values[idx].mean()
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def summarise(rows: list[dict]) -> dict:
    yrs, fov = annual_from_rows(rows, "F_ov")
    _, raw = annual_from_rows(rows, "F_ov_raw")
    _, faz = annual_from_rows(rows, "F_az")
    _, vnet = annual_from_rows(rows, "V_net_Sv")
    fit = ols_santer(yrs.astype(float), fov)
    ci = block_mean_ci(fov)
    return {
        "section_latitude": rows[0]["lat"],
        "lon_range": [rows[0]["lon_min"], rows[0]["lon_max"]],
        "n_points": rows[0]["n_points"],
        "record": [int(yrs[0]), int(yrs[-1])],
        "n_years": int(len(yrs)),
        "mean_F_ov_Sv": float(fov.mean()),
        "ci_F_ov_Sv": ci,
        "ci_excludes_zero": bool(ci[0] * ci[1] > 0),
        "trend_F_ov_mSv_per_yr": fit["slope"] * 1e3,
        "p_santer": fit["p_santer"],
        "n_eff": fit["n_eff"],
        "mean_F_ov_uncorrected_Sv": float(raw.mean()),
        "barotropic_correction_Sv": float(fov.mean() - raw.mean()),
        "mean_V_net_Sv": float(vnet.mean()),
        "mean_F_az_Sv": float(faz.mean()),
        "trend_F_az_mSv_per_yr": ols_santer(yrs.astype(float), faz)["slope"] * 1e3,
        "years": yrs.tolist(),
        "F_ov_Sv": fov.tolist(),
        "F_az_Sv": faz.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--oras5-start", type=int, default=1993)
    ap.add_argument("--products", default="glorys12,oras5")
    args = ap.parse_args()

    # Merge into whatever is already on disk, so the two products can be run in
    # separate invocations without the second erasing the first.
    path = OUT_DIR / "latitude_sensitivity.json"
    res: dict = json.loads(path.read_text()) if path.exists() else {}
    res["latitudes_requested"] = list(LATITUDES)
    res["anchor_lon"] = ANCHOR_LON
    wanted = set(args.products.split(","))

    if "glorys12" in wanted:
        print("GLORYS12V1 ...", flush=True)
        raw = run_glorys(LATITUDES, args.workers)
        res["glorys12"] = {f"{t}": summarise(rows) for t, rows in raw.items()}

    if "oras5" in wanted:
        print(f"ORAS5 from {args.oras5_start} ...", flush=True)
        raw = run_oras5(LATITUDES, args.workers, args.oras5_start)
        res["oras5"] = {f"{t}": summarise(rows) for t, rows in raw.items()}

    for prod in ("glorys12", "oras5"):
        if prod not in res:
            continue
        print(f"\n=== {prod} ===")
        print(
            f"{'lat':>8} {'npts':>6} {'lon range':>18} {'mean F_ov':>26} "
            f"{'trend (mSv/yr)':>20} {'F_az':>9} {'BT corr':>9}"
        )
        for s in res[prod].values():
            print(
                f"{s['section_latitude']:8.2f} {s['n_points']:6d} "
                f"{s['lon_range'][0]:8.1f}{s['lon_range'][1]:9.1f}   "
                f"{s['mean_F_ov_Sv']:+.4f} "
                f"[{s['ci_F_ov_Sv'][0]:+.4f},{s['ci_F_ov_Sv'][1]:+.4f}]"
                f"{'*' if s['ci_excludes_zero'] else ' '} "
                f"{s['trend_F_ov_mSv_per_yr']:+8.2f} p={s['p_santer']:5.3f} "
                f"{s['mean_F_az_Sv']:+8.4f} {s['barotropic_correction_Sv']:+8.4f}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "latitude_sensitivity.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
