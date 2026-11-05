#!/usr/bin/env python3
"""EN4 row sensitivity of the fixed-circulation limb salinity trend (2005 to 2024).

The observation-only test of ``analysis_paper3v2_en4.py`` is run on the 34.0 S
whole-degree row. The reanalysis sections lie at 34.5 S to 34.56 S, so the EN4
row is 0.5 degrees from them. This script repeats the test on two further rows:

* 33.0 S, a whole-degree row north of Cape Agulhas (34.83 S), on which the
  contiguity walk closes at the African coast without modification;
* 34.5 S, obtained by linear interpolation of the EN4 salinity between the 34 S
  and 35 S rows at every level. A longitude is kept only where both rows are
  ocean at the surface, and the level salinity is interpolated only where both
  rows are defined. The contiguity walk runs on this interpolated row. The raw
  35 S row is not used on its own: it lies south of Cape Agulhas and is not a
  closed Atlantic section.

The velocity structure is frozen at each reanalysis record-mean profile
(ORAS5 and GLORYS12V1), exactly as in ``analysis_paper3v2_en4.py``. For each
row and circulation the monthly limb quantities S_north, S_south, dS, T and
F_ovS are averaged to calendar years, and the annual series over 2005 to 2024
are tested with the bias-corrected lag-one trend test of ``paper3r1_stats``
(effective sample size with its floor; 95 per cent interval from the inverse of
that test). Trends are reported per decade. The 34.0 S row is recomputed with
the same code and checked against the series stored in ``en4.json``.

The script draws no random numbers. The only stochastic element of the stats
module is its AR(1) bias table, which uses its own fixed seed.

Inputs:  data/en4/EN.4.2.2.f.analysis.g10.*.nc (240 monthly files);
         data/results/paper3r1_section_{oras5,glorys12}.nc (record-mean V_bc);
         PAPER_3_v2/analysis/en4.json (validation only).
Output:  PAPER_3_v2/analysis/jc_r2_G06_en4rows_rowsens.json
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_en4 import (  # noqa: E402
    ANCHOR_LON,
    EN4_DIR,
    atlantic_segment,
    frozen_velocity,
    limb_quantities,
)
from paper3r1_stats import trend_test  # noqa: E402

OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G06_en4rows_rowsens.json"
EN4_JSON = REPO / "PAPER_3_v2" / "analysis" / "en4.json"
WINDOW = (2005, 2024)
CIRCS = {"ORAS5": "paper3r1_section_oras5.nc", "GLORYS12V1": "paper3r1_section_glorys12.nc"}
FIELDS = ("S_north", "S_south", "dS", "T_Sv", "F_ov_Sv")

# name: (latitude used for the segment, interpolation pair or None)
ROWS = {
    "33.0": (-33.0, None),
    "34.0": (-34.0, None),
    "34.5_interpolated": (-34.5, (-34.0, -35.0)),
}


def _row_profile(lon, latv, sal_row, ocean_mask_sal):
    """Width-weighted zonal-mean salinity on an Atlantic segment of one row."""
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    lat_row = np.full(lon.shape, float(latv))
    ocean = np.isfinite(ocean_mask_sal[0])
    if not ocean.any():
        return None
    sel, e1t = atlantic_segment(lon180, lat_row, ocean)
    s = sal_row[:, sel]
    w = np.where(np.isfinite(s), e1t, 0.0)
    ws = w.sum(axis=1)
    sbar = np.where(
        ws > 0, (np.nan_to_num(s) * w).sum(axis=1) / np.where(ws > 0, ws, 1), np.nan
    )
    return {
        "S_bar": sbar,
        "n_points": int(len(sel)),
        "lon_min": float(lon180[sel].min()),
        "lon_max": float(lon180[sel].max()),
        "lat": float(latv),
    }


def month_rows(path: Path) -> dict:
    """All row profiles of one monthly EN4 file (one file open)."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values.astype(float)
        depth = ds["depth"].values.astype(float)
        bnds = ds["depth_bnds"].values
        e3t = np.abs(bnds[:, 1] - bnds[:, 0]).astype(float)
        time = pd.Timestamp(ds["time"].values[0])
        sal_full = ds["salinity"].isel(time=0).values  # (depth, lat, lon)

    def take(latv):
        j = int(np.argmin(np.abs(lat - latv)))
        if abs(float(lat[j]) - latv) > 1e-6:
            raise RuntimeError(f"row {latv} not on the EN4 grid")
        return sal_full[:, j, :]

    out = {}
    for name, (latv, pair) in ROWS.items():
        if pair is None:
            sal = take(latv)
            prof = _row_profile(lon, latv, sal, sal)
        else:
            a, b = take(pair[0]), take(pair[1])
            both = np.isfinite(a) & np.isfinite(b)
            sal = np.where(both, 0.5 * (a + b), np.nan)
            # The surface mask of the interpolated row is the intersection of the
            # two rows, so the contiguity walk sees the Cape Agulhas closure of 34 S.
            prof = _row_profile(lon, latv, sal, sal)
        if prof is not None:
            out[name] = prof
    return {"time": time, "depth": depth, "e3t": e3t, "rows": out}


def annual_limb_tests(months: list[dict], row: str, v: np.ndarray, e3t: np.ndarray):
    """Annual means of the frozen-circulation limb quantities and their trend tests."""
    times = pd.DatetimeIndex([m["time"] for m in months])
    years = times.year.values
    uniq = np.arange(WINDOW[0], WINDOW[1] + 1)
    sbar = np.array([m["rows"][row]["S_bar"] for m in months])
    monthly = [limb_quantities(v, sbar[i], e3t) for i in range(len(months))]
    ann = {}
    for field in FIELDS:
        vals = np.array(
            [m[field] if m is not None else np.nan for m in monthly], dtype=float
        )
        ann[field] = np.array(
            [np.nanmean(vals[years == y]) for y in uniq], dtype=float
        )
    res = {
        "n_valid_months": int(sum(m is not None for m in monthly)),
        "T_Sv_mean": float(np.mean(ann["T_Sv"])),
        "F_ov_Sv_mean": float(np.mean(ann["F_ov_Sv"])),
    }
    for field in ("S_north", "S_south", "dS"):
        t = trend_test(uniq.astype(float), ann[field])
        res[field] = {
            "mean": float(np.mean(ann[field])),
            "trend_per_decade": t["value"] * 10.0,
            "ci_per_decade": [t["ci"][0] * 10.0, t["ci"][1] * 10.0],
            "p": t["p"],
            "n_eff": t["n_eff"],
            "n_eff_at_floor": t["n_eff_at_floor"],
            "lag1_residual": t["lag1"],
            "n_years": int(len(uniq)),
            "annual": ann[field].tolist(),
        }
    return res


def main() -> None:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(month_rows, files, chunksize=8))
    results.sort(key=lambda r: r["time"])
    depth = results[0]["depth"]
    e3t = results[0]["e3t"]
    times = pd.DatetimeIndex([r["time"] for r in results])
    keep = [i for i, t in enumerate(times) if WINDOW[0] <= t.year <= WINDOW[1]]
    months = [results[i] for i in keep]

    rows_out: dict = {}
    for row in ROWS:
        if not all(row in m["rows"] for m in months):
            raise RuntimeError(f"row {row} missing in some months")
        first = months[0]["rows"][row]
        entry = {
            "section_latitude": first["lat"],
            "n_points": first["n_points"],
            "lon_range": [first["lon_min"], first["lon_max"]],
            "circulations": {},
        }
        for cname, fname in CIRCS.items():
            v = frozen_velocity(fname, depth, e3t)
            entry["circulations"][cname] = annual_limb_tests(months, row, v, e3t)
        rows_out[row] = entry

    # Validation: the 34.0 S row must reproduce the stored en4.json series.
    stored = json.loads(EN4_JSON.read_text())
    checks = {}
    for cname, key in (("ORAS5", "oras5"), ("GLORYS12V1", "glorys12")):
        for field in ("S_north", "S_south", "dS"):
            a = np.array(rows_out["34.0"]["circulations"][cname][field]["annual"])
            b = np.array(stored[key][field]["series"])
            checks[f"34.0_{cname}_{field}_max_abs_diff_vs_en4_json"] = float(
                np.max(np.abs(a - b))
            )
    checks["34.0_n_points_matches_en4_json"] = bool(
        rows_out["34.0"]["n_points"] == stored["n_points"]
    )
    checks["34.0_latitude_matches_en4_json"] = bool(
        abs(rows_out["34.0"]["section_latitude"] - stored["section_latitude"]) < 1e-9
    )

    dS_diff = {}
    for row in ROWS:
        for cname in CIRCS:
            dS_diff[f"{row} minus 34.0, {cname}, dS trend per decade"] = float(
                rows_out[row]["circulations"][cname]["dS"]["trend_per_decade"]
                - rows_out["34.0"]["circulations"][cname]["dS"]["trend_per_decade"]
            )

    out = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G06_en4rows_rowsens.py",
            "action_ids": ["A103"],
            "cluster": "C52",
            "inputs": [
                "data/en4/EN.4.2.2.f.analysis.g10.*.nc (EN4.2.2 g10, monthly)",
                "data/results/paper3r1_section_oras5.nc (record-mean V_bc)",
                "data/results/paper3r1_section_glorys12.nc (record-mean V_bc)",
                "PAPER_3_v2/analysis/en4.json (validation only)",
            ],
            "window_years": list(WINDOW),
            "seeds": "none drawn in this script; stats module AR(1) bias table seed 20261103",
            "replicate_counts": {
                "EN4_months_in_window": len(months),
                "years": WINDOW[1] - WINDOW[0] + 1,
            },
            "rows": {
                "33.0": "whole-degree row, 33 S",
                "34.0": "whole-degree row, 34 S (current manuscript row)",
                "34.5_interpolated": (
                    "linear interpolation of the 34 S and 35 S EN4 rows at each level; "
                    "surface ocean mask is the intersection of both rows; "
                    "contiguity walk on the interpolated row, anchor at "
                    f"{ANCHOR_LON} E"
                ),
            },
            "units": {
                "S_north, S_south, dS": "PSU (trend per decade in PSU per decade)",
                "T_Sv": "Sv",
                "F_ov_Sv": "Sv",
            },
            "intervals": "95 per cent, bias-corrected lag-one test of paper3r1_stats (trend_test)",
            "method": (
                "monthly limb quantities with the frozen record-mean V_bc of each "
                "circulation, averaged to calendar years, trend_test on 20 annual values"
            ),
        },
        "rows": rows_out,
        "dS_trend_difference_from_34.0_row": dS_diff,
        "validation": checks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for row, e in rows_out.items():
        print(f"row {row}: lat {e['section_latitude']}, n_points {e['n_points']}, lon {e['lon_range']}")
        for cname, c in e["circulations"].items():
            for f in ("S_north", "S_south", "dS"):
                d = c[f]
                print(
                    f"   {cname:10s} {f:8s} {d['trend_per_decade']:+.4f} "
                    f"[{d['ci_per_decade'][0]:+.4f}, {d['ci_per_decade'][1]:+.4f}] "
                    f"n_eff {d['n_eff']:.2f}"
                )
    print("validation", checks)


if __name__ == "__main__":
    main()
