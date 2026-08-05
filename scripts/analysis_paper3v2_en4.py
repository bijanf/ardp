#!/usr/bin/env python3
"""Does the limb salinification appear in a salinity field no model has touched?

The paper's central positive claim is that the transport-weighted salinity of
the northward limb at 34.5 S is rising. That claim is made from two
sequential-assimilation reanalyses, and the obvious alternative explanation is
that they are tracking the arrival of the Argo array rather than the ocean. The
discriminating test is to hold the circulation fixed and let only an
observation-only salinity analysis vary.

EN4.2.2 (Met Office objective analysis of profile observations, g10 corrections)
is used here. It is an optimal interpolation of profiles onto a 1 degree grid;
no ocean model integrates it forward, so a limb salinity computed from it cannot
be an assimilation increment.

The velocity structure is frozen at each reanalysis product's record-mean
barotropic-corrected profile, interpolated onto the EN4 levels and renormalised
so its depth integral still vanishes. The limbs are then exactly as in the main
analysis, and

    S_up(t) = int_{V_bc>0} V_bc S_EN4(z,t) dz / T

varies only because the observed salinity varies. Any trend in it is a trend in
the observed water masses weighted by a fixed circulation.

Writes ``PAPER_3_v2/analysis/en4.json``.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
EN4_DIR = REPO / "data" / "en4"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"

S0 = 35.0
# EN4 is on whole-degree latitudes. 34.5 S snaps to 35.0 S, which lies SOUTH of
# Cape Agulhas (34.83 S), so the basin is not closed to the east there and the
# contiguity walk runs across the entire Indian Ocean to Australia. 34.0 S is
# the nearest row on which Africa closes the section; the 35.0 S row is retained
# as a documented sensitivity.
TARGET_LAT = -34.0
# The 35.0 S row is not used; the comment above records why.
ANCHOR_LON = -25.0


def atlantic_segment(lon_row, lat_row, ocean):
    order = np.argsort(lon_row)
    lon_s, ocean_s = lon_row[order], ocean[order]
    anchor = int(np.abs(lon_s - ANCHOR_LON).argmin())
    if not ocean_s[anchor]:
        raise RuntimeError("anchor is not ocean")
    lo = anchor
    while lo - 1 >= 0 and ocean_s[lo - 1]:
        lo -= 1
    hi = anchor
    while hi + 1 < len(lon_s) and ocean_s[hi + 1]:
        hi += 1
    sel = order[lo : hi + 1]
    lon_sel = lon_row[sel]
    dlon = np.diff(lon_sel)
    dlon = np.append(dlon, dlon[-1] if len(dlon) else 1.0)
    e1t = np.clip(np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(lat_row[sel])), 1.0, None)
    return sel, e1t


def en4_month(path: Path, target_lat: float = TARGET_LAT) -> dict | None:
    """Width-weighted zonal-mean salinity profile at the section from one file."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        depth = ds["depth"].values
        # Cell thicknesses from the bounds, not from differences of the centres.
        bnds = ds["depth_bnds"].values
        e3t = np.abs(bnds[:, 1] - bnds[:, 0])
        j = int(np.abs(lat - target_lat).argmin())
        sal = ds["salinity"].isel(time=0, lat=j).values  # (depth, lon)
        time = pd.Timestamp(ds["time"].values[0])
    # EN4 longitudes run 0 to 360; shift to -180..180 for the contiguity test.
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    lat_row = np.full(lon.shape, float(lat[j]))
    ocean = np.isfinite(sal[0])
    if not ocean.any():
        return None
    sel, e1t = atlantic_segment(lon180, lat_row, ocean)
    s = sal[:, sel]
    w = np.where(np.isfinite(s), e1t, 0.0)
    wsum = w.sum(axis=1)
    sbar = np.where(
        wsum > 0,
        (np.nan_to_num(s) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1),
        np.nan,
    )
    return {
        "time": time,
        "S_bar": sbar,
        "depth": depth,
        "e3t": e3t,
        "lat": float(lat[j]),
        "n_points": int(len(sel)),
        "lon_min": float(lon180[sel].min()),
        "lon_max": float(lon180[sel].max()),
    }


def frozen_velocity(
    product_file: str, en4_depth: np.ndarray, e3_en4: np.ndarray
) -> np.ndarray:
    """Record-mean V_bc interpolated onto the EN4 levels, integral renormalised."""
    with xr.open_dataset(RESULTS / product_file) as ds:
        depth = ds["depth"].values
        vbc = ds["V_bc"].values.mean(axis=0)
    v = np.interp(en4_depth, depth, vbc, left=vbc[0], right=0.0)
    # Restore the vanishing depth integral that the limb split relies on.
    v = v - (v * e3_en4).sum() / e3_en4.sum()
    return v


def limb_quantities(v: np.ndarray, sbar: np.ndarray, e3t: np.ndarray) -> dict | None:
    """Limb transports and transport-weighted salinities on the valid levels only.

    Levels with no salinity are dropped before the sign split; zeroing them
    instead would place them in the southward limb and multiply 0 by NaN.
    """
    ok = np.isfinite(sbar) & np.isfinite(v)
    if ok.sum() < 10:
        return None
    # Renormalise over the levels actually used, not over all EN4 levels: the
    # deepest levels are empty at this section, and leaving them in would leave
    # the depth integral of the retained profile non-zero by about 1%.
    vv = v[ok]
    ee = e3t[ok]
    vv = vv - (vv * ee).sum() / ee.sum()
    trans = vv * ee
    s = sbar[ok]
    north = trans > 0
    t_n = float(trans[north].sum())
    t_s = float(-trans[~north].sum())
    if t_n <= 0 or t_s <= 0:
        return None
    s_n = float((trans[north] * s[north]).sum() / t_n)
    s_s = float((-trans[~north] * s[~north]).sum() / t_s)
    # F_ov implied by the observed salinity under this frozen circulation.
    f_ov = -(1.0 / S0) * (t_n * (s_n - s_s)) / 1e6
    return {
        "T_Sv": t_n / 1e6,
        "S_north": s_n,
        "S_south": s_s,
        "dS": s_n - s_s,
        "F_ov_Sv": f_ov,
    }


def main() -> None:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(en4_month, f) for f in files]):
            r = fut.result()
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["time"])
    depth = rows[0]["depth"]
    e3t = rows[0]["e3t"]
    times = pd.DatetimeIndex([r["time"] for r in rows])
    sbar = np.array([r["S_bar"] for r in rows])

    out: dict = {
        "source": "EN.4.2.2.f.analysis.g10 (Met Office objective analysis)",
        "n_months": len(rows),
        "record": [int(times[0].year), int(times[-1].year)],
        "section_latitude": rows[0]["lat"],
        "n_points": rows[0]["n_points"],
        "lon_range": [rows[0]["lon_min"], rows[0]["lon_max"]],
        "note": (
            "Velocity structure frozen at each reanalysis record-mean profile; "
            "only the EN4 salinity varies, so any trend is a trend in observed "
            "water masses under a fixed circulation."
        ),
    }

    years = np.array([t.year for t in times])
    uniq = np.unique(years)
    for key, fname, label in (
        ("oras5", "paper3v2_section_oras5.nc", "ORAS5 velocity"),
        ("glorys12", "paper3v2_section_glorys12.nc", "GLORYS12V1 velocity"),
    ):
        v = frozen_velocity(fname, depth, e3t)
        monthly = [limb_quantities(v, sbar[i], e3t) for i in range(len(rows))]
        good = [m is not None for m in monthly]
        ann = {}
        for field in ("S_north", "S_south", "dS", "T_Sv", "F_ov_Sv"):
            vals = np.array(
                [m[field] if m is not None else np.nan for m in monthly], dtype=float
            )
            ann[field] = np.array(
                [np.nanmean(vals[years == y]) for y in uniq], dtype=float
            )
        x = uniq.astype(float)
        entry = {
            "label": label,
            "n_years": int(len(uniq)),
            "T_Sv": float(ann["T_Sv"].mean()),
            "implied_F_ov_Sv": float(ann["F_ov_Sv"].mean()),
        }
        for field in ("S_north", "S_south", "dS"):
            fit = ols_santer(x, ann[field])
            entry[field] = {
                "mean": float(ann[field].mean()),
                "trend_per_decade": fit["slope"] * 10,
                "p_santer": fit["p_santer"],
                "n_eff": fit["n_eff"],
                "series": ann[field].tolist(),
            }
        entry["n_valid_months"] = int(sum(good))
        out[key] = entry

    # The reanalyses' own limb salinity over the identical 2005-2024 window, so
    # the comparison is like for like.
    out["reanalysis_same_window"] = {}
    for key, fname in (
        ("oras5", "paper3v2_section_oras5.nc"),
        ("glorys12", "paper3v2_section_glorys12.nc"),
    ):
        with xr.open_dataset(RESULTS / fname) as ds:
            a = ds.groupby("time.year").mean()
        yy = a["year"].values.astype(int)
        m = (yy >= uniq[0]) & (yy <= uniq[-1])
        fit = ols_santer(yy[m].astype(float), a["S_north"].values[m])
        fitd = ols_santer(yy[m].astype(float), a["dS_limb"].values[m])
        out["reanalysis_same_window"][key] = {
            "window": [int(uniq[0]), int(uniq[-1])],
            "S_north_trend_per_decade": fit["slope"] * 10,
            "S_north_p": fit["p_santer"],
            "dS_trend_per_decade": fitd["slope"] * 10,
            "dS_p": fitd["p_santer"],
        }
    out["years"] = uniq.tolist()

    print(
        f"EN4 {out['record'][0]}-{out['record'][1]}, {out['n_months']} months, "
        f"row {out['section_latitude']:.1f}, {out['n_points']} points"
    )
    for key in ("oras5", "glorys12"):
        e = out[key]
        print(
            f"\n  frozen {e['label']} (T = {e['T_Sv']:.2f} Sv, implied F_ov "
            f"{e['implied_F_ov_Sv']:+.4f} Sv)"
        )
        for f in ("S_north", "S_south", "dS"):
            d = e[f]
            print(
                f"     {f:8s} mean {d['mean']:8.4f}  trend {d['trend_per_decade']:+.4f}"
                f" /dec  p={d['p_santer']:.4f}  N_eff={d['n_eff']:.1f}"
            )
        r = out["reanalysis_same_window"][key]
        print(
            f"     reanalysis own S_north over {r['window'][0]}-{r['window'][1]}: "
            f"{r['S_north_trend_per_decade']:+.4f}/dec (p={r['S_north_p']:.4f})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "en4.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
