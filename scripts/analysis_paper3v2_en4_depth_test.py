#!/usr/bin/env python3
"""Is the EN4 limb asymmetry a water-mass signal or an observing-density artefact?

The observation-only test in ``analysis_paper3v2_en4.py`` finds a salinifying
northward limb against a flat southward one. The northward limb is the upper
cell and lies inside the core-Argo layer; the southward limb runs from the base
of that cell to the abyssal crossing, so most of its transport weight sits below
2000 m, where core Argo does not sample and EN4 relaxes towards climatology. A
field relaxed towards a fixed background has both its variance and any
low-frequency signal damped, so the limb asymmetry the paper reports could in
principle be produced by the sampling asymmetry rather than by the ocean.

This script tests that directly, in four ways:

1. It integrates the EN4 salinity observation weights over each limb mask, year
   by year, so the actual observational constraint on each limb is measurable
   rather than assumed.
2. It reports the fraction of southward-limb transport weight lying below
   2000 m.
3. It repeats the whole frozen-velocity test with both limbs redefined inside
   the upper 2000 m, where both are observed. The net transport removal is
   re-applied within the truncated layer so the factorisation stays exact there,
   exactly as in the abyssal-exclusion test of the main analysis.
4. It reports trends over 2005-2017 and 2018-2024 separately as well as over the
   full record, because the full-record significance may rest on a level
   difference between the two halves rather than on a sustained trend.

Writes ``PAPER_3_v2/analysis/en4_depth_test.json``.
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
from analysis_paper3v2_en4 import (  # noqa: E402
    ANCHOR_LON,
    EN4_DIR,
    RESULTS,
    S0,
    TARGET_LAT,
    atlantic_segment,
    frozen_velocity,
)

OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
ARGO_FLOOR = 2000.0  # core-Argo profiling depth


def en4_month_full(path: Path) -> dict | None:
    """Zonal-mean salinity, observation weight and uncertainty at the section."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        depth = ds["depth"].values
        bnds = ds["depth_bnds"].values
        e3t = np.abs(bnds[:, 1] - bnds[:, 0])
        j = int(np.abs(lat - TARGET_LAT).argmin())
        sal = ds["salinity"].isel(time=0, lat=j).values
        wgt = ds["salinity_observation_weights"].isel(time=0, lat=j).values
        unc = ds["salinity_uncertainty"].isel(time=0, lat=j).values
        time = pd.Timestamp(ds["time"].values[0])
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    lat_row = np.full(lon.shape, float(lat[j]))
    ocean = np.isfinite(sal[0])
    if not ocean.any():
        return None
    sel, e1t = atlantic_segment(lon180, lat_row, ocean)

    def zonal(field):
        f = field[:, sel]
        w = np.where(np.isfinite(f), e1t, 0.0)
        ws = w.sum(axis=1)
        return np.where(ws > 0, (np.nan_to_num(f) * w).sum(axis=1) / np.where(ws > 0, ws, 1), np.nan)

    return {
        "time": time,
        "S_bar": zonal(sal),
        "W_bar": zonal(wgt),
        "U_bar": zonal(unc),
        "depth": depth,
        "e3t": e3t,
    }


def limbs(v, sbar, e3t, extra=None, zmax=None, depth=None):
    """Limb transports/salinities, optionally restricted to depth <= zmax.

    When zmax is set the profile is truncated first and the net transport
    removal re-applied within the truncated layer, so the factorisation is again
    exact there. `extra` is a dict of additional profiles to average over each
    limb with the same transport weights.
    """
    ok = np.isfinite(sbar) & np.isfinite(v)
    if zmax is not None:
        ok = ok & (depth <= zmax)
    if ok.sum() < 8:
        return None
    vv, ee, s = v[ok], e3t[ok], sbar[ok]
    vv = vv - (vv * ee).sum() / ee.sum()  # re-close within the retained layer
    trans = vv * ee
    north = trans > 0
    t_n = float(trans[north].sum())
    t_s = float(-trans[~north].sum())
    if t_n <= 0 or t_s <= 0:
        return None
    s_n = float((trans[north] * s[north]).sum() / t_n)
    s_s = float((-trans[~north] * s[~north]).sum() / t_s)
    out = {
        "T_Sv": t_n / 1e6,
        "S_north": s_n,
        "S_south": s_s,
        "dS": s_n - s_s,
        "F_ov_Sv": -(1.0 / S0) * (t_n * (s_n - s_s)) / 1e6,
    }
    if extra:
        for name, prof in extra.items():
            p = prof[ok]
            if np.isfinite(p).sum() == 0:
                continue
            out[f"{name}_north"] = float(
                (trans[north] * np.nan_to_num(p[north])).sum() / t_n
            )
            out[f"{name}_south"] = float(
                (-trans[~north] * np.nan_to_num(p[~north])).sum() / t_s
            )
    # fraction of southward-limb transport weight below the Argo floor
    if zmax is None:
        deep = depth[ok] > ARGO_FLOOR
        ts_deep = float(-trans[(~north) & deep].sum())
        tn_deep = float(trans[north & deep].sum())
        out["frac_south_below_argo"] = ts_deep / t_s if t_s > 0 else np.nan
        out["frac_north_below_argo"] = tn_deep / t_n if t_n > 0 else np.nan
    return out


def trend_block(years, values):
    y = np.asarray(years, float)
    v = np.asarray(values, float)
    m = np.isfinite(v)
    if m.sum() < 5:
        return None
    res = ols_santer(y[m], v[m])
    return {
        "trend_per_decade": float(res["slope"] * 10.0),
        "p_santer": float(res["p_santer"]),
        "n_eff": float(res["n_eff"]),
        "n": int(m.sum()),
    }


def main() -> None:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(en4_month_full, f) for f in files]):
            r = fut.result()
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["time"])
    depth = rows[0]["depth"]
    e3t = rows[0]["e3t"]
    times = pd.DatetimeIndex([r["time"] for r in rows])
    years_m = np.array([t.year for t in times])
    sbar = np.array([r["S_bar"] for r in rows])
    wbar = np.array([r["W_bar"] for r in rows])
    ubar = np.array([r["U_bar"] for r in rows])
    uniq = np.unique(years_m)

    out = {
        "purpose": (
            "Test whether the EN4 limb asymmetry is a water-mass signal or an "
            "artefact of the observing-density contrast between the upper "
            "(Argo-sampled) and deep (climatology-relaxed) limbs."
        ),
        "argo_floor_m": ARGO_FLOOR,
        "record": [int(uniq[0]), int(uniq[-1])],
        "n_months": len(rows),
        "en4_levels_total": int(depth.size),
        "en4_levels_above_floor": int((depth <= ARGO_FLOOR).sum()),
        "windows": {"full": None, "early": [2005, 2017], "late": [2018, 2024]},
    }

    for key, fname, label in (
        ("oras5", "paper3v2_section_oras5.nc", "ORAS5 velocity"),
        ("glorys12", "paper3v2_section_glorys12.nc", "GLORYS12V1 velocity"),
    ):
        v = frozen_velocity(fname, depth, e3t)
        blocks: dict[str, dict[str, list]] = {}
        for tag, zmax in (("full_column", None), ("upper_2000m", ARGO_FLOOR)):
            series: dict[str, list] = {}
            for yr in uniq:
                m = years_m == yr
                s_y = np.nanmean(sbar[m], axis=0)
                extra = {
                    "obsw": np.nanmean(wbar[m], axis=0),
                    "unc": np.nanmean(ubar[m], axis=0),
                }
                q = limbs(v, s_y, e3t, extra=extra, zmax=zmax, depth=depth)
                if q is None:
                    continue
                for k, val in q.items():
                    series.setdefault(k, []).append(val)
                series.setdefault("year", []).append(int(yr))
            blocks[tag] = series

        entry = {"label": label}
        for tag, series in blocks.items():
            yrs = np.array(series["year"], float)
            sub = {"years": series["year"]}
            for var in ("S_north", "S_south", "dS"):
                sub[var] = {"series": series[var]}
                for wname, sel in (
                    ("full", np.ones_like(yrs, bool)),
                    ("early", (yrs >= 2005) & (yrs <= 2017)),
                    ("late", yrs >= 2018),
                ):
                    t = trend_block(yrs[sel], np.array(series[var])[sel])
                    if t:
                        sub[var][wname] = t
            for var in ("T_Sv", "F_ov_Sv", "obsw_north", "obsw_south", "unc_north", "unc_south"):
                if var in series:
                    sub[var] = {
                        "mean": float(np.mean(series[var])),
                        "series": [float(x) for x in series[var]],
                    }
            if "frac_south_below_argo" in series:
                sub["frac_south_below_argo_mean"] = float(
                    np.mean(series["frac_south_below_argo"])
                )
                sub["frac_north_below_argo_mean"] = float(
                    np.mean(series["frac_north_below_argo"])
                )
            entry[tag] = sub
        out[key] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "en4_depth_test.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_DIR / 'en4_depth_test.json'}")

    for key in ("oras5", "glorys12"):
        e = out[key]
        print(f"\n=== {e['label']} ===")
        fc, up = e["full_column"], e["upper_2000m"]
        print(
            f"  southward-limb transport weight below {ARGO_FLOOR:.0f} m: "
            f"{fc['frac_south_below_argo_mean']*100:.1f}%   "
            f"(northward limb: {fc['frac_north_below_argo_mean']*100:.1f}%)"
        )
        print(f"  mean EN4 obs weight  north={fc['obsw_north']['mean']:.4f}  south={fc['obsw_south']['mean']:.4f}")
        print(f"  mean EN4 uncertainty north={fc['unc_north']['mean']:.4f}  south={fc['unc_south']['mean']:.4f}")
        for tag, blk in (("FULL COLUMN", fc), ("UPPER 2000 m", up)):
            print(f"  -- {tag} --")
            for var in ("S_north", "S_south", "dS"):
                bits = []
                for w in ("full", "early", "late"):
                    if w in blk[var]:
                        t = blk[var][w]
                        bits.append(f"{w}={t['trend_per_decade']:+.4f} (p={t['p_santer']:.3f})")
                print(f"     {var:8s} " + "  ".join(bits))


if __name__ == "__main__":
    main()
