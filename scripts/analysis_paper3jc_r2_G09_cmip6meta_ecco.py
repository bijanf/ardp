#!/usr/bin/env python3
"""ECCO-V4r4 limb quantities on the 34.75 S and 34.25 S grid rows (round-2 group G09).

Both ECCO-V4r4 0.5 degree rows that bracket 34.5 S are extracted in one pass
over the monthly velocity and salinity files of 1992 to 2017, with the same
contiguous Atlantic run, level thicknesses (ECCO Z_bnds) and limb
factorisation as ``scripts/extract_paper3r1_section_fields.py`` and
``scripts/compute_paper3r1_section_quantities.py``. The 34.75 S row is the row
used in the paper (``data/results/paper3r1_section_ecco.nc``); the 34.25 S row
is the equally distant alternative named in Text S1. For each row the script
gives the annual means of F_ovS, T, dS, F_az and the net section transport
removed, and summarises 1992 to 2017 with the record mean, the linear trend and
their intervals (``analysis_paper3jc_r2_G03_headline_construction.summarise``,
which uses ``paper3r1_stats`` with 10,000 sieve resamples and seed 20261104).

Validation: the recomputed 34.75 S annual F_ovS is compared with the stored
section file, and the maximum absolute difference is written to the JSON.

Inputs: data/ecco/vel/**, data/ecco/sal/** (monthly mean files, 0.5 degree),
data/results/paper3r1_section_ecco.nc (validation only).
Output: PAPER_3_v2/analysis/jc_r2_G09_cmip6meta_ecco.json.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3jc_r2_G03_headline_construction import summarise  # noqa: E402
from compute_paper3r1_section_quantities import section_series  # noqa: E402
from compute_paper3v2_section_profiles import atlantic_segment  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G09_cmip6meta_ecco.json"
ROWS = {"34.75S": -34.75, "34.25S": -34.25}
WINDOW = (1992, 2017)
SEED = 20261104
N_BOOT = 10_000
VARS = ("F_ov", "F_az", "T_limb_Sv", "dS_limb", "V_net_Sv")


def tag_of(p: Path) -> str:
    return p.name.split("_mon_mean_")[1][:7]


def month_rows(args):
    """Both latitude rows of one month: sections of v and S, thickness and lon."""
    v_path, s_path = args
    dv = xr.open_dataset(v_path)
    dsal = xr.open_dataset(s_path)
    lat = dsal["latitude"].values
    lon = dsal["longitude"].values
    out = {}
    for name, target in ROWS.items():
        j = int(np.abs(lat - target).argmin())
        s = dsal["SALT"].isel(latitude=j).values[0]
        v = dv["NVEL"].isel(latitude=j).values[0]
        s = np.where(s == 0.0, np.nan, s)
        sel, e1t = atlantic_segment(lon, np.full(lon.shape, float(lat[j])), ~np.isnan(s[0]))
        out[name] = {
            "lat": float(lat[j]),
            "v": v[:, sel],
            "s": s[:, sel],
            "e1t": e1t,
            "lon": lon[sel],
        }
    dv.close()
    dsal.close()
    return out


def annual_block(monthly: list[dict], e3t: np.ndarray, n_points: int) -> dict:
    """Monthly section diagnostics and their calendar-year means."""
    rows = []
    for m in monthly:
        out = section_series(
            m["v"][None],
            m["s"][None],
            m["e1t"],
            e3t,
            np.full(n_points, -1),
        )
        rows.append(
            {
                "year": m["year"],
                **{k: float(out[k][0]) for k in ("F_ov", "T_limb_Sv", "dS_limb", "F_az", "V_net_Sv")},
            }
        )
    df = pd.DataFrame(rows)
    ann = df.groupby("year").mean()
    return {
        "years": ann.index.values.astype(int),
        **{k: ann[k].values.astype(float) for k in VARS},
        "e3t_sum_m": float(e3t.sum()),
        "n_monthly": int(len(df)),
    }


def main() -> None:
    sal = {tag_of(p): p for p in (REPO / "data" / "ecco" / "sal").rglob("*.nc")}
    vel = {tag_of(p): p for p in (REPO / "data" / "ecco" / "vel").rglob("*.nc")}
    tags = sorted(set(sal) & set(vel))
    with xr.open_dataset(sal[tags[0]]) as d0:
        depth = np.abs(d0["Z"].values)
        zb = d0["Z_bnds"].values
    e3t = np.abs(zb[:, 1] - zb[:, 0])

    with ProcessPoolExecutor(max_workers=4) as ex:
        parts = list(ex.map(month_rows, [(vel[t], sal[t]) for t in tags], chunksize=8))
    for t, p in zip(tags, parts):
        p["year"] = int(t[:4])

    result = {"rows": {}}
    for name in ROWS:
        monthly = [
            {"year": p["year"], "v": p[name]["v"], "s": p[name]["s"], "e1t": p[name]["e1t"]}
            for p in parts
        ]
        n_points = len(parts[0][name]["lon"])
        data = annual_block(monthly, e3t, n_points)
        summ = summarise(data, WINDOW[0], WINDOW[1], None)
        result["rows"][name] = {
            "section_latitude": parts[0][name]["lat"],
            "n_points": n_points,
            "lon_range": [float(parts[0][name]["lon"].min()), float(parts[0][name]["lon"].max())],
            "depth_levels": int(len(depth)),
            "annual": {
                "years": [int(y) for y in data["years"]],
                **{k: [float(x) for x in data[k]] for k in VARS},
            },
            "summary_1992_2017": summ,
        }

    # Validation of the 34.75 S row against the stored section file.
    with xr.open_dataset(RESULTS / "paper3r1_section_ecco.nc") as ds:
        stored = ds["F_ov"].groupby("time.year").mean()
        stored_years = stored["year"].values.astype(int)
        stored_vals = stored.values.astype(float)
    mine = dict(
        zip(result["rows"]["34.75S"]["annual"]["years"], result["rows"]["34.75S"]["annual"]["F_ov"])
    )
    diffs = [abs(mine[int(y)] - v) for y, v in zip(stored_years, stored_vals) if int(y) in mine]
    result["validation_34_75S_vs_stored"] = {
        "stored_file": "data/results/paper3r1_section_ecco.nc",
        "n_years_compared": int(len(diffs)),
        "max_abs_diff_F_ov_Sv": float(max(diffs)),
    }

    payload = {
        "meta": {
            "date": "2026-11-05",
            "group": "G09_cmip6meta",
            "actions": ["A84 (ECCO 34.25 S row)"],
            "script": "scripts/analysis_paper3jc_r2_G09_cmip6meta_ecco.py",
            "inputs": [
                "data/ecco/vel/**/OCEAN_VELOCITY_mon_mean_*_ECCO_V4r4_latlon_0p50deg.nc",
                "data/ecco/sal/**/OCEAN_TEMPERATURE_SALINITY_mon_mean_*_ECCO_V4r4_latlon_0p50deg.nc",
                "data/results/paper3r1_section_ecco.nc (validation)",
            ],
            "window": list(WINDOW),
            "seed": SEED,
            "sieve_resamples": N_BOOT,
            "replicates": {"sieve": N_BOOT},
            "intervals": "95%: ci_bias_corrected (corrected N_eff test) and ci_sieve (AR(1) sieve percentiles)",
            "units": {
                "F_ovS, F_az, V_net, T_limb": "Sv",
                "dS_limb": "PSU",
                "trend": "mSv per year for F_ovS",
            },
        },
        **result,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    for name, r in result["rows"].items():
        s = r["summary_1992_2017"]
        print(name, r["n_points"], round(s["F_ovS_mean_Sv"]["value"], 4), round(s["F_ovS_trend_mSv_per_yr"]["value"], 3))
    print("validation", result["validation_34_75S_vs_stored"])


if __name__ == "__main__":
    main()
