#!/usr/bin/env python3
"""Per-model section metadata and the Jensen term at 34.5 S (round-2 group G09).

For each of the 25 CMIP6 models of ``r1_cmip6.json`` this script re-derives the
34.5 S section exactly as ``scripts/analysis_paper3r1_cmip6.py`` does (the
contiguous Atlantic run containing 25 W, bounded by 70 W and 20 E, level
bounds reconstructed from level centres, removal of the net section transport,
limb factorisation), and adds the metadata that the per-model table did not
hold:

* lat_south, lat_north, lon_west, lon_east of the section row used, n_points;
* the grid description inferred from the stored section file (number of
  longitude points on the row and of vertical levels; the grid label is not
  stored in the section files, see the note);
* the net transport removed, V_net = sum over the section of the level
  transports q_k (monthly, averaged over the window, Sv);
* the Jensen term, J_y = mean over the months of year y of F_ovS(month)
  minus F_ovS(annual-mean fields of year y), averaged over the window (Sv), and
  as a percentage of the monthly-mean F_ovS.

Windows: 1950-1980 and 1993-2014 (historical) and the full record. The
recomputed 1993-2014 monthly-mean F_ovS is compared with ``r1_cmip6.json``
(validation, maximum absolute difference written to the JSON).

No new downloads and no new random draws; the intervals of the per-model
Jensen term are not computed (the Jensen term is a deterministic difference of
two section calculations on the same data). Cross-model summaries are medians
and ranges.

Inputs: data/cmip6_sections/{model}_{historical,ssp585}_{vo,so}.nc,
data/results/fovs_decomposition_cmip6_summary.csv (model list),
PAPER_3_v2/analysis/r1_cmip6.json (validation only).
Output: PAPER_3_v2/analysis/jc_r2_G09_cmip6meta_models.json.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3r1_cmip6 import bounds_thickness, load  # noqa: E402
from compute_paper3r1_section_quantities import section_series  # noqa: E402
from compute_paper3v2_section_profiles import atlantic_segment  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G09_cmip6meta_models.json"
REF = REPO / "PAPER_3_v2" / "analysis" / "r1_cmip6.json"
WINDOWS = {"1950-1980": (1950, 1980), "1993-2014": (1993, 2014), "record": (1850, 2100)}
EXPERIMENTS = ("historical", "ssp585")


def section_arrays(model: str, exp: str):
    """Preprocessed monthly section arrays with the selection of analysis_paper3r1_cmip6."""
    d = load(model, exp)
    if d is None:
        return None
    depth = d["depth"]
    order = np.argsort(depth)
    # Pair the velocity and salinity months by their time stamps; some files
    # hold a different number of time steps.
    tv = list(d["v"]["time"].values)
    ts = list(d["s"]["time"].values)
    pos_s = {t: i for i, t in enumerate(ts)}
    keep = [i for i, t in enumerate(tv) if t in pos_s]
    iv = np.array(keep)
    is_ = np.array([pos_s[tv[i]] for i in keep])
    common_t = [tv[i] for i in keep]
    s_full = d["s"].values[is_].astype(float)
    s0 = s_full[0]
    s0 = np.where((s0 > 0) & (s0 < 100), s0, np.nan)
    lon180 = ((d["lon"] + 180.0) % 360.0) - 180.0
    in_box = (lon180 >= -70.0) & (lon180 <= 20.0)
    sel, e1t = atlantic_segment(lon180, d["lat"], np.isfinite(s0[order[0]]) & in_box)
    e3 = bounds_thickness(depth)[order]
    v = d["v"].values[iv].astype(float)[:, :, sel][:, order, :]
    s = s_full[:, :, sel][:, order, :]
    s = np.where((s > 0) & (s < 100), s, np.nan)
    v = np.where(np.isfinite(s), v, np.nan)
    v = np.where(np.abs(v) < 1e3, v, np.nan)
    return {
        "v": v,
        "s": s,
        "e1t": e1t,
        "e3": e3,
        "years": np.array([t.year for t in common_t]).astype(int),
        "months": np.array([t.month for t in common_t]).astype(int),
        "lon": lon180[sel],
        "lat": np.asarray(d["lat"])[sel],
        "n_levels": int(len(depth)),
        "n_i": int(d["lon"].shape[0]),
    }


def model_years(model: str):
    """Monthly and annual-mean section diagnostics for the two experiments."""
    years_all = {}
    meta = {}
    for exp in EXPERIMENTS:
        a = section_arrays(model, exp)
        if a is None:
            continue
        n = len(a["e1t"])
        monthly = section_series(a["v"], a["s"], a["e1t"], a["e3"], np.full(n, -1))
        yrs = a["years"]
        uy = np.unique(yrs)
        vann, sann = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for y in uy:
                idx = np.where(yrs == y)[0]
                vann.append(np.nanmean(a["v"][idx], axis=0))
                sann.append(np.nanmean(a["s"][idx], axis=0))
        vann = np.stack(vann)
        sann = np.stack(sann)
        vann = np.where(np.isfinite(sann), vann, np.nan)
        ann = section_series(vann, sann, a["e1t"], a["e3"], np.full(n, -1))
        df = pd.DataFrame(
            {
                "year": yrs,
                "F_month": monthly["F_ov"],
                "V_net": monthly["V_net_Sv"],
            }
        )
        g = df.groupby("year").mean()
        for k, y in enumerate(uy):
            if int(y) in years_all:
                continue
            years_all[int(y)] = {
                "F_month": float(g.loc[y, "F_month"]),
                "V_net": float(g.loc[y, "V_net"]),
                "F_annual_fields": float(ann["F_ov"][k]),
                "n_months": int((yrs == y).sum()),
            }
        meta[exp] = {
            "n_points": n,
            "lon_west": float(a["lon"].min()),
            "lon_east": float(a["lon"].max()),
            "lat_south": float(a["lat"].min()),
            "lat_north": float(a["lat"].max()),
            "n_levels": a["n_levels"],
            "n_i": a["n_i"],
            "lat_varies_along_row": bool(np.ptp(a["lat"]) > 1e-6),
        }
    if not years_all:
        return {"model": model, "error": "no section files"}
    return {"model": model, "meta": meta, "years": years_all}


def window_block(years_all: dict, y0: int, y1: int):
    ys = [y for y in years_all if y0 <= y <= y1]
    if len(ys) < 0.8 * (y1 - y0 + 1):
        return None
    fm = np.array([years_all[y]["F_month"] for y in ys])
    fa = np.array([years_all[y]["F_annual_fields"] for y in ys])
    vn = np.array([years_all[y]["V_net"] for y in ys])
    j = fm - fa
    return {
        "n_years": len(ys),
        "F_ovS_monthly_mean_Sv": float(fm.mean()),
        "F_ovS_annual_fields_mean_Sv": float(fa.mean()),
        "jensen_term_Sv": float(j.mean()),
        "jensen_term_pct_of_F_ovS": float(100.0 * j.mean() / fm.mean()) if fm.mean() != 0 else None,
        "jensen_term_abs_mean_Sv": float(np.abs(j).mean()),
        "V_net_removed_mean_Sv": float(vn.mean()),
    }


def main() -> None:
    models = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")["model"].tolist()
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(model_years, models))

    ref = json.loads(REF.read_text())
    ref_pm = {r["model"]: r for r in ref["per_model"]}

    per_model = []
    errors = {}
    for r in results:
        if "error" in r:
            errors[r["model"]] = r["error"]
            continue
        hist = r["meta"].get("historical", r["meta"].get("ssp585"))
        row = {
            "model": r["model"],
            "section_file_experiments": sorted(r["meta"].keys()),
            "lat_south": hist["lat_south"],
            "lat_north": hist["lat_north"],
            "lon_west": hist["lon_west"],
            "lon_east": hist["lon_east"],
            "n_points": hist["n_points"],
            "lat_varies_along_row": hist["lat_varies_along_row"],
            "grid_description": {
                "n_longitude_points_global_i": hist["n_i"],
                "n_vertical_levels": hist["n_levels"],
                "grid_label": None,
            },
            "windows": {},
        }
        for name, (y0, y1) in WINDOWS.items():
            b = window_block(r["years"], y0, y1)
            if b is not None:
                row["windows"][name] = b
        ref_row = ref_pm.get(r["model"], {})
        if "1993-2014" in row["windows"] and "F_1993-2014" in ref_row:
            row["validation_F_1993_2014_minus_r1_cmip6_Sv"] = float(
                row["windows"]["1993-2014"]["F_ovS_monthly_mean_Sv"]
                - ref_row["F_1993-2014"]
            ) if "F_1993-2014" in ref_row else None
        for key in ("lat_south", "lat_north", "lon_west", "lon_east", "n_points"):
            if key in ref_row:
                row[f"validation_{key}_minus_r1_cmip6"] = float(row[key] - ref_row[key])
        per_model.append(row)

    summary = {}
    for name in ("1950-1980", "1993-2014"):
        js = [r["windows"][name]["jensen_term_Sv"] for r in per_model if name in r["windows"]]
        ps = [r["windows"][name]["jensen_term_pct_of_F_ovS"] for r in per_model if name in r["windows"]]
        vs = [r["windows"][name]["V_net_removed_mean_Sv"] for r in per_model if name in r["windows"]]
        summary[name] = {
            "n": len(js),
            "jensen_term_Sv_median": float(np.median(js)),
            "jensen_term_Sv_min": float(np.min(js)),
            "jensen_term_Sv_max": float(np.max(js)),
            "jensen_term_pct_median": float(np.median(ps)),
            "jensen_term_pct_min": float(np.min(ps)),
            "jensen_term_pct_max": float(np.max(ps)),
            "V_net_removed_Sv_median": float(np.median(vs)),
            "V_net_removed_Sv_min": float(np.min(vs)),
            "V_net_removed_Sv_max": float(np.max(vs)),
        }
    vmax = max(
        (abs(r["validation_F_1993_2014_minus_r1_cmip6_Sv"]) for r in per_model
         if r.get("validation_F_1993_2014_minus_r1_cmip6_Sv") is not None),
        default=None,
    )

    payload = {
        "meta": {
            "date": "2026-11-05",
            "group": "G09_cmip6meta",
            "actions": ["A84 (per-model metadata, net removal, Jensen term)"],
            "script": "scripts/analysis_paper3jc_r2_G09_cmip6meta_models.py",
            "inputs": [
                "data/cmip6_sections/{model}_{historical,ssp585}_{vo,so}.nc",
                "data/results/fovs_decomposition_cmip6_summary.csv (model list)",
                "PAPER_3_v2/analysis/r1_cmip6.json (validation only)",
            ],
            "seed": None,
            "sieve_resamples": None,
            "replicates": {"models": len(models), "experiments": list(EXPERIMENTS)},
            "windows": {k: list(v) for k, v in WINDOWS.items()},
            "units": {
                "F_ovS, jensen_term, V_net_removed": "Sv",
                "jensen_term_pct_of_F_ovS": "percent",
                "lat, lon": "degrees",
            },
            "definitions": {
                "V_net_removed": "sum over the section of the level transports before removal, monthly mean over the window",
                "jensen_term": "mean over the months of year y of F_ovS(monthly) minus F_ovS(annual-mean v and S of year y), averaged over the window",
            },
            "note": "grid_label is not stored in the section files; see the note file.",
        },
        "n_models": int(len(per_model)),
        "errors": errors,
        "per_model": per_model,
        "summary": summary,
        "max_abs_validation_F_1993_2014_Sv": vmax,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("models", len(per_model), "errors", errors)
    print("validation max abs", vmax)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
