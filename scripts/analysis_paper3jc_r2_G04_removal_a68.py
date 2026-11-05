#!/usr/bin/env python3
"""Net-removal checks for the revision of paper 3 (action A68, cluster C24).

1. Salinity placement test. The ORAS5 velocity row lies at 34.56 S, between
   the tracer rows at 34.67 S (row j) and 34.46 S (row j+1). The baseline
   takes velocity and salinity from the same row index j. Text S1 states that
   averaging the two tracer rows onto the velocity row changes the record-mean
   FovS by -2.4% and its trend by about 1% "in 63 months sampled across the
   record". This script re-reads the ORAS5 tracer rows j and j+1 of the section
   from the monthly vosaline files for 63 months, every 13th month of the 816
   (indices 0, 13, ..., 806), and recomputes the monthly section diagnostics
   with

     - baseline: salinity from row j (checked against the stored section);
     - velocity_row_average: 0.5 (S_j + S_j+1) onto the velocity row;
     - next_row: salinity from row j+1 alone.

   Velocities are those stored in data/results/paper3r1_section3d_oras5.nc.
   The section integrals use compute_paper3r1_section_quantities.section_series
   with the baseline thickness convention ('bounds') and uniform removal.
   The full 816-month re-read was started and stopped: each file is read in
   chunks of 15 levels by 205 rows by 289 columns, and the shared disk
   delivered about 2 MB/s, so the full record was not feasible in the time
   budget. The 63-month design is the one the Supplement describes.
   Changes are reported as the signed change of the mean in percent of |F_base|,
   and as the change of the magnitude.

2. Proportional-to-|V| removal. The monthly estimator of
   analysis_paper3r1_cells.abs_v_removal (the one behind Table S2(a)) is
   applied to all three products over the full records and the
   alternative_removal block is regenerated. The annual-mean estimator of
   final_checks.json is carried alongside for the record.

Inputs: data/results/paper3r1_section3d_oras5.nc, data/results/paper3r1_section_<p>.nc,
data/oras5/vosaline_control_monthly_highres_3D_<yyyymm>_CONS_v0.1.nc (63 files),
PAPER_3_v2/analysis/final_checks.json (read only).
Output: PAPER_3_v2/analysis/jc_r2_G04_removal_a68.json.
Seeds: none (deterministic). Replicates: none.
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

from analysis_paper3r1_cells import abs_v_removal, summarise_variant  # noqa: E402
from analysis_paper3v2_attribution import annual, trend_stats  # noqa: E402
from compute_paper3r1_section_quantities import section_series, thickness  # noqa: E402
from compute_paper3v2_section_profiles import TARGET_LAT, atlantic_segment  # noqa: E402

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G04_removal_a68.json"
OR_DIR = REPO / "data" / "oras5"
PRODUCTS = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}
VARIANTS = ("baseline", "velocity_row_average", "next_row")
STRIDE = 13


def _yyyymm(p: Path) -> str:
    return p.name.split("_3D_")[1][:6]


def read_rows(args):
    """Salinity of tracer rows j and j+1 on the section columns, one month."""
    path, j, sel = args
    with xr.open_dataset(path) as ds:
        block = ds["vosaline"].isel(time_counter=0, y=slice(j, j + 2)).values
    # block has dimensions (depth, y, x) with y = rows j and j+1.
    block = np.where(block == 0.0, np.nan, block)[:, :, sel].astype(float)
    return block[:, 0, :], block[:, 1, :]


def variant_series(s_j: np.ndarray, s_j1: np.ndarray, kind: str) -> np.ndarray:
    if kind == "baseline":
        return s_j
    if kind == "velocity_row_average":
        both = np.isfinite(s_j) & np.isfinite(s_j1)
        return np.where(both, 0.5 * (s_j + s_j1), np.nan)
    if kind == "next_row":
        return s_j1
    raise ValueError(kind)


def main() -> None:
    f3 = xr.open_dataset(RESULTS / "paper3r1_section3d_oras5.nc")
    depth = f3["depth"].values
    e1t = f3["e1t"].values
    kbot = f3["kbot"].values
    lon_stored = f3["lon"].values
    times_all = f3["time"].values
    e3t = thickness(depth, "oras5", f3["e3t"].values, "bounds")
    tags_all = [f"{t.year:04d}{t.month:02d}" for t in pd.DatetimeIndex(times_all)]
    idx = np.arange(0, len(times_all), STRIDE)
    times = times_all[idx]
    tags = [tags_all[i] for i in idx]
    v = f3["v"].values[idx].astype(float)
    s_stored = f3["s"].values[idx].astype(float)
    f_ov_full_stored = xr.open_dataset(RESULTS / "paper3r1_section_oras5.nc")["F_ov"].values

    # Velocity row index and grid latitudes, as in extract_paper3r1_section_fields.py.
    first_v = sorted(OR_DIR.glob("vomecrty_*_3D_*.nc"))[0]
    with xr.open_dataset(first_v) as dv:
        lat_1d = np.nanmean(dv["nav_lat"].values, axis=1)
        nav_lat = dv["nav_lat"].values.copy()
        nav_lon = dv["nav_lon"].values.copy()
    j = int(np.abs(lat_1d - TARGET_LAT).argmin())
    lat_vrow = float(np.nanmean(nav_lat[j]))

    s_files = {_yyyymm(p): p for p in OR_DIR.glob("vosaline_*_3D_*.nc")}
    first_s = s_files[tags_all[0]]
    with xr.open_dataset(first_s) as ds0:
        nav_lat_t = ds0["nav_lat"].values.copy()
        s_surf = ds0["vosaline"].isel(time_counter=0, y=j).values[0]
    lat_row_j = float(np.nanmean(nav_lat_t[j]))
    lat_row_j1 = float(np.nanmean(nav_lat_t[j + 1]))
    s_surf = np.where(s_surf == 0.0, np.nan, s_surf)
    sel, _ = atlantic_segment(nav_lon[j], nav_lat[j], ~np.isnan(s_surf))
    if not np.allclose(nav_lon[j][sel], lon_stored):
        raise RuntimeError("section columns do not match the stored section")

    files = [s_files[t] for t in tags]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(read_rows, [(p, j, sel) for p in files], chunksize=4))
    s_j = np.stack([r[0] for r in rows])
    s_j1 = np.stack([r[1] for r in rows])

    wet = np.isfinite(s_stored) & np.isfinite(v)
    base_diff = float(np.nanmax(np.abs(np.where(wet, s_j - s_stored, 0.0))))

    results, fov = {}, {}
    for kind in VARIANTS:
        s_var = variant_series(s_j, s_j1, kind)
        n_lost = int(np.sum(np.isfinite(v) & np.isfinite(s_j) & ~np.isfinite(s_var)))
        res = section_series(v, s_var, e1t, e3t, kbot, 1.0)
        fov[kind] = res["F_ov"]
        results[kind] = {
            "mean_F_ov_Sv": float(res["F_ov"].mean()),
            "mean_T_limb_Sv": float(res["T_limb_Sv"].mean()),
            "mean_dS_limb_PSU": float(res["dS_limb"].mean()),
            "mean_F_az_Sv": float(res["F_az"].mean()),
            "wet_cells_lost_vs_baseline_total": n_lost,
        }

    fb = results["baseline"]["mean_F_ov_Sv"]
    for kind in VARIANTS[1:]:
        fv = results[kind]["mean_F_ov_Sv"]
        results[kind]["change_vs_baseline"] = {
            "mean_F_ov_difference_Sv": fv - fb,
            "mean_F_ov_signed_pct": 100.0 * (fv - fb) / abs(fb),
            "mean_F_ov_magnitude_pct": 100.0 * (abs(fv) - abs(fb)) / abs(fb),
        }

    # Monthly-series change and its sign, for the sampled months.
    month_change = 100.0 * (fov["velocity_row_average"] - fov["baseline"]) / np.abs(fov["baseline"])
    sub = {
        "design": "63 months, indices 0, 13, ..., 806 of the 816-month record (every 13th month)",
        "n_months": int(len(idx)),
        "stored_full_record_baseline_mean_F_ov_Sv": float(f_ov_full_stored.mean()),
        "stored_subsample_baseline_mean_F_ov_Sv": float(np.mean(f_ov_full_stored[idx])),
        "reconstructed_baseline_mean_F_ov_Sv": fb,
        "reconstructed_baseline_check_max_abs_diff_salinity_PSU": base_diff,
        "velocity_row_average_mean_F_ov_Sv": results["velocity_row_average"]["mean_F_ov_Sv"],
        "signed_pct_of_mean": results["velocity_row_average"]["change_vs_baseline"]["mean_F_ov_signed_pct"],
        "magnitude_pct_of_mean": results["velocity_row_average"]["change_vs_baseline"]["mean_F_ov_magnitude_pct"],
        "monthly_change_pct_median": float(np.median(month_change)),
        "monthly_change_pct_min": float(np.min(month_change)),
        "monthly_change_pct_max": float(np.max(month_change)),
        "note": "trend change cannot be formed from 63 unevenly spaced months; the Supplement's trend change (about 1%) is not reproduced here",
    }

    # Proportional-to-|V| removal, monthly estimator, full records, all products.
    alt = {}
    for key, label in PRODUCTS.items():
        with xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc") as ds_sec:
            ds_sec.load()
            s_alt = abs_v_removal(ds_sec)
            s_alt.load()
            base_summ = summarise_variant(ds_sec)
        summ = summarise_variant(s_alt)
        ann_alt = annual(s_alt)
        f_tr = trend_stats(ann_alt["year"].values, ann_alt["F_ov"].values)
        alt[key] = {
            "product": label,
            "scheme": "net transport removed in proportion to |V_int(z)| (monthly estimator, Table S2(a))",
            "estimator": "F = -(1/S0) sum_z q'_k (S_k - S0) on each monthly profile; annual means for trends",
            "mean_T_Sv": summ["mean_T_Sv"],
            "mean_dS_PSU": summ["mean_dS_PSU"],
            "mean_F_ov_Sv": summ["mean_F_ov_Sv"],
            "F_ov_trend_mSv_per_yr": f_tr["trend"],
            "F_ov_trend_ci_mSv_per_yr": f_tr["ci"],
            "F_ov_trend_n_eff": f_tr["n_eff"],
            "F_ov_trend_p_paper_test": f_tr["p_santer"],
            "F_ov_trend_summarise_variant_mSv_per_yr": summ["trend_F_ov_mSv_per_yr"],
            "F_ov_trend_p_summarise_variant": summ["p_F_ov"],
            "transport_share_T_term": summ["share_T"],
            "transport_share_T_term_ci": summ["share_T_ci"],
            "max_abs_c": float(s_alt.attrs["max_abs_c"]),
            "uniform_removal_baseline": {
                "mean_F_ov_Sv": base_summ["mean_F_ov_Sv"],
                "F_ov_trend_mSv_per_yr": base_summ["trend_F_ov_mSv_per_yr"],
                "transport_share_T_term": base_summ["share_T"],
            },
        }

    with (ANALYSIS / "final_checks.json").open() as fh:
        prev_fc = json.load(fh)
    previous = {k: prev_fc[k]["alternative_removal"] for k in PRODUCTS}

    out = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G04_removal_a68.py",
            "action": "A68 (cluster C24)",
            "inputs": [
                "data/results/paper3r1_section3d_oras5.nc (velocity, stored salinity, grid)",
                "data/results/paper3r1_section_{oras5,glorys12,ecco}.nc",
                "data/oras5/vosaline_control_monthly_highres_3D_<yyyymm>_{CONS,OPER}_v0.1.nc (63 months, tracer rows j and j+1)",
                "PAPER_3_v2/analysis/final_checks.json (read only, previous annual-mean estimator)",
            ],
            "seeds": None,
            "replicates": None,
            "sign_convention": "F_ovS northward-positive freshwater transport in Sv; F < 0 is a northward salt-flux deficit",
            "units": {"F": "Sv", "T": "Sv", "dS": "PSU", "trend": "mSv per yr", "pct": "percent of |F_base|"},
            "trend_test": "analysis_paper3v2_attribution.trend_stats (OLS, corrected N_eff, interval = inverse of the test)",
            "removal": "uniform removal (baseline) for the placement test; proportional-to-|V| for the alternative block",
        },
        "placement_test_oras5": {
            "velocity_row_index_j": j,
            "velocity_row_lat_deg": lat_vrow,
            "tracer_row_j_lat_deg": lat_row_j,
            "tracer_row_j1_lat_deg": lat_row_j1,
            "variants_63_months": results,
            "subsample_summary": sub,
        },
        "alternative_removal": alt,
        "alternative_removal_previous_annual_mean_estimator": previous,
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("placement (63 months):", results["velocity_row_average"]["change_vs_baseline"])
    print("next row:", results["next_row"]["change_vs_baseline"])
    print("baseline check", base_diff, "stored subsample mean", sub["stored_subsample_baseline_mean_F_ov_Sv"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
