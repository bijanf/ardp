#!/usr/bin/env python3
"""Finite-difference transient ratio at the southern boundary for CMIP6 (Table S4).

For each CMIP6 model with AMOC output at 26.5N, the change of F_ovS between the
historical window (1950-1980) and the SSP5-8.5 window (2080-2100) and its three
contributions (velocity, salinity, cross) are divided by the change of the AMOC
strength Psi (maximum of the overturning streamfunction at 26.5N below 500 m)
between the same windows:

    L_S = dF_ovS / dPsi = L_v + L_s + L_cross,
    L_v = dF_v / dPsi,  L_s = dF_s / dPsi,  L_cross = dF_cross / dPsi.

For a proportional weakening at fixed salinity L_v equals F_ovS(t1) / Psi(t1).
Counts are reported for the models whose |dF_ovS| reaches the 10 mSv
classification floor.

Run from the repository root:
    python scripts/compute_cmip6_transient_ratio.py --out data/results
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

T1 = (1950, 1980)
T2 = (2080, 2100)
T2_WEAKENING = (2081, 2100)
FLOOR_SV = 0.010
SHARE_THRESHOLD = 60.0


def classify(row: pd.Series) -> str:
    if abs(row["delta_total"]) < FLOOR_SV:
        return "ill-defined"
    if row["velocity_share_pct"] > SHARE_THRESHOLD:
        return "velocity"
    if row["salinity_share_pct"] > SHARE_THRESHOLD:
        return "salinity"
    return "mixed"


def window_mean(years: np.ndarray, x: np.ndarray, w: tuple[int, int]) -> float:
    m = (years >= w[0]) & (years <= w[1]) & np.isfinite(x)
    return float(np.mean(x[m])) if m.any() else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="data/results")
    p.add_argument("--out", default="data/results")
    a = p.parse_args()
    res = Path(a.results)
    df = pd.read_csv(res / "fovs_decomposition_cmip6_summary.csv")
    amoc = np.load(res / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    models_amoc = [str(m) for m in amoc["models"]]

    rows = []
    for _, r in df.iterrows():
        m = r["model"]
        row = {
            "model": m,
            "class": classify(r),
            "F1_mSv": 1e3 * r["F_ov_baseline"],
            "dF_mSv": 1e3 * r["delta_total"],
            "dFv_mSv": 1e3 * r["delta_v"],
            "dFs_mSv": 1e3 * r["delta_s"],
            "dFx_mSv": 1e3 * r["delta_cross"],
            "has_amoc": m in models_amoc,
        }
        if m in models_amoc:
            y = np.asarray(amoc[f"{m}_years"]).astype(int)
            x = np.asarray(amoc[f"{m}_amoc"]).astype(float)
            psi1 = window_mean(y, x, T1)
            psi2 = window_mean(y, x, T2)
            dpsi = psi2 - psi1
            row.update({
                "Psi1_Sv": psi1,
                "Psi2_Sv": psi2,
                "dPsi_Sv": dpsi,
                "weakening_pct_2081_2100": 100 * (1 - window_mean(y, x, T2_WEAKENING) / psi1),
                "L_S": row["dF_mSv"] / dpsi,
                "L_v": row["dFv_mSv"] / dpsi,
                "L_s": row["dFs_mSv"] / dpsi,
                "L_cross": row["dFx_mSv"] / dpsi,
                "F1_over_Psi1": row["F1_mSv"] / psi1,
            })
            row["Lv_over_prop"] = row["L_v"] / row["F1_over_Psi1"]
        rows.append(row)
    t = pd.DataFrame(rows)

    above = t[t["class"] != "ill-defined"]
    sub = above[above["has_amoc"]]
    neg_above = above[above["F1_mSv"] < 0]
    neg_sub = sub[sub["F1_mSv"] < 0]
    pos_ls = sub[sub["L_S"] > 0]
    counts = {
        "n_models": int(len(t)),
        "n_with_amoc": int(t["has_amoc"].sum()),
        "n_above_floor": int(len(above)),
        "n_above_floor_with_amoc": int(len(sub)),
        "all_amoc_decline_in_sub": bool((sub["dPsi_Sv"] < 0).all()),
        "dPsi_range_Sv": [float(sub["dPsi_Sv"].min()), float(sub["dPsi_Sv"].max())],
        "LS_positive": int((sub["L_S"] > 0).sum()),
        "LS_positive_models": sorted(pos_ls["model"]),
        "LS_negative_models": sorted(sub[sub["L_S"] <= 0]["model"]),
        "LS_positive_range": [float(pos_ls["L_S"].min()), float(pos_ls["L_S"].max())],
        "Lv_sign_as_baseline": int((np.sign(sub["L_v"]) == np.sign(sub["F1_mSv"])).sum()),
        "Lv_sign_not_as_baseline_models": sorted(sub[np.sign(sub["L_v"]) != np.sign(sub["F1_mSv"])]["model"]),
        "Lv_within_factor2_of_prop": int(((sub["Lv_over_prop"] >= 0.5) & (sub["Lv_over_prop"] <= 2.0)).sum()),
        "Lv_within_50pct_of_prop": int(((sub["Lv_over_prop"] >= 0.5) & (sub["Lv_over_prop"] <= 1.5)).sum()),
        "Lv_over_prop_median": float(sub["Lv_over_prop"].median()),
        "Ls_positive": int((sub["L_s"] > 0).sum()),
        "Ls_positive_models": sorted(sub[sub["L_s"] > 0]["model"]),
        "neg_baseline_with_amoc_above_floor": sorted(neg_sub["model"]),
        "neg_baseline_more_negative": sorted(neg_sub[neg_sub["dF_mSv"] < 0]["model"]),
        "neg_baseline_above_floor_all": sorted(neg_above["model"]),
        "neg_baseline_opposite_signs": sorted(neg_above[np.sign(neg_above["dFv_mSv"]) != np.sign(neg_above["dFs_mSv"])]["model"]),
        "neg_baseline_salt_prevails": sorted(neg_above[(np.sign(neg_above["dFv_mSv"]) != np.sign(neg_above["dFs_mSv"]))
                                                       & (neg_above["dFs_mSv"].abs() > neg_above["dFv_mSv"].abs())]["model"]),
        "salinity_class_neg_baseline": sorted(neg_above[neg_above["class"] == "salinity"]["model"]),
        "salinity_neg_baseline_Lv_neg_Ls_pos_larger": sorted(
            neg_sub[(neg_sub["class"] == "salinity") & (neg_sub["L_v"] < 0) & (neg_sub["L_s"] > 0)
                    & (neg_sub["L_s"].abs() > neg_sub["L_v"].abs())]["model"]),
        "full_ensemble_circ_sign_expected_above_floor": int((np.sign(above["dFv_mSv"]) == -np.sign(above["F1_mSv"])).sum()),
        "full_ensemble_dFs_negative_above_floor": int((above["dFs_mSv"] < 0).sum()),
        "windows": {"t1": T1, "t2": T2},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "cmip6_transient_ratio_table.csv", index=False, float_format="%.4f")
    (out / "cmip6_transient_ratio.json").write_text(json.dumps({"counts": counts, "table": t.to_dict(orient="records")},
                                                                   indent=1, default=float))
    pd.set_option("display.width", 220)
    print(t.round(2).to_string())
    print(json.dumps(counts, indent=1, default=float))


if __name__ == "__main__":
    main()
