#!/usr/bin/env python3
"""Two AMOC estimators at 26.5N, exact Hodges-Lehmann interval and SODA 1998 sensitivity (paper 2, Table S5).

The AMOC series of the analysis (yearly_amoc26n_cmip6.npz) is the published Atlantic streamfunction for every model
that provides it; yearly_amoc26n_cmip6_velocity_based.npz keeps the series integrated from the meridional velocity.

1. Optionally retrieves the published Atlantic streamfunction (msftmz or msftyz) for the original-ensemble models
   that have it for both experiments, with the functions of compute_amoc26n_expansion.py; a file that cannot be
   retrieved from any replica is skipped (MPI-ESM1-2-HR: year 2100).
2. Compares baseline strength (1950-1980) and weakening (2081-2100) of the published streamfunction with the
   velocity-based series (Table S5). compute_amoc26n_published.py retrieves the published series of all models.
3. Class medians, gap, Hodges-Lehmann estimate with its exact 95% interval and the exact Mann-Whitney p-value for
   the velocity-dominant and salinity-dominant classes, for the velocity-based series and with the published values
   substituted where they exist; Spearman and Pearson relations of the weakening with f_v and with dF_s (17 models).
4. Effect of the SODA 3.15.2 1998 annual mean on the SODA mean, trend and early-window mean.

Run from the repository root:
    python scripts/compare_amoc26n_estimators.py --out data/results [--download]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
TARGETS = {"MPI-ESM1-2-LR": "msftmz", "MIROC6": "msftmz", "FGOALS-g3": "msftmz", "ACCESS-CM2": "msftmz",
           "MPI-ESM1-2-HR": "msftmz", "CanESM5": "msftmz", "CMCC-CM2-SR5": "msftyz", "IPSL-CM6A-LR": "msftyz"}
V_CLASS = ["ACCESS-ESM1-5", "CESM2", "CESM2-WACCM", "FIO-ESM-2-0", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "NESM3"]
S_CLASS = ["CNRM-CM6-1", "FGOALS-g3", "GFDL-CM4", "HadGEM3-GC31-LL", "MIROC6", "MRI-ESM2-0", "UKESM1-0-LL"]
MIXED = ["ACCESS-CM2", "FGOALS-f3-L", "GISS-E2-1-G"]


def _expansion_module():
    spec = importlib.util.spec_from_file_location("expansion", HERE / "compute_amoc26n_expansion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def retrieve(out_npz: Path, scratch: Path) -> None:
    exp = _expansion_module()
    res = dict(np.load(out_npz, allow_pickle=True)) if out_npz.exists() else {}
    for model, var in TARGETS.items():
        if f"{model}_years" in res:
            continue
        work = scratch / model
        work.mkdir(parents=True, exist_ok=True)
        years, amoc = [], []
        try:
            for e in ("historical", "ssp585"):
                paths = []
                for f in exp._solr_search(model, e, var):
                    p = work / f["filename"]
                    if exp._download(f["url"], p):
                        paths.append(p)
                ds = xr.open_mfdataset(sorted(paths), combine="by_coords", parallel=False)
                y, a = exp._extract_amoc(ds, var)
                years.append(y)
                amoc.append(a)
        except Exception as exc:  # model skipped, reported in the output
            print(f"{model}: not retrieved ({exc})")
            shutil.rmtree(work, ignore_errors=True)
            continue
        y = np.concatenate(years)
        a = np.concatenate(amoc)
        order = np.argsort(y)
        y, a = y[order], a[order]
        _, first = np.unique(y, return_index=True)
        res[f"{model}_years"], res[f"{model}_amoc"] = y[sorted(first)], a[sorted(first)]
        np.savez_compressed(out_npz, **res)
        shutil.rmtree(work, ignore_errors=True)


def _base_weak(src, model):
    y = np.asarray(src[f"{model}_years"]).astype(int)
    x = np.asarray(src[f"{model}_amoc"], float)
    base = x[(y >= 1950) & (y <= 1980)].mean()
    end = x[(y >= 2081) & (y <= 2100)].mean()
    return float(base), float(100 * (1 - end / base))


def _exact_hl(v, s, level=0.95):
    d = np.sort([b - a for a in v for b in s])
    n1, n2 = len(v), len(s)
    counts = {}
    idx = range(n1 + n2)
    for c in combinations(idx, n1):
        rest = set(idx) - set(c)
        u = sum(1 for i in c for j in rest if i > j)
        counts[u] = counts.get(u, 0) + 1
    total = math.comb(n1 + n2, n1)
    cum, ucrit = 0.0, -1
    for u in sorted(counts):
        if (cum + counts[u]) / total > (1 - level) / 2:
            break
        cum += counts[u]
        ucrit = u
    k = ucrit + 1  # the k-th smallest and k-th largest pairwise difference
    return float(np.median(d)), float(d[k - 1]), float(d[len(d) - k])


def class_stats(weak: dict) -> dict:
    v = [weak[m] for m in V_CLASS]
    s = [weak[m] for m in S_CLASS]
    hl, lo, hi = _exact_hl(v, s)
    return {"median_v": float(np.median(v)), "median_s": float(np.median(s)),
            "gap": float(np.median(s) - np.median(v)), "hodges_lehmann": hl, "hl_ci95": [lo, hi],
            "mann_whitney_p": float(mannwhitneyu(s, v, alternative="two-sided", method="exact").pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    res = Path(a.out)
    pub_npz = res / "amoc26n_published_streamfunction_check.npz"
    if a.download:
        retrieve(pub_npz, res / "tmp_amoc26n_published")
    vo_file = res / "yearly_amoc26n_cmip6_velocity_based.npz"
    vo = np.load(vo_file if vo_file.exists() else res / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    pub = np.load(pub_npz, allow_pickle=True)
    pub_models = sorted(k[:-6] for k in pub.files if k.endswith("_years"))
    table = {}
    for m in pub_models:
        b_vo, w_vo = _base_weak(vo, m)
        b_pub, w_pub = _base_weak(pub, m)
        table[m] = {"psi1_velocity_based": b_vo, "weakening_velocity_based": w_vo,
                    "psi1_published": b_pub, "weakening_published": w_pub}
    models = V_CLASS + S_CLASS + MIXED
    w_vel = {m: _base_weak(vo, m)[1] for m in models}
    w_pub = {m: (table[m]["weakening_published"] if m in table else w_vel[m]) for m in models}
    summ = pd.read_csv(res / "fovs_decomposition_cmip6_summary.csv").set_index("model")
    fv = summ.loc[models, "velocity_share_pct"].values
    dfs = 1e3 * summ.loc[models, "delta_s"].values
    out = {"table_S5": table}
    for name, w in (("velocity_based", w_vel), ("published_where_available", w_pub)):
        ws = [w[m] for m in models]
        out[name] = class_stats(w)
        out[name]["spearman_fv_weakening"] = list(map(float, spearmanr(fv, ws)))
        out[name]["pearson_dFs_weakening"] = list(map(float, pearsonr(dfs, ws)))
        out[name]["spearman_dFs_weakening"] = list(map(float, spearmanr(dfs, ws)))
    soda = xr.open_dataset(res / "soda_f_ovs.nc")
    yr, f = soda["year"].values, soda["F_ovS"].values
    keep = yr != 1998
    early = (yr >= 1993) & (yr <= 2005)
    out["soda_1998"] = {
        "value_1998_Sv": float(f[yr == 1998][0]),
        "mean_mSv_with": 1e3 * float(f.mean()), "mean_mSv_without": 1e3 * float(f[keep].mean()),
        "trend_mSv_per_yr_with": 1e3 * float(np.polyfit(yr, f, 1)[0]),
        "trend_mSv_per_yr_without": 1e3 * float(np.polyfit(yr[keep], f[keep], 1)[0]),
        "early_window_mean_mSv_with": 1e3 * float(f[early].mean()),
        "early_window_mean_mSv_without": 1e3 * float(f[early & keep].mean())}
    (res / "amoc26n_estimator_comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("velocity_based", "published_where_available", "soda_1998")}, indent=1))


if __name__ == "__main__":
    main()
