"""Change-on-change relation of the CMIP6 overturning changes over the 17 common models.

Inputs: per-model values of PAPER_3_v2/analysis/jc_n11_cmip6_relation.json (key "per_model":
damoc26, the change of the overturning at 26.5 N, and dpsi34, the change of the 34.5 S
streamfunction maximum, both from 1850 to 1900 to 2081 to 2100 under SSP5-8.5).

Statistics over the models that have both changes:
* Pearson correlation of the two changes and the OLS slope of the 26.5 N change on the
  34.5 S change;
* medians of each change;
* 95% intervals from a family-cluster bootstrap (families of the genealogy grouping of
  analysis_paper3jc_n11_cmip6_relation.py are resampled with replacement, all members of a
  drawn family enter the sample), 10,000 resamples, seed 20261104.

Writes PAPER_3_v2/analysis/jc_r2_F1_cmip6_change.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
SRC = ANALYSIS / "jc_n11_cmip6_relation.json"
OUT = ANALYSIS / "jc_r2_F1_cmip6_change.json"
N_BOOT = 10_000
SEED = 20261104

FAMILIES = {
    "CESM2": ["CESM2", "CESM2-WACCM", "TaiESM1"],
    "MPI": ["MPI-ESM1-2-LR", "MPI-ESM1-2-HR"],
    "ACCESS": ["ACCESS-CM2", "ACCESS-ESM1-5"],
    "FGOALS": ["FGOALS-f3-L", "FGOALS-g3"],
    "CMCC": ["CMCC-CM2-SR5", "CMCC-ESM2"],
    "UKMO": ["UKESM1-0-LL", "HadGEM3-GC31-LL"],
}


def clusters_of(members):
    out, used = [], set()
    for fam in FAMILIES.values():
        present = [m for m in fam if m in members]
        if present:
            out.append(present)
            used.update(present)
    out += [[m] for m in members if m not in used]
    return out


def slope(x, y):
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def stats_of(x, y):
    return {
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope_26N_on_34S": slope(x, y),
        "median_34S_change": float(np.median(x)),
        "median_26N_change": float(np.median(y)),
    }


def main():
    src = json.loads(SRC.read_text())
    rows = [r for r in src["per_model"] if r.get("damoc26") is not None and r.get("dpsi34") is not None]
    models = [r["model"] for r in rows]
    x = np.array([r["dpsi34"] for r in rows], dtype=float)
    y = np.array([r["damoc26"] for r in rows], dtype=float)
    n = len(models)
    point = stats_of(x, y)
    cl = [[models.index(m) for m in c] for c in clusters_of(models)]
    K = len(cl)
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in ("r", "slope_26N_on_34S", "median_34S_change", "median_26N_change")}
    for _ in range(N_BOOT):
        pick = rng.integers(0, K, K)
        idx = np.array([j for k in pick for j in cl[k]], dtype=int)
        if len(idx) < 4 or np.ptp(x[idx]) == 0:
            continue
        s = stats_of(x[idx], y[idx])
        for k in draws:
            draws[k].append(s[k])
    boot = {}
    for k, v in draws.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        boot[k] = {"ci95": [float(lo), float(hi)], "n_valid": int(v.size),
                   "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    out = {
        "description": ("Change-on-change relation over the models with both the 26.5 N and the 34.5 S "
                        "change (SSP5-8.5, 2081 to 2100 minus 1850 to 1900)."),
        "source": "PAPER_3_v2/analysis/jc_n11_cmip6_relation.json (per_model)",
        "script": "scripts/analysis_paper3jc_r2_F1_cmip6_change.py",
        "units": {"changes": "Sv", "slope": "Sv per Sv", "r": "dimensionless"},
        "models": models,
        "n_models": n,
        "family_units": K,
        "n_boot": N_BOOT,
        "seed": SEED,
        "point": point,
        "bootstrap_family_cluster": boot,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"point": point, "n": n, "units": K, "boot": boot}, indent=1))


if __name__ == "__main__":
    main()
