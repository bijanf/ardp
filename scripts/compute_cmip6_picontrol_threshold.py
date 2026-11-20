#!/usr/bin/env python3
"""Control-run thresholds for |dF_ovS| matched to the reanalysis windows.

For each CMIP6 model with piControl sections, annual-mean velocity and salinity
sections are formed and every pair of windows of a given length and start
separation is evaluated (no random draws). F_ovS of each window mean uses the
same kernel as the decomposition (baroclinic velocity, ardp.physics), so
|dF_ovS| = |F_ovS(late) - F_ovS(early)|.

Window settings:
  13-year windows whose starts are 20 years apart (the reanalysis main windows,
  1993-2005 and 2013-2025), and non-overlapping 30-year windows (the CMIP6
  windows; all pairs with start separation >= 30 years).
Each setting is evaluated on the raw control sections and on sections from
which a linear trend in time has been removed at every grid point.

Reported: pooled and per-model 95th percentiles; the number of models whose
forced change (1950-1980 to 2080-2100) exceeds the 95th percentile of their own
control run; the position of the reanalysis changes relative to the thresholds.

Run from the repository root:
    python scripts/compute_cmip6_picontrol_threshold.py --out data/results
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "scripts"))
from ardp.constants import S0, SAMBA_LAT  # noqa: E402
from ardp.physics.fovs_decomposition import decompose_fovs_trend  # noqa: E402
from compute_cmip6_fovs_decomposition import CMIP6_DIR, _grid_metrics, _load_section  # noqa: E402

REANALYSIS_MAIN_MSV = {"GLORYS12V1": 91.4, "ORAS5": 33.6, "SODA 3.15.2": 13.1, "ECCO-V4r4": 2.1}


def annual_sections(path_v: Path, path_s: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    da_v, depth, lon_180, _, _ = _load_section(path_v, "vo")
    da_v = da_v.load()
    da_v.close()
    da_s, _, _, _, _ = _load_section(path_s, "so")
    da_s = da_s.load()
    da_s.close()
    atl, e1t, e3t = _grid_metrics(lon_180, depth, SAMBA_LAT)
    yv = da_v["time"].dt.year.values
    years = np.unique(yv)
    v = np.stack([da_v.isel(time=np.where(yv == y)[0]).mean("time", skipna=True).values[:, atl] for y in years])
    ys = da_s["time"].dt.year.values
    s = np.stack([da_s.isel(time=np.where(ys == y)[0]).mean("time", skipna=True).values[:, atl] for y in np.unique(ys)])
    n = min(len(v), len(s))
    v, s = v[:n], s[:n]
    s = s.astype(float)
    s[(s <= 0) | (s > 100)] = np.nan
    v = np.nan_to_num(v.astype(float), nan=0.0)
    return v, s, e1t[atl], e3t


def detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(x.shape[0], dtype=float)
    t -= t.mean()
    flat = x.reshape(x.shape[0], -1)
    good = np.all(np.isfinite(flat), axis=0)
    out = flat.copy()
    slope = (t[:, None] * flat[:, good]).sum(0) / (t**2).sum()
    out[:, good] = flat[:, good] - t[:, None] * slope[None, :]
    return out.reshape(x.shape)


def window_fov(v: np.ndarray, s: np.ndarray, e1t: np.ndarray, e3t: np.ndarray, length: int) -> np.ndarray:
    n = v.shape[0] - length + 1
    f = np.empty(n)
    for i in range(n):
        vm = v[i:i + length].mean(0)
        with np.errstate(invalid="ignore"):
            sm = np.nanmean(s[i:i + length], axis=0)
        f[i] = decompose_fovs_trend(vm, sm, vm, sm, e1t, e3t, s0=S0)["F_ov_1"]
    return f


def pair_changes(f: np.ndarray, sep: int, exact: bool) -> np.ndarray:
    n = len(f)
    if exact:
        return np.abs(f[sep:] - f[: n - sep])
    out = [np.abs(f[j] - f[i]) for i in range(n) for j in range(i + sep, n)]
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="data/results")
    p.add_argument("--out", default="data/results")
    a = p.parse_args()
    summary = pd.read_csv(Path(a.results) / "fovs_decomposition_cmip6_summary.csv").set_index("model")
    old_null = pd.read_csv(Path(a.results) / "fovs_decomposition_cmip6_null.csv")

    settings = {"13y_sep20": (13, 20, True), "30y_nonoverlap": (30, 30, False)}
    models = sorted(f.name[: -len("_piControl_vo.nc")] for f in CMIP6_DIR.glob("*_piControl_vo.nc"))
    per_model: dict[str, dict] = {}
    pooled: dict[str, list] = {f"{k}_{d}": [] for k in settings for d in ("raw", "detrended")}
    for m in models:
        v, s, e1t, e3t = annual_sections(CMIP6_DIR / f"{m}_piControl_vo.nc", CMIP6_DIR / f"{m}_piControl_so.nc")
        rec = {"n_years": int(v.shape[0])}
        for d, (vv, ss) in {"raw": (v, s), "detrended": (detrend(v), detrend(s))}.items():
            for k, (length, sep, exact) in settings.items():
                ch = 1e3 * pair_changes(window_fov(vv, ss, e1t, e3t, length), sep, exact)
                pooled[f"{k}_{d}"].append(ch)
                rec[f"{k}_{d}_p95_mSv"] = float(np.percentile(ch, 95))
                rec[f"{k}_{d}_n"] = int(len(ch))
        old = old_null[old_null["model"] == m]["delta_total"].abs() * 1e3
        rec["random_draw_null_p95_mSv"] = float(np.percentile(old, 95)) if len(old) else None
        rec["forced_abs_change_mSv"] = float(abs(summary.loc[m, "delta_total"]) * 1e3)
        per_model[m] = rec
        print(m, {k: round(x, 1) if isinstance(x, float) else x for k, x in rec.items()}, flush=True)

    out = {"models": models, "per_model": per_model, "pooled": {}, "own_model_exceedance": {}, "reanalyses": {}}
    for key, chunks in pooled.items():
        allc = np.concatenate(chunks)
        p95m = [per_model[m][f"{key}_p95_mSv"] for m in models]
        out["pooled"][key] = {"p95_mSv": float(np.percentile(allc, 95)), "n_pairs": int(len(allc)),
                              "per_model_p95_range_mSv": [float(min(p95m)), float(max(p95m))],
                              "per_model_p95_median_mSv": float(np.median(p95m))}
        others = [per_model[m][f"{key}_p95_mSv"] for m in models if m != "CMCC-CM2-SR5"]
        out["pooled"][key]["p95_without_CMCC-CM2-SR5_mSv"] = float(
            np.percentile(np.concatenate([c for m, c in zip(models, chunks) if m != "CMCC-CM2-SR5"]), 95))
        out["pooled"][key]["max_per_model_p95_without_CMCC-CM2-SR5_mSv"] = float(max(others))
    old_all = old_null["delta_total"].abs() * 1e3
    out["pooled"]["random_draw_30y"] = {"p95_mSv": float(np.percentile(old_all, 95)), "n_pairs": int(len(old_all))}

    for key in ["30y_nonoverlap_raw", "30y_nonoverlap_detrended", "random_draw"]:
        col = "random_draw_null_p95_mSv" if key == "random_draw" else f"{key}_p95_mSv"
        exceed = [m for m in models if per_model[m]["forced_abs_change_mSv"] > per_model[m][col]]
        out["own_model_exceedance"][key] = {"n_exceed": len(exceed), "n_models": len(models),
                                            "not_exceeding": sorted(set(models) - set(exceed))}
    for prod, val in REANALYSIS_MAIN_MSV.items():
        r = {}
        for key in ["13y_sep20_raw", "13y_sep20_detrended"]:
            r[key] = {"above_pooled_p95": bool(val > out["pooled"][key]["p95_mSv"]),
                      "n_models_own_p95_exceeded": int(sum(val > per_model[m][f"{key}_p95_mSv"] for m in models))}
        r["above_random_draw_pooled_p95"] = bool(val > out["pooled"]["random_draw_30y"]["p95_mSv"])
        out["reanalyses"][prod] = {"abs_change_mSv": val, **r}

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "picontrol_matched_thresholds.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("pooled", "own_model_exceedance", "reanalyses")}, indent=1))


if __name__ == "__main__":
    main()
