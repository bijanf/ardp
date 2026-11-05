#!/usr/bin/env python3
"""Calibrated thresholds of the bias-corrected trend test under fitted noise.

For the three headline ORAS5 annual series (F_ovS, T and dS, 1958 to 2025,
n = 68) the null distribution of the corrected p-value of
``paper3r1_stats.trend_test`` is simulated under the fitted lag-one noise
models stored in ``PAPER_3_v2/analysis/jc_n04_ar_recalibration.json``
(``fitted_ar2`` and ``fitted_arma11``). Two versions are run for each case:

1. Fixed coefficients: the stored AR(2) or ARMA(1,1) coefficients are the
   noise model for every replicate.
2. Refitted coefficients (coefficient estimation error included): each
   replicate is a two-stage parametric bootstrap. Stage 1 draws a series from
   the stored model; the noise model of the same order is refitted by
   maximum likelihood to the ordinary least-squares trend residuals of that
   series; Stage 2 draws the series used for the test from the refitted
   model. Replicates whose refit is non-stationary fall back to the stored
   coefficients, and their number is reported.

The null is zero trend. For each case the output gives the 5 per cent
critical value of the corrected p-value (the 5 per cent quantile of the
simulated p-values), the simulated size at nominal 5 per cent, and whether
the stored headline p-value of the series is below the calibrated threshold.

Seeds: ``numpy.random.SeedSequence(20261104)`` spawned once per case, and
once per chunk of 5,000 replicates within a case. Replicates: 20,000 per case
and version. The computation runs in one process.

Inputs: jc_n04_ar_recalibration.json (noise parameters and headline
p-values), paper3r1_stats.py (test), analysis_paper3jc_n04_ar_recalibration.py
(noise model class and burn-in).

Writes ``PAPER_3_v2/analysis/jc_r2_G02_calibration_bootstrap.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3jc_n04_ar_recalibration import Model  # noqa: E402
from paper3r1_stats import trend_design, trend_test_many  # noqa: E402

ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
SRC = ANALYSIS / "jc_n04_ar_recalibration.json"
OUT = ANALYSIS / "jc_r2_G02_calibration_bootstrap.json"

SEED = 20261104
N_REP = 20_000
CHUNK = 5_000
N = 68
ALPHA = 0.05
VARIABLES = ("F_ovS", "T", "dS")
ORDERS = {"AR(2)": (2, 0, 0), "ARMA(1,1)": (1, 0, 1)}


def is_stationary(ar) -> bool:
    """True when all roots of z^p - phi_1 z^(p-1) - ... lie inside the unit circle."""
    if len(ar) == 0:
        return True
    roots = np.roots(np.r_[1.0, -np.asarray(ar, dtype=float)])
    return bool(np.all(np.abs(roots) < 1.0))


def stored_cases(src: dict) -> list[dict]:
    """The stored ORAS5 noise fits and headline tests of the three series."""
    cases = []
    for s in src["series"]:
        if s["product"] != "ORAS5" or s["variable"] not in VARIABLES:
            continue
        cases.append({
            "variable": s["variable"],
            "window": s["window"],
            "n": s["n"],
            "headline_p": float(s["paper_trend_test"]["p"]),
            "headline_value": float(s["paper_trend_test"]["value"]),
            "headline_unit": s["paper_trend_test"]["unit"],
            "headline_n_eff": float(s["paper_trend_test"]["n_eff"]),
            "models": {
                "AR(2)": {"ar": [float(v) for v in s["fitted_ar2"]], "ma": []},
                "ARMA(1,1)": {"ar": [float(v) for v in s["fitted_arma11"]["ar"]],
                              "ma": [float(v) for v in s["fitted_arma11"]["ma"]]},
            },
        })
    return cases


def chunk_worker(ar, ma, label: str, nrep: int, seed_seq) -> dict:
    """Fixed-coefficient and refitted two-stage p-values for one chunk."""
    rng = np.random.default_rng(seed_seq)
    years = np.arange(N, dtype=float)
    x = trend_design(years)
    proj = np.eye(N) - x @ np.linalg.pinv(x)
    model = Model(label, "fitted", ar=ar, ma=ma)
    order = ORDERS[label]

    y_fixed = np.ascontiguousarray(model.draw(nrep, N, rng).T)
    p_fixed = trend_test_many(years, y_fixed)["p"]

    y1 = model.draw(nrep, N, rng)
    res1 = y1 @ proj.T
    refit_ar = np.full((nrep, order[0]), np.nan)
    refit_ma = np.full((nrep, order[2]), np.nan)
    fallback = 0
    y2 = np.empty((nrep, N))
    for i in range(nrep):
        th_ar, th_ma = list(ar), list(ma)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(res1[i], order=order, trend="n").fit()
            cand_ar = [float(v) for v in fit.arparams]
            cand_ma = [float(v) for v in fit.maparams]
            if not is_stationary(cand_ar) or not np.all(np.isfinite(cand_ar + cand_ma)):
                raise ValueError("non-stationary refit")
            th_ar, th_ma = cand_ar, cand_ma
        except Exception:
            fallback += 1
        if order[0]:
            refit_ar[i] = th_ar
        if order[2]:
            refit_ma[i] = th_ma
        m2 = Model(label, "fitted", ar=th_ar, ma=th_ma)
        y2[i] = m2.draw(1, N, rng)[0]
    p_refit = trend_test_many(years, np.ascontiguousarray(y2.T))["p"]
    return {
        "p_fixed": p_fixed,
        "p_refit": p_refit,
        "fallback": fallback,
        "refit_ar_mean": np.nanmean(refit_ar, axis=0) if order[0] else np.array([]),
        "refit_ar_sd": np.nanstd(refit_ar, axis=0) if order[0] else np.array([]),
        "refit_ma_mean": np.nanmean(refit_ma, axis=0) if order[2] else np.array([]),
        "refit_ma_sd": np.nanstd(refit_ma, axis=0) if order[2] else np.array([]),
    }


def summarise_p(p: np.ndarray, headline: float) -> dict:
    p = p[np.isfinite(p)]
    crit = float(np.quantile(p, ALPHA))
    size = float(np.mean(p < ALPHA))
    return {
        "n_valid": int(p.size),
        "critical_p_5pct": crit,
        "size_at_nominal_5pct": size,
        "size_mc_se": float(np.sqrt(size * (1 - size) / p.size)),
        "headline_p": headline,
        "headline_passes_calibrated_threshold": bool(headline < crit),
        "headline_p_over_critical": float(headline / crit) if crit > 0 else None,
    }


def main() -> None:
    t0 = time.time()
    src = json.loads(SRC.read_text())
    cases = stored_cases(src)
    seeds = np.random.SeedSequence(SEED).spawn(len(cases) * len(ORDERS))
    out: dict = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G02_calibration_bootstrap.py",
            "action": "A16 (cluster C03)",
            "inputs": [
                "PAPER_3_v2/analysis/jc_n04_ar_recalibration.json (series[*].fitted_ar2, fitted_arma11, paper_trend_test)",
                "scripts/paper3r1_stats.py (trend_test_many, bias-corrected N_eff)",
                "scripts/analysis_paper3jc_n04_ar_recalibration.py (Model, burn-in)",
            ],
            "seed": SEED,
            "seed_scheme": "SeedSequence(20261104).spawn(6), one child per (series, noise model); each child spawns one stream per 5,000-replicate chunk",
            "replicates_per_case_and_version": N_REP,
            "n": N,
            "alpha": ALPHA,
            "null": "zero trend",
            "units": "p-values dimensionless; noise coefficients of the stored ARMA convention, unit stationary variance",
            "versions": {
                "fixed": "noise coefficients fixed at the stored fit",
                "refit": "two-stage parametric bootstrap with ARIMA refit of the same order on the OLS residuals of stage 1; non-stationary refits fall back to the stored fit",
            },
        },
        "cases": [],
    }
    k = 0
    for c in cases:
        for label in ORDERS:
            ar = c["models"][label]["ar"]
            ma = c["models"][label]["ma"]
            chunks = [
                chunk_worker(ar, ma, label, CHUNK, ch)
                for ch in seeds[k].spawn(N_REP // CHUNK)
            ]
            k += 1
            p_fixed = np.concatenate([r["p_fixed"] for r in chunks])
            p_refit = np.concatenate([r["p_refit"] for r in chunks])
            fallback = int(sum(r["fallback"] for r in chunks))
            row = {
                "variable": c["variable"],
                "window": c["window"],
                "n": c["n"],
                "noise_model": label,
                "stored_noise_ar": ar,
                "stored_noise_ma": ma,
                "headline_value": c["headline_value"],
                "headline_unit": c["headline_unit"],
                "headline_n_eff": c["headline_n_eff"],
                "fixed_coefficients": summarise_p(p_fixed, c["headline_p"]),
                "refitted_coefficients": summarise_p(p_refit, c["headline_p"]),
                "refit_fallbacks": fallback,
                "refit_fallback_fraction": fallback / N_REP,
                "refit_ar_mean": np.mean([r["refit_ar_mean"] for r in chunks], axis=0).tolist()
                if order_has_ar(label) else [],
                "refit_ar_sd_across_replicates": np.mean([r["refit_ar_sd"] for r in chunks], axis=0).tolist()
                if order_has_ar(label) else [],
                "refit_ma_mean": np.mean([r["refit_ma_mean"] for r in chunks], axis=0).tolist()
                if label == "ARMA(1,1)" else [],
                "refit_ma_sd_across_replicates": np.mean([r["refit_ma_sd"] for r in chunks], axis=0).tolist()
                if label == "ARMA(1,1)" else [],
            }
            out["cases"].append(row)
            print(c["variable"], label,
                  "fixed crit", round(row["fixed_coefficients"]["critical_p_5pct"], 6),
                  "refit crit", round(row["refitted_coefficients"]["critical_p_5pct"], 6),
                  "fallbacks", fallback, f"{time.time() - t0:.0f}s", flush=True)
    out["summary"] = {
        f"{r['variable']} {r['noise_model']}": {
            "critical_p_5pct_refit": r["refitted_coefficients"]["critical_p_5pct"],
            "critical_p_5pct_fixed": r["fixed_coefficients"]["critical_p_5pct"],
            "headline_p": r["refitted_coefficients"]["headline_p"],
            "headline_passes_refit": r["refitted_coefficients"]["headline_passes_calibrated_threshold"],
            "headline_passes_fixed": r["fixed_coefficients"]["headline_passes_calibrated_threshold"],
        } for r in out["cases"]
    }
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT, flush=True)


def order_has_ar(label: str) -> bool:
    return ORDERS[label][0] > 0


if __name__ == "__main__":
    main()
