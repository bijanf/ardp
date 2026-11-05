#!/usr/bin/env python3
"""Coverage of the sieve share intervals under fitted noise, with inflation.

The transport share of the ORAS5 F_ovS trend is the trend of the transport
term over the trend of the total, as in ``analysis_paper3v2_attribution.py``:

    F = -(1/S0) T dS,   f_T = -(1/S0) dbar T,   share = |b(f_T)| / |b(F)|,

with dbar the record mean of dS. Three share specifications are checked:

* unstepped: F and T on a linear trend, 1958 to 2025;
* 2015_only: a linear trend with a level shift at 2015;
* joint_2004_2015: a linear trend with level shifts at 2004 and 2015.

For each specification a data-generating model is built from the fitted
trend (or step) coefficients of the ORAS5 annual series, with residual
standard deviations and residual correlation of that specification. The
noise is AR(2) for F_ovS (stored fit in jc_n04_ar_recalibration.json,
fitted_ar2) and ARMA(1,1) for T (fitted_arma11). The dS noise enters only the
salinity term, which is not a share, and is therefore not simulated. For
each of 5,000 simulated record pairs the paper's 95 per cent sieve interval
of the share (``paper3r1_stats.sieve_coefficients`` with 10,000 resamples and
``ratio_summary`` percentiles, the same procedure as the attribution script)
is formed, and the coverage of the true share is recorded.

A sensitivity case repeats the unstepped share with the innovation
correlation fixed at 0.5, the value of the stylised study in
jc_n04_ar_recalibration.json. The data give a correlation near zero.

If the coverage is below 95 per cent, the interval is inflated about its
point estimate, [p - k (p - lo), p + k (hi - p)], and the smallest k on a
grid of step 0.01 that attains 95 per cent coverage in the simulations is
reported. A Gaussian-scale factor z(0.975) / z(coverage) is reported for
comparison. The inflated intervals of the stored unstepped, 2015-only and
joint shares (from jc_r2_G01_steps_oras5.json, A11) are also written, so the
manuscript numbers come from one file.

Seeds: ``numpy.random.SeedSequence(20261104).spawn(3)``, one stream per
specification for the simulation and the sieve. Single process.

Inputs: data/results/paper3r1_section_oras5.nc (annual F_ov, T_limb_Sv,
dS_limb), jc_n04_ar_recalibration.json (noise fits), attribution.json (record
means), jc_r2_G01_steps_oras5.json (stored shares and intervals),
paper3r1_stats.py (sieve, ratio summary, step design).

Writes ``PAPER_3_v2/analysis/jc_r2_G02_calibration_sieve.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3jc_n04_ar_recalibration import Model  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    N_BOOT,
    ratio_summary,
    sieve_coefficients,
    step_design,
    trend_design,
)

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
SRC_NOISE = ANALYSIS / "jc_n04_ar_recalibration.json"
SRC_ATTR = ANALYSIS / "attribution.json"
SRC_G01 = ANALYSIS / "jc_r2_G01_steps_oras5.json"
OUT = ANALYSIS / "jc_r2_G02_calibration_sieve.json"

S0 = 35.0
SEED = 20261104
N_DATASETS = 5_000
BURN = 1000
NOMINAL = 0.95
K_GRID = np.round(np.arange(1.00, 4.0 + 1e-9, 0.01), 2)


def annual_oras5() -> dict:
    """Calendar-year means of the ORAS5 section fields, as in the attribution script."""
    with xr.open_dataset(RESULTS / "paper3r1_section_oras5.nc") as ds:
        a = ds.groupby("time.year").mean()
        return {
            "years": a["year"].values.astype(float),
            "F": a["F_ov"].values.astype(float),
            "T": a["T_limb_Sv"].values.astype(float),
            "dS": a["dS_limb"].values.astype(float),
        }


def specification(name: str, years: np.ndarray) -> np.ndarray:
    """Design matrix of a share specification; the trend is column 1."""
    if name == "unstepped":
        return trend_design(years)
    if name == "2015_only":
        return step_design(years, 2015)
    if name == "joint_2004_2015":
        t = years - years.mean()
        return np.column_stack([np.ones_like(t), t,
                                (years >= 2004).astype(float),
                                (years >= 2015).astype(float)])
    raise ValueError(name)


def inflate_k(p: float, lo: float, hi: float, k: float) -> tuple[float, float]:
    return p - k * (p - lo), p + k * (hi - p)


def coverage_for_k(points, lows, highs, truth, k) -> float:
    lo_k = points - k * (points - lows)
    hi_k = points + k * (highs - points)
    return float(np.mean((lo_k <= truth) & (truth <= hi_k)))


def run_specification(name: str, data: dict, noise_f: Model, noise_t: Model,
                      stored_sd: dict, rng: np.random.Generator,
                      rho_override: float | None = None, label: str | None = None) -> dict:
    years = data["years"]
    x = specification(name, years)
    n, m = x.shape
    dbar = float(data["dS"].mean())
    pinv = np.linalg.pinv(x)
    bf = pinv @ data["F"]
    bt = pinv @ data["T"]
    f_t_data = -(dbar / S0) * data["T"]
    bft = pinv @ f_t_data
    truth = abs(bft[1] / bf[1])
    r_f = data["F"] - x @ bf
    r_t = data["T"] - x @ bt
    sd_f = float(np.std(r_f, ddof=m))
    sd_t = float(np.std(r_t, ddof=m))
    rho_data = float(np.corrcoef(r_f, r_t)[0, 1])
    rho = rho_data if rho_override is None else float(rho_override)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    fitted_f = x @ bf
    fitted_t = x @ bt

    points, lows, highs, cover_flags = [], [], [], []
    miss_below = miss_above = 0
    for _ in range(N_DATASETS):
        w = rng.standard_normal((n + BURN, 2)) @ chol.T
        e_f = lfilter(noise_f.b, noise_f.a, w[:, 0])[BURN:] * noise_f.scale
        e_t = lfilter(noise_t.b, noise_t.a, w[:, 1])[BURN:] * noise_t.scale
        f_sim = fitted_f + sd_f * e_f
        t_sim = fitted_t + sd_t * e_t
        ys = np.column_stack([f_sim, -(dbar / S0) * t_sim])
        draws = sieve_coefficients(x, ys, 1, rng, n_boot=N_BOOT)
        b_f = (pinv @ f_sim)[1]
        b_ft = (pinv @ ys[:, 1])[1]
        point = abs(b_ft / b_f)
        rs = ratio_summary(draws[:, 1], draws[:, 0], point)
        lo, hi = rs["ci"]
        points.append(point)
        lows.append(lo)
        highs.append(hi)
        ok = lo <= truth <= hi
        cover_flags.append(ok)
        if truth < lo:
            miss_below += 1
        elif truth > hi:
            miss_above += 1
    points = np.asarray(points)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    cov = float(np.mean(cover_flags))
    mc_se = float(np.sqrt(cov * (1 - cov) / N_DATASETS))

    k_table = []
    k95 = None
    for k in K_GRID:
        c = coverage_for_k(points, lows, highs, truth, k)
        k_table.append((float(k), c))
        if k95 is None and c >= NOMINAL:
            k95 = float(k)
    z95 = float(stats.norm.ppf(0.975))
    gauss_k = float(z95 / stats.norm.ppf(0.5 + cov / 2)) if 0 < cov < 1 else None

    return {
        "specification": name,
        "case": label or name,
        "truth_share": float(truth),
        "n": int(n),
        "n_datasets": N_DATASETS,
        "n_boot": N_BOOT,
        "dS_mean_PSU": dbar,
        "F_trend_fit_Sv_per_yr": float(bf[1]),
        "fT_trend_fit_Sv_per_yr": float(bft[1]),
        "sd_residual_F_Sv": sd_f,
        "sd_residual_T_Sv": sd_t,
        "residual_corr_F_T_data": rho_data,
        "innovation_correlation_used": rho,
        "coverage_share_interval": cov,
        "coverage_mc_se": mc_se,
        "miss_below": miss_below / N_DATASETS,
        "miss_above": miss_above / N_DATASETS,
        "below_90pct": bool(cov < 0.90),
        "median_point_estimate": float(np.median(points)),
        "median_interval_width": float(np.median(highs - lows)),
        "inflation_factor_empirical_95pct": k95,
        "inflation_factor_gaussian_95pct": gauss_k,
        "coverage_by_inflation_factor": {f"{k:.2f}": c for k, c in k_table
                                         if round(k, 2) in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                             1.6, 1.8, 2.0, 2.5, 3.0)},
        "stored_noise_F": {"model": "AR(2)", "ar": list(noise_f.ar), "ma": []},
        "stored_noise_T": {"model": "ARMA(1,1)", "ar": list(noise_t.ar), "ma": list(noise_t.ma)},
        "stored_sd_residual_trend_design_F_Sv": stored_sd["F"],
        "stored_sd_residual_trend_design_T_Sv": stored_sd["T"],
    }


def g01_stored(key: str) -> dict:
    g = json.loads(SRC_G01.read_text())["A11_transport_share"]
    row = g[key]
    return {"share_point": row["share_point"],
            "share_ci_sieve_2.5_97.5": row["share_ci_sieve_2.5_97.5"]}


def main() -> None:
    t0 = time.time()
    noise = {s["variable"]: s for s in json.loads(SRC_NOISE.read_text())["series"]
             if s["product"] == "ORAS5"}
    attr = json.loads(SRC_ATTR.read_text())["oras5"]
    data = annual_oras5()
    f_noise = Model("AR(2) F_ovS", "AR", ar=noise["F_ovS"]["fitted_ar2"])
    t_noise = Model("ARMA(1,1) T", "ARMA", ar=noise["T"]["fitted_arma11"]["ar"],
                    ma=noise["T"]["fitted_arma11"]["ma"])
    stored_sd = {"F": float(noise["F_ovS"]["sd_residual"]), "T": float(noise["T"]["sd_residual"])}

    # Three share specifications at the data residual correlation, and one
    # sensitivity case (unstepped, innovation correlation 0.5) to compare with
    # the stylised study of jc_n04_ar_recalibration.json (sieve_coverage).
    specs = [("unstepped", None, None), ("2015_only", None, None),
             ("joint_2004_2015", None, None),
             ("unstepped", 0.5, "unstepped_rho_0.5_sensitivity")]
    seeds = np.random.SeedSequence(SEED).spawn(len(specs))
    rows = []
    for (name, rho_fix, label), ss in zip(specs, seeds):
        rng = np.random.default_rng(ss)
        row = run_specification(name, data, f_noise, t_noise, stored_sd, rng,
                                rho_override=rho_fix, label=label)
        rows.append(row)
        print(name, "truth", round(row["truth_share"], 5), "coverage",
              round(row["coverage_share_interval"], 4), "k95",
              row["inflation_factor_empirical_95pct"], f"{time.time() - t0:.0f}s",
              flush=True)

    # Cross-checks against the stored attribution and G01 values.
    checks = {
        "attribution_unstepped_share_point": attr["continuous"]["own record"]["share_T_point"],
        "attribution_dS_mean_PSU": attr["continuous"]["own record"]["dS_mean_PSU"],
        "computed_dS_mean_PSU": float(data["dS"].mean()),
        "G01_stored_points": {k: g01_stored(k)["share_point"]
                              for k in ("unstepped", "2015_only", "joint_2004_2015")},
        "computed_truth_shares": {r["specification"]: r["truth_share"] for r in rows},
    }

    # Inflated stored intervals (A11 values from the G01 JSON).
    inflated = {}
    for r in rows:
        key = r["specification"]
        stored = g01_stored(key)
        p = stored["share_point"]
        lo, hi = stored["share_ci_sieve_2.5_97.5"]
        k = r["inflation_factor_empirical_95pct"]
        if k is None:
            inflated[key] = {"k": None, "note": "95 per cent not reached on the grid"}
            continue
        lo_k, hi_k = inflate_k(p, lo, hi, k)
        inflated[key] = {
            "k": k,
            "stored_point": p,
            "stored_interval": [lo, hi],
            "inflated_interval": [lo_k, hi_k],
            "inflated_interval_percent": [100 * lo_k, 100 * hi_k],
            "formula": "[p - k (p - lo), p + k (hi - p)]",
        }

    out = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G02_calibration_sieve.py",
            "action": "A18 (cluster C27)",
            "inputs": [
                "data/results/paper3r1_section_oras5.nc",
                "PAPER_3_v2/analysis/jc_n04_ar_recalibration.json (fitted_ar2 for F_ovS, fitted_arma11 for T, sd_residual)",
                "PAPER_3_v2/analysis/attribution.json (oras5 continuous, dS_mean_PSU)",
                "PAPER_3_v2/analysis/jc_r2_G01_steps_oras5.json (stored shares and intervals for the inflation)",
                "scripts/paper3r1_stats.py (sieve_coefficients, ratio_summary, step_design)",
            ],
            "seed": SEED,
            "seed_scheme": "SeedSequence(20261104).spawn(3), one stream per specification",
            "datasets_per_specification": N_DATASETS,
            "sieve_resamples_per_dataset": N_BOOT,
            "nominal_coverage": NOMINAL,
            "inflation_grid": "1.00 to 4.00 step 0.01",
            "units": "shares dimensionless; trends Sv per year; residual sd Sv",
            "runtime_s": None,
        },
        "specifications": rows,
        "inflated_stored_shares": inflated,
        "checks": checks,
        "note": "The noise of the simulated F is the stored AR(2) fit, that of T the stored ARMA(1,1) fit; the innovation correlation is the residual correlation of each specification.",
    }
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
