#!/usr/bin/env python3
"""Recalibration of the trend test of ``paper3r1_stats.py`` under AR(p) noise.

The trend test of ``paper3r1_stats.trend_test`` corrects the lag-1 residual
autocorrelation for its small-sample bias, under an AR(1) model with the same
design, and inverts the test for the interval. This script measures how the
test behaves when the noise is not AR(1):

1. Size (nominal 5 per cent) of the test, with 20,000 replicates per cell, for
   record lengths 11, 13, 15, 20, 22, 25, 26, 33 and 68 years.
   (a) AR(1) noise on a grid of lag-1 autocorrelation from 0 to 0.98.
   (b) AR(2), ARMA(1,1) and the AIC-selected model (AR(0) to AR(3) or
       ARMA(1,1)) fitted to the linear-trend residuals of the main series:
       ORAS5 F_ovS, T and dS (1958-2025), GLORYS12V1 (1993-2025), ECCO-V4r4
       (1992-2017) and the EN4 contrast (2005-2024, ORAS5 circulation).
   The size with the uncorrected residual autocorrelation (the Santer form
   used before the correction) is reported beside it for reference.
2. Power at 0.5 and 1 times the minimum detectable trend. The minimum
   detectable trend of a model is the median half-width of the 95 per cent
   interval of the test over noise-only replicates, in units of the
   stationary standard deviation of the noise per year, so it scales with the
   standard deviation of each series.
3. Coverage of the joint AR(1) sieve bootstrap for trend shares |b_T| / |b_F|
   on 500 AR(2) datasets for each of several record lengths and two true
   shares, with the 95 per cent percentile intervals of ``ratio_summary``.

Writes ``PAPER_3_v2/analysis/jc_n04_ar_recalibration.json``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import (  # noqa: E402
    N_BOOT,
    lag1,
    ratio_summary,
    sieve_coefficients,
    trend_design,
    trend_test,
    trend_test_many,
)

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_n04_ar_recalibration.json"

S0 = 35.0
SEED = 20261104
N_REP = 20_000
CHUNK = 2_000
ALPHA = 0.05
LENGTHS = (11, 13, 15, 20, 22, 25, 26, 33, 68)
PHI_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.98)
POWER_FRACTIONS = (0.5, 1.0)
BURN = 1000
PSI_LEN = 4000

SIEVE_LENGTHS = (20, 26, 33, 68)
SIEVE_SHARES = (0.1, 0.5)
SIEVE_DATASETS = 500
SIEVE_RHO = 0.5  # innovation correlation between the two series
SIEVE_SEED = 20261105


# ── data ──────────────────────────────────────────────────────────────


def annual(ds: xr.Dataset) -> xr.Dataset:
    """Calendar-year means (same grouping as the attribution script)."""
    return ds.groupby("time.year").mean()


def load_main_series() -> list[dict]:
    """Annual F_ovS, T and dS of the three sections, plus the EN4 contrast."""
    specs = (
        ("ORAS5", "paper3r1_section_oras5.nc"),
        ("GLORYS12V1", "paper3r1_section_glorys12.nc"),
        ("ECCO-V4r4", "paper3r1_section_ecco.nc"),
    )
    series = []
    for product, fname in specs:
        with xr.open_dataset(RESULTS / fname) as ds:
            ann = annual(ds)
            years = ann["year"].values.astype(int)
            fields = {
                "F_ovS": (ann["F_ov"].values.astype(float), "Sv"),
                "T": (ann["T_limb_Sv"].values.astype(float), "Sv"),
                "dS": (ann["dS_limb"].values.astype(float), "PSU"),
            }
        for var, (vals, unit) in fields.items():
            series.append(
                {"product": product, "variable": var, "unit": unit,
                 "years": years, "y": vals}
            )
    en4 = json.loads((ANALYSIS / "en4.json").read_text())
    series.append(
        {
            "product": "EN4 (ORAS5 circulation)",
            "variable": "dS",
            "unit": "PSU",
            "years": np.asarray(en4["years"], dtype=int),
            "y": np.asarray(en4["oras5"]["dS"]["series"], dtype=float),
        }
    )
    return series


# ── noise models ──────────────────────────────────────────────────────


class Model:
    """Stationary ARMA noise scaled to unit stationary variance."""

    def __init__(self, label: str, kind: str, ar=(), ma=()):
        self.label = label
        self.kind = kind
        self.ar = [float(v) for v in ar]
        self.ma = [float(v) for v in ma]
        self.a = np.r_[1.0, -np.asarray(self.ar, dtype=float)]
        self.b = np.r_[1.0, np.asarray(self.ma, dtype=float)]
        impulse = np.zeros(PSI_LEN)
        impulse[0] = 1.0
        psi = lfilter(self.b, self.a, impulse)
        self.scale = 1.0 / np.sqrt(np.sum(psi * psi))

    def draw(self, m: int, n: int, rng: np.random.Generator) -> np.ndarray:
        w = rng.standard_normal((m, n + BURN))
        x = lfilter(self.b, self.a, w, axis=-1)[:, BURN:]
        return x * self.scale


def fit_noise(resid: np.ndarray) -> dict:
    """Fit candidate noise models to the residuals and tabulate their AIC."""
    specs = (
        ("AR(0)", (0, 0, 0)),
        ("AR(1)", (1, 0, 0)),
        ("AR(2)", (2, 0, 0)),
        ("AR(3)", (3, 0, 0)),
        ("ARMA(1,1)", (1, 0, 1)),
    )
    fits = {}
    for label, order in specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(resid, order=order, trend="n").fit()
            fits[label] = {
                "aic": float(res.aic),
                "ar": [float(v) for v in res.arparams],
                "ma": [float(v) for v in res.maparams],
                "sigma2": float(res.params[-1]),
            }
        except Exception as exc:  # pragma: no cover - reported in the output
            fits[label] = {"aic": float("inf"), "error": str(exc)}
    selected = min(fits, key=lambda k: fits[k]["aic"])
    return {"fits": fits, "selected": selected}


def model_from_fit(label: str, f: dict) -> Model:
    return Model(label, "fitted", ar=f["ar"], ma=f["ma"])


# ── test under simulation ─────────────────────────────────────────────


def p_uncorrected(years: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p-value with the raw residual lag-1 (no bias correction), for reference.

    Same form as the uncorrected comparison in ``paper3r1_stats.calibration``:
    the raw lag-1 is clipped to +-0.99 and N_eff = n (1 - r1) / (1 + r1) is
    floored at 3. y has shape (n, m)."""
    x = trend_design(years)
    n = len(years)
    coef = np.linalg.pinv(x) @ y
    resid = y - x @ coef
    sxx = float(np.sum(x[:, 1] ** 2))
    se = np.sqrt((resid * resid).sum(0) / (n - 2) / sxx)
    r1 = np.clip(lag1(resid.T), -0.99, 0.99)
    ne = np.maximum(n * (1 - r1) / (1 + r1), 3.0)
    se_adj = se * np.sqrt((n - 2) / (ne - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * stats.t.sf(np.abs(coef[1]) / se_adj, ne - 2)


def run_cell(model: Model, n: int, rng: np.random.Generator, slope: float = 0.0,
             unc: bool = False) -> dict:
    """Trend test on N_REP replicates of model noise plus slope * year."""
    years = np.arange(n, dtype=float)
    p_c, p_u, hw, floor = [], [], [], []
    left = N_REP
    while left > 0:
        m = min(CHUNK, left)
        y = model.draw(m, n, rng) + slope * years
        yt = np.ascontiguousarray(y.T)
        r = trend_test_many(years, yt)
        p_c.append(r["p"])
        hw.append(r["half_width"])
        floor.append(r["n_eff"] <= 3.0 + 1e-9)
        if unc:
            p_u.append(p_uncorrected(years, yt))
        left -= m
    p_c = np.concatenate(p_c)
    rate = float(np.mean(p_c < ALPHA))
    out = {
        "p_below_alpha": rate,
        "mc_se": float(np.sqrt(rate * (1 - rate) / N_REP)),
        "n_rep": N_REP,
        "frac_neff_at_floor": float(np.mean(np.concatenate(floor))),
    }
    if unc:
        pu = np.concatenate(p_u)
        out["size_uncorrected"] = float(np.mean(pu < ALPHA))
    out["half_width_median_unit"] = float(np.median(np.concatenate(hw)))
    return out


def check_vectorised_matches_scalar(rng: np.random.Generator) -> dict:
    """Agreement of trend_test_many with the scalar trend_test of the paper."""
    worst = 0.0
    for n in (11, 26, 68):
        years = np.arange(n, dtype=float)
        y = Model("chk", "AR", ar=(0.6,)).draw(60, n, rng)
        r = trend_test_many(years, np.ascontiguousarray(y.T))
        for j in range(60):
            s = trend_test(years, y[j])
            worst = max(worst, abs(s["p"] - r["p"][j]), abs(s["half_width"] - r["half_width"][j]))
    return {"max_abs_difference_p_or_half_width": float(worst)}


# ── sieve bootstrap coverage ──────────────────────────────────────────


def sieve_coverage(model: Model, rng: np.random.Generator) -> list[dict]:
    """Coverage of the 95 per cent sieve interval of |b_T| / |b_F|."""
    rows = []
    chol = np.linalg.cholesky(np.array([[1.0, SIEVE_RHO], [SIEVE_RHO, 1.0]]))
    for n in SIEVE_LENGTHS:
        years = np.arange(n, dtype=float)
        x = trend_design(years)
        for share in SIEVE_SHARES:
            cover, lo_miss, hi_miss, widths, pts, f_cover = 0, 0, 0, [], [], 0
            for _ in range(SIEVE_DATASETS):
                w = rng.standard_normal((n + BURN, 2)) @ chol.T
                e = lfilter(model.b, model.a, w, axis=0)[BURN:] * model.scale
                f_series = years + e[:, 0]          # F_ovS, true slope 1
                t_series = share * years + e[:, 1]  # transport term, slope share
                ys = np.column_stack([f_series, t_series])
                draws = sieve_coefficients(x, ys, 1, rng, n_boot=N_BOOT)
                b_f = float(np.linalg.lstsq(x, f_series, rcond=None)[0][1])
                b_t = float(np.linalg.lstsq(x, t_series, rcond=None)[0][1])
                point = abs(b_t) / abs(b_f)
                rs = ratio_summary(draws[:, 1], draws[:, 0], point)
                lo, hi = rs["ci"]
                if lo <= share <= hi:
                    cover += 1
                elif share < lo:
                    lo_miss += 1
                else:
                    hi_miss += 1
                widths.append(hi - lo)
                pts.append(point)
                tf = trend_test(years, f_series)["ci"]
                if tf[0] <= 1.0 <= tf[1]:
                    f_cover += 1
            cov = cover / SIEVE_DATASETS
            rows.append(
                {
                    "n": int(n),
                    "true_share": float(share),
                    "n_datasets": SIEVE_DATASETS,
                    "coverage_share_interval": float(cov),
                    "mc_se": float(np.sqrt(cov * (1 - cov) / SIEVE_DATASETS)),
                    "miss_below": lo_miss / SIEVE_DATASETS,
                    "miss_above": hi_miss / SIEVE_DATASETS,
                    "median_point_estimate": float(np.median(pts)),
                    "median_interval_width": float(np.median(widths)),
                    "coverage_F_trend_interval": f_cover / SIEVE_DATASETS,
                }
            )
            print(f"sieve n={n} share={share}: coverage {cov:.3f}", flush=True)
    return rows


# ── main ──────────────────────────────────────────────────────────────


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def summarise(out: dict) -> dict:
    """Ranges (per cent) used in the text of the manuscript and its supplement."""

    def pct_range(rows, key):
        vals = [100.0 * r[key] for r in rows]
        return [min(vals), max(vals)]

    grid = out["size_power_ar1_grid"]
    main_ns = (22, 25, 26, 33, 68)
    s: dict = {}
    a = [r for r in grid if r["phi"] <= 0.8 + 1e-9 and r["n"] in main_ns]
    s["ar1_phi_0_to_0p8_n22_to_68_size_corrected"] = pct_range(a, "size_corrected")
    s["ar1_phi_0_to_0p8_n22_to_68_size_uncorrected"] = pct_range(a, "size_uncorrected")
    b = [r for r in grid if r["phi"] >= 0.9 - 1e-9 and r["n"] == 22]
    s["ar1_phi_0p9_to_0p98_n22_size_corrected"] = pct_range(b, "size_corrected")
    s["ar1_phi_0p9_to_0p98_n22_size_uncorrected"] = pct_range(b, "size_uncorrected")
    c = [r for r in grid if r["phi"] <= 0.8 + 1e-9 and r["n"] in (11, 13, 15, 20)]
    s["ar1_phi_0_to_0p8_n11_to_20_size_corrected"] = pct_range(c, "size_corrected")
    s["ar1_phi_0_to_0p8_n11_to_20_size_uncorrected"] = pct_range(c, "size_uncorrected")
    d = [r for r in grid if r["phi"] <= 0.8 + 1e-9]
    s["ar1_phi_0_to_0p8_all_n_size_corrected"] = pct_range(d, "size_corrected")
    e = [r for r in grid if r["phi"] <= 0.8 + 1e-9 and r["n"] in main_ns]
    s["ar1_phi_0_to_0p8_n22_to_68_power_0p5xMDT"] = pct_range(
        [{"v": r["power"]["0.5xMDT"]} for r in e], "v")
    s["ar1_phi_0_to_0p8_n22_to_68_power_1xMDT"] = pct_range(
        [{"v": r["power"]["1xMDT"]} for r in e], "v")
    s["ar1_neff_at_floor_by_phi_n68"] = {
        f"{r['phi']:g}": r["frac_neff_at_floor"] for r in grid if r["n"] == 68
    }
    s["ar1_mdt_unit_by_phi_n68"] = {
        f"{r['phi']:g}": r["mdt_unit"] for r in grid if r["n"] == 68
    }

    fitted = out["size_fitted_noise"]
    keys = []
    for r in fitted:
        k = (r["product"], r["variable"], r["model"], r["is_selected"])
        if k not in keys:
            keys.append(k)
    s["fitted_size_n22_to_68_corrected"] = {}
    s["fitted_size_n11_to_20_corrected"] = {}
    for k in keys:
        rows = [r for r in fitted if (r["product"], r["variable"], r["model"], r["is_selected"]) == k]
        name = f"{k[0]} {k[1]} {k[2]}{' (AIC-selected)' if k[3] else ''}"
        s["fitted_size_n22_to_68_corrected"][name] = pct_range(
            [r for r in rows if r["n"] in main_ns], "size_corrected")
        s["fitted_size_n11_to_20_corrected"][name] = pct_range(
            [r for r in rows if r["n"] in (11, 13, 15, 20)], "size_corrected")

    sel_power = out["power_selected_fitted_noise"]
    sel_main = [r for r in sel_power if r["n"] in main_ns]
    s["selected_power_n22_to_68_0p5xMDT"] = [
        100.0 * min(r["power"]["0.5xMDT"] for r in sel_main),
        100.0 * max(r["power"]["0.5xMDT"] for r in sel_main)]
    s["selected_power_n22_to_68_1xMDT"] = [
        100.0 * min(r["power"]["1xMDT"] for r in sel_main),
        100.0 * max(r["power"]["1xMDT"] for r in sel_main)]

    cov = out["sieve_coverage"]
    cvals = [r["coverage_share_interval"] for r in cov]
    s["sieve_share_coverage_pct"] = [100.0 * min(cvals), 100.0 * max(cvals)]
    s["sieve_share_coverage_F_trend_pct"] = [
        100.0 * min(r["coverage_F_trend_interval"] for r in cov),
        100.0 * max(r["coverage_F_trend_interval"] for r in cov)]
    # Interval half-width factor that would give 95 per cent coverage if the
    # sieve intervals were Gaussian with the observed coverage.
    z95 = stats.norm.ppf(0.975)
    s["sieve_halfwidth_inflation_for_95pct"] = [
        float(z95 / stats.norm.ppf(0.5 + c / 2)) for c in cvals]
    return s


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {
        "description": (
            "Simulated size and power of the bias-corrected trend test of "
            "paper3r1_stats.trend_test under AR(p) and ARMA noise, with "
            "sieve-bootstrap coverage for trend shares."
        ),
        "settings": {
            "n_rep": N_REP,
            "alpha": ALPHA,
            "lengths": list(LENGTHS),
            "phi_grid": list(PHI_GRID),
            "power_fractions": list(POWER_FRACTIONS),
            "burn_in": BURN,
            "seed": SEED,
            "order_candidates": ["AR(0)", "AR(1)", "AR(2)", "AR(3)", "ARMA(1,1)"],
            "selection": "minimum AIC of the candidates on the linear-trend residuals",
            "mdt_definition": (
                "median half-width of the 95 per cent interval over noise-only "
                "replicates, in units of the stationary noise sd per year"
            ),
        },
    }
    out["checks"] = check_vectorised_matches_scalar(np.random.default_rng(7))
    print("check", out["checks"], flush=True)

    # Series fits and the paper's test on the data.
    series_out = []
    fitted: list[tuple[dict, dict]] = []
    for s in load_main_series():
        years = s["years"]
        y = s["y"]
        x = trend_design(years.astype(float))
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        nf = fit_noise(resid)
        paper = trend_test(years.astype(float), y)
        scale = 1e3 if s["variable"] == "F_ovS" else 1.0
        unit = "mSv per yr" if s["variable"] == "F_ovS" else f"{s['unit']} per yr"
        row = {
            "product": s["product"],
            "variable": s["variable"],
            "window": [int(years[0]), int(years[-1])],
            "n": int(len(y)),
            "sd_residual": float(np.std(resid, ddof=2)),
            "lag1_residual": float(lag1(resid)),
            "aic": {k: v["aic"] for k, v in nf["fits"].items()},
            "selected_order": nf["selected"],
            "selected_ar": nf["fits"][nf["selected"]]["ar"],
            "selected_ma": nf["fits"][nf["selected"]]["ma"],
            "fitted_ar2": nf["fits"]["AR(2)"]["ar"],
            "fitted_arma11": {
                "ar": nf["fits"]["ARMA(1,1)"]["ar"],
                "ma": nf["fits"]["ARMA(1,1)"]["ma"],
            },
            "paper_trend_test": {
                "unit": unit,
                "value": paper["value"] * scale,
                "ci": [c * scale for c in paper["ci"]],
                "p": paper["p"],
                "phi_corrected": paper["phi_corrected"],
                "n_eff": paper["n_eff"],
                "half_width": paper["half_width"] * scale,
            },
        }
        series_out.append(row)
        fitted.append((row, nf))
        print(
            s["product"], s["variable"], row["window"], "selected", nf["selected"],
            "AIC", {k: round(v["aic"], 2) for k, v in nf["fits"].items()}, flush=True,
        )
    out["series"] = series_out

    # Simulated sizes and powers.
    grid_rows = []
    for phi in PHI_GRID:
        model = Model(f"AR(1) phi={phi:.2f}", "AR(1)", ar=(phi,) if phi else ())
        for n in LENGTHS:
            null = run_cell(model, n, rng, unc=True)
            row = {"phi": phi, "n": n, "size_corrected": null["p_below_alpha"],
                   "mc_se": null["mc_se"], "size_uncorrected": null["size_uncorrected"],
                   "mdt_unit": null["half_width_median_unit"],
                   "frac_neff_at_floor": null["frac_neff_at_floor"], "power": {}}
            for c in POWER_FRACTIONS:
                pw = run_cell(model, n, rng, slope=c * null["half_width_median_unit"])
                row["power"][f"{c:g}xMDT"] = pw["p_below_alpha"]
            grid_rows.append(row)
        print("grid phi", phi, "done", flush=True)
    out["size_power_ar1_grid"] = grid_rows

    fit_rows = []
    power_rows = []
    for row, nf in fitted:
        models = {}
        for label in ("AR(2)", "ARMA(1,1)", nf["selected"]):
            if label in models:
                continue
            models[label] = model_from_fit(label, nf["fits"][label])
        for label, model in models.items():
            for n in LENGTHS:
                null = run_cell(model, n, rng, unc=True)
                fit_rows.append(
                    {
                        "product": row["product"],
                        "variable": row["variable"],
                        "model": label,
                        "is_selected": label == nf["selected"],
                        "ar": model.ar,
                        "ma": model.ma,
                        "n": n,
                        "size_corrected": null["p_below_alpha"],
                        "mc_se": null["mc_se"],
                        "size_uncorrected": null["size_uncorrected"],
                        "mdt_unit": null["half_width_median_unit"],
                        "frac_neff_at_floor": null["frac_neff_at_floor"],
                    }
                )
                if label == nf["selected"]:
                    pw = {}
                    for c in POWER_FRACTIONS:
                        r_pw = run_cell(model, n, rng, slope=c * null["half_width_median_unit"])
                        pw[f"{c:g}xMDT"] = r_pw["p_below_alpha"]
                    power_rows.append(
                        {
                            "product": row["product"],
                            "variable": row["variable"],
                            "model": label,
                            "n": n,
                            "mdt_unit": null["half_width_median_unit"],
                            "mdt_model_physical": null["half_width_median_unit"] * row["sd_residual"],
                            "power": pw,
                        }
                    )
        print("fitted", row["product"], row["variable"], "done", flush=True)
    out["size_fitted_noise"] = fit_rows
    out["power_selected_fitted_noise"] = power_rows

    # Sieve coverage on AR(2) datasets, with the ORAS5 F_ovS AR(2) coefficients.
    f_oras = fitted[0][1]["fits"]["AR(2)"]
    ar2 = Model("AR(2) fitted to ORAS5 F_ovS residuals", "AR(2)",
                ar=f_oras["ar"], ma=f_oras["ma"])
    out["sieve_model"] = {"label": ar2.label, "ar": ar2.ar,
                          "innovation_correlation": SIEVE_RHO,
                          "datasets_per_cell": SIEVE_DATASETS}
    out["sieve_coverage"] = sieve_coverage(ar2, np.random.default_rng(SIEVE_SEED))
    out["summary"] = summarise(out)

    out = clean(out)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
