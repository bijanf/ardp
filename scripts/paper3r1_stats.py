"""Tests for short, autocorrelated annual series (paper 3, revision).

The lag-1 autocorrelation r1 of regression residuals is biased low in short
records, which makes the effective sample size of Santer et al. (2000) too
large and the trend test too liberal. Here the coefficient is corrected by
inverting its simulated mean under an AR(1) process with the same design
matrix, truncated at zero, and

    N_eff = N (1 - phi) / (1 + phi),     k + 1 <= N_eff <= N,

with k the number of regression coefficients. The standard error of a
coefficient is inflated by sqrt((N - k) / (N_eff - k)) and the t test uses
N_eff - k degrees of freedom. Intervals are the inverse of that test, so a
coefficient is significant exactly when its interval excludes zero, and the
minimum detectable trend is the half-width of the interval.

Ratios of trends (transport shares) use an AR(1) sieve bootstrap: residuals
are filtered with the corrected coefficient, the innovations are resampled
jointly across the series at the same time indices, and the series are
rebuilt and refitted.

Correlations between two autocorrelated series use the effective sample size
N (1 - phi_a phi_b) / (1 + phi_a phi_b) of Bretherton et al. (1999).

Run this file to write the simulated size of the trend test for the record
lengths of the paper to PAPER_3_v2/analysis/r1_stats_calibration.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

PHI_GRID = np.linspace(0.0, 0.98, 50)
N_SIM_BIAS = 4000
N_BOOT = 10_000
_BIAS_CACHE: dict = {}


# ── AR(1) noise and the bias of the residual autocorrelation ──────────


def ar1_noise(n_sims: int, n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) series with unit innovation variance, shape (n_sims, n)."""
    e = np.empty((n_sims, n))
    e[:, 0] = rng.standard_normal(n_sims) / np.sqrt(1.0 - phi**2)
    w = rng.standard_normal((n_sims, n))
    for t in range(1, n):
        e[:, t] = phi * e[:, t - 1] + w[:, t]
    return e


def lag1(r: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation along the last axis (Pearson form)."""
    r = np.asarray(r, dtype=float)
    a = r[..., :-1] - r[..., :-1].mean(-1, keepdims=True)
    b = r[..., 1:] - r[..., 1:].mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(-1) / den, 0.0)


def bias_table(x: np.ndarray) -> np.ndarray:
    """Mean lag-1 autocorrelation of OLS residuals on design x, per PHI_GRID."""
    x = np.ascontiguousarray(x, dtype=float)
    key = (x.shape, x.tobytes())
    if key not in _BIAS_CACHE:
        rng = np.random.default_rng(20261103)
        resid_maker = np.eye(x.shape[0]) - x @ np.linalg.pinv(x)
        means = np.array(
            [
                lag1(ar1_noise(N_SIM_BIAS, x.shape[0], phi, rng) @ resid_maker.T).mean()
                for phi in PHI_GRID
            ]
        )
        _BIAS_CACHE[key] = np.maximum.accumulate(means)
    return _BIAS_CACHE[key]


def corrected_phi(r1: np.ndarray | float, x: np.ndarray) -> np.ndarray:
    """Bias-corrected AR(1) coefficient of residuals on design x, in [0, 0.98]."""
    table = bias_table(x)
    return np.interp(r1, table, PHI_GRID, left=0.0, right=PHI_GRID[-1])


def trend_design(years: np.ndarray) -> np.ndarray:
    t = np.asarray(years, dtype=float)
    return np.column_stack([np.ones_like(t), t - t.mean()])


def step_design(years: np.ndarray, step_year: int) -> np.ndarray:
    t = np.asarray(years, dtype=float)
    return np.column_stack(
        [np.ones_like(t), t - t.mean(), (t >= step_year).astype(float)]
    )


# ── tests ─────────────────────────────────────────────────────────────


def regression_test(x: np.ndarray, y: np.ndarray, j: int) -> dict:
    """OLS on design x; test of coefficient j with the corrected N_eff."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    s2 = float(resid @ resid) / (n - k)
    se = float(np.sqrt(s2 * np.linalg.inv(x.T @ x)[j, j]))
    r1 = float(lag1(resid))
    phi = float(corrected_phi(r1, x))
    neff_raw = n * (1.0 - phi) / (1.0 + phi)
    neff = float(min(max(neff_raw, k + 1.0), n))
    dof = neff - k
    se_adj = se * np.sqrt((n - k) / dof)
    value = float(coef[j])
    p = float(2.0 * stats.t.sf(abs(value) / se_adj, dof)) if se_adj > 0 else 0.0
    tcrit = float(stats.t.ppf(0.975, dof))
    p_ols = float(2.0 * stats.t.sf(abs(value) / se, n - k)) if se > 0 else 0.0
    return {
        "coef": [float(c) for c in coef],
        "value": value,
        "n": int(n),
        "k": int(k),
        "stderr_ols": se,
        "p_ols": p_ols,
        "lag1": r1,
        "phi_corrected": phi,
        "n_eff": neff,
        "n_eff_at_floor": bool(neff_raw < k + 1.0),
        "stderr_adj": float(se_adj),
        "p": p,
        "ci": [value - tcrit * se_adj, value + tcrit * se_adj],
        "half_width": tcrit * se_adj,
    }


def trend_test(years: np.ndarray, y: np.ndarray) -> dict:
    """Linear trend per unit of years."""
    return regression_test(trend_design(years), y, 1)


def step_test(years: np.ndarray, y: np.ndarray, step_year: int) -> dict:
    """Trend and level shift fitted together; returns both tests."""
    x = step_design(years, step_year)
    return {"trend": regression_test(x, y, 1), "step": regression_test(x, y, 2)}


def mean_test(y: np.ndarray) -> dict:
    """Mean with its interval from the corrected N_eff of the de-meaned series."""
    y = np.asarray(y, dtype=float)
    return regression_test(np.ones((len(y), 1)), y, 0)


def fitted_mean_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Mean of the fitted values of the regression on design x (equal to the
    sample mean when x has an intercept), with its standard error from the
    residuals of that regression and their corrected N_eff. With a trend in
    the design, the trend counts as signal and only the variability about it
    enters the interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    s2 = float(resid @ resid) / (n - k)
    w = x.mean(axis=0)
    se = float(np.sqrt(s2 * w @ np.linalg.inv(x.T @ x) @ w))
    phi = float(corrected_phi(float(lag1(resid)), x))
    neff_raw = n * (1.0 - phi) / (1.0 + phi)
    neff = float(min(max(neff_raw, k + 1.0), n))
    dof = neff - k
    se_adj = se * np.sqrt((n - k) / dof)
    mean = float(w @ coef)
    tcrit = float(stats.t.ppf(0.975, dof))
    return {
        "mean": mean,
        "ci": [mean - tcrit * se_adj, mean + tcrit * se_adj],
        "n_eff": neff,
        "n_eff_at_floor": bool(neff_raw < k + 1.0),
        "design_columns": int(k),
    }


def corr_test(a: np.ndarray, b: np.ndarray, detrend: bool = True) -> dict:
    """Pearson correlation with the N_eff of Bretherton et al. (1999)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    x = trend_design(np.arange(n)) if detrend else np.ones((n, 1))
    proj = np.eye(n) - x @ np.linalg.pinv(x)
    ra, rb = proj @ a, proj @ b
    r = float(np.corrcoef(ra, rb)[0, 1])
    pa = float(corrected_phi(lag1(ra), x))
    pb = float(corrected_phi(lag1(rb), x))
    neff = float(min(max(n * (1 - pa * pb) / (1 + pa * pb), 3.0), n))
    tval = r * np.sqrt((neff - 2) / max(1 - r * r, 1e-12))
    p_ols = float(stats.pearsonr(ra, rb)[1])
    return {
        "r": r,
        "n": int(n),
        "phi_a": pa,
        "phi_b": pb,
        "n_eff": neff,
        "p": float(2 * stats.t.sf(abs(tval), neff - 2)),
        "p_ols": p_ols,
    }


def trend_test_many(years: np.ndarray, y: np.ndarray) -> dict:
    """Vectorised trend_test for y of shape (n, m); NaN columns give NaN."""
    x = trend_design(years)
    n = x.shape[0]
    y = np.asarray(y, dtype=float)
    ok = np.all(np.isfinite(y), axis=0)
    out = {
        k: np.full(y.shape[1], np.nan) for k in ("slope", "p", "n_eff", "half_width")
    }
    if not ok.any():
        return out
    yy = y[:, ok]
    coef = np.linalg.pinv(x) @ yy
    resid = yy - x @ coef
    sxx = float(np.sum(x[:, 1] ** 2))
    se = np.sqrt((resid * resid).sum(0) / (n - 2) / sxx)
    phi = corrected_phi(lag1(resid.T), x)
    neff = np.clip(n * (1 - phi) / (1 + phi), 3.0, n)
    dof = neff - 2
    se_adj = se * np.sqrt((n - 2) / dof)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(coef[1]) / se_adj, dof)
    out["slope"][ok] = coef[1]
    out["p"][ok] = p
    out["n_eff"][ok] = neff
    out["half_width"][ok] = stats.t.ppf(0.975, dof) * se_adj
    return out


# ── joint AR(1) sieve bootstrap for ratios of trends ──────────────────


def sieve_coefficients(
    x: np.ndarray,
    ys: np.ndarray,
    j: int,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> np.ndarray:
    """Bootstrap draws of coefficient j for the columns of ys, shape (n_boot, m).

    Residuals of each series are filtered with its corrected AR(1)
    coefficient; the innovations are resampled at the same time indices for
    all series, which keeps their cross-correlation, and the series are
    rebuilt from the stationary state and refitted on the same design.
    """
    x = np.asarray(x, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n, m = ys.shape
    pinv = np.linalg.pinv(x)
    fitted = x @ (pinv @ ys)
    resid = ys - fitted
    phi = np.array([float(corrected_phi(lag1(resid[:, i]), x)) for i in range(m)])
    innov = resid[1:] - phi * resid[:-1]
    innov = innov - innov.mean(axis=0)
    idx = rng.integers(0, n - 1, size=(n_boot, n))
    u = innov[idx]  # (n_boot, n, m)
    e = np.empty_like(u)
    e[:, 0] = u[:, 0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        e[:, t] = phi * e[:, t - 1] + u[:, t]
    boot = fitted[None, :, :] + e
    return np.einsum("t,btm->bm", pinv[j], boot)


def ratio_summary(num: np.ndarray, den: np.ndarray, point: float) -> dict:
    """Share |num| / |den| from paired bootstrap draws."""
    ratio = np.abs(num) / np.abs(den)
    return {
        "point": float(point),
        "ci": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
        "prob_below_0p25": float(np.mean(ratio < 0.25)),
        "prob_opposite_sign": float(np.mean(np.sign(num) != np.sign(den))),
    }


# ── simulated size of the trend test ──────────────────────────────────


def calibration(
    lengths=(22, 25, 26, 33, 68), phis=(0.0, 0.3, 0.5, 0.65, 0.8), n_sims=2000
) -> dict:
    rng = np.random.default_rng(42)
    rows = []
    for n in lengths:
        years = np.arange(n)
        for phi in phis:
            sims = ar1_noise(n_sims, n, phi, rng)
            rej = np.mean([trend_test(years, s)["p"] < 0.05 for s in sims])
            raw = []
            for s in sims:
                fit = stats.linregress(years, s)
                res = s - (fit.intercept + fit.slope * years)
                r1 = min(max(float(np.corrcoef(res[:-1], res[1:])[0, 1]), -0.99), 0.99)
                ne = max(n * (1 - r1) / (1 + r1), 3.0)
                se = fit.stderr * np.sqrt((n - 2) / (ne - 2))
                raw.append(2 * stats.t.sf(abs(fit.slope) / se, ne - 2) < 0.05)
            rows.append(
                {
                    "n": n,
                    "phi": phi,
                    "size_corrected": float(rej),
                    "size_uncorrected": float(np.mean(raw)),
                }
            )
    sc = [r["size_corrected"] for r in rows]
    su = [r["size_uncorrected"] for r in rows]
    return {
        "n_sims": n_sims,
        "nominal": 0.05,
        "rows": rows,
        "range_corrected": [min(sc), max(sc)],
        "range_uncorrected": [min(su), max(su)],
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "PAPER_3_v2" / "analysis"
    res = calibration()
    (out / "r1_stats_calibration.json").write_text(json.dumps(res, indent=1))
    for r in res["rows"]:
        print(
            f"n={r['n']:>3} phi={r['phi']:.2f}  corrected {r['size_corrected']:.3f}"
            f"  uncorrected {r['size_uncorrected']:.3f}"
        )
    print(
        "range corrected",
        res["range_corrected"],
        "uncorrected",
        res["range_uncorrected"],
    )
