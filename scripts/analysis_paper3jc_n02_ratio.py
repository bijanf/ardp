#!/usr/bin/env python3
"""Intervals for the reanalysis-to-EN4 trend ratios of the limb salinity contrast.

The limb contrast dS = S_N - S_S at 34.5 S (northward limb minus southward
limb, transport-weighted salinities) is trended over 2005-2024 in four ways:

  reanalysis, own circulation    dS_limb of paper3r1_section_<product>.nc, with
                                 the monthly transports of the reanalysis;
  reanalysis, fixed circulation  the 3-D section salinity of the reanalysis
                                 weighted by the record-mean limb weights, as in
                                 analysis_paper3r1_salinification_structure.py;
  EN4, fixed circulation         EN4.2.2 salinity with the record-mean V_bc of
                                 the reanalysis, as in analysis_paper3v2_en4.py;
  EN4, own circulation           EN4.2.2 salinity with the monthly V_bc of the
                                 reanalysis interpolated onto the EN4 levels.

The ratios of interest are the reanalysis trend over the EN4 trend. Each
numerator and denominator is an annual series over 2005-2024 (n = 20). Trend
intervals are those of paper3r1_stats.trend_test, and ratio intervals come from
the joint AR(1) sieve bootstrap of paper3r1_stats (sieve_coefficients and
ratio_summary), in which numerator and denominator are resampled with the same
year indices.

Writes ``PAPER_3_v2/analysis/jc_n02_ratio.json``.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_en4 import (  # noqa: E402
    EN4_DIR,
    en4_month,
    frozen_velocity,
    limb_quantities,
)
from compute_paper3r1_section_quantities import thickness  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    N_BOOT,
    ratio_summary,
    sieve_coefficients,
    trend_design,
    trend_test,
)

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_n02_ratio.json"

WINDOW = (2005, 2024)
YEARS = np.arange(WINDOW[0], WINDOW[1] + 1)
PRODUCTS = ("oras5", "glorys12")
RNG_SEED = 20261104
N_WORKERS = 2


def in_window(years: np.ndarray) -> np.ndarray:
    return (years >= WINDOW[0]) & (years <= WINDOW[1])


def annual_mean(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of monthly values within each calendar year of the window."""
    return np.array([np.nanmean(values[years == y]) for y in YEARS], dtype=float)


def trend_summary(years: np.ndarray, y: np.ndarray) -> dict:
    """OLS trend per decade with the interval of paper3r1_stats.trend_test.

    The uncorrected interval (ordinary OLS standard error, n - 2 degrees of
    freedom) is given for reference only; it is not the test used in the paper.
    """
    r = trend_test(years.astype(float), y)
    fit = stats.linregress(years.astype(float), y)
    half = float(stats.t.ppf(0.975, len(y) - 2) * fit.stderr)
    return {
        "slope_per_decade": float(10.0 * r["value"]),
        "ci_per_decade": [float(10.0 * r["ci"][0]), float(10.0 * r["ci"][1])],
        "p": float(r["p"]),
        "n": int(r["n"]),
        "n_eff": float(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
        "lag1": float(r["lag1"]),
        "phi_corrected": float(r["phi_corrected"]),
        "ols_ci_per_decade_reference_only": [
            float(10.0 * (fit.slope - half)),
            float(10.0 * (fit.slope + half)),
        ],
    }


def ratio_block(
    years: np.ndarray,
    num: np.ndarray,
    den: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Ratio of OLS trends, |num| / |den|, with the joint sieve bootstrap."""
    x = years.astype(float)
    tn = trend_summary(years, num)["slope_per_decade"]
    td = trend_summary(years, den)["slope_per_decade"]
    point = abs(tn) / abs(td)
    draws = sieve_coefficients(trend_design(x), np.column_stack([num, den]), 1, rng)
    res = ratio_summary(draws[:, 0], draws[:, 1], point)
    ratio = np.abs(draws[:, 0]) / np.abs(draws[:, 1])
    res["prob_below_1"] = float(np.mean(ratio < 1.0))
    res["ci_excludes_1"] = bool(res["ci"][0] > 1.0 or res["ci"][1] < 1.0)
    res["numerator_trend_per_decade"] = tn
    res["denominator_trend_per_decade"] = td
    res["n_boot"] = int(draws.shape[0])
    return res


# ── reanalysis series ────────────────────────────────────────────────


def reanalysis_own(product: str) -> dict:
    """Own-circulation dS_limb: annual series and monthly transports V_bc."""
    with xr.open_dataset(RESULTS / f"paper3r1_section_{product}.nc") as ds:
        t = pd.DatetimeIndex(ds["time"].values)
        dS = ds["dS_limb"].values.astype(float)
        vbc = ds["V_bc"].values.astype(float)
        depth = ds["depth"].values.astype(float)
    m = in_window(t.year.values)
    return {
        "annual_dS": annual_mean(t.year.values[m], dS[m]),
        "months": [(int(a), int(b)) for a, b in zip(t.year[m], t.month[m])],
        "vbc_monthly": vbc[m],
        "depth": depth,
    }


def reanalysis_fixed(product: str) -> dict:
    """Fixed-circulation dS of the reanalysis salinity.

    Mirrors the frozen-circulation construction of
    analysis_paper3r1_salinification_structure.run: the record-mean V_bc of the
    section gives the limb weights, and the annual salinity of the 3-D section
    is weighted by the area of each row and by these weights.
    """
    sec = xr.open_dataset(RESULTS / f"paper3r1_section_{product}.nc")
    f3 = xr.open_dataset(RESULTS / f"paper3r1_section3d_{product}.nc")
    depth = f3["depth"].values
    dz = thickness(depth, product, f3["e3t"].values, "bounds")
    e1t = f3["e1t"].values
    s = f3["s"]
    wet = np.isfinite(s.isel(time=0).values)
    area = np.where(wet, e1t[None, :] * dz[:, None], 0.0)
    a_k = area.sum(axis=1)
    frac = np.where(a_k[:, None] > 0, area / np.where(a_k > 0, a_k, 1)[:, None], 0.0)

    q = sec["V_bc"].values.mean(axis=0) * sec["e3t"].values
    qp = np.clip(q, 0, None)
    qn = np.clip(-q, 0, None)
    w_n = qp / qp.sum()
    w_s = qn / qn.sum()

    sub = s.sel(time=slice(f"{WINDOW[0]}-01-01", f"{WINDOW[1]}-12-31")).load()
    ann = sub.groupby("time.year").mean().load()
    years = ann["year"].values
    vals = np.where(wet[None], ann.values, 0.0)
    sk = (vals * frac[None]).sum(axis=2)  # (year, depth)
    s_n = sk @ w_n
    s_s = sk @ w_s
    sec.close()
    f3.close()
    return {
        "years": years,
        "annual_dS": s_n - s_s,
        "annual_S_north": s_n,
        "annual_S_south": s_s,
    }


# ── EN4 series ───────────────────────────────────────────────────────


def en4_files() -> list[Path]:
    files = []
    for p in sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc")):
        year = int(p.name.split(".")[-2][:4])
        if WINDOW[0] <= year <= WINDOW[1]:
            files.append(p)
    return files


def en4_limb_series(product: str, en4: dict, own_months: dict | None) -> dict:
    """EN4 dS per month, with the fixed or the monthly reanalysis circulation.

    own_months is None for the fixed circulation (record-mean V_bc of the
    reanalysis, as in analysis_paper3v2_en4.frozen_velocity). Otherwise it holds
    the monthly V_bc of the reanalysis, its depths and its (year, month) keys.
    """
    e3 = en4["e3t"]
    sbar = en4["sbar"]
    if own_months is None:
        v_fixed = frozen_velocity(f"paper3r1_section_{product}.nc", en4["depth"], e3)
        vs = [v_fixed] * len(sbar)
    else:
        index = {k: i for i, k in enumerate(own_months["months"])}
        vs = []
        for yr, mo in zip(en4["year"], en4["month"]):
            j = index.get((int(yr), int(mo)))
            if j is None:
                vs.append(None)
                continue
            vb = own_months["vbc_monthly"][j]
            if not np.all(np.isfinite(vb)):
                vs.append(None)
                continue
            vs.append(
                np.interp(
                    en4["depth"],
                    own_months["depth"],
                    vb,
                    left=vb[0],
                    right=0.0,
                )
            )
    dS = np.full(len(sbar), np.nan)
    n_valid = 0
    for i, v in enumerate(vs):
        if v is None:
            continue
        m = limb_quantities(v, sbar[i], e3)
        if m is not None:
            dS[i] = m["dS"]
            n_valid += 1
    return {
        "annual_dS": annual_mean(en4["year"], dS),
        "n_valid_months": n_valid,
    }


def _en4_row(path: Path):
    row = en4_month(path)
    return row


def load_en4_window() -> dict:
    files = en4_files()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = [r for r in ex.map(_en4_row, files, chunksize=8) if r is not None]
    rows.sort(key=lambda r: r["time"])
    times = pd.DatetimeIndex([r["time"] for r in rows])
    return {
        "year": np.array(times.year),
        "month": np.array(times.month),
        "sbar": np.array([r["S_bar"] for r in rows]),
        "depth": rows[0]["depth"].astype(float),
        "e3t": rows[0]["e3t"].astype(float),
        "n_months": len(rows),
    }


# ── checks against the existing outputs ──────────────────────────────


def existing_values() -> dict:
    """Values already reported by the existing analyses, for checks only."""
    with open(ANALYSIS / "r1_salinification_structure.json") as fh:
        sal = json.load(fh)
    with open(ANALYSIS / "en4.json") as fh:
        en4 = json.load(fh)
    return {
        # The stored S_north and S_south trends are already per decade.
        "frozen_dS_2005_2024_per_decade": {
            p: float(
                sal[p]["windows"]["2005-2024"]["S_north_frozen_trend_direct"]
                - sal[p]["windows"]["2005-2024"]["S_south_frozen_trend_direct"]
            )
            for p in PRODUCTS
        },
        "en4_frozen_dS_series": {p: en4[p]["dS"]["series"] for p in PRODUCTS},
        "en4_frozen_dS_slope": {
            p: float(en4[p]["dS"]["trend_per_decade"]) for p in PRODUCTS
        },
        "reanalysis_same_window_dS_per_decade": {
            p: float(en4["reanalysis_same_window"][p]["dS_trend_per_decade"])
            for p in PRODUCTS
        },
    }


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    existing = existing_values()
    en4 = load_en4_window()

    out: dict = {
        "id": "N02",
        "description": (
            "Intervals for the reanalysis-to-EN4 trend ratios of the limb "
            "salinity contrast dS over 2005-2024."
        ),
        "window": list(WINDOW),
        "units": "PSU per decade for trends; ratios are dimensionless",
        "n_years": int(len(YEARS)),
        "method": {
            "trend_interval": "paper3r1_stats.trend_test (corrected N_eff, 95 %)",
            "ratio_interval": (
                "joint AR(1) sieve bootstrap, paper3r1_stats.sieve_coefficients "
                "with ratio_summary, 2.5 and 97.5 percentiles"
            ),
            "n_boot": N_BOOT,
            "seed": RNG_SEED,
            "ratio_definition": "|trend of numerator| / |trend of denominator|",
        },
        "en4_months_in_window": en4["n_months"],
        "products": {},
        "checks": {},
    }

    for product in PRODUCTS:
        own = reanalysis_own(product)
        fixed = reanalysis_fixed(product)
        if not np.array_equal(fixed["years"], YEARS):
            raise RuntimeError("reanalysis years do not match the window")
        en4_fixed = en4_limb_series(product, en4, None)
        en4_own = en4_limb_series(product, en4, own)

        series = {
            "reanalysis_own_dS": own["annual_dS"],
            "reanalysis_fixed_dS": fixed["annual_dS"],
            "en4_fixed_dS": en4_fixed["annual_dS"],
            "en4_own_dS": en4_own["annual_dS"],
        }
        trends = {k: trend_summary(YEARS, v) for k, v in series.items()}
        ratios = {
            "fixed_over_en4_fixed": ratio_block(
                YEARS,
                series["reanalysis_fixed_dS"],
                series["en4_fixed_dS"],
                rng,
            ),
            "own_over_en4_fixed": ratio_block(
                YEARS,
                series["reanalysis_own_dS"],
                series["en4_fixed_dS"],
                rng,
            ),
            "own_over_en4_own": ratio_block(
                YEARS,
                series["reanalysis_own_dS"],
                series["en4_own_dS"],
                rng,
            ),
        }
        out["products"][product] = {
            "en4_valid_months": en4_fixed["n_valid_months"],
            "en4_own_valid_months": en4_own["n_valid_months"],
            "series": {k: [float(x) for x in v] for k, v in series.items()},
            "trends_per_decade": trends,
            "ratios": ratios,
            "reanalysis_fixed_S_north_S_south_per_decade": {
                "S_north": trend_summary(YEARS, fixed["annual_S_north"])[
                    "slope_per_decade"
                ],
                "S_south": trend_summary(YEARS, fixed["annual_S_south"])[
                    "slope_per_decade"
                ],
            },
        }

        # Checks: the fixed reanalysis contrast against the value in
        # r1_salinification_structure.json, and the EN4 fixed series against
        # en4.json, and the EN4 interval against the one printed in the text.
        out["checks"][product] = {
            "reanalysis_fixed_dS_slope_minus_existing": float(
                trends["reanalysis_fixed_dS"]["slope_per_decade"]
                - existing["frozen_dS_2005_2024_per_decade"][product]
            ),
            "reanalysis_own_dS_slope_minus_en4_json_same_window": float(
                trends["reanalysis_own_dS"]["slope_per_decade"]
                - existing["reanalysis_same_window_dS_per_decade"][product]
            ),
            "en4_fixed_dS_series_max_abs_diff_vs_en4_json": float(
                np.max(
                    np.abs(
                        series["en4_fixed_dS"]
                        - np.array(existing["en4_frozen_dS_series"][product])
                    )
                )
            ),
            "en4_fixed_dS_interval_per_decade": trends["en4_fixed_dS"]["ci_per_decade"],
            "en4_fixed_dS_slope_per_decade": trends["en4_fixed_dS"]["slope_per_decade"],
        }

    out["checks"]["text_intervals_pdf_L418_420"] = {
        "oras5_text": [0.011, 0.037],
        "glorys12_text": [0.013, 0.038],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))

    for product in PRODUCTS:
        print(f"== {product}")
        for k, t in out["products"][product]["trends_per_decade"].items():
            print(
                f"  trend {k:22s} {t['slope_per_decade']:+.4f} "
                f"[{t['ci_per_decade'][0]:+.4f}, {t['ci_per_decade'][1]:+.4f}]"
            )
        for k, r in out["products"][product]["ratios"].items():
            print(
                f"  ratio {k:22s} {r['point']:.3f} "
                f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] P(<1)={r['prob_below_1']:.3f}"
            )


if __name__ == "__main__":
    main()
