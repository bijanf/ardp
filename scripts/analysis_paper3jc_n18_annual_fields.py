#!/usr/bin/env python3
"""Annual-field versus monthly transport decomposition at 34.5 S.

For each product the northward-limb transport T, the limb salinity contrast
dS and the overturning freshwater transport F_ovS are evaluated in two ways.

* Monthly: the monthly section values, averaged over the twelve months of
  each year (``*_monthly``).
* Annual fields: the section diagnostics recomputed from the annual-mean
  meridional velocity and salinity of each year, with the same routine that
  gives the monthly values (``section_series``), so that the two differ only
  in the averaging order (``*_fields``).

The Jensen term is the annual mean of the monthly transport minus the
transport of the annual-mean field,

    J_T = mean_{m in y}(T_m) - T(annual-mean fields),

and likewise J_F for F_ovS. Its size is given relative to the mean transport.
Each annual series is tested over its record with ``trend_test`` of
``paper3r1_stats`` (lag-1 autocorrelation corrected for bias, effective sample
size N_eff, intervals as the inverse of the test). Slopes are per decade;
transports are in Sv, F_ovS in mSv and dS in PSU.

Window-mean covariance term of F_ovS. With monthly values over a window,

    C = mean_m(F_m) - ( -(1/S0) mean_m(T_m) mean_m(dS_m) )
      = -(1/S0) [ mean_m(T_m dS_m) - Tbar dSbar ],

which is the covariance of T and dS in the window, scaled by -1/S0. The
annual terms c_y = -(1/S0) [ mean_{m in y}(T_m dS_m) - Tbar dSbar ] average to
C exactly for records of whole years. Their interval is the window mean of the
annual terms with the trend treated as signal (``fitted_mean_test``), with the
``mean_test`` interval of the same terms reported alongside; both treat the
window means Tbar and dSbar as fixed. The term is split into a within-year part (monthly covariance inside
each year) and a between-year part (covariance of the annual means).

Writes PAPER_3_v2/analysis/jc_n18_annual_fields.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compute_paper3r1_section_quantities import section_series  # noqa: E402
from paper3r1_stats import fitted_mean_test, mean_test, trend_test  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n18_annual_fields.json"
S0 = 35.0

# key: (label, first year, last year of the record)
PRODUCTS = {
    "oras5": ("ORAS5", 1958, 2025),
    "glorys12": ("GLORYS12V1", 1993, 2025),
    "ecco": ("ECCO-V4r4", 1992, 2017),
}
TREND_WINDOWS = [(1958, 2025), (1993, 2014), (1993, 2017), (2004, 2014), (2004, 2025)]
COV_WINDOWS = [(1958, 2025), (1993, 2014), (1993, 2017)]


def windows_of(y0: int, y1: int) -> list:
    """The full record of a product followed by the common sub-windows inside it."""
    return [(y0, y1)] + [w for w in TREND_WINDOWS if w != (y0, y1) and w[0] >= y0 and w[1] <= y1]


def f(x) -> float:
    return float(x)


def decade_trend(values: np.ndarray, years: np.ndarray) -> dict:
    """Trend per decade with the paper's test (paper3r1_stats.trend_test)."""
    r = trend_test(years.astype(float), np.asarray(values, dtype=float))
    return {
        "per_decade": f(r["value"] * 10.0),
        "ci_95_per_decade": [f(r["ci"][0] * 10.0), f(r["ci"][1] * 10.0)],
        "p": f(r["p"]),
        "n_years": int(r["n"]),
        "n_eff": f(r["n_eff"]),
        "phi_corrected": f(r["phi_corrected"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
    }


def mean_interval(values: np.ndarray) -> dict:
    """Mean of annual values with the paper's corrected-N_eff interval."""
    r = mean_test(np.asarray(values, dtype=float))
    return {
        "mean": f(r["value"]),
        "ci_95": [f(r["ci"][0]), f(r["ci"][1])],
        "n_eff": f(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
    }


def fitted_interval(values: np.ndarray, years: np.ndarray) -> dict:
    """Window mean with the trend treated as signal (paper3r1_stats.fitted_mean_test)."""
    yy = np.asarray(years, dtype=float)
    x = np.column_stack([np.ones_like(yy), yy - yy.mean()])
    r = fitted_mean_test(x, np.asarray(values, dtype=float))
    return {
        "mean": f(r["mean"]),
        "ci_95": [f(r["ci"][0]), f(r["ci"][1])],
        "n_eff": f(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
    }


def annual_mean(x: np.ndarray, years: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.array([x[years == y].mean(axis=0) for y in uy])


def process(key: str) -> dict:
    label, y0, y1 = PRODUCTS[key]
    sec = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
    f3 = xr.open_dataset(RESULTS / f"paper3r1_section3d_{key}.nc")

    years = sec["time"].dt.year.values
    uy = np.unique(years)
    counts = np.array([np.sum(years == y) for y in uy])
    if not np.all(counts == 12):
        raise ValueError(f"{label}: records are not whole years")
    if not np.array_equal(f3["time"].values, sec["time"].values):
        raise ValueError(f"{label}: time axes of the two files differ")

    e1t = f3["e1t"].values.astype(np.float64)
    kbot = f3["kbot"].values
    e3t = sec["e3t"].values.astype(np.float64)
    v = f3["v"].values.astype(np.float64)
    s = f3["s"].values.astype(np.float64)

    # The wet mask must be the same in every month for the annual means.
    wet = np.isfinite(s) & np.isfinite(v)
    if not np.array_equal(wet, np.broadcast_to(wet[:1], wet.shape)):
        raise ValueError(f"{label}: wet mask changes in time")

    # Monthly diagnostics recomputed from the 3D fields, to confirm the stored
    # section values are reproduced by the routine used for the annual fields.
    mon = section_series(v, s, e1t, e3t, kbot, 1.0)
    T_m = sec["T_limb_Sv"].values.astype(float)
    dS_m = sec["dS_limb"].values.astype(float)
    F_m = sec["F_ov"].values.astype(float)
    repro = {
        "T_max_abs_diff_Sv": f(np.max(np.abs(mon["T_limb_Sv"] - T_m))),
        "F_max_abs_diff_Sv": f(np.max(np.abs(mon["F_ov"] - F_m))),
        "dS_max_abs_diff_PSU": f(np.max(np.abs(mon["dS_limb"] - dS_m))),
    }
    identity = F_m - (-(1.0 / S0) * T_m * dS_m)
    repro["identity_max_abs_F_minus_minusTdS_over_S0_Sv"] = f(np.max(np.abs(identity)))
    del mon

    # Annual-mean fields and the same diagnostics on them.
    v_a = annual_mean(v, years, uy)
    s_a = annual_mean(s, years, uy)
    del v, s
    fld = section_series(v_a, s_a, e1t, e3t, kbot, 1.0)
    T_f = fld["T_limb_Sv"]
    dS_f = fld["dS_limb"]
    F_f = fld["F_ov"]

    # Annual means of the monthly values.
    T_a = annual_mean(T_m, years, uy)
    dS_a = annual_mean(dS_m, years, uy)
    F_a = annual_mean(F_m, years, uy)

    # Jensen-type terms.
    J_T = T_a - T_f
    J_F_mSv = (F_a - F_f) * 1e3
    rel_J_T = J_T / T_a
    # Within-year covariance: mean_{m in y}(T_m dS_m) - T_y dS_y, with the
    # monthly values, times -1/S0.
    TdS_a = annual_mean(T_m * dS_m, years, uy)
    cov_within = -(1.0 / S0) * (TdS_a - T_a * dS_a)
    cov_within_mSv = cov_within * 1e3

    jensen = {
        "mean_T_monthly_Sv": f(T_a.mean()),
        "mean_T_fields_Sv": f(T_f.mean()),
        "mean_J_T_Sv": f(J_T.mean()),
        "rel_J_T_to_T_mean_percent": f(100.0 * J_T.mean() / T_a.mean()),
        "rel_J_T_annual_min_percent": f(100.0 * rel_J_T.min()),
        "rel_J_T_annual_max_percent": f(100.0 * rel_J_T.max()),
        "rel_mean_abs_J_T_percent": f(100.0 * np.mean(np.abs(J_T)) / T_a.mean()),
        "mean_J_F_mSv": f(J_F_mSv.mean()),
        "mean_F_monthly_mSv": f(F_a.mean() * 1e3),
        "rel_J_F_to_F_mean_percent": f(100.0 * J_F_mSv.mean() / (F_a.mean() * 1e3)),
        "mean_cov_within_year_F_mSv": f(cov_within_mSv.mean()),
        "mean_dS_monthly_PSU": f(dS_a.mean()),
        "mean_dS_fields_PSU": f(dS_f.mean()),
    }

    series_for_trend = {
        "T_monthly_Sv": (T_a, "Sv"),
        "T_fields_Sv": (T_f, "Sv"),
        "J_T_Sv": (J_T, "Sv"),
        "dS_monthly_PSU": (dS_a, "PSU"),
        "dS_fields_PSU": (dS_f, "PSU"),
        "F_monthly_mSv": (F_a * 1e3, "mSv"),
        "F_fields_mSv": (F_f * 1e3, "mSv"),
        "J_F_mSv": (J_F_mSv, "mSv"),
        "cov_within_year_F_mSv": (cov_within_mSv, "mSv"),
    }
    trends = {}
    for name, (arr, unit) in series_for_trend.items():
        trends[name] = {"unit": f"{unit} per decade", **decade_trend(arr, uy)}

    # The same trends over the common sub-windows, so that the annual-field
    # and monthly declines can be compared where the paper quotes them.
    sub = {}
    for a, b in windows_of(y0, y1):
        m = (uy >= a) & (uy <= b)
        if m.sum() < 11:
            continue
        key_w = f"{a}_{b}"
        sub[key_w] = {
            "n_years": int(m.sum()),
            "T_monthly_Sv": decade_trend(T_a[m], uy[m]),
            "T_fields_Sv": decade_trend(T_f[m], uy[m]),
            "J_T_Sv": decade_trend(J_T[m], uy[m]),
        }

    # Window-mean covariance term of F_ovS, monthly values.
    cov = {}
    for a, b in COV_WINDOWS:
        key_w = f"{a}_{b}"
        if a < y0 or b > y1:
            cov[key_w] = {"available": False, "reason": f"record {y0}-{y1}"}
            continue
        mm = (years >= a) & (years <= b)
        ym = uy[(uy >= a) & (uy <= b)]
        Tbar = T_m[mm].mean()
        dSbar = dS_m[mm].mean()
        Fbar = F_m[mm].mean()
        F_of_means = -(1.0 / S0) * Tbar * dSbar
        C = Fbar - F_of_means
        ii = np.isin(uy, ym)
        c_tot = -(1.0 / S0) * (TdS_a[ii] - Tbar * dSbar)
        c_between = -(1.0 / S0) * (T_a[ii] * dS_a[ii] - Tbar * dSbar)
        c_within = cov_within[ii]
        if not np.isclose(c_tot.mean(), C, rtol=0, atol=1e-12):
            raise ValueError(f"{label}: window covariance decomposition does not close")
        tot_ci = fitted_interval(c_tot * 1e3, ym)
        tot_ci_const = mean_interval(c_tot * 1e3)
        cov[key_w] = {
            "available": True,
            "n_years": int(len(ym)),
            "n_months": int(mm.sum()),
            "Tbar_Sv": f(Tbar),
            "dSbar_PSU": f(dSbar),
            "mean_F_monthly_mSv": f(Fbar * 1e3),
            "F_of_window_means_mSv": f(F_of_means * 1e3),
            "covariance_term_mSv": f(C * 1e3),
            "covariance_term_ci_95_mSv": tot_ci["ci_95"],
            "covariance_term_ci_95_mSv_constant_mean_test": tot_ci_const["ci_95"],
            "covariance_term_n_eff": tot_ci["n_eff"],
            "covariance_term_n_eff_at_floor": tot_ci["n_eff_at_floor"],
            "covariance_term_share_of_F_percent": f(100.0 * C / Fbar),
            "within_year_part_mSv": f(c_within.mean() * 1e3),
            "within_year_part_ci_95_mSv": fitted_interval(c_within * 1e3, ym)["ci_95"],
            "between_year_part_mSv": f(c_between.mean() * 1e3),
            "between_year_part_ci_95_mSv": fitted_interval(c_between * 1e3, ym)["ci_95"],
            "window_mean_J_T_Sv": f(Tbar - T_f[ii].mean()),
            "window_mean_J_F_mSv": f((Fbar - F_f[ii].mean()) * 1e3),
            "interval_note": (
                "interval of the window mean of the annual terms (fitted_mean_test, "
                "trend as signal, corrected N_eff); window means treated as fixed"
            ),
        }

    # Paper convention for the trend of F_ovS: slope of F = transport term
    # + salinity term + residual, with transport term -(1/S0) slope(T) mean(dS)
    # and salinity term -(1/S0) mean(T) slope(dS), all from annual means.
    resid = {}
    for a, b in windows_of(y0, y1):
        m = (uy >= a) & (uy <= b)
        if m.sum() < 11:
            continue
        yy = uy[m].astype(float)
        sT = trend_test(yy, T_a[m])["value"]
        sdS = trend_test(yy, dS_a[m])["value"]
        sF = trend_test(yy, F_a[m])["value"]
        Tb, dSb = T_a[m].mean(), dS_a[m].mean()
        t_term = -(1.0 / S0) * sT * dSb
        s_term = -(1.0 / S0) * Tb * sdS
        r_term = sF - t_term - s_term
        resid[f"{a}_{b}"] = {
            "n_years": int(m.sum()),
            "F_trend_mSv_per_decade": f(sF * 1e4),
            "transport_term_mSv_per_decade": f(t_term * 1e4),
            "salinity_term_mSv_per_decade": f(s_term * 1e4),
            "residual_mSv_per_decade": f(r_term * 1e4),
            "residual_share_of_transport_term_percent": f(100.0 * abs(r_term) / abs(t_term)),
            "residual_share_of_F_trend_percent": f(100.0 * abs(r_term) / abs(sF)),
        }

    annual = {"years": [int(y) for y in uy]}
    for name, (arr, _unit) in series_for_trend.items():
        annual[name] = [f(x) for x in arr]

    rec = {
        "label": label,
        "record": [y0, y1],
        "n_years": int(len(uy)),
        "reproduction": repro,
        "jensen": jensen,
        "trends_full_record": trends,
        "trends_subwindows": sub,
        "covariance_windows": cov,
        "trend_decomposition_windows": resid,
        "annual_series": annual,
    }
    sec.close()
    f3.close()
    return rec


def main() -> None:
    out = {
        "analysis": "annual-field versus monthly transport decomposition (N18)",
        "S0_PSU": S0,
        "units": {
            "T": "Sv",
            "dS": "PSU",
            "F_ovS": "mSv",
            "trend": "per decade, paper3r1_stats.trend_test with corrected N_eff",
        },
        "products": {},
    }
    for key in PRODUCTS:
        out["products"][key] = process(key)
        r = out["products"][key]
        j = r["jensen"]
        print(f"\n{r['label']} {r['record'][0]}-{r['record'][1]} (n={r['n_years']})", flush=True)
        print(
            f"  T monthly {j['mean_T_monthly_Sv']:.3f} Sv, fields {j['mean_T_fields_Sv']:.3f} Sv, "
            f"J_T {j['mean_J_T_Sv']:+.3f} Sv ({j['rel_J_T_to_T_mean_percent']:+.2f} percent)",
            flush=True,
        )
        for name in ("T_monthly_Sv", "T_fields_Sv", "J_T_Sv", "J_F_mSv", "cov_within_year_F_mSv"):
            t = r["trends_full_record"][name]
            print(
                f"  trend {name:24s} {t['per_decade']:+.4f} [{t['ci_95_per_decade'][0]:+.4f}, "
                f"{t['ci_95_per_decade'][1]:+.4f}] p={t['p']:.3f} n_eff={t['n_eff']:.1f}",
                flush=True,
            )
        for w, c in r["covariance_windows"].items():
            if c.get("available"):
                print(
                    f"  cov window {w}: C={c['covariance_term_mSv']:+.3f} mSv "
                    f"[{c['covariance_term_ci_95_mSv'][0]:+.3f}, {c['covariance_term_ci_95_mSv'][1]:+.3f}] "
                    f"({c['covariance_term_share_of_F_percent']:+.2f} percent of F)",
                    flush=True,
                )
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
