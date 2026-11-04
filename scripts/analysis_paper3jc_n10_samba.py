#!/usr/bin/env python3
"""SAMBA comparison of the overturning cells and the northward-limb transport at 34.5 S.

The SAMBA array series in data/external are the monthly and daily MOC transport
anomalies of Kersale et al. (2020). Both files cover September 2013 to July 2017
(47 months, 1405 days). The record from 2009 is not in the repository, so the
comparison is made on the 47 months that SAMBA and all three reanalyses share.

Design
  * Anomalies. Every series (SAMBA and reanalysis) is fitted over the common
    window by least squares on a constant, a linear trend and the annual and
    semi-annual harmonics. The residuals are the anomalies with the seasonal
    cycle and the trend removed. The same treatment is applied to both sides.
  * Layers. The reanalysis transports are integrated over the SAMBA layers:
    surface to 1315 dbar (upper), 1315 to 3155 dbar (NADW return layer, not
    observed by SAMBA in this file) and 3155 to 4295 dbar (abyssal). The layers
    are taken from the net-removed velocity V_bc, which defines the cells of
    analysis_paper3r1_cells.py, and from the uncorrected velocity V_int.
  * Cell definition. The upper cell psi_max and the abyssal cell -psi_min of
    paper3r1_cells_<product>.nc, and the northward-limb transport T_limb.
  * Sign convention of SAMBA: upper cell northward positive, abyssal cell
    southward positive. The NADW return layer and the abyssal layers are reported
    southward positive. The reanalysis series are given the same signs.
  * Tests. The correlation of anomalies uses corr_test (Bretherton et al. 1999
    effective sample size) from paper3r1_stats. A design-matched check replaces
    the constant design by the harmonic-and-trend design. Trends use
    regression_test with the bias-corrected effective sample size. The ratio of
    standard deviations uses an F interval with the effective degrees of freedom
    of each anomaly series (approximate).
  * Sensitivity. Calendar-month means in place of the harmonic fit.

Writes PAPER_3_v2/analysis/jc_n10_samba.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import gsw  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import (  # noqa: E402
    corr_test,
    corrected_phi,
    lag1,
    regression_test,
    trend_design,
)

RESULTS = REPO / "data" / "results"
EXTERNAL = REPO / "data" / "external"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n10_samba.json"

SAMBA_MONTHLY = EXTERNAL / "samba_moc_monthly.nc"
SAMBA_DAILY = EXTERNAL / "samba_moc_daily.nc"
PRODUCTS = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}
SAMBA_LAT = -34.5
SAMBA_PRESSURES = (1315.0, 3155.0, 4295.0)  # dbar: upper, NADW return and abyssal limits
LONG_WINDOW = (1993, 2017)

PAIRS = [
    ("up_cell", "samba_up", "upper cell psi_max (cell definition) vs SAMBA upper cell"),
    ("up_layer", "samba_up", "upper layer 0 to 1315 dbar (V_bc) vs SAMBA upper cell"),
    ("up_layer_vint", "samba_up", "upper layer 0 to 1315 dbar (V_int) vs SAMBA upper cell"),
    ("tlimb", "samba_up", "northward-limb transport T_limb vs SAMBA upper cell"),
    ("ab_cell", "samba_ab", "abyssal cell -psi_min (cell definition) vs SAMBA abyssal cell"),
    ("ab_layer", "samba_ab", "abyssal layer 3155 to 4295 dbar (V_bc) vs SAMBA abyssal cell"),
    ("ab_layer_vint", "samba_ab", "abyssal layer 3155 to 4295 dbar (V_int) vs SAMBA abyssal cell"),
]
REANALYSIS_SERIES = [
    "up_cell",
    "ab_cell",
    "tlimb",
    "up_layer",
    "nadw_layer",
    "ab_layer",
    "up_layer_vint",
    "nadw_layer_vint",
    "ab_layer_vint",
]


# ── time and layer helpers ────────────────────────────────────────────


def month_keys(times) -> list[str]:
    return list(pd.DatetimeIndex(times).strftime("%Y-%m"))


def decimal_year(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return np.asarray(t.year + (t.month - 1) / 12.0, dtype=float)


def layer_weights(e3t: np.ndarray) -> tuple[np.ndarray, ...]:
    """Thickness of each level inside the upper, NADW return, abyssal and bottom layers (m).

    The bottom weight covers the column below the abyssal limit, so that the four
    weights partition the full column and their sum is e3t.
    """
    zw = np.concatenate([[0.0], np.cumsum(e3t)])
    d1, d2, d3 = (-float(gsw.z_from_p(p, SAMBA_LAT)) for p in SAMBA_PRESSURES)

    def overlap(z_top: float, z_bot: float) -> np.ndarray:
        return np.clip(np.minimum(zw[1:], z_bot) - np.maximum(zw[:-1], z_top), 0, None)

    return (
        overlap(0.0, d1),
        overlap(d1, d2),
        overlap(d2, d3),
        overlap(d3, zw[-1] + 1.0),
    )


def layer_series(sec: xr.Dataset, vname: str) -> dict[str, np.ndarray]:
    """Monthly layer transports in Sv, with the SAMBA sign convention.

    upper: northward positive. nadw_layer and abyssal: southward positive.
    bottom_north: northward transport below the abyssal limit (closure check only).
    The integration is the one of analysis_paper3r1_cells.py (V times the overlap
    of each level with the layer, summed over depth).
    """
    e3t = sec["e3t"].values
    w_up, w_nadw, w_ab, w_bot = layer_weights(e3t)
    v = sec[vname].values
    up = np.nansum(v * w_up, axis=1) / 1e6
    nadw_north = np.nansum(v * w_nadw, axis=1) / 1e6
    ab_north = np.nansum(v * w_ab, axis=1) / 1e6
    bot_north = np.nansum(v * w_bot, axis=1) / 1e6
    return {
        "up_layer": up,
        "nadw_layer": -nadw_north,
        "ab_layer": -ab_north,
        "bottom_north": bot_north,
    }


# ── series ────────────────────────────────────────────────────────────


def product_frame(key: str) -> pd.DataFrame:
    """Full monthly record of one reanalysis on the cell and layer definitions."""
    cells = xr.open_dataset(RESULTS / f"paper3r1_cells_{key}.nc")
    sec = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
    keys = month_keys(cells["time"].values)
    if keys != month_keys(sec["time"].values):
        raise RuntimeError(f"time axes of {key} cells and section differ")
    bc = layer_series(sec, "V_bc")
    vi = layer_series(sec, "V_int")
    df = pd.DataFrame(
        {
            "t_dec": decimal_year(cells["time"].values),
            "year": pd.DatetimeIndex(cells["time"].values).year,
            "month": pd.DatetimeIndex(cells["time"].values).month,
            "up_cell": cells["psi_max"].values,
            "ab_cell": -cells["psi_min"].values,
            "tlimb": cells["T_limb"].values,
            "up_layer": bc["up_layer"],
            "nadw_layer": bc["nadw_layer"],
            "ab_layer": bc["ab_layer"],
            "up_layer_vint": vi["up_layer"],
            "nadw_layer_vint": vi["nadw_layer"],
            "ab_layer_vint": vi["ab_layer"],
            "bottom_north": bc["bottom_north"],
            "z_psimax_m": cells["z_psimax"].values,
            "z0_m": cells["z0"].values,
        },
        index=keys,
    )
    return df


def samba_frame() -> pd.DataFrame:
    sb = xr.open_dataset(SAMBA_MONTHLY)
    keys = month_keys(sb["time"].values)
    return pd.DataFrame(
        {
            "t_dec": decimal_year(sb["time"].values),
            "year": pd.DatetimeIndex(sb["time"].values).year,
            "month": pd.DatetimeIndex(sb["time"].values).month,
            "samba_up": sb["upper_cell"].values.astype(float),
            "samba_ab": sb["abyssal_cell"].values.astype(float),
        },
        index=keys,
    )


# ── statistics ────────────────────────────────────────────────────────


def harmonic_design(t_dec: np.ndarray) -> np.ndarray:
    """Constant, linear trend, annual and semi-annual harmonics."""
    t = np.asarray(t_dec, dtype=float)
    w = 2.0 * np.pi * (t - np.floor(t))
    return np.column_stack(
        [
            np.ones_like(t),
            t - t.mean(),
            np.cos(w),
            np.sin(w),
            np.cos(2 * w),
            np.sin(2 * w),
        ]
    )


def fit_anomalies(t_dec: np.ndarray, y: np.ndarray) -> dict:
    """Anomalies (seasonal cycle and trend removed) and the trend-kept deseasonalised series.

    The standard deviation of the anomalies uses the n - k residual degrees of
    freedom. The effective sample size follows the bias-corrected lag-1
    coefficient of the residuals on the same design (as in paper3r1_stats).
    """
    x = harmonic_design(t_dec)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    seasonal = x[:, 2:] @ coef[2:]
    r1 = float(lag1(resid))
    phi = float(corrected_phi(r1, x))
    neff_raw = n * (1.0 - phi) / (1.0 + phi)
    neff = float(min(max(neff_raw, k + 1.0), n))
    return {
        "resid": resid,
        "deseasonalised": y - seasonal,
        "std": float(np.sqrt(resid @ resid / (n - k))),
        "lag1_resid": r1,
        "phi": phi,
        "n_eff": neff,
        "n_eff_at_floor": bool(neff_raw < k + 1.0),
        "df": neff - k,
        "n": int(n),
        "k": int(k),
    }


def trend_block(t_dec: np.ndarray, y_deseasonalised: np.ndarray) -> dict:
    """Linear trend of the deseasonalised series, per decade, with the paper's test."""
    res = regression_test(trend_design(t_dec), y_deseasonalised, 1)
    return {
        "per_decade": 10.0 * res["value"],
        "ci_per_decade": [10.0 * res["ci"][0], 10.0 * res["ci"][1]],
        "half_width_per_decade": 10.0 * res["half_width"],
        "p": res["p"],
        "n": res["n"],
        "n_eff": res["n_eff"],
        "phi_corrected": res["phi_corrected"],
        "units": "Sv per decade",
    }


def pair_block(t_dec: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Anomaly statistics and correlation of two series over the same months."""
    fx = fit_anomalies(t_dec, x)
    fy = fit_anomalies(t_dec, y)
    ct = corr_test(fx["resid"], fy["resid"], detrend=False)
    r = float(ct["r"])
    z = float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    se = 1.0 / np.sqrt(max(ct["n_eff"] - 3.0, 1e-9))
    ratio = fx["std"] / fy["std"]
    var_ratio = ratio**2
    f_lo = float(stats.f.ppf(0.025, fx["df"], fy["df"]))
    f_hi = float(stats.f.ppf(0.975, fx["df"], fy["df"]))
    ci_ratio = [float(np.sqrt(var_ratio / f_hi)), float(np.sqrt(var_ratio / f_lo))]
    # Design-matched check of the same correlation: harmonic-and-trend design.
    neff_m = float(
        min(
            max(
                fx["n"] * (1 - fx["phi"] * fy["phi"]) / (1 + fx["phi"] * fy["phi"]),
                3.0,
            ),
            fx["n"],
        )
    )
    r_m = float(np.corrcoef(fx["resid"], fy["resid"])[0, 1])
    t_m = r_m * np.sqrt((neff_m - 2) / max(1 - r_m * r_m, 1e-12))
    return {
        "n": fx["n"],
        "std_x": fx["std"],
        "std_y": fy["std"],
        "std_ratio_x_over_y": float(ratio),
        "std_ratio_ci_95": ci_ratio,
        "r": r,
        "r_ci_95_fisher_approx": [
            float(np.tanh(z - 1.96 * se)),
            float(np.tanh(z + 1.96 * se)),
        ],
        "p": float(ct["p"]),
        "n_eff": float(ct["n_eff"]),
        "phi_x_corr_test_design": float(ct["phi_a"]),
        "phi_y_corr_test_design": float(ct["phi_b"]),
        "lag1_x_resid": fx["lag1_resid"],
        "lag1_y_resid": fy["lag1_resid"],
        "n_eff_x_harmonic_design": fx["n_eff"],
        "n_eff_y_harmonic_design": fy["n_eff"],
        "design_matched_r": r_m,
        "design_matched_n_eff": neff_m,
        "design_matched_p": float(2 * stats.t.sf(abs(t_m), neff_m - 2)),
    }


def calendar_block(keys_months: np.ndarray, t_dec: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Sensitivity: calendar-month means removed, then the linear trend removed."""

    def anomaly(v: np.ndarray) -> np.ndarray:
        clim = np.array([v[keys_months == m].mean() for m in range(1, 13) if np.any(keys_months == m)])
        months = [m for m in range(1, 13) if np.any(keys_months == m)]
        a = v.copy()
        for m, c in zip(months, clim):
            a[keys_months == m] = v[keys_months == m] - c
        x_tr = trend_design(t_dec)
        coef, *_ = np.linalg.lstsq(x_tr, a, rcond=None)
        return a - x_tr @ coef

    ax = anomaly(np.asarray(x, dtype=float))
    ay = anomaly(np.asarray(y, dtype=float))
    ct = corr_test(ax, ay, detrend=False)
    n = len(ax)
    return {
        "std_x": float(np.sqrt(ax @ ax / (n - 2))),
        "std_y": float(np.sqrt(ay @ ay / (n - 2))),
        "std_ratio_x_over_y": float(np.sqrt(ax @ ax / (n - 2)) / np.sqrt(ay @ ay / (n - 2))),
        "r": float(ct["r"]),
        "n_eff": float(ct["n_eff"]),
        "p": float(ct["p"]),
    }


def reanalysis_only_block(df_window: pd.DataFrame, series: str) -> dict:
    """Mean, anomaly standard deviation and trend of one reanalysis series."""
    t = df_window["t_dec"].values
    y = df_window[series].values.astype(float)
    fit = fit_anomalies(t, y)
    return {
        "raw_mean_Sv": float(y.mean()),
        "raw_std_Sv": float(y.std(ddof=1)),
        "anomaly_std_Sv": fit["std"],
        "lag1_anomaly": fit["lag1_resid"],
        "n_eff_anomaly": fit["n_eff"],
        "trend_deseasonalised": trend_block(t, fit["deseasonalised"]),
        "n": int(len(y)),
    }


def annual_trend_block(df_window: pd.DataFrame, series: str) -> dict:
    """Trend of the annual means over complete years, with the paper's trend test.

    Annual means average out the seasonal cycle. The lag-1 correction of the
    monthly test is generous for a slow decline, so the annual test is reported
    alongside the monthly one.
    """
    ann = df_window.groupby("year").agg(
        value=(series, "mean"), n_months=(series, "size"), t=("t_dec", "mean")
    )
    ann = ann[ann["n_months"] == 12]
    years = ann.index.values.astype(float)
    res = regression_test(trend_design(years), ann["value"].values.astype(float), 1)
    return {
        "per_decade": 10.0 * res["value"],
        "ci_per_decade": [10.0 * res["ci"][0], 10.0 * res["ci"][1]],
        "half_width_per_decade": 10.0 * res["half_width"],
        "p": res["p"],
        "n": res["n"],
        "n_eff": res["n_eff"],
        "units": "Sv per decade",
        "years": [int(ann.index.min()), int(ann.index.max())],
    }


# ── main ──────────────────────────────────────────────────────────────


def check_daily() -> dict:
    """Consistency of the daily file with the monthly file and its own spread."""
    dd = xr.open_dataset(SAMBA_DAILY)
    daily = pd.DataFrame(
        {
            "up": dd["upper_cell"].values.astype(float),
            "ab": dd["abyssal_cell"].values.astype(float),
        },
        index=pd.DatetimeIndex(dd["time"].values),
    )
    monthly_from_daily = daily.resample("MS").mean().dropna(how="all")
    mon = xr.open_dataset(SAMBA_MONTHLY)
    mon_df = pd.DataFrame(
        {
            "up": mon["upper_cell"].values.astype(float),
            "ab": mon["abyssal_cell"].values.astype(float),
        },
        index=pd.DatetimeIndex(mon["time"].values),
    )
    joined = mon_df.join(monthly_from_daily, rsuffix="_daily", how="inner")
    return {
        "n_days": int(len(daily)),
        "daily_span": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "daily_std_upper_Sv": float(daily["up"].std(ddof=1)),
        "daily_std_abyssal_Sv": float(daily["ab"].std(ddof=1)),
        "daily_min_max_upper_Sv": [float(daily["up"].min()), float(daily["up"].max())],
        "daily_min_max_abyssal_Sv": [float(daily["ab"].min()), float(daily["ab"].max())],
        "n_months_joined": int(len(joined)),
        "max_abs_monthly_minus_daily_mean_upper_Sv": float((joined["up"] - joined["up_daily"]).abs().max()),
        "max_abs_monthly_minus_daily_mean_abyssal_Sv": float((joined["ab"] - joined["ab_daily"]).abs().max()),
        "note": "The daily file is the same 2013-2017 record; the monthly file is its monthly mean.",
    }


def main() -> None:
    samba = samba_frame()
    skeys = list(samba.index)
    t_c = samba["t_dec"].values
    m_c = samba["month"].values

    products = {key: product_frame(key) for key in PRODUCTS}
    common = {}
    for key, df in products.items():
        missing = [k for k in skeys if k not in df.index]
        if missing:
            raise RuntimeError(f"{key} lacks SAMBA months {missing[:3]}")
        common[key] = df.loc[skeys]

    samba_fit_up = fit_anomalies(t_c, samba["samba_up"].values)
    samba_fit_ab = fit_anomalies(t_c, samba["samba_ab"].values)

    result: dict = {
        "id": "N10",
        "script": "scripts/analysis_paper3jc_n10_samba.py",
        "record": {
            "samba_source": "Kersale et al. (2020), doi:10.1126/sciadv.aba7573, stored in data/external",
            "samba_monthly_span": [str(samba.index[0]), str(skeys[-1])],
            "n_months": int(len(skeys)),
            "full_record_from_2009_in_repository": False,
            "full_record_note": (
                "The repository holds only the Sep 2013 to Jul 2017 record. The NOAA FTP "
                "location and the HTTPS mirror were not reachable in this session, so the "
                "2009 to 2013 part was not obtained."
            ),
            "common_window": [skeys[0], skeys[-1]],
            "products_common_window_months": {k: int(len(v)) for k, v in common.items()},
            "reanalysis_window_long": list(LONG_WINDOW),
        },
        "conventions": {
            "units": "Sv",
            "sign": "SAMBA upper cell northward positive; abyssal cell and NADW return layer southward positive (reanalysis series given the same sign)",
            "layers_dbar": list(SAMBA_PRESSURES),
            "layer_depth_m": [
                float(-gsw.z_from_p(p, SAMBA_LAT)) for p in SAMBA_PRESSURES
            ],
            "anomaly": "least squares on constant, linear trend, annual and semi-annual harmonics over the window",
            "std": "residual standard deviation with n - k degrees of freedom",
            "std_ratio_ci": "approximate, F interval with the effective degrees of freedom of each anomaly series",
            "correlation_test": "paper3r1_stats.corr_test with detrend=False on the anomalies (Bretherton et al. 1999 n_eff); design-matched check reported",
        },
        "daily_check": check_daily(),
        "samba_series": {
            "upper": {
                "raw_mean_stored_Sv": float(samba["samba_up"].mean()),
                "raw_std_Sv": float(samba["samba_up"].std(ddof=1)),
                "anomaly_std_Sv": samba_fit_up["std"],
                "lag1_anomaly": samba_fit_up["lag1_resid"],
                "n_eff_anomaly": samba_fit_up["n_eff"],
                "trend_deseasonalised": trend_block(t_c, samba_fit_up["deseasonalised"]),
            },
            "abyssal": {
                "raw_mean_stored_Sv": float(samba["samba_ab"].mean()),
                "raw_std_Sv": float(samba["samba_ab"].std(ddof=1)),
                "anomaly_std_Sv": samba_fit_ab["std"],
                "lag1_anomaly": samba_fit_ab["lag1_resid"],
                "n_eff_anomaly": samba_fit_ab["n_eff"],
                "trend_deseasonalised": trend_block(t_c, samba_fit_ab["deseasonalised"]),
            },
            "calendar_month_counts": {int(m): int(np.sum(m_c == m)) for m in range(1, 13) if np.any(m_c == m)},
        },
        "pairs": {},
        "reanalysis_only_common_window": {},
        "reanalysis_long_window": {},
        "calendar_month_sensitivity": {},
        "diagnostics": {},
    }

    samba_series = {"samba_up": samba["samba_up"].values, "samba_ab": samba["samba_ab"].values}
    for key, cdf in common.items():
        pairs_out = {}
        for xname, yname, label in PAIRS:
            x = cdf[xname].values.astype(float)
            y = samba_series[yname]
            block = pair_block(t_c, x, y)
            block["label"] = label
            block["raw_mean_reanalysis_Sv"] = float(x.mean())
            block["raw_mean_samba_stored_Sv"] = float(y.mean())
            block["calendar_month_sensitivity"] = calendar_block(m_c, t_c, x, y)
            pairs_out[f"{xname}__{yname}"] = block
        result["pairs"][PRODUCTS[key]] = pairs_out

        result["reanalysis_only_common_window"][PRODUCTS[key]] = {
            s: reanalysis_only_block(cdf, s) for s in REANALYSIS_SERIES
        }
        result["reanalysis_only_common_window"][PRODUCTS[key]]["nadw_layer_label"] = (
            "NADW return layer 1315 to 3155 dbar, southward positive; not in the SAMBA file"
        )

        full = products[key]
        window = full[(full["year"] >= LONG_WINDOW[0]) & (full["year"] <= LONG_WINDOW[1])]
        long_block = {
            s: reanalysis_only_block(window, s)
            for s in ["up_cell", "ab_cell", "tlimb", "up_layer", "nadw_layer", "ab_layer"]
        }
        for s in ["up_cell", "ab_cell", "tlimb", "up_layer", "nadw_layer", "ab_layer"]:
            long_block[s]["trend_annual_means"] = annual_trend_block(window, s)
        result["reanalysis_long_window"][PRODUCTS[key]] = long_block

        closure = (cdf["up_layer"] - cdf["nadw_layer"] - cdf["ab_layer"] + cdf["bottom_north"]).values
        diff_cell_layer = (cdf["up_cell"] - cdf["up_layer"]).values
        result["diagnostics"][PRODUCTS[key]] = {
            "cell_upper_vs_layer_upper": pair_block(t_c, cdf["up_cell"].values, cdf["up_layer"].values),
            "cell_abyssal_vs_layer_abyssal": pair_block(t_c, cdf["ab_cell"].values, cdf["ab_layer"].values),
            "upper_cell_vs_limb_transport": pair_block(t_c, cdf["up_cell"].values, cdf["tlimb"].values),
            "upper_cell_minus_upper_layer_mean_Sv": float(diff_cell_layer.mean()),
            "upper_cell_minus_upper_layer_max_abs_Sv": float(np.abs(diff_cell_layer).max()),
            "layer_closure_max_abs_Sv": float(np.abs(closure).max()),
            "bottom_layer_below_abyssal_limit_mean_northward_Sv": float(cdf["bottom_north"].mean()),
            "psi_max_depth_m": {
                "min": float(cdf["z_psimax_m"].min()),
                "mean": float(cdf["z_psimax_m"].mean()),
                "max": float(cdf["z_psimax_m"].max()),
            },
            "zero_crossing_depth_m": {
                "min": float(cdf["z0_m"].min()),
                "mean": float(cdf["z0_m"].mean()),
                "max": float(cdf["z0_m"].max()),
            },
        }

    result["calendar_month_sensitivity"] = {
        "note": "Calendar-month means in place of the harmonic fit; see pairs[*].calendar_month_sensitivity",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(result), indent=1))
    print(f"wrote {OUT}")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    return o


if __name__ == "__main__":
    main()
