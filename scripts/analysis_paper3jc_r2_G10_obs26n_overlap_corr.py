#!/usr/bin/env python3
"""Overlap means and offsets of the 26.5 N overturning, and monthly correlations with RAPID.

Action A107 (cluster C55), and the numbers of Fig. S5 (FigS_rapid).

Part 1, annual overlaps. For each product X in GLORYS12V1, ECCO-V4r4, SODA 3.15.2
and RAPID, over the years common to ORAS5 and X:
  * the ORAS5 and X means over the overlap;
  * the offset as used in scripts/plot_paper3r1_si_figures.py, offset = mean(ORAS5)
    minus mean(X) over the overlap, so that X + offset matches ORAS5 on average;
  * the difference X minus ORAS5 (the negative of the offset).
RAPID annual means are the means of the monthly moc_mar_hc10 of complete years
(as in the figure script). The annual npz of RAPID (data/results/rapid_amoc26n.npz)
is reported as a cross-check. The ORAS5 1958 to 2009 reference mean (the anomaly
baseline of Fig. S4) and the ORAS5 full-record mean are also given.

Part 2, monthly correlations with RAPID over the 240 months 2004-04 to 2024-03
(data/external/rapid_moc_monthly.nc). ORAS5 (data/results/oras5_moc_26N.nc,
moc_upper) and GLORYS12V1 (data/results/glorys12_moc_26N.nc, moc_upper) are
restricted to the same months. Each series is deseasonalised by subtracting its
calendar-month means over the 240 months (primary) or over its full record (ORAS5
sensitivity). Correlations use paper3r1_stats.corr_test, with the Bretherton et al.
(1999) effective sample size. Primary: detrend=False (removes the mean only), on
raw and deseasonalised series. Sensitivity: detrend=True (removes a linear trend too).
The raw correlations reproduce the values printed in Fig. S5 (0.75 and 0.71).

No random draws are made. Writes PAPER_3_v2/analysis/jc_r2_G10_obs26n_overlap_corr.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import corr_test, lag1, mean_test  # noqa: E402

RESULTS = REPO / "data" / "results"
EXTERNAL = REPO / "data" / "external"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G10_obs26n_overlap_corr.json"


def load_yearly(key: str) -> pd.Series:
    z = np.load(RESULTS / f"yearly_amoc26n_{key}.npz")
    return pd.Series(z["amoc"].astype(float), index=z["years"].astype(int))


def deseasonalise(s: pd.Series, clim_source: pd.Series | None = None) -> pd.Series:
    src = s if clim_source is None else clim_source
    clim = src.groupby(src.index.month).mean()
    return s - clim.reindex(s.index.month).values


def main() -> None:
    oras5 = load_yearly("oras5")
    glorys = load_yearly("glorys12")
    ecco = load_yearly("ecco")
    soda = load_yearly("soda")

    rp = xr.open_dataset(EXTERNAL / "rapid_moc_monthly.nc")["moc_mar_hc10"].to_series()
    rp.index = pd.DatetimeIndex(rp.index)
    cnt = rp.groupby(rp.index.year).count()
    rapid_annual = rp.groupby(rp.index.year).mean()[cnt == 12]
    rz = np.load(RESULTS / "rapid_amoc26n.npz")
    rapid_npz = pd.Series(rz["amoc"].astype(float), index=rz["years"].astype(int))

    def overlap_block(name: str, s: pd.Series, source: str, window_note: str) -> dict:
        ov = s.index.intersection(oras5.index)
        mo = float(oras5.loc[ov].mean())
        ms = float(s.loc[ov].mean())
        # Interval of the offset: mean of the paired difference, with the corrected N_eff of paper3r1_stats.
        diff_test = mean_test((oras5.loc[ov] - s.loc[ov]).values)
        return {
            "offset_ci95_Sv": [float(c) for c in diff_test["ci"]],
            "offset_n_eff": float(diff_test["n_eff"]),
            "offset_n_eff_at_floor": bool(diff_test["n_eff_at_floor"]),
            "product": name,
            "source": source,
            "overlap_years": [int(ov.min()), int(ov.max())],
            "n_years": int(len(ov)),
            "mean_ORAS5_over_overlap_Sv": mo,
            "mean_product_over_overlap_Sv": ms,
            "offset_ORAS5_minus_product_Sv": mo - ms,
            "difference_product_minus_ORAS5_Sv": ms - mo,
            "window_note": window_note,
        }

    overlaps = {
        "GLORYS12V1": overlap_block(
            "GLORYS12V1", glorys, "data/results/yearly_amoc26n_glorys12.npz", "1993 to 2025"
        ),
        "ECCO-V4r4": overlap_block(
            "ECCO-V4r4", ecco, "data/results/yearly_amoc26n_ecco.npz", "1992 to 2017"
        ),
        "SODA 3.15.2": overlap_block(
            "SODA 3.15.2", soda, "data/results/yearly_amoc26n_soda.npz", "1980 to 2020"
        ),
        "RAPID": overlap_block(
            "RAPID",
            rapid_annual,
            "data/external/rapid_moc_monthly.nc (annual means of complete years)",
            "2005 to 2023",
        ),
    }
    overlaps["RAPID"]["cross_check_npz_mean_over_overlap_Sv"] = float(
        rapid_npz.loc[overlaps["RAPID"]["overlap_years"][0] : overlaps["RAPID"]["overlap_years"][1]].mean()
    )
    overlaps["RAPID"]["cross_check_npz_offset_ORAS5_minus_product_Sv"] = (
        overlaps["RAPID"]["mean_ORAS5_over_overlap_Sv"]
        - overlaps["RAPID"]["cross_check_npz_mean_over_overlap_Sv"]
    )

    ref_1958_2009 = float(oras5.loc[1958:2009].mean())
    oras5_full = float(oras5.mean())

    # Monthly correlations with RAPID
    rapid_m = rp.copy()
    t0, t1 = rapid_m.index[0], rapid_m.index[-1]
    oras5_m = xr.open_dataset(RESULTS / "oras5_moc_26N.nc")["moc_upper"].to_series()
    oras5_m.index = pd.DatetimeIndex(oras5_m.index)
    glorys_m = xr.open_dataset(RESULTS / "glorys12_moc_26N.nc")["moc_upper"].to_series()
    glorys_m.index = pd.DatetimeIndex(glorys_m.index)

    # align on calendar months; RAPID carries the 1st of the month, reanalyses mid-month
    def to_month(s: pd.Series) -> pd.Series:
        out = s.copy()
        out.index = pd.DatetimeIndex(out.index).to_period("M").to_timestamp()
        return out

    rap = to_month(rapid_m)
    orm = to_month(oras5_m)
    glm = to_month(glorys_m)
    months = rap.index[(rap.index >= t0) & (rap.index <= t1)]
    rap_w = rap.reindex(months)
    or_w = orm.reindex(months)
    gl_w = glm.reindex(months)
    assert not (rap_w.isna().any() or or_w.isna().any() or gl_w.isna().any())

    or_des = deseasonalise(or_w)
    gl_des = deseasonalise(gl_w)
    rap_des = deseasonalise(rap_w)
    or_des_full = deseasonalise(or_w, clim_source=orm)
    gl_des_full = deseasonalise(gl_w, clim_source=glm)
    rap_des_full = deseasonalise(rap_w, clim_source=rap)

    def pack(res: dict) -> dict:
        return {k: (float(v) if not isinstance(v, bool) else v) for k, v in res.items()}

    def cor(a, b, detrend: bool) -> dict:
        out = pack(corr_test(a.values, b.values, detrend=detrend))
        # 95% interval of r by the Fisher transform with the effective sample size n_eff.
        z = np.arctanh(out["r"])
        half = 1.959964 / np.sqrt(out["n_eff"] - 3.0)
        out["r_ci95"] = [float(np.tanh(z - half)), float(np.tanh(z + half))]
        return out

    monthly = {
        "months": [str(months[0].date()), str(months[-1].date()), int(len(months))],
        "primary_detrend_false": {
            "ORAS5_vs_RAPID_raw": cor(or_w, rap_w, False),
            "GLORYS12V1_vs_RAPID_raw": cor(gl_w, rap_w, False),
            "ORAS5_vs_RAPID_deseasonalised": cor(or_des, rap_des, False),
            "GLORYS12V1_vs_RAPID_deseasonalised": cor(gl_des, rap_des, False),
        },
        "sensitivity_detrend_true": {
            "ORAS5_vs_RAPID_raw": cor(or_w, rap_w, True),
            "GLORYS12V1_vs_RAPID_raw": cor(gl_w, rap_w, True),
            "ORAS5_vs_RAPID_deseasonalised": cor(or_des, rap_des, True),
            "GLORYS12V1_vs_RAPID_deseasonalised": cor(gl_des, rap_des, True),
        },
        "sensitivity_climatology_full_record": {
            "note": "Climatology of each reanalysis over its full record; RAPID over its 240 months.",
            "ORAS5_vs_RAPID_deseasonalised": cor(or_des_full, rap_des_full, False),
            "GLORYS12V1_vs_RAPID_deseasonalised": cor(gl_des_full, rap_des_full, False),
        },
        "lag1_of_deseasonalised_series": {
            "RAPID": float(lag1(rap_des.values)),
            "ORAS5": float(lag1(or_des.values)),
            "GLORYS12V1": float(lag1(gl_des.values)),
        },
        "seasonal_amplitude_Sv": {
            "RAPID_std_of_climatology": float(rap_w.groupby(rap_w.index.month).mean().std()),
            "ORAS5_std_of_climatology": float(or_w.groupby(or_w.index.month).mean().std()),
            "GLORYS12V1_std_of_climatology": float(gl_w.groupby(gl_w.index.month).mean().std()),
        },
    }

    fig_rapid = {
        "figure": "FigS_rapid (Fig. S5 panels b and c)",
        "printed_in_caption": {"ORAS5_vs_RAPID": 0.75, "GLORYS12V1_vs_RAPID": 0.71},
        "reproduced_raw_r": {
            "ORAS5_vs_RAPID": monthly["primary_detrend_false"]["ORAS5_vs_RAPID_raw"]["r"],
            "GLORYS12V1_vs_RAPID": monthly["primary_detrend_false"]["GLORYS12V1_vs_RAPID_raw"]["r"],
        },
        "deseasonalised_r": {
            "ORAS5_vs_RAPID": monthly["primary_detrend_false"]["ORAS5_vs_RAPID_deseasonalised"]["r"],
            "GLORYS12V1_vs_RAPID": monthly["primary_detrend_false"]["GLORYS12V1_vs_RAPID_deseasonalised"]["r"],
        },
        "n_months": int(len(months)),
        "n_eff_raw": {
            "ORAS5_vs_RAPID": monthly["primary_detrend_false"]["ORAS5_vs_RAPID_raw"]["n_eff"],
            "GLORYS12V1_vs_RAPID": monthly["primary_detrend_false"]["GLORYS12V1_vs_RAPID_raw"]["n_eff"],
        },
        "n_eff_deseasonalised": {
            "ORAS5_vs_RAPID": monthly["primary_detrend_false"]["ORAS5_vs_RAPID_deseasonalised"]["n_eff"],
            "GLORYS12V1_vs_RAPID": monthly["primary_detrend_false"]["GLORYS12V1_vs_RAPID_deseasonalised"]["n_eff"],
        },
    }

    result = {
        "meta": {
            "id": "jc_r2_G10_obs26n_overlap_corr",
            "group": "G10_obs26n",
            "actions": ["A107"],
            "cluster": "C55",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G10_obs26n_overlap_corr.py",
            "inputs": [
                "data/results/yearly_amoc26n_{oras5,glorys12,ecco,soda}.npz",
                "data/results/rapid_amoc26n.npz (cross-check)",
                "data/external/rapid_moc_monthly.nc (moc_mar_hc10)",
                "data/results/oras5_moc_26N.nc (moc_upper)",
                "data/results/glorys12_moc_26N.nc (moc_upper)",
            ],
            "seeds": "none (no random draws)",
            "replicates": "none (analytic effective sample sizes, Bretherton et al. 1999)",
            "units": "Sv; correlations dimensionless; n_eff in months or years as stated",
            "offset_convention": "offset = mean(ORAS5) - mean(product) over the overlap (as in the figure script)",
        },
        "annual_overlaps": {
            "products": overlaps,
            "ORAS5_reference_1958_2009_mean_Sv": ref_1958_2009,
            "ORAS5_full_record_1958_2025_mean_Sv": oras5_full,
            "ORAS5_mean_1993_2025_Sv": float(oras5.loc[1993:2025].mean()),
            "ORAS5_mean_2005_2023_Sv": float(oras5.loc[2005:2023].mean()),
        },
        "monthly_correlations_with_RAPID": monthly,
        "FigS_rapid": fig_rapid,
    }

    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
