"""Effective number of epoch configurations and start-by-end trend matrices (group G07, action A62).

1. Lag-one autocorrelation of the annual ORAS5 series in attribution.json
   (oras5.years, F_ov_Sv, T_limb_Sv, dS_limb_PSU): raw lag-one coefficient of
   the series, and the bias-corrected coefficient of the residuals of a linear
   trend fit (scripts/paper3r1_stats.py). The effective sample size of the
   record is N_eff = N (1 - phi) / (1 + phi), bounded to [3, N]. The number of
   independent epoch configurations quoted in the text is taken as the
   effective sample size of the FovS residual series, rounded to the nearest
   integer; the other series are reported for comparison.

2. Re-evaluation of the epoch census of attribution.json (oras5.census) from
   the stored annual series with the census() function of
   scripts/analysis_paper3v2_attribution.py: pairs of non-overlapping early and
   late epochs of 10, 13 and 15 years, admissible when |total change| > 5 mSv.
   Reports the number of pairs, the usable pairs, the share of usable pairs in
   which the transport term opposes the total, and the share with a transport
   share below one quarter.

3. Start-by-end matrices of the linear trend over the stored annual series:
   - ORAS5 FovS (attribution.json oras5.F_ov_Sv), start years 1958 to 2016 and
     end years up to 2025, minimum length 10 years, trend in mSv per year;
   - EN4 limb salinity contrast (en4.json, oras5.dS and glorys12.dS series,
     2005-2024), start years 2005 to 2015 and end years 2014 to 2024, minimum
     length 10 years, trend in PSU per decade.
   Each cell gives the trend, its 95% interval, the p value and the effective
   sample size of the bias-corrected N_eff test (paper3r1_stats.trend_test).

No random numbers are drawn; the matrices are deterministic.

Inputs: PAPER_3_v2/analysis/attribution.json, PAPER_3_v2/analysis/en4.json.
Outputs: PAPER_3_v2/analysis/jc_r2_G07_frozen_epochs_census.json

Run: python3 scripts/analysis_paper3jc_r2_G07_frozen_epochs_census.py
     (from the worktree root, with OMP_NUM_THREADS=4 and related variables set to 4)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_attribution import census  # noqa: E402
from paper3r1_stats import corrected_phi, lag1, trend_design, trend_test  # noqa: E402

ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G07_frozen_epochs_census.json"
MIN_LEN = 10


def nan_to_none(v):
    return None if (v is None or not np.isfinite(v)) else float(v)


def lag1_block(years: np.ndarray, series: np.ndarray) -> dict:
    x = trend_design(years)
    coef, *_ = np.linalg.lstsq(x, series, rcond=None)
    resid = series - x @ coef
    r_raw = float(lag1(series - series.mean()))
    r_res = float(lag1(resid))
    phi = float(corrected_phi(r_res, x))
    n = len(series)
    neff_raw = n * (1 - phi) / (1 + phi)
    neff = float(min(max(neff_raw, 3.0), n))
    neff_naive = float(n * (1 - r_res) / (1 + r_res))
    return {
        "n": int(n),
        "lag1_raw_series": r_raw,
        "lag1_residuals_uncorrected": r_res,
        "phi_corrected": phi,
        "n_eff_corrected": neff,
        "n_eff_corrected_rounded": int(round(neff)),
        "n_eff_uncorrected_santer": neff_naive,
    }


def matrix(years: np.ndarray, series: np.ndarray, starts, ends, scale: float) -> dict:
    trend = np.full((len(starts), len(ends)), np.nan)
    lo = trend.copy()
    hi = trend.copy()
    pv = trend.copy()
    neff = trend.copy()
    for i, s in enumerate(starts):
        for j, e in enumerate(ends):
            if e - s + 1 < MIN_LEN:
                continue
            m = (years >= s) & (years <= e)
            r = trend_test(years[m].astype(float), series[m])
            trend[i, j] = r["value"] * scale
            lo[i, j] = r["ci"][0] * scale
            hi[i, j] = r["ci"][1] * scale
            pv[i, j] = r["p"]
            neff[i, j] = r["n_eff"]

    def tolist(a):
        return [[nan_to_none(v) for v in row] for row in a]

    return {
        "starts": [int(s) for s in starts],
        "ends": [int(e) for e in ends],
        "trend": tolist(trend),
        "ci_95_low": tolist(lo),
        "ci_95_high": tolist(hi),
        "p_value": tolist(pv),
        "n_eff": tolist(neff),
        "min_length_years": MIN_LEN,
    }


def main() -> None:
    attr = json.loads((ANALYSIS / "attribution.json").read_text())["oras5"]
    years = np.array(attr["years"], dtype=float)
    fov = np.array(attr["F_ov_Sv"], dtype=float)
    tl = np.array(attr["T_limb_Sv"], dtype=float)
    dsl = np.array(attr["dS_limb_PSU"], dtype=float)

    out: dict = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G07_frozen_epochs_census.py",
            "group": "G07_frozen_epochs",
            "action": "A62",
            "inputs": [
                "PAPER_3_v2/analysis/attribution.json (oras5.years, F_ov_Sv, T_limb_Sv, dS_limb_PSU, census)",
                "PAPER_3_v2/analysis/en4.json (oras5.dS.series, glorys12.dS.series, years)",
            ],
            "seed": None,
            "replicates": None,
            "units": "F_ov in Sv; trends of FovS in mSv per year; EN4 contrast trends in PSU per decade; n_eff counts annual values",
            "trend_test": "scripts/paper3r1_stats.py trend_test (bias-corrected N_eff, t intervals)",
            "epoch_lengths": [10, 13, 15],
            "admissibility_threshold_mSv": 5.0,
        }
    }

    out["lag1_oras5_annual"] = {
        "record": [int(years[0]), int(years[-1])],
        "F_ov_Sv": lag1_block(years, fov),
        "T_limb_Sv": lag1_block(years, tl),
        "dS_limb_PSU": lag1_block(years, dsl),
    }
    n_fov = out["lag1_oras5_annual"]["F_ov_Sv"]["n_eff_corrected_rounded"]
    out["epoch_configurations"] = {
        "definition": "effective sample size of the FovS annual residual series, bias-corrected, rounded",
        "N": n_fov,
        "alternatives": {
            "T_limb_Sv": out["lag1_oras5_annual"]["T_limb_Sv"]["n_eff_corrected_rounded"],
            "dS_limb_PSU": out["lag1_oras5_annual"]["dS_limb_PSU"]["n_eff_corrected_rounded"],
            "F_ov_Sv_uncorrected_santer": int(round(out["lag1_oras5_annual"]["F_ov_Sv"]["n_eff_uncorrected_santer"])),
        },
    }

    # Census re-evaluation from the stored annual series.
    c = census(years.astype(int), tl, dsl)
    stored = attr["census"]
    out["census"] = {
        "recomputed": c,
        "stored": stored,
        "matches_stored": bool(
            c["n_pairs"] == stored["n_pairs"]
            and c["n_pairs_usable"] == stored["n_pairs_usable"]
            and abs(c["frac_T_term_opposes_total"] - stored["frac_T_term_opposes_total"]) < 1e-12
            and abs(c["frac_share_below_0p25"] - stored["frac_share_below_0p25"]) < 1e-12
        ),
        "percent": {
            "opposes_total": round(100 * c["frac_T_term_opposes_total"], 1),
            "share_below_0p25": round(100 * c["frac_share_below_0p25"], 1),
            "usable_pairs": c["n_pairs_usable"],
            "all_pairs": c["n_pairs"],
        },
    }

    starts_f = np.arange(1958, 2017)
    ends_f = np.arange(1967, 2026)
    out["start_by_end_fovs_mSv_per_yr"] = matrix(years, fov, starts_f, ends_f, 1e3)

    en4 = json.loads((ANALYSIS / "en4.json").read_text())
    ye = np.array(en4["years"], dtype=float)
    starts_e = np.arange(2005, 2016)
    ends_e = np.arange(2014, 2025)
    out["start_by_end_en4_contrast"] = {}
    for prod, key in (("oras5", "ORAS5 circulation"), ("glorys12", "GLORYS12V1 circulation")):
        dse = np.array(en4[prod]["dS"]["series"], dtype=float)
        out["start_by_end_en4_contrast"][prod] = {
            "label": key,
            **matrix(ye, dse, starts_e, ends_e, 10.0),
            "units": "PSU per decade",
        }

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT}")
    print("lag1:", json.dumps(out["lag1_oras5_annual"], indent=1)[:1500])
    print("N:", n_fov, out["epoch_configurations"]["alternatives"])
    print("census:", out["census"]["percent"], out["census"]["matches_stored"])


if __name__ == "__main__":
    main()
