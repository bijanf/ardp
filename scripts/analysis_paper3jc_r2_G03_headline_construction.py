#!/usr/bin/env python3
"""Headline record statistics from one vertical construction (round-2 group G03).

Computes, for ORAS5, GLORYS12V1 and ECCO-V4r4, the annual-mean limb quantities
F_ovS, T (northward-limb transport), dS (limb salinity contrast), F_az, the net
section transport removed (V_net) and the transport share of the F_ovS trend,
under two vertical constructions:

* ``bounds`` (baseline): level thicknesses are the midpoints between level
  centres with the surface at zero, i.e. the section files
  ``data/results/paper3r1_section_<product>.nc``;
* ``centres`` (variant): thickness equal to the spacing of consecutive level
  centres, ``data/results/paper3r1_section_<product>_centres.nc``.

For each product, construction and window the script gives the record mean with
a sieve interval and with the corrected-N_eff interval of
``paper3r1_stats.mean_test``; the linear trend with the bias-corrected test
(``ols_santer``; its interval is the inverse of that test) and a sieve interval;
the transport share of the F_ovS trend (continuous attribution with both factors
frozen at their window means, as in
``analysis_paper3v2_attribution.continuous``) with a joint sieve interval; the
record-mean covariance residual of Eq. (1), F_ovS minus -(1/S0) T dS of the
means; and the net section transport removed.

Sieve intervals use ``paper3r1_stats.sieve_coefficients`` with 10,000 resamples
and seed 20261104; each call creates its own generator with that seed, so the
intervals do not depend on call order.

Cross-check: headline values stored in the repository (attribution.json,
r1_common_window.json, r1_cells.json, r1_vertical_grid.json, final_checks.json,
and CANONICAL_FACTS.md sections 21 to 23 typed in below) are compared with both
constructions and labelled with the construction whose value they reproduce.

Inputs: data/results/paper3r1_section_{oras5,glorys12,ecco}[_centres].nc and the
stored JSON files read by the cross-check.
Outputs: PAPER_3_v2/analysis/jc_r2_G03_headline.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from analysis_paper3v2_attribution import annual, trend_stats  # noqa: E402
from paper3r1_stats import (  # noqa: E402
    fitted_mean_test,
    ratio_summary,
    sieve_coefficients,
    step_design,
    trend_design,
)

S0 = 35.0
SEED = 20261104
N_BOOT = 10_000
RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
OUT_NAME = "jc_r2_G03_headline.json"

PRODUCTS = {
    "oras5": ("ORAS5", "paper3r1_section_oras5.nc", "paper3r1_section_oras5_centres.nc"),
    "glorys12": ("GLORYS12V1", "paper3r1_section_glorys12.nc", "paper3r1_section_glorys12_centres.nc"),
    "ecco": ("ECCO-V4r4", "paper3r1_section_ecco.nc", "paper3r1_section_ecco_centres.nc"),
}
CONSTRUCTIONS = ("bounds", "centres")
VARS = ("F_ov", "F_az", "T_limb_Sv", "dS_limb", "V_net_Sv")


def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def sieve_interval(x: np.ndarray, y: np.ndarray, j: int, scale: float) -> list:
    """Percentile 95% interval of coefficient j from sieve draws, scaled."""
    d = sieve_coefficients(x, y[:, None], j, rng(), n_boot=N_BOOT)[:, 0] * scale
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


STEP_YEARS = {"oras5": 2015, "glorys12": 2004, "ecco": None}


def mean_block(yf: np.ndarray, f: np.ndarray, step_year: int | None) -> dict:
    """Record mean as the mean of the fitted values of the design used by
    analysis_paper3v2_attribution.mean_stats (trend, plus the observing-system
    step where it falls inside the window). The corrected-N_eff interval is
    fitted_mean_test; the sieve interval draws the same weighted coefficient
    combination from one bootstrap sample (identical seed for each column)."""
    use_step = step_year is not None and yf.min() < step_year <= yf.max()
    design = step_design(yf, step_year) if use_step else trend_design(yf)
    r = fitted_mean_test(design, f)
    w = design.mean(axis=0)
    draws = sum(
        w[j] * sieve_coefficients(design, f[:, None], j, rng(), n_boot=N_BOOT)[:, 0]
        for j in range(design.shape[1])
    )
    return {
        "value": float(f.mean()),
        "fitted_mean": float(r["mean"]),
        "design": "step" if use_step else "trend",
        "step_year": int(step_year) if use_step else None,
        "ci_bias_corrected": [float(c) for c in r["ci"]],
        "ci_sieve": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_eff": float(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
    }


def load(fname: str) -> dict:
    with xr.open_dataset(RESULTS / fname) as ds:
        ann = annual(ds)
        out = {"years": ann["year"].values.astype(int)}
        for v in VARS:
            out[v] = ann[v].values.astype(float)
        out["e3t_sum_m"] = float(ds["e3t"].values.sum())
        out["n_monthly"] = int(ds.sizes["time"])
        out["section_latitude"] = float(ds.attrs["section_latitude"])
        out["thickness_convention"] = str(ds.attrs.get("thickness_convention", ""))
    return out


def trend_block(y: np.ndarray, series: np.ndarray, scale: float) -> dict:
    ts = trend_stats(y, series, scale=scale)
    return {
        "value": ts["trend"],
        "ci_bias_corrected": ts["ci"],
        "ci_sieve": sieve_interval(trend_design(y.astype(float)), series, 1, scale),
        "p_santer": ts["p_santer"],
        "p_ols": ts["p_ols"],
        "n_eff": ts["n_eff"],
        "n_eff_at_floor": ts["n_eff_at_floor"],
        "min_detectable": ts["min_detectable_trend"],
    }


def summarise(data: dict, y0: int, y1: int, step_year: int | None) -> dict:
    years = data["years"]
    m = (years >= y0) & (years <= y1)
    y = years[m]
    yf = y.astype(float)
    f, t, d = data["F_ov"][m], data["T_limb_Sv"][m], data["dS_limb"][m]
    fa, vn = data["F_az"][m], data["V_net_Sv"][m]

    mean_f = float(f.mean())
    mb = mean_block(yf, f, step_year)
    cov_pred = -(1.0 / S0) * float(t.mean()) * float(d.mean())

    fit_f = ols_santer(yf, f)
    slope_f = fit_f["slope"]
    f_t = -(1.0 / S0) * t * float(d.mean())
    f_d = -(1.0 / S0) * float(t.mean()) * d
    fit_T_term = ols_santer(yf, f_t)
    fit_S_term = ols_santer(yf, f_d)
    share_point = abs(fit_T_term["slope"] / slope_f)
    draws = sieve_coefficients(trend_design(yf), np.column_stack([f, f_t]), 1, rng(), n_boot=N_BOOT)
    share_ratio = ratio_summary(draws[:, 1], draws[:, 0], share_point)

    fit_T = ols_santer(yf, t)
    share_fc = abs(-(1.0 / S0) * fit_T["slope"] * float(d.mean()) / slope_f)

    return {
        "window": [int(y[0]), int(y[-1])],
        "n_years": int(len(y)),
        "F_ovS_mean_Sv": mb,
        "F_ovS_trend_mSv_per_yr": trend_block(y, f, 1e3),
        "F_az_mean_Sv": float(fa.mean()),
        "F_az_trend_mSv_per_yr": trend_block(y, fa, 1e3),
        "T_limb_mean_Sv": float(t.mean()),
        "T_limb_trend_Sv_per_decade": trend_block(y, t, 10.0),
        "dS_limb_mean_PSU": float(d.mean()),
        "dS_limb_trend_PSU_per_decade": trend_block(y, d, 10.0),
        "F_ovS_pred_minus_T_dS_Sv": cov_pred,
        "covariance_residual_Eq1_Sv": mean_f - cov_pred,
        "transport_share_continuous": {
            "point": float(share_point),
            "ci_sieve": [float(c) for c in share_ratio["ci"]],
            "prob_below_0p25": float(share_ratio["prob_below_0p25"]),
            "T_term_trend_mSv_per_yr": float(fit_T_term["slope"] * 1e3),
            "T_term_p_santer": float(fit_T_term["p_santer"]),
            "S_term_trend_mSv_per_yr": float(fit_S_term["slope"] * 1e3),
        },
        "transport_share_T_term_with_mean_dS": {
            "point": float(share_fc),
            "definition": "abs(-(1/S0) * slope(T) * mean(dS) / slope(F_ovS)), as in analysis_paper3v2_final_checks.py",
        },
        "V_net_mean_Sv": float(vn.mean()),
    }


def cross_check(results: dict) -> list:
    """Compare stored headline numbers with both constructions.

    Each stored value is rounded to its printed precision; the label is the
    construction whose value rounds to it, or 'neither'.
    """
    att = json.loads((OUT_DIR / "attribution.json").read_text())
    rcw = json.loads((OUT_DIR / "r1_common_window.json").read_text())
    cel = json.loads((OUT_DIR / "r1_cells.json").read_text())
    vg = json.loads((OUT_DIR / "r1_vertical_grid.json").read_text())
    fc = json.loads((OUT_DIR / "final_checks.json").read_text())

    def get(key, con, win, field):
        v = results[key]["constructions"][con]["windows"][win][field]
        return v["value"] if isinstance(v, dict) else v

    def get_trend(key, con, win, field):
        return results[key]["constructions"][con]["windows"][win][field]["value"]

    rows = []

    def add(src, product, window, stored, decimals, fn, note=""):
        vb, vc = fn("bounds"), fn("centres")
        if round(stored, decimals) == round(vb, decimals):
            lab = "bounds"
        elif round(stored, decimals) == round(vc, decimals):
            lab = "centres"
        else:
            lab = "neither"
        rows.append(
            {
                "source": src,
                "product": product,
                "window": window,
                "stored": stored,
                "decimals": decimals,
                "bounds_value": vb,
                "centres_value": vc,
                "matches": lab,
                "note": note,
            }
        )

    for key in PRODUCTS:
        add("attribution.json trends['own record'].F_ov.trend (mSv/yr)", key, "record",
            att[key]["trends"]["own record"]["F_ov"]["trend"], 3,
            lambda c, k=key: get_trend(k, c, "record", "F_ovS_trend_mSv_per_yr"))
        add("attribution.json mean_F_ov_Sv", key, "record", att[key]["mean_F_ov_Sv"], 4,
            lambda c, k=key: get(k, c, "record", "F_ovS_mean_Sv"))
        add("attribution.json mean_stats.mean (fitted mean)", key, "record", att[key]["mean_stats"]["mean"], 4,
            lambda c, k=key: results[k]["constructions"][c]["windows"]["record"]["F_ovS_mean_Sv"]["fitted_mean"])
        add("attribution.json mean_V_net_Sv", key, "record", att[key]["mean_V_net_Sv"], 2,
            lambda c, k=key: get(k, c, "record", "V_net_mean_Sv"))
        add("r1_common_window.json own mean_F_ov_Sv", key, "record", rcw[key]["own"]["mean_F_ov_Sv"], 4,
            lambda c, k=key: get(k, c, "record", "F_ovS_mean_Sv"))
        add("r1_common_window.json own trend_F_ov.trend (mSv/yr)", key, "record",
            rcw[key]["own"]["trend_F_ov"]["trend"], 3,
            lambda c, k=key: get_trend(k, c, "record", "F_ovS_trend_mSv_per_yr"))
        add("r1_common_window.json own trend_F_az.trend (mSv/yr)", key, "record",
            rcw[key]["own"]["trend_F_az"]["trend"], 3,
            lambda c, k=key: get_trend(k, c, "record", "F_az_trend_mSv_per_yr"))
        add("r1_common_window.json common mean_F_ov_Sv (1993-2017)", key, "1993_2017",
            rcw[key]["common"]["mean_F_ov_Sv"], 4,
            lambda c, k=key: get(k, c, "1993_2017", "F_ovS_mean_Sv"))
        add("r1_common_window.json common trend_F_ov.trend (mSv/yr)", key, "1993_2017",
            rcw[key]["common"]["trend_F_ov"]["trend"], 3,
            lambda c, k=key: get_trend(k, c, "1993_2017", "F_ovS_trend_mSv_per_yr"))
        add("r1_cells.json means F_ov", key, "record", cel[key]["means"]["F_ov"], 4,
            lambda c, k=key: get(k, c, "record", "F_ovS_mean_Sv"))
        for slot in ("used", "alternative"):
            add(f"r1_vertical_grid.json {slot} mean_F_ov_Sv", key, "record",
                vg[key][slot]["mean_F_ov_Sv"], 4,
                lambda c, k=key: get(k, c, "record", "F_ovS_mean_Sv"))
            add(f"r1_vertical_grid.json {slot} trend_F_ov_mSv_per_yr", key, "record",
                vg[key][slot]["trend_F_ov_mSv_per_yr"], 3,
                lambda c, k=key: get_trend(k, c, "record", "F_ovS_trend_mSv_per_yr"))
            add(f"r1_vertical_grid.json {slot} share_T", key, "record",
                vg[key][slot]["share_T"], 3,
                lambda c, k=key: results[k]["constructions"][c]["windows"]["record"]["transport_share_continuous"]["point"])
        w = fc[key].get("window_1993_2025")
        if w:
            note = "ECCO key: record ends 2017, effective window 1993 to 2017" if key == "ecco" else ""
            add("final_checks.json window_1993_2025 F_ov_trend_mSv_per_yr", key, "1993_2025",
                w["F_ov_trend_mSv_per_yr"], 3,
                lambda c, k=key: get_trend(k, c, "1993_2025", "F_ovS_trend_mSv_per_yr"), note)
            add("final_checks.json window_1993_2025 transport_share", key, "1993_2025",
                w["transport_share"], 3,
                lambda c, k=key: results[k]["constructions"][c]["windows"]["1993_2025"]["transport_share_T_term_with_mean_dS"]["point"], note)
    canon = [
        ("CANONICAL_FACTS sec 21 mean F_ovS", "oras5", "record", -0.0383, 4, "F_ovS_mean_Sv"),
        ("CANONICAL_FACTS sec 21 mean F_ovS", "glorys12", "record", -0.0837, 4, "F_ovS_mean_Sv"),
        ("CANONICAL_FACTS sec 21 mean F_ovS", "ecco", "record", -0.1321, 4, "F_ovS_mean_Sv"),
        ("CANONICAL_FACTS sec 21 net section transport", "oras5", "record", -1.22, 2, "V_net_mean_Sv"),
        ("CANONICAL_FACTS sec 21 net section transport", "glorys12", "record", -1.17, 2, "V_net_mean_Sv"),
        ("CANONICAL_FACTS sec 21 net section transport", "ecco", "record", -0.98, 2, "V_net_mean_Sv"),
        ("CANONICAL_FACTS sec 21 T", "oras5", "record", 18.69, 2, "T_limb_mean_Sv"),
        ("CANONICAL_FACTS sec 21 T", "glorys12", "record", 21.26, 2, "T_limb_mean_Sv"),
        ("CANONICAL_FACTS sec 21 T", "ecco", "record", 14.58, 2, "T_limb_mean_Sv"),
        ("CANONICAL_FACTS sec 21 dS", "oras5", "record", 0.0676, 4, "dS_limb_mean_PSU"),
        ("CANONICAL_FACTS sec 21 dS", "glorys12", "record", 0.1363, 4, "dS_limb_mean_PSU"),
        ("CANONICAL_FACTS sec 21 dS", "ecco", "record", 0.3066, 4, "dS_limb_mean_PSU"),
    ]
    for src, key, win, stored, dec, field in canon:
        add(src, key, win, stored, dec, lambda c, k=key, w=win, f=field: get(k, c, w, f))
    add("CANONICAL_FACTS sec 22 F_ovS trend 1958-2025", "oras5", "record", -1.366, 3,
        lambda c: get_trend("oras5", c, "record", "F_ovS_trend_mSv_per_yr"))
    add("CANONICAL_FACTS sec 22 F_ovS trend 1993-2025", "glorys12", "record", -4.362, 3,
        lambda c: get_trend("glorys12", c, "record", "F_ovS_trend_mSv_per_yr"))
    add("CANONICAL_FACTS sec 22 F_ovS trend 1993-2017", "oras5", "1993_2017", -0.775, 3,
        lambda c: get_trend("oras5", c, "1993_2017", "F_ovS_trend_mSv_per_yr"))
    add("CANONICAL_FACTS sec 23 share_T full record", "oras5", "record", 0.068, 3,
        lambda c: results["oras5"]["constructions"][c]["windows"]["record"]["transport_share_continuous"]["point"])
    add("CANONICAL_FACTS sec 23 share_T full record", "glorys12", "record", 0.012, 3,
        lambda c: results["glorys12"]["constructions"][c]["windows"]["record"]["transport_share_continuous"]["point"])
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, (_, bfile, cfile) in PRODUCTS.items():
        data[key] = {"bounds": load(bfile), "centres": load(cfile)}

    results = {}
    for key, (label, bfile, cfile) in PRODUCTS.items():
        results[key] = {"label": label, "constructions": {}}
        for con in CONSTRUCTIONS:
            dd = data[key][con]
            yrs = dd["years"]
            wins = {
                "record": (int(yrs[0]), int(yrs[-1])),
                "1993_2017": (1993, 2017),
                "1993_2025": (1993, 2025),
            }
            results[key]["constructions"][con] = {
                "section_file": bfile if con == "bounds" else cfile,
                "thickness_convention": dd["thickness_convention"],
                "column_total_m": dd["e3t_sum_m"],
                "section_latitude": dd["section_latitude"],
                "n_monthly": dd["n_monthly"],
                "windows": {w: summarise(dd, a, b, STEP_YEARS[key]) for w, (a, b) in wins.items()},
            }

    b_rec = results["oras5"]["constructions"]["bounds"]["windows"]["record"]
    expected = {
        "ORAS5 record-mean F_ovS (Sv), expected -0.0433": b_rec["F_ovS_mean_Sv"]["value"],
        "ORAS5 record trend (mSv/yr), expected -1.42": b_rec["F_ovS_trend_mSv_per_yr"]["value"],
        "ORAS5 record transport share, expected 0.075": b_rec["transport_share_continuous"]["point"],
        "ECCO record-mean F_ovS (Sv), expected -0.132": results["ecco"]["constructions"]["bounds"]["windows"]["record"]["F_ovS_mean_Sv"]["value"],
        "ORAS5 net removal (Sv), expected -0.68": b_rec["V_net_mean_Sv"],
        "GLORYS12V1 net removal (Sv), expected -0.42": results["glorys12"]["constructions"]["bounds"]["windows"]["record"]["V_net_mean_Sv"],
        "ECCO net removal (Sv), expected -0.98": results["ecco"]["constructions"]["bounds"]["windows"]["record"]["V_net_mean_Sv"],
    }

    payload = {
        "meta": {
            "date": "2026-11-05",
            "group": "G03_headline",
            "actions": ["A27", "A28 (covariance residual values)", "A73 (items 2 and 3; the key rename is applied in the note only)"],
            "script": "scripts/analysis_paper3jc_r2_G03_headline_construction.py",
            "inputs": [f"data/results/{v[1]}" for v in PRODUCTS.values()]
            + [f"data/results/{v[2]}" for v in PRODUCTS.values()]
            + [
                "PAPER_3_v2/analysis/attribution.json",
                "PAPER_3_v2/analysis/r1_common_window.json",
                "PAPER_3_v2/analysis/r1_cells.json",
                "PAPER_3_v2/analysis/r1_vertical_grid.json",
                "PAPER_3_v2/analysis/final_checks.json",
                "CANONICAL_FACTS.md sections 21 to 23 (values typed in the script)",
            ],
            "seed": SEED,
            "sieve_resamples": N_BOOT,
            "replicates": {"sieve": N_BOOT, "bias_table_draws": 4000},
            "intervals": "95%: ci_bias_corrected (corrected N_eff test) and ci_sieve (AR(1) sieve percentiles)",
            "units": {
                "F_ovS_mean, F_az_mean, V_net_mean, covariance_residual": "Sv",
                "T_limb_mean": "Sv",
                "dS_limb_mean": "PSU",
                "F_ovS_trend, F_az_trend": "mSv per year",
                "T_limb_trend, dS_limb_trend": "Sv per decade, PSU per decade",
                "transport_share": "dimensionless",
            },
            "constructions": {
                "bounds": "midpoints between level centres, surface at zero (baseline, headline)",
                "centres": "spacing of consecutive level centres (variant)",
            },
            "windows": {
                "record": "own record of each section file (ORAS5 1958-2025, GLORYS12V1 1993-2025, ECCO-V4r4 1992-2017)",
                "1993_2017": "common window of the three products",
                "1993_2025": "requested window; ECCO has no data after 2017, so its effective window is 1993-2017",
            },
            "p_values": "stored in JSON only (p_santer, p_ols)",
        },
        "products": results,
        "expected_check": expected,
        "cross_check_stored_values": cross_check(results),
    }
    path = OUT_DIR / OUT_NAME
    path.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {path}")
    for k, v in expected.items():
        print(f"{k}: {v:+.5f}")


if __name__ == "__main__":
    main()
