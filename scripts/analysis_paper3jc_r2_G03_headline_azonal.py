#!/usr/bin/env python3
"""ORAS5 1993 to 2025 F_ovS row and the azonal (F_az) reconciliation (round-2 group G03).

Items (action A69 of the round-2 plan, and CANONICAL_FACTS section 29):

(a) ORAS5 F_ovS trend over 1993 to 2025 with the same bias-corrected test and
    interval as the full-record row (``analysis_paper3v2_attribution.trend_stats``,
    i.e. ``analysis_paper3_r2_gate_checks.ols_santer``), a sieve interval
    (10,000 resamples, seed 20261104), and the transport share of the trend under
    the two definitions used in the repository: the transport term with the
    window-mean salinity contrast (``analysis_paper3v2_final_checks.py``) and the
    continuous attribution with both factors frozen at their window means
    (``analysis_paper3v2_attribution.continuous``). Both constructions are given.

(b) The GLORYS12V1 azonal trend. The +6.68 mSv per year (p = 0.049) quoted in
    CANONICAL_FACTS section 29 is searched over every start in {1993, 2004} and
    end in {2014, 2017, 2020, 2022, 2025} of the GLORYS12V1 record, for both
    constructions, and the same search is run for ORAS5 (starts 1958, 1993,
    2004; ends 2014, 2017, 2020, 2022, 2025) to test the ORAS5 value +1.84
    (p = 0.0002). Each candidate is reported with the corrected and naive p-values
    so that the source of the stored numbers can be matched or ruled out.

Inputs: data/results/paper3r1_section_{oras5,glorys12}[_centres].nc through the
functions of analysis_paper3jc_r2_G03_headline_construction.py.
Outputs: PAPER_3_v2/analysis/jc_r2_G03_headline_a69_azonal.json.
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

from analysis_paper3jc_r2_G03_headline_construction import (  # noqa: E402
    N_BOOT,
    OUT_DIR,
    PRODUCTS,
    SEED,
    STEP_YEARS,
    load,
    summarise,
    trend_block,
)
from analysis_paper3v2_attribution import trend_stats  # noqa: E402  (test of the full-record row)

OUT_NAME = "jc_r2_G03_headline_a69_azonal.json"
TARGET_AZ_GLORYS = 6.68
TARGET_AZ_ORAS5 = 1.84


def total_trend(dd: dict, y0: int, y1: int) -> dict:
    """Trend of F_ovS + F_az over the window, with the same test as the F_ovS row."""
    m = (dd["years"] >= y0) & (dd["years"] <= y1)
    return trend_block(dd["years"][m].astype(float), (dd["F_ov"] + dd["F_az"])[m], 1e3)


def grid_search(data: dict, starts: list, ends: list, available: tuple) -> list:
    years = data["years"]
    out = []
    for a in starts:
        for b in ends:
            if b > available[1] or a < available[0]:
                continue
            m = (years >= a) & (years <= b)
            if m.sum() < 15:
                continue
            ts = trend_stats(years[m], data["F_az"][m], scale=1e3)
            out.append(
                {
                    "window": [a, b],
                    "n_years": int(m.sum()),
                    "F_az_trend_mSv_per_yr": ts["trend"],
                    "ci_bias_corrected": [float(c) for c in ts["ci"]],
                    "p_santer": ts["p_santer"],
                    "p_ols": ts["p_ols"],
                    "n_eff": ts["n_eff"],
                    "n_eff_at_floor": ts["n_eff_at_floor"],
                }
            )
    return out


def matches(rows: list, target: float, p_target: float | None) -> list:
    hits = []
    for r in rows:
        v = r["F_az_trend_mSv_per_yr"]
        close = abs(abs(v) - target) < 0.01
        p_hit = p_target is not None and any(
            abs(r[k] - p_target) <= 0.00005 for k in ("p_santer", "p_ols")
        )
        if close or p_hit:
            hits.append({"window": r["window"], "constructions": None, "value": v,
                         "p_santer": r["p_santer"], "p_ols": r["p_ols"], "value_close": bool(close),
                         "p_close": bool(p_hit)})
    return hits


def main() -> None:
    out: dict = {"meta": {}, "oras5_1993_2025": {}, "azonal_search": {}}
    data = {}
    for key, (_, bfile, cfile) in PRODUCTS.items():
        data[key] = {"bounds": load(bfile), "centres": load(cfile)}

    # (a) ORAS5 1993 to 2025 row, both constructions.
    for con in ("bounds", "centres"):
        dd = data["oras5"][con]
        s = summarise(dd, 1993, 2025, STEP_YEARS["oras5"])
        out["oras5_1993_2025"][con] = {
            "window": s["window"],
            "n_years": s["n_years"],
            "F_ovS_trend_mSv_per_yr": s["F_ovS_trend_mSv_per_yr"],
            "F_ovS_mean_Sv": s["F_ovS_mean_Sv"],
            "F_az_trend_mSv_per_yr": s["F_az_trend_mSv_per_yr"],
            "F_ovS_plus_F_az_trend_mSv_per_yr": total_trend(dd, 1993, 2025),
            "T_limb_trend_Sv_per_decade": s["T_limb_trend_Sv_per_decade"],
            "dS_limb_trend_PSU_per_decade": s["dS_limb_trend_PSU_per_decade"],
            "transport_share_T_term_with_mean_dS": s["transport_share_T_term_with_mean_dS"],
            "transport_share_continuous": s["transport_share_continuous"],
            "covariance_residual_Eq1_Sv": s["covariance_residual_Eq1_Sv"],
        }

    # (b) azonal searches.
    g_rows = {}
    for con in ("bounds", "centres"):
        g_rows[con] = grid_search(
            data["glorys12"][con], [1993, 2004], [2014, 2017, 2020, 2022, 2025],
            (1993, 2025),
        )
    all_g = []
    for con, rows in g_rows.items():
        for h in matches(rows, TARGET_AZ_GLORYS, 0.049):
            h["constructions"] = con
            all_g.append(h)
    o_rows = {}
    for con in ("bounds", "centres"):
        o_rows[con] = grid_search(
            data["oras5"][con], [1958, 1993, 2004], [2014, 2017, 2020, 2022, 2025],
            (1958, 2025),
        )
    all_o = []
    for con, rows in o_rows.items():
        for h in matches(rows, TARGET_AZ_ORAS5, 0.0002):
            h["constructions"] = con
            all_o.append(h)

    out["azonal_search"] = {
        "glorys12_target_mSv_per_yr": TARGET_AZ_GLORYS,
        "glorys12_target_p": 0.049,
        "glorys12_grid": g_rows,
        "glorys12_matches": all_g,
        "oras5_target_mSv_per_yr": TARGET_AZ_ORAS5,
        "oras5_target_p": 0.0002,
        "oras5_grid": o_rows,
        "oras5_matches": all_o,
        "glorys12_headline_1993_2025": {
            con: {
                "F_az_trend_mSv_per_yr": summarise(data["glorys12"][con], 1993, 2025, STEP_YEARS["glorys12"])["F_az_trend_mSv_per_yr"],
            }
            for con in ("bounds", "centres")
        },
    }

    out["meta"] = {
        "date": "2026-11-05",
        "group": "G03_headline",
        "action": "A69 (a) and (b); CANONICAL_FACTS section 29 azonal values",
        "script": "scripts/analysis_paper3jc_r2_G03_headline_azonal.py",
        "imports": "scripts/analysis_paper3jc_r2_G03_headline_construction.py (load, summarise, trend_block)",
        "inputs": [f"data/results/{v[1]}" for v in PRODUCTS.values()]
        + [f"data/results/{v[2]}" for v in PRODUCTS.values()],
        "seed": SEED,
        "sieve_resamples": N_BOOT,
        "intervals": "95%: bias-corrected corrected-N_eff test and AR(1) sieve percentiles",
        "units": "trends: mSv per year (F), Sv per decade (T), PSU per decade (dS); shares dimensionless; p-values in JSON only",
        "search_grid": "GLORYS12V1 starts 1993, 2004 x ends 2014, 2017, 2020, 2022, 2025; ORAS5 starts 1958, 1993, 2004 x same ends",
    }
    path = OUT_DIR / OUT_NAME
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")

    o = out["oras5_1993_2025"]["bounds"]
    ft = o["F_ovS_trend_mSv_per_yr"]
    print(f"ORAS5 1993-2025 bounds F_ovS trend {ft['value']:+.3f} ci {ft['ci_bias_corrected']} p {ft['p_santer']:.4f} neff {ft['n_eff']:.2f}")
    print(f"  share (T term, mean dS) {o['transport_share_T_term_with_mean_dS']['point']:.4f}; continuous {o['transport_share_continuous']['point']:.4f} ci {o['transport_share_continuous']['ci_sieve']}")
    gh = out["azonal_search"]["glorys12_headline_1993_2025"]
    print(f"GLORYS12V1 F_az 1993-2025 bounds {gh['bounds']['F_az_trend_mSv_per_yr']['value']:+.3f}; centres {gh['centres']['F_az_trend_mSv_per_yr']['value']:+.3f}")
    print(f"matches for +6.68 (p 0.049) in GLORYS grid: {len(all_g)}; for +1.84 (p 0.0002) in ORAS5 grid: {len(all_o)}")


if __name__ == "__main__":
    main()
