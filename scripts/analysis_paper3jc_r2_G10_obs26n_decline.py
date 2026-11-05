#!/usr/bin/env python3
"""Early decline of the ORAS5 overturning at 26.5 N (action A15, cluster C02).

Computes
  * the ORAS5 annual overturning at 26.5 N for 1993 to 2005 from
    data/results/yearly_amoc26n_oras5.npz (keys years and amoc), with the
    year-to-year changes and the means of the epochs 1993-1995, 1996-2000,
    2001-2004 and 2005-2008;
  * the bias-corrected trend (paper3r1_stats.trend_test) over 1993-2000 and
    1996-2004 and over the full 1958-2025 record for reference;
  * the timing of the decline against the documented forcing and
    observing-system dates. The forcing switches are read from
    PAPER_3_v2/analysis/jc_n16_forcing.json (switch_dates); the RAPID
    record span is read from data/external/rapid_moc_monthly.nc; the 2002
    transport step selection statistics are read from
    PAPER_3_v2/analysis/jc_n03_steps.json (a_step_scans.oras5).

No random draws are made. Writes PAPER_3_v2/analysis/jc_r2_G10_obs26n_decline.json.
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

from paper3r1_stats import trend_test  # noqa: E402

RESULTS = REPO / "data" / "results"
EXTERNAL = REPO / "data" / "external"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G10_obs26n_decline.json"


def epoch_mean(years: np.ndarray, amoc: np.ndarray, a: int, b: int) -> dict:
    m = (years >= a) & (years <= b)
    return {"window": [a, b], "mean_Sv": float(amoc[m].mean()), "n_years": int(m.sum())}


def trend_block(years: np.ndarray, amoc: np.ndarray, a: int, b: int) -> dict:
    m = (years >= a) & (years <= b)
    r = trend_test(years[m].astype(float), amoc[m])
    return {
        "window": [a, b],
        "n": int(m.sum()),
        "trend_Sv_per_decade": float(r["value"] * 10.0),
        "ci_Sv_per_decade": [float(c * 10.0) for c in r["ci"]],
        "p": float(r["p"]),
        "n_eff": float(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
        "lag1_residual_corrected": float(r["phi_corrected"]),
    }


def main() -> None:
    z = np.load(RESULTS / "yearly_amoc26n_oras5.npz")
    years = z["years"].astype(int)
    amoc = z["amoc"].astype(float)

    sel = (years >= 1993) & (years <= 2005)
    rows = []
    prev = None
    for y, v in zip(years[sel], amoc[sel]):
        rows.append(
            {
                "year": int(y),
                "amoc_Sv": float(v),
                "change_from_previous_Sv": None if prev is None else float(v - prev),
            }
        )
        prev = v

    diffs = np.diff(amoc[sel])
    yr_next = years[sel][1:]
    largest = int(np.argmin(diffs))

    epochs = {
        "1993_1995": epoch_mean(years, amoc, 1993, 1995),
        "1996_2000": epoch_mean(years, amoc, 1996, 2000),
        "2001_2004": epoch_mean(years, amoc, 2001, 2004),
        "2005_2008": epoch_mean(years, amoc, 2005, 2008),
    }

    trends = {
        "1993_2000": trend_block(years, amoc, 1993, 2000),
        "1996_2004": trend_block(years, amoc, 1996, 2004),
        "1958_2025_reference": trend_block(years, amoc, 1958, 2025),
    }

    forc = json.loads((ANALYSIS / "jc_n16_forcing.json").read_text())
    switch_rows = [
        {
            "product": s["product"],
            "date": s["date"],
            "from": s["from"],
            "to": s["to"],
            "status": s["status"],
        }
        for s in forc["switch_dates"]
    ]
    rp = xr.open_dataset(EXTERNAL / "rapid_moc_monthly.nc")
    rp_time = rp["time"].values
    rapid_first = str(np.datetime_as_string(rp_time[0], unit="M"))
    rapid_last = str(np.datetime_as_string(rp_time[-1], unit="M"))

    n03 = json.loads((ANALYSIS / "jc_n03_steps.json").read_text())
    t_block = n03["a_step_scans"]["oras5"]["variables"]["T"]
    t_families = {}
    for fam_name, fam in t_block["families"].items():
        t_families[fam_name] = {
            k: v for k, v in fam.items() if not isinstance(v, (list, dict))
        }

    result = {
        "meta": {
            "id": "jc_r2_G10_obs26n_decline",
            "group": "G10_obs26n",
            "action": "A15",
            "cluster": "C02",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G10_obs26n_decline.py",
            "inputs": [
                "data/results/yearly_amoc26n_oras5.npz (keys years, amoc)",
                "data/external/rapid_moc_monthly.nc (time)",
                "PAPER_3_v2/analysis/jc_n16_forcing.json (switch_dates)",
                "PAPER_3_v2/analysis/jc_n03_steps.json (a_step_scans.oras5.variables.T.families)",
            ],
            "seeds": "none (no random draws)",
            "replicates": "none (analytic trend tests with the bias-corrected n_eff of paper3r1_stats)",
            "units": "Sv for overturning; Sv per decade for trends; years for dates",
        },
        "oras5_annual_1993_2005": {
            "rows": rows,
            "largest_one_year_fall": {
                "from_year": int(yr_next[largest] - 1),
                "to_year": int(yr_next[largest]),
                "change_Sv": float(diffs[largest]),
            },
            "epoch_means": epochs,
            "value_2000_Sv": float(amoc[years == 2000][0]),
            "mean_1993_1995_Sv": epochs["1993_1995"]["mean_Sv"],
        },
        "trends_bias_corrected": trends,
        "observing_system_and_forcing_dates": {
            "forcing_switch_dates_from_jc_n16": switch_rows,
            "rapid_monthly_record_first_month": rapid_first,
            "rapid_monthly_record_last_month": rapid_last,
            "argo_era_start_used_in_manuscript": 2004,
            "note": (
                "The repository documents no observing-system or forcing change between "
                "1996 and 2000 in ORAS5. The documented changes are the ERA-40 to ERA-Interim "
                "switch in 1979, the ERA-Interim to operational switch in 2015, and the start "
                "of the RAPID record (April 2004) and of the Argo era (2004)."
            ),
        },
        "transport_step_2002_selection_from_jc_n03": {
            "T_family_scans": t_families,
            "note": (
                "Single-step supremum scan of the northward-limb transport T in ORAS5, as stored "
                "in jc_n03_steps.json; scalar entries only."
            ),
        },
    }

    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
