#!/usr/bin/env python3
"""Inputs of the new Supplement Table S8 (paper 3, revision round 2, group G11).

Collects, from the stored JSON outputs only, the quantities that the Supplement
table needs and writes them with their keys and values:

1. Jensen term per product (jc_n18_annual_fields.json, products/<p>/jensen):
   the mean of the monthly-minus-annual-mean term J_T as a percentage of the
   mean northward-limb transport, and its range over the annual values.
2. Depth share of the southward-limb weight below 2000 m
   (jc_n06_layer_damping.json, damping/oras5/all_levels/levels): the share of
   |transport_weight_Sv| of the southward levels whose depth exceeds 2000 m.
   The same share is given for the northward limb.
3. EN4 and reanalysis trends by depth band (r1_salinity_profiles.json, layers),
   with the reanalysis-to-EN4 ratios computed from them.
4. Own-circulation EN4 means of the limb quantities over 2005 to 2024
   (jc_n05_limb_levels_rg09.json, summary/full_column), with the reanalysis
   own-circulation means and the relative deviation of EN4 from them.

Inputs: PAPER_3_v2/analysis/{jc_n18_annual_fields, jc_n06_layer_damping,
r1_salinity_profiles, jc_n05_limb_levels_rg09}.json.
Output: PAPER_3_v2/analysis/jc_r2_G11_tables_s8.json.
No random numbers are drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANA = REPO / "PAPER_3_v2" / "analysis"
OUT = ANA / "jc_r2_G11_tables_s8.json"

LABEL = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}
VARIABLES = ("S_north", "S_south", "dS", "T_Sv", "F_ov_Sv")


def load(name: str) -> dict:
    return json.loads((ANA / name).read_text())


def jensen_block(n18: dict) -> dict:
    out = {}
    for key, label in LABEL.items():
        j = n18["products"][key]["jensen"]
        out[label] = {
            "source_key": f"products/{key}/jensen",
            "mean_T_monthly_Sv": j["mean_T_monthly_Sv"],
            "mean_T_fields_Sv": j["mean_T_fields_Sv"],
            "mean_J_T_Sv": j["mean_J_T_Sv"],
            "rel_J_T_to_T_mean_percent": j["rel_J_T_to_T_mean_percent"],
            "rel_J_T_annual_min_percent": j["rel_J_T_annual_min_percent"],
            "rel_J_T_annual_max_percent": j["rel_J_T_annual_max_percent"],
        }
    return out


def depth_share_block(n06: dict) -> dict:
    levels = n06["damping"]["oras5"]["all_levels"]["levels"]
    depth = np.array([lv["depth_m"] for lv in levels])
    limb = np.array([lv["limb"] for lv in levels])
    w = np.array([abs(lv["transport_weight_Sv"]) for lv in levels])
    out = {"source_key": "damping/oras5/all_levels/levels", "n_levels": len(levels)}
    for side in ("southward", "northward"):
        sel = limb == side
        tot = float(w[sel].sum())
        deep = float(w[sel & (depth > 2000.0)].sum())
        out[side] = {
            "total_weight_Sv": tot,
            "weight_below_2000m_Sv": deep,
            "share_below_2000m_percent": 100.0 * deep / tot,
        }
    out["southward_share_below_2000m_percent"] = out["southward"][
        "share_below_2000m_percent"
    ]
    out["definition"] = (
        "share of |transport_weight_Sv| on levels with depth_m > 2000 among "
        "the levels of the stated limb"
    )
    return out


def layers_block(sal: dict) -> dict:
    out = {
        "source_key": "layers",
        "window": sal["window"],
        "units": "PSU per decade (layer means of the trend profile)",
        "bands": {},
    }
    for band, d in sal["layers"].items():
        en4 = d["en4"]
        rec = {
            "obs_weight_en4": d["obs_weight_en4"],
            "trend_EN4_PSU_per_decade": en4,
            "trend_ORAS5_PSU_per_decade": d["oras5"],
            "trend_GLORYS12V1_PSU_per_decade": d["glorys12"],
        }
        if en4 != 0:
            rec["ratio_ORAS5_to_EN4"] = d["oras5"] / en4
            rec["ratio_GLORYS12V1_to_EN4"] = d["glorys12"] / en4
        out["bands"][band] = rec
    return out


def own_circulation_block(n05: dict) -> dict:
    fc = n05["summary"]["full_column"]
    out = {
        "source_key": "summary/full_column",
        "window": n05["window"],
        "units": {
            "S_north": "PSU",
            "S_south": "PSU",
            "dS": "PSU",
            "T_Sv": "Sv",
            "F_ov_Sv": "Sv",
        },
        "note": (
            "EN4 salinity with the frozen circulation of the stated product, "
            "full column; reanalysis own-circulation means for comparison"
        ),
        "circulation": {},
    }
    for circ in ("ORAS5", "GLORYS12V1"):
        en4 = {v: fc[circ]["EN4"][v]["mean"] for v in VARIABLES}
        en4_ci = {v: fc[circ]["EN4"][v]["mean_ci95"] for v in VARIABLES}
        own_key = f"{circ}_own_monthly"
        own = {v: fc[circ][own_key][v]["mean"] for v in VARIABLES}
        out["circulation"][circ] = {
            "EN4_mean": en4,
            "EN4_mean_ci95": en4_ci,
            "own_monthly_key": f"{circ}/{own_key}",
            "own_monthly_mean": own,
            "EN4_minus_own_percent_F_ov": 100.0
            * (en4["F_ov_Sv"] - own["F_ov_Sv"])
            / abs(own["F_ov_Sv"]),
        }
    return out


def main() -> None:
    n18 = load("jc_n18_annual_fields.json")
    n06 = load("jc_n06_layer_damping.json")
    sal = load("r1_salinity_profiles.json")
    n05 = load("jc_n05_limb_levels_rg09.json")
    res = {
        "meta": {
            "id": "jc_r2_G11_tables_s8",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G11_tables_s8.py",
            "inputs": [
                "PAPER_3_v2/analysis/jc_n18_annual_fields.json",
                "PAPER_3_v2/analysis/jc_n06_layer_damping.json",
                "PAPER_3_v2/analysis/r1_salinity_profiles.json",
                "PAPER_3_v2/analysis/jc_n05_limb_levels_rg09.json",
            ],
            "seeds": None,
            "replicates": None,
            "units": (
                "percent of mean northward-limb transport for the Jensen term; "
                "percent of weight for depth shares; PSU per decade for band "
                "trends; Sv and PSU for means"
            ),
            "purpose": "inputs of the new Supplement Table S8",
        },
        "jensen_term_by_product": jensen_block(n18),
        "southward_weight_depth_share": depth_share_block(n06),
        "salinity_trends_by_depth_band": layers_block(sal),
        "own_circulation_en4_means_2005_2024": own_circulation_block(n05),
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                "jensen_rel_mean_percent": {
                    k: v["rel_J_T_to_T_mean_percent"]
                    for k, v in res["jensen_term_by_product"].items()
                },
                "southward_share_below_2000m_percent": res[
                    "southward_weight_depth_share"
                ]["southward_share_below_2000m_percent"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
