#!/usr/bin/env python3
"""CMIP6 family intervals, level-offset decomposition and spread of the forced change.

Paper 3 (Journal of Climate revision), review group G08_cmip6, actions A38, A79
and A82. Computed from the per-model values stored in
``PAPER_3_v2/analysis/jc_n12_cmip6_forced.json`` (25 models; 1993 to 2014 and
2081 to 2100 under SSP5-8.5; 2081 to 2100 under SSP2-4.5 for 13 models) and the
1993 to 2014 reanalysis window means stored in the same file.

Intervals are 2.5 and 97.5 percentiles of a family bootstrap with 10,000
resamples and seed 20261104. The resampled units are the six families of
Text S4 and the single models: 18 units for the 25 models, 12 units for the 13
SSP2-4.5 models.

A38. Family-grouped median (median of the unit means) and pooled 25-model median
     of the limb salinity contrast dS and of F_ovS over 1993 to 2014, and the
     offsets of both from ORAS5, GLORYS12V1 and ECCO-V4r4 (pooled and
     family-grouped; the reanalysis value is subtracted from the median).
A79. Level-offset decomposition of the model-minus-reanalysis offset of F_ovS at
     1993 to 2014. For each model m and reference r, with T the northward-limb
     transport (Sv) and dS the limb salinity contrast (PSU):
         transport part = -(T_m - T_r) dS_r / S0
         contrast part  = -T_r (dS_m - dS_r) / S0
         cross part     = -(T_m - T_r) (dS_m - dS_r) / S0
     The three parts sum to the identity offset -(T_m dS_m - T_r dS_r)/S0. The
     actual offset F_m - F_r of the stored F_ovS values and its residual from the
     identity are reported as well. Medians of the parts are taken over the
     models (pooled and family-grouped) with family-bootstrap intervals.
A82. Family-bootstrap standard deviations (ddof 1, over the 25 models) of F_ovS,
     T and dS in 1993 to 2014 and 2081 to 2100 under SSP5-8.5, with their ratios
     (2081 to 2100 over 1993 to 2014). The SSP2-4.5 row is the change of F_ovS
     from 1993 to 2014 to 2081 to 2100 over 13 models (12 units), pooled and
     family-grouped. Each stored quantity is recomputed from the per-model values
     and compared with the stored summary.

Inputs: PAPER_3_v2/analysis/jc_n12_cmip6_forced.json (per_model,
reanalysis_1993_2014, a_table_and_medians, c_offsets_from_reanalyses_1993_2014).
The family grouping, the unit construction, S0, the seed, the number of draws and
the draw order are imported from scripts/analysis_paper3jc_n12_cmip6_forced.py,
so the draws reproduce its family_bootstrap for identical inputs.

Output: PAPER_3_v2/analysis/jc_r2_G08_cmip6.json.
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

import analysis_paper3jc_n12_cmip6_forced as n12  # noqa: E402

IN_JSON = REPO / "PAPER_3_v2" / "analysis" / "jc_n12_cmip6_forced.json"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G08_cmip6.json"
SEED = n12.SEED
N_BOOT = n12.N_BOOT
S0 = n12.S0
REFS = ("oras5", "glorys12", "ecco")
REF_LABEL = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}


def pct(a: np.ndarray) -> list[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def bootstrap(series: dict, stat) -> dict:
    """Family bootstrap over the units of the models finite in every series.

    ``series`` maps a name to {model: value}. ``stat(pool, umeans)`` receives, for
    one set of models, the pooled values per series name and the unit means per
    series name, and returns a dict of named floats. Point values use each unit
    once; intervals use N_BOOT resamples of the units drawn from one generator
    seeded with SEED, in the draw order of n12.family_bootstrap.
    """
    names = list(series)
    models = [m for m in series[names[0]]
              if all(np.isfinite(series[k].get(m, np.nan)) for k in names)]
    units = n12.units_for(models)
    nU = len(units)
    unit_vals = {k: [np.array([float(series[k][m]) for m in u]) for u in units] for k in names}
    full_pool = {k: np.array([float(series[k][m]) for m in models]) for k in names}
    full_um = {k: np.array([v.mean() for v in unit_vals[k]]) for k in names}
    point = stat(full_pool, full_um)
    draws = {key: np.empty(N_BOOT) for key in point}
    rng = np.random.default_rng(SEED)
    for b in range(N_BOOT):
        idx = rng.integers(0, nU, nU)
        pool = {k: np.concatenate([unit_vals[k][i] for i in idx]) for k in names}
        um = {k: np.array([unit_vals[k][i].mean() for i in idx]) for k in names}
        s = stat(pool, um)
        for key in point:
            draws[key][b] = s[key]
    out = {key: {"point": float(point[key]), "ci": pct(draws[key])} for key in point}
    return {
        "models": models,
        "n_models": len(models),
        "n_units": nU,
        "units": [list(u) for u in units],
        "stats": out,
    }


def med_stats(name: str, pool: dict, um: dict, out: dict) -> None:
    out[f"{name}_median"] = float(np.median(pool[name]))
    out[f"{name}_family_grouped_median"] = float(np.median(um[name]))


def reproduce(label: str, mine: float, stored: float, tol: float = 1e-9) -> dict:
    diff = abs(mine - stored)
    return {"quantity": label, "recomputed": float(mine), "stored": float(stored),
            "abs_diff": float(diff), "agrees": bool(diff <= tol)}


def main() -> None:
    d = json.loads(IN_JSON.read_text())
    per_model = {r["model"]: r for r in d["per_model"]}
    models25 = [r["model"] for r in d["per_model"]]
    rean = d["meta"]["reanalysis_1993_2014"]
    stored_a = d["a_table_and_medians"]
    stored_c = d["c_offsets_from_reanalyses_1993_2014"]
    checks = []

    # ── A38: family-grouped medians and offsets ─────────────────────────
    F = {m: per_model[m]["F_ov_1993-2014"] for m in models25}
    DS = {m: per_model[m]["dS_1993-2014"] for m in models25}
    series38 = {"F_ov": F, "dS": DS}

    def stat38(p, u):
        out = {}
        med_stats("F_ov", p, u, out)
        med_stats("dS", p, u, out)
        for r in REFS:
            rvF = rean[r]["1993-2014"]["F_ov"]
            rvD = rean[r]["1993-2014"]["dS"]
            out[f"F_ov_offset_pooled_{r}"] = float(np.median(p["F_ov"]) - rvF)
            out[f"F_ov_offset_family_grouped_{r}"] = float(np.median(u["F_ov"]) - rvF)
            out[f"dS_offset_pooled_{r}"] = float(np.median(p["dS"]) - rvD)
            out[f"dS_offset_family_grouped_{r}"] = float(np.median(u["dS"]) - rvD)
        return out

    a38 = bootstrap(series38, stat38)
    s38 = a38["stats"]
    sfg = stored_a["F_ov_1993-2014"]
    checks.append(reproduce("F_ov family-grouped median 1993-2014",
                            s38["F_ov_family_grouped_median"]["point"], sfg["family_grouped_median"]))
    checks.append(reproduce("F_ov family-grouped median, lower bound",
                            s38["F_ov_family_grouped_median"]["ci"][0], sfg["ci_family_grouped_bootstrap"][0]))
    checks.append(reproduce("F_ov family-grouped median, upper bound",
                            s38["F_ov_family_grouped_median"]["ci"][1], sfg["ci_family_grouped_bootstrap"][1]))
    checks.append(reproduce("dS family-grouped median 1993-2014",
                            s38["dS_family_grouped_median"]["point"], stored_c["dS"]["model_family_grouped_median"]))
    checks.append(reproduce("dS pooled median 1993-2014",
                            s38["dS_median"]["point"], stored_c["dS"]["model_median"]))
    for r in REFS:
        st = stored_c["F_ov"]["reanalysis"][r]
        checks.append(reproduce(f"F_ov pooled offset vs {REF_LABEL[r]}",
                                s38[f"F_ov_offset_pooled_{r}"]["point"], st["offset_median_minus_reanalysis"]))
        checks.append(reproduce(f"F_ov pooled offset vs {REF_LABEL[r]}, lower bound",
                                s38[f"F_ov_offset_pooled_{r}"]["ci"][0], st["offset_ci"][0]))
        checks.append(reproduce(f"F_ov pooled offset vs {REF_LABEL[r]}, upper bound",
                                s38[f"F_ov_offset_pooled_{r}"]["ci"][1], st["offset_ci"][1]))

    # ── A79: level-offset decomposition at 1993 to 2014 ─────────────────
    series79 = {}
    decomp_point = {}
    for r in REFS:
        Tr = rean[r]["1993-2014"]["T"]
        Dr = rean[r]["1993-2014"]["dS"]
        Fr = rean[r]["1993-2014"]["F_ov"]
        tr, co, cr, idn, act, res, direct = {}, {}, {}, {}, {}, {}, {}
        for m in models25:
            Tm = per_model[m]["T_1993-2014"]
            Dm = per_model[m]["dS_1993-2014"]
            Fm = per_model[m]["F_ov_1993-2014"]
            tr[m] = -(Tm - Tr) * Dr / S0
            co[m] = -Tr * (Dm - Dr) / S0
            cr[m] = -(Tm - Tr) * (Dm - Dr) / S0
            idn[m] = tr[m] + co[m] + cr[m]
            direct[m] = -(Tm * Dm - Tr * Dr) / S0
            act[m] = Fm - Fr
            res[m] = act[m] - idn[m]
        series79[f"transport_{r}"] = tr
        series79[f"contrast_{r}"] = co
        series79[f"cross_{r}"] = cr
        series79[f"identity_{r}"] = idn
        series79[f"actual_F_ov_offset_{r}"] = act
        series79[f"residual_actual_minus_identity_{r}"] = res
        decomp_point[r] = {
            "reanalysis_T_Sv": Tr,
            "reanalysis_dS_PSU": Dr,
            "reanalysis_F_ov_Sv": Fr,
            "max_abs_identity_check_error_Sv": float(max(abs(idn[m] - direct[m]) for m in models25)),
            "n_models_identity_offset_positive": int(sum(idn[m] > 0 for m in models25)),
            "n_models_actual_offset_positive": int(sum(act[m] > 0 for m in models25)),
        }

    def stat79(p, u):
        out = {}
        for name in series79:
            med_stats(name, p, u, out)
        return out

    a79 = bootstrap(series79, stat79)
    s79 = a79["stats"]
    for r in REFS:
        checks.append(reproduce(f"actual F_ov offset vs {REF_LABEL[r]} equals stored pooled offset",
                                s79[f"actual_F_ov_offset_{r}_median"]["point"],
                                stored_c["F_ov"]["reanalysis"][r]["offset_median_minus_reanalysis"]))
        decomp_point[r]["contrast_share_of_identity_median_point"] = (
            s79[f"contrast_{r}_median"]["point"] / s79[f"identity_{r}_median"]["point"])

    # ── A82: family-bootstrap spreads and SSP2-4.5 row ──────────────────
    m_ssp = [m for m in models25 if all(k in per_model[m] for k in
             ("F_ov_ssp585_2081-2100", "T_ssp585_2081-2100", "dS_ssp585_2081-2100"))]
    series82 = {
        "F_ov_1993-2014": {m: per_model[m]["F_ov_1993-2014"] for m in m_ssp},
        "F_ov_2081-2100": {m: per_model[m]["F_ov_ssp585_2081-2100"] for m in m_ssp},
        "T_1993-2014": {m: per_model[m]["T_1993-2014"] for m in m_ssp},
        "T_2081-2100": {m: per_model[m]["T_ssp585_2081-2100"] for m in m_ssp},
        "dS_1993-2014": {m: per_model[m]["dS_1993-2014"] for m in m_ssp},
        "dS_2081-2100": {m: per_model[m]["dS_ssp585_2081-2100"] for m in m_ssp},
    }

    def stat82(p, u):
        out = {}
        for k in p:
            out[f"sd_{k}"] = float(p[k].std(ddof=1))
        for var in ("F_ov", "T", "dS"):
            ratio = out[f"sd_{var}_2081-2100"] / out[f"sd_{var}_1993-2014"]
            out[f"sd_ratio_2081_over_1993_{var}"] = float(ratio)
            out[f"sd_reduction_fraction_{var}"] = float(1.0 - ratio)
        return out

    a82 = bootstrap(series82, stat82)
    s82 = a82["stats"]
    spread = stored_a["spread_ssp585"]
    checks.append(reproduce("SD of F_ov 1993-2014 over 25 models vs stored spread_ssp585",
                            s82["sd_F_ov_1993-2014"]["point"], spread["sd_1993_2014"]))
    checks.append(reproduce("SD of F_ov 2081-2100 over 25 models vs stored spread_ssp585",
                            s82["sd_F_ov_2081-2100"]["point"], spread["sd_2081_2100"]))
    checks.append(reproduce("SD reduction fraction of F_ov vs stored spread_ssp585",
                            s82["sd_reduction_fraction_F_ov"]["point"], spread["sd_reduction_fraction"]))

    m245 = [m for m in models25 if "F_ov_ssp245_2081-2100" in per_model[m]]
    dF245 = {m: per_model[m]["F_ov_ssp245_2081-2100"] - per_model[m]["F_ov_1993-2014"] for m in m245}
    series245 = {"dF_ov_ssp245": dF245}

    def stat245(p, u):
        out = {}
        med_stats("dF_ov_ssp245", p, u, out)
        return out

    a245 = bootstrap(series245, stat245)
    s245 = a245["stats"]
    st245 = stored_a["dF_ov_ssp245_vs_1993-2014"]
    checks.append(reproduce("SSP2-4.5 dF_ov median vs stored",
                            s245["dF_ov_ssp245_median"]["point"], st245["median"]))
    checks.append(reproduce("SSP2-4.5 dF_ov family-grouped median vs stored",
                            s245["dF_ov_ssp245_family_grouped_median"]["point"], st245["family_grouped_median"]))
    checks.append(reproduce("SSP2-4.5 dF_ov median, lower bound vs stored",
                            s245["dF_ov_ssp245_median"]["ci"][0], st245["ci_family_bootstrap"][0]))
    checks.append(reproduce("SSP2-4.5 dF_ov median, upper bound vs stored",
                            s245["dF_ov_ssp245_median"]["ci"][1], st245["ci_family_bootstrap"][1]))
    checks.append(reproduce("SSP2-4.5 dF_ov family-grouped median, lower bound vs stored",
                            s245["dF_ov_ssp245_family_grouped_median"]["ci"][0], st245["ci_family_grouped_bootstrap"][0]))
    checks.append(reproduce("SSP2-4.5 dF_ov family-grouped median, upper bound vs stored",
                            s245["dF_ov_ssp245_family_grouped_median"]["ci"][1], st245["ci_family_grouped_bootstrap"][1]))

    worst = max(c["abs_diff"] for c in checks)
    n_ok = sum(c["agrees"] for c in checks)

    result = {
        "meta": {
            "date": "2026-11-05",
            "group": "G08_cmip6",
            "actions": ["A38", "A79", "A82"],
            "script": "scripts/analysis_paper3jc_r2_G08_cmip6.py",
            "inputs": [
                "PAPER_3_v2/analysis/jc_n12_cmip6_forced.json (per_model, reanalysis_1993_2014, a_table_and_medians, c_offsets_from_reanalyses_1993_2014)",
                "scripts/analysis_paper3jc_n12_cmip6_forced.py (units_for, FAMILIES, SEED, N_BOOT, S0)",
            ],
            "seed": SEED,
            "n_boot": N_BOOT,
            "interval": "2.5 and 97.5 percentiles of the family bootstrap; resampled units are the six families and the single models",
            "units": "Sv for F_ov, T, F_ov offsets and F_ov standard deviations; PSU for dS and its offsets and standard deviations; ratios and fractions are dimensionless",
            "S0_PSU": S0,
            "window_1993_2014": [1993, 2014],
            "window_2081_2100": [2081, 2100],
            "p_values": "none: no hypothesis test is performed in this file",
            "reproduction_check": {"n_checks": len(checks), "n_agree_1e-9": n_ok, "max_abs_diff": worst},
        },
        "A38_family_grouped_contrast_and_offsets": {
            "description": "Family-grouped (median of 18 unit means) and pooled 25-model medians of F_ov and dS over 1993 to 2014, and offsets from ORAS5, GLORYS12V1 and ECCO-V4r4 (reanalysis value subtracted from the median).",
            "n_models": a38["n_models"],
            "n_units": a38["n_units"],
            "units": a38["units"],
            "stats": s38,
        },
        "A79_level_offset_decomposition_1993_2014": {
            "description": "Level offset of F_ov at 1993 to 2014 per reference, split as -(T_m dS_m - T_r dS_r)/S0 into transport, contrast and cross parts. Medians over models (pooled and family-grouped) with family-bootstrap intervals. Actual offset = F_ov_m - F_ov_r; residual = actual minus identity. The medians of the parts do not sum to the median of the identity offset.",
            "n_models": a79["n_models"],
            "n_units": a79["n_units"],
            "decomposition_point": decomp_point,
            "stats": s79,
        },
        "A82_spread_and_ssp245": {
            "sd_ssp585": {
                "description": "Family-bootstrap standard deviations (ddof 1) of F_ov, T and dS over the 25 models in 1993 to 2014 and 2081 to 2100 under SSP5-8.5, with ratios and reduction fractions.",
                "models": a82["models"],
                "n_models": a82["n_models"],
                "n_units": a82["n_units"],
                "units": a82["units"],
                "stats": s82,
            },
            "ssp245": {
                "description": "Change of F_ov from 1993 to 2014 to 2081 to 2100 under SSP2-4.5, 13 models; family bootstrap over 12 units.",
                "models": a245["models"],
                "n_models": a245["n_models"],
                "n_units": a245["n_units"],
                "units": a245["units"],
                "excluded_no_2081_2100_window": stored_a["ssp245_excluded_no_2081_2100_window"],
                "stats": s245,
                "per_model_dF_ov_ssp245_minus_1993_2014": dF245,
            },
        },
        "reproduction_checks": checks,
        "not_done": [
            {
                "item": "observed-window (2005 to 2024) model contrast and FovS compared with EN4",
                "reason": "needs a spliced series of historical 2005 to 2014 and SSP2-4.5 2015 to 2024 section values for matched models; not built in this group",
            }
        ],
    }
    OUT.write_text(json.dumps(n12.clean(result), indent=2))
    print(json.dumps({"written": str(OUT), "checks": len(checks), "agree": n_ok, "max_abs_diff": worst}))


if __name__ == "__main__":
    main()
