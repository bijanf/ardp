#!/usr/bin/env python3
"""Forced change of the overturning freshwater transport at 34.5 S in CMIP6.

Paper 3 (Journal of Climate version): forced change of the CMIP6 F_ovS, its
decomposition and the offsets of the models from the reanalyses.

Inputs are the annual CMIP6 section series of the 25 models in
``data/results/paper3r1_cmip6_<model>.nc`` (historical and SSP5-8.5, computed by
``analysis_paper3r1_cmip6.py``). The SSP2-4.5 window 2081 to 2100 is not in
those files; for the models that provide SSP2-4.5 sections it is computed here
with the same monthly algorithm (``load`` and ``monthly_series`` of
``analysis_paper3r1_cmip6.py``), applied to the 2081 to 2100 slice of the
SSP2-4.5 files only.

Quantities, all in Sv, with F_ov = -(1/S0) T dS, S0 = 35 PSU:
  F_ov   overturning freshwater transport across 34.5 S (F_ovS)
  F_az   azonal (gyre) freshwater transport
  T      northward-limb transport
  dS     limb salinity contrast (PSU)

Windows: 1850 to 1900, 1993 to 2014 (historical), 2081 to 2100 (SSP5-8.5 and,
where present, SSP2-4.5). Forced changes are taken relative to 1993 to 2014 and
to 1850 to 1900 (the baseline of the manuscript). Window means are means of the
annual values; every window must hold at least 80 per cent of its years.

Family bootstrap. Independent units are the families of the Text S4 grouping
and the single models. Units are resampled with replacement (10,000 draws,
fixed seed). For the median over models, the statistic is the median of the
pooled models of the resampled units; the family-grouped median is the median
of the unit means. Intervals are the 2.5 and 97.5 percentiles of the draws.

Decomposition. For period means T0, dS0 and T1, dS1 with F_hat = -(T dS)/S0,
the change of F_hat is exactly
    transport term  = -(T1 - T0) dS0 / S0
    contrast term   = -T0 (dS1 - dS0) / S0
    product term    = -(T1 - T0)(dS1 - dS0) / S0
The change of the window mean of F_ov differs from the change of F_hat by the
within-window covariance of T and dS; that residual is reported separately.

Writes PAPER_3_v2/analysis/jc_n12_cmip6_forced.json.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analysis_paper3r1_cmip6 as base  # noqa: E402

RESULTS = REPO / "data" / "results"
CMIP6_DIR = REPO / "data" / "cmip6_sections"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n12_cmip6_forced.json"
R1_JSON = REPO / "PAPER_3_v2" / "analysis" / "r1_cmip6.json"

S0 = 35.0
N_BOOT = 10_000
SEED = 20261104
VARS = ("F_ov", "F_az", "T", "dS")
BASE_WINDOWS = {
    "1850-1900": (1850, 1900),
    "1993-2014": (1993, 2014),
    "2081-2100": (2081, 2100),
}
FUTURE = (2081, 2100)
# Same grouping as the Text S4 genealogy in analysis_paper3r1_cmip6.py.
FAMILIES = {
    "CESM2": ["CESM2", "CESM2-WACCM", "TaiESM1"],
    "MPI": ["MPI-ESM1-2-LR", "MPI-ESM1-2-HR"],
    "ACCESS": ["ACCESS-CM2", "ACCESS-ESM1-5"],
    "FGOALS": ["FGOALS-f3-L", "FGOALS-g3"],
    "CMCC": ["CMCC-CM2-SR5", "CMCC-ESM2"],
    "UKMO": ["UKESM1-0-LL", "HadGEM3-GC31-LL"],
}


# ── helpers ───────────────────────────────────────────────────────────


def clean(obj):
    """Replace NaN and infinite numbers by None so the JSON is standard."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def nc_window_means(model: str) -> dict:
    """Window means of the annual section series stored for a model."""
    ds = xr.open_dataset(RESULTS / f"paper3r1_cmip6_{model}.nc")
    out = {}
    for name, (y0, y1) in BASE_WINDOWS.items():
        sub = ds.sel(year=slice(y0, y1))
        n = int(sub["F_ov"].notnull().sum())
        if n < 0.8 * (y1 - y0 + 1):
            continue
        out[name] = {v: float(sub[v].mean()) for v in VARS}
        out[name]["n_years"] = n
    ds.close()
    return out


def ssp245_future_window(model: str) -> dict | None:
    """Window 2081 to 2100 of SSP2-4.5 from the sections, with the monthly
    algorithm of analysis_paper3r1_cmip6.monthly_series."""
    d = base.load(model, "ssp245")
    if d is None:
        return None
    years = np.asarray(d["years"])
    idx = np.flatnonzero((years >= FUTURE[0]) & (years <= FUTURE[1]))
    # Files that do not hold the window are excluded: GISS-E2-1-G salinity
    # covers 2020 to 2024 only, and HadGEM3-GC31-LL covers 2015 to 2020 only.
    if d["s"].sizes["time"] != d["v"].sizes["time"] or idx.size < 0.8 * 12 * (FUTURE[1] - FUTURE[0] + 1):
        return None
    d["v"] = d["v"].isel(time=idx)
    d["s"] = d["s"].isel(time=idx)
    d["years"] = years[idx]
    d["months"] = np.asarray(d["months"])[idx]
    df, _, _ = base.monthly_series(d)
    sub = df[(df["year"] >= FUTURE[0]) & (df["year"] <= FUTURE[1])]
    n = int(sub["year"].nunique())
    if n < 0.8 * (FUTURE[1] - FUTURE[0] + 1):
        return None
    out = {v: float(sub[v].mean()) for v in VARS}
    out["n_years"] = n
    return out


def ssp585_future_window_from_sections(model: str) -> dict:
    """Same slice-based path as ssp245_future_window, for SSP5-8.5. Used only
    to check that the slice-based path reproduces the stored values."""
    d = base.load(model, "ssp585")
    years = np.asarray(d["years"])
    idx = np.flatnonzero((years >= FUTURE[0]) & (years <= FUTURE[1]))
    d["v"] = d["v"].isel(time=idx)
    d["s"] = d["s"].isel(time=idx)
    d["years"] = years[idx]
    d["months"] = np.asarray(d["months"])[idx]
    df, _, _ = base.monthly_series(d)
    sub = df[(df["year"] >= FUTURE[0]) & (df["year"] <= FUTURE[1])]
    return {v: float(sub[v].mean()) for v in VARS}


def worker(task: tuple) -> dict:
    kind, model = task
    if kind == "ssp245":
        return {"model": model, "kind": kind, "window": ssp245_future_window(model)}
    return {
        "model": model,
        "kind": kind,
        "window": ssp585_future_window_from_sections(model),
    }


def units_for(models: list[str]) -> list[list[str]]:
    """Independent units: each family present in the set, and each single."""
    fam_members = {m for v in FAMILIES.values() for m in v}
    units = []
    for members in FAMILIES.values():
        present = [m for m in members if m in models]
        if present:
            units.append(present)
    units += [[m] for m in models if m not in fam_members]
    return units


def family_bootstrap(values: dict) -> dict:
    """Median over models, the family-grouped median and their family-bootstrap
    intervals. ``values`` maps model to a number; NaN entries are dropped."""
    models = [m for m, v in values.items() if np.isfinite(v)]
    rng = np.random.default_rng(SEED)
    units = units_for(models)
    vals = [np.array([values[m] for m in u], float) for u in units]
    unit_means = np.array([v.mean() for v in vals])
    nU = len(units)
    pooled = np.array([values[m] for m in models], float)
    d_pool = np.empty(N_BOOT)
    d_units = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, nU, nU)
        d_pool[b] = np.median(np.concatenate([vals[i] for i in idx]))
        d_units[b] = np.median(unit_means[idx])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "n_models": len(models),
        "n_units": nU,
        "n_positive": int((pooled > 0).sum()),
        "n_negative": int((pooled < 0).sum()),
        "median": float(np.median(pooled)),
        "ci_family_bootstrap": ci(d_pool),
        "family_grouped_median": float(np.median(unit_means)),
        "ci_family_grouped_bootstrap": ci(d_units),
        "min": float(pooled.min()),
        "max": float(pooled.max()),
    }


def decompose(t0, d0, t1, d1, f0, f1) -> dict:
    """Exact discrete split of the change of -(T dS)/S0 between two periods,
    plus the covariance residual to the change of the window mean F_ov."""
    dT, dD = t1 - t0, d1 - d0
    transport = -dT * d0 / S0
    contrast = -t0 * dD / S0
    product = -dT * dD / S0
    fhat0, fhat1 = -t0 * d0 / S0, -t1 * d1 / S0
    residual = (f1 - f0) - (fhat1 - fhat0)
    return {
        "dF_ovS": float(f1 - f0),
        "transport_term": float(transport),
        "contrast_term": float(contrast),
        "product_term": float(product),
        "sum_of_terms": float(transport + contrast + product),
        "identity_error": float(transport + contrast + product - (fhat1 - fhat0)),
        "covariance_residual": float(residual),
    }


def reanalysis_window_values() -> dict:
    """1993 to 2014 window means of the three reanalyses (annual means first,
    as in analysis_paper3r1_cmip6.reanalysis_windows), including F_az."""
    out = base.reanalysis_windows()
    for key in ("oras5", "glorys12", "ecco"):
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
        a = ds.groupby("time.year").mean()
        y = a["year"].values
        m = (y >= 1993) & (y <= 2014)
        out[key]["1993-2014"]["F_az"] = float(a["F_az"].values[m].mean())
        ds.close()
    return out


def genealogy_counts(values: dict) -> dict:
    """Positive counts over every choice of family representatives, as in
    the genealogy of analysis_paper3r1_cmip6.py."""
    import itertools

    models = [m for m, v in values.items() if np.isfinite(v)]
    fams = {k: [m for m in v if m in models] for k, v in FAMILIES.items()}
    fams = {k: v for k, v in fams.items() if v}
    singles = [m for m in models if not any(m in v for v in fams.values())]
    counts, medians, by_choice = [], [], {}
    for choice in itertools.product(*fams.values()):
        members = singles + list(choice)
        counts.append(int(sum(values[m] > 0 for m in members)))
        med = float(np.median([values[m] for m in members]))
        medians.append(med)
        by_choice["|".join(choice)] = {"n_positive": counts[-1], "median": med}
    return {
        "n_independent": len(singles) + len(fams),
        "n_positive_range": [min(counts), max(counts)],
        "median_range_over_representatives": [min(medians), max(medians)],
        "median_by_representative_choice": by_choice,
    }


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    r1 = json.loads(R1_JSON.read_text())
    models = [r["model"] for r in r1["per_model"]]

    # SSP2-4.5 windows where the sections exist (computed here) and the check
    # of the slice-based path on SSP5-8.5 for one model against the stored file.
    ssp245_models = [
        m
        for m in models
        if (CMIP6_DIR / f"{m}_ssp245_vo.nc").exists()
        and (CMIP6_DIR / f"{m}_ssp245_so.nc").exists()
    ]
    tasks = [("ssp245", m) for m in ssp245_models] + [("ssp585", "MPI-ESM1-2-LR")]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, tasks))
    ssp245 = {r["model"]: r["window"] for r in results if r["kind"] == "ssp245"}
    ssp245_excluded = [m for m, w in ssp245.items() if w is None]
    ssp245_models = [m for m, w in ssp245.items() if w is not None]
    check = next(r for r in results if r["kind"] == "ssp585")
    stored = nc_window_means("MPI-ESM1-2-LR")["2081-2100"]
    check_diff = {v: abs(check["window"][v] - stored[v]) for v in VARS}

    rean = reanalysis_window_values()

    # Per-model window values.
    win = {}
    for m in models:
        w = nc_window_means(m)
        win[m] = w
        if ssp245.get(m) is not None:
            win[m]["ssp245_2081-2100"] = ssp245[m]

    def get(m, period, var):
        """Value of a window of a model, or NaN if it is missing."""
        if period == "ssp245_2081-2100":
            s = win[m].get(period)
        elif period == "2081-2100":
            s = nc_stored_future.get(m)
        else:
            s = win[m].get(period)
        return float(s[var]) if s is not None else np.nan

    # SSP5-8.5 2081 to 2100 from the stored sections (the nc window).
    nc_stored_future = {}
    for m in models:
        ds = xr.open_dataset(RESULTS / f"paper3r1_cmip6_{m}.nc")
        sub = ds.sel(year=slice(FUTURE[0], FUTURE[1]))
        if int(sub["F_ov"].notnull().sum()) >= 16:
            nc_stored_future[m] = {v: float(sub[v].mean()) for v in VARS}
        ds.close()

    futures = {"ssp585": "2081-2100", "ssp245": "ssp245_2081-2100"}
    baselines = {"1993-2014": "1993-2014", "1850-1900": "1850-1900"}

    # (a) per-model table.
    per_model = []
    for m in models:
        row = {"model": m}
        for var in ("F_ov", "F_az", "T", "dS"):
            row[f"{var}_1993-2014"] = get(m, "1993-2014", var)
        row["Sum_1993-2014"] = row["F_ov_1993-2014"] + row["F_az_1993-2014"]
        row["n_years_1993-2014"] = win[m].get("1993-2014", {}).get("n_years")
        row["n_years_1850-1900"] = win[m].get("1850-1900", {}).get("n_years")
        for scen, fkey in futures.items():
            if fkey not in win[m] and not (scen == "ssp585" and m in nc_stored_future):
                continue
            for var in ("F_ov", "F_az", "T", "dS"):
                row[f"{var}_{scen}_2081-2100"] = get(m, fkey, var) if scen == "ssp245" else nc_stored_future[m][var]
            fv = row[f"F_ov_{scen}_2081-2100"]
            av = row[f"F_az_{scen}_2081-2100"]
            row[f"Sum_{scen}_2081-2100"] = fv + av
            for bname, bkey in baselines.items():
                for var in ("F_ov", "F_az"):
                    b = get(m, bkey, var)
                    row[f"d{var}_{scen}_vs_{bname}"] = (
                        row[f"{var}_{scen}_2081-2100"] - b
                    )
                row[f"dSum_{scen}_vs_{bname}"] = (
                    row[f"Sum_{scen}_2081-2100"] - (get(m, bkey, "F_ov") + get(m, bkey, "F_az"))
                )
                # decomposition of the change of F_ovS
                t0, d0 = get(m, bkey, "T"), get(m, bkey, "dS")
                t1 = row[f"T_{scen}_2081-2100"]
                d1 = row[f"dS_{scen}_2081-2100"]
                f0, f1 = get(m, bkey, "F_ov"), row[f"F_ov_{scen}_2081-2100"]
                if all(np.isfinite([t0, d0, t1, d1, f0, f1])):
                    row[f"decomp_{scen}_vs_{bname}"] = decompose(t0, d0, t1, d1, f0, f1)
        per_model.append(row)
    per_model_by = {r["model"]: r for r in per_model}

    def vals(key):
        return {m: per_model_by[m].get(key, np.nan) for m in models}

    # (a) medians of levels and changes.
    summary_a = {}
    for key in ("F_ov_1993-2014", "F_az_1993-2014", "Sum_1993-2014"):
        summary_a[key] = family_bootstrap(vals(key))
    for scen in futures:
        for bname in baselines:
            for var in ("F_ov", "F_az", "Sum"):
                key = f"d{var}_{scen}_vs_{bname}" if var != "Sum" else f"dSum_{scen}_vs_{bname}"
                summary_a[key] = family_bootstrap(vals(key))
        summary_a[f"{scen}_2081-2100_F_ov"] = family_bootstrap(vals(f"F_ov_{scen}_2081-2100"))
        summary_a[f"{scen}_2081-2100_F_az"] = family_bootstrap(vals(f"F_az_{scen}_2081-2100"))
        summary_a[f"{scen}_2081-2100_Sum"] = family_bootstrap(vals(f"Sum_{scen}_2081-2100"))
    # MPI-ESM1-2-LR and HR values, quoted separately.
    summary_a["MPI_members"] = {
        m: {k: per_model_by[m].get(k) for k in per_model_by[m] if k != "model"}
        for m in ("MPI-ESM1-2-LR", "MPI-ESM1-2-HR")
    }
    summary_a["genealogy_1993-2014_F_ov"] = genealogy_counts(vals("F_ov_1993-2014"))
    # Spread between models and its change in SSP5-8.5 (same model set).
    both = [m for m in models if np.isfinite(per_model_by[m].get("F_ov_ssp585_2081-2100", np.nan)) and np.isfinite(per_model_by[m]["F_ov_1993-2014"])]
    f0s = np.array([per_model_by[m]["F_ov_1993-2014"] for m in both])
    f1s = np.array([per_model_by[m]["F_ov_ssp585_2081-2100"] for m in both])
    summary_a["spread_ssp585"] = {
        "models": both,
        "sd_1993_2014": float(f0s.std(ddof=1)),
        "sd_2081_2100": float(f1s.std(ddof=1)),
        "sd_reduction_fraction": float(1 - f1s.std(ddof=1) / f0s.std(ddof=1)),
    }
    summary_a["ssp245_models"] = ssp245_models
    summary_a["ssp245_excluded_no_2081_2100_window"] = ssp245_excluded

    # (b) decomposition medians.
    summary_b = {}
    for scen in futures:
        for bname in baselines:
            terms = {}
            for term in ("dF_ovS", "transport_term", "contrast_term", "product_term", "covariance_residual", "identity_error"):
                v = {}
                for m in models:
                    dec = per_model_by[m].get(f"decomp_{scen}_vs_{bname}")
                    if dec is not None:
                        v[m] = dec[term]
                if v:
                    terms[term] = family_bootstrap(v)
                    terms[term]["per_model"] = v
            summary_b[f"{scen}_vs_{bname}"] = terms

    # (c) model offsets from the reanalyses over 1993 to 2014.
    summary_c = {}
    for var, key in (("F_ov", "F_ov_1993-2014"), ("F_az", "F_az_1993-2014"), ("T", "T_1993-2014"), ("dS", "dS_1993-2014")):
        boot = family_bootstrap(vals(key))
        entry = {
            "model_median": boot["median"],
            "model_median_ci": boot["ci_family_bootstrap"],
            "model_family_grouped_median": boot["family_grouped_median"],
            "reanalysis": {},
        }
        for rk in ("oras5", "glorys12", "ecco"):
            rv = rean[rk]["1993-2014"][var]
            rel = vals(key)
            arr = np.array([v for v in rel.values() if np.isfinite(v)])
            entry["reanalysis"][rk] = {
                "value": rv,
                "offset_median_minus_reanalysis": boot["median"] - rv,
                "offset_ci": [boot["ci_family_bootstrap"][0] - rv, boot["ci_family_bootstrap"][1] - rv],
                "fraction_of_models_above": float(np.mean(arr > rv)),
            }
        summary_c[var] = entry

    # Claim checks against sentences of the manuscript (Main.tex PDF lines).
    f93 = vals("F_ov_1993-2014")
    f93_pos = [m for m in models if np.isfinite(f93[m]) and f93[m] > 0]
    t93 = vals("T_1993-2014")
    rean_T_max = max(rean[k]["1993-2014"]["T"] for k in rean)
    claims = {
        "PDF_L479_L480": {
            "n_models": int(np.isfinite(list(f93.values())).sum()),
            "n_positive": len(f93_pos),
            "median": float(np.nanmedian(list(f93.values()))),
            "reanalysis_F": {k: rean[k]["1993-2014"]["F_ov"] for k in rean},
        },
        "PDF_L481_fraction_of_models_above_reanalysis": {
            k: float(np.mean([f93[m] > rean[k]["1993-2014"]["F_ov"] for m in models if np.isfinite(f93[m])])) for k in rean
        },
        "PDF_L495_L497": {
            "T_range_1993_2014": [float(np.nanmin(list(t93.values()))), float(np.nanmax(list(t93.values())))],
            "reanalysis_T_max": rean_T_max,
            "n_positive_F_with_T_above_all_reanalyses": int(sum(t93[m] > rean_T_max for m in f93_pos)),
        },
        "PDF_L512_L515": {
            "median_F_ssp585_2081_2100": summary_a["ssp585_2081-2100_F_ov"]["median"],
            "n_positive_1993_2014": summary_a["F_ov_1993-2014"]["n_positive"],
            "n_positive_ssp585_2081_2100": summary_a["ssp585_2081-2100_F_ov"]["n_positive"],
            "sd_reduction_fraction": summary_a["spread_ssp585"]["sd_reduction_fraction"],
        },
    }

    out = {
        "meta": {
            "analysis": "CMIP6 forced change of F_ovS: per-model table, decomposition, offsets",
            "S0_PSU": S0,
            "n_boot": N_BOOT,
            "seed": SEED,
            "windows": BASE_WINDOWS,
            "units": "Sv for fluxes, PSU for dS, Sv for T",
            "families": FAMILIES,
            "units_used": units_for(models),
            "models_25": models,
            "ssp245_models": ssp245_models,
            "consistency_check_MPI-ESM1-2-LR_ssp585_slice_vs_stored": check_diff,
            "reanalysis_1993_2014": rean,
            "interval_definition": "family bootstrap, 2.5 and 97.5 percentiles",
            "not_done": [
                "regression of the dS change on the gyre E-P change: no gyre E-P series is among the inputs of this analysis",
                "large-ensemble noise band for the forced change: no large-ensemble members are among the inputs",
            ],
        },
        "per_model": per_model,
        "a_table_and_medians": summary_a,
        "b_decomposition": summary_b,
        "c_offsets_from_reanalyses_1993_2014": summary_c,
        "claim_checks": claims,
    }
    OUT.write_text(json.dumps(clean(out), indent=1))
    print(json.dumps(clean({"a": summary_a["F_ov_1993-2014"], "b": {k: v["dF_ovS"]["median"] for k, v in summary_b.items()}, "claims": claims, "check": check_diff}), indent=1))


if __name__ == "__main__":
    main()
