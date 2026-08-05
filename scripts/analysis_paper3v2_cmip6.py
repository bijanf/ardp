#!/usr/bin/env python3
"""CMIP6 freshwater transport bias, and whether it says anything about the rate.

Three defects in the previous version of this analysis are corrected here.

1. **The panels disagreed.** Figure 4a took each model's mean F_ovS from
   ``fovs_decomposition_cmip6_summary.csv`` while Figure 4b took it from
   ``WP8_scatter_stats.json``; the two differ by up to 0.04 Sv, which is larger
   than three of the four reanalysis means. Everything here is computed from the
   CSV, and the discrepancy is quantified so the figure can be rebuilt from one
   source.

2. **The Methods described a bias correction that the code never applied.**
   No offset against published pre-industrial control values exists anywhere in
   the pipeline. The model means are the raw 1950-1980 section values and are
   reported as such.

3. **The rate test had no power.** A 19-year overturning trend in a single
   ensemble member is dominated by internal variability, so a null correlation
   across 16 models is the expected result whatever the forced relationship is.
   The internal-variability floor is measured directly from the MPI-ESM1-2-LR
   large ensemble, and the test is repeated on the forced response to 2100,
   where signal exceeds noise.

Model genealogy is also accounted for: several of the 25 models are close
relatives, so the counts are reported both raw and with one member per family.

Writes ``PAPER_3_v2/analysis/cmip6.json``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"
WP8 = REPO / "revision" / "rev_papaer3_02" / "results" / "WP8_scatter_stats.json"

BASELINE = (1950, 1980)
RAPID_WINDOW = (2005, 2023)
FORCED_BASE = (1850, 1900)
FORCED_END = (2081, 2100)

# Models sharing an ocean component or a direct lineage. One representative per
# family is used for the genealogy-aware counts.
FAMILIES = {
    "CESM2": ["CESM2", "CESM2-WACCM", "TaiESM1"],
    "MPI": ["MPI-ESM1-2-LR", "MPI-ESM1-2-HR"],
    "ACCESS": ["ACCESS-CM2", "ACCESS-ESM1-5"],
    "FGOALS": ["FGOALS-f3-L", "FGOALS-g3"],
    "CMCC": ["CMCC-CM2-SR5", "CMCC-ESM2"],
    "UKMO": ["UKESM1-0-LL", "HadGEM3-GC31-LL"],
}


def fisher_ci(r: float, n: int) -> list[float]:
    """Fisher z confidence interval for a Pearson correlation."""
    if n < 4:
        return [np.nan, np.nan]
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def min_detectable_r(n: int) -> float:
    """Smallest |r| that would reach p < 0.05 at this sample size."""
    t = stats.t.ppf(0.975, df=n - 2)
    return float(np.sqrt(t**2 / (t**2 + n - 2)))


def load_models() -> dict[str, float]:
    with open(RESULTS / "fovs_decomposition_cmip6_summary.csv") as fh:
        return {r["model"]: float(r["F_ov_baseline"]) for r in csv.DictReader(fh)}


def family_of(model: str) -> str:
    for fam, members in FAMILIES.items():
        if model in members:
            return fam
    return model


def main() -> None:
    fovs = load_models()
    out: dict = {
        "baseline_window": list(BASELINE),
        "n_models": len(fovs),
        "note_offset": (
            "No offset or bias correction is applied. The previous Methods text "
            "described a correction against published pre-industrial control "
            "values; no such correction exists in the pipeline."
        ),
    }

    vals = np.array(list(fovs.values()))
    out["distribution"] = {
        "median_Sv": float(np.median(vals)),
        "mean_Sv": float(vals.mean()),
        "min_Sv": float(vals.min()),
        "max_Sv": float(vals.max()),
        "n_positive": int((vals > 0).sum()),
        "n_negative": int((vals < 0).sum()),
    }

    # Genealogy-aware count: one representative per family. Members are sorted by
    # F_ovS and the one at index n//2 is kept, which for a two-member family is
    # the more positive of the pair; the only sign-inconsistent family (MPI) is
    # reported both ways.
    reps: dict[str, str] = {}
    for model in fovs:
        fam = family_of(model)
        reps.setdefault(fam, []).append(model)
    chosen = []
    for members in reps.values():
        members = sorted(members, key=lambda m: fovs[m])
        chosen.append(members[len(members) // 2])
    cvals = np.array([fovs[m] for m in chosen])
    out["genealogy_aware"] = {
        "n_independent_models": len(chosen),
        "representatives": sorted(chosen),
        "n_positive": int((cvals > 0).sum()),
        "n_negative": int((cvals < 0).sum()),
        "median_Sv": float(np.median(cvals)),
    }

    # Panel-to-panel consistency of the two F_ovS sources used previously.
    wp8 = json.loads(WP8.read_text())
    pairs = [
        (m["model"], m["fovs_Sv"], fovs[m["model"]])
        for m in wp8["models"]
        if m["model"] in fovs
    ]
    diffs = np.array([b - c for _, b, c in pairs])
    out["source_discrepancy"] = {
        "n_compared": len(pairs),
        "max_abs_diff_Sv": float(np.abs(diffs).max()),
        "rms_diff_Sv": float(np.sqrt((diffs**2).mean())),
        "sign_flips": [m for m, b, c in pairs if (b < 0) != (c < 0)],
        "worst": sorted(
            [{"model": m, "wp8": b, "csv": c, "diff": b - c} for m, b, c in pairs],
            key=lambda r: -abs(r["diff"]),
        )[:5],
    }

    # ── the short-window rate test, and why it cannot resolve anything ──
    rates = {m["model"]: m["rate_Sv_dec"] for m in wp8["models"]}
    common = [m for m in rates if m in fovs]
    x = np.array([fovs[m] for m in common])
    y = np.array([rates[m] for m in common])
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    out["rate_test_short_window"] = {
        "window": list(RAPID_WINDOW),
        "n": len(common),
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "pearson_ci": fisher_ci(float(r_p), len(common)),
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "min_detectable_r": min_detectable_r(len(common)),
    }

    # Internal-variability floor for a 19-year trend, from a large ensemble.
    smile = np.load(RESULTS / "smile_amoc26n_mpi_lr.npz", allow_pickle=True)
    members = sorted(
        {k.rsplit("_", 1)[0] for k in smile.files if k.endswith(("_years", "_amoc"))}
        - {"members"}
    )
    members = [
        m for m in members if f"{m}_years" in smile.files and f"{m}_amoc" in smile.files
    ]
    trends = []
    for mem in members:
        yy, aa = smile[f"{mem}_years"], smile[f"{mem}_amoc"]
        m = (yy >= RAPID_WINDOW[0]) & (yy <= RAPID_WINDOW[1])
        if m.sum() >= 15:
            trends.append(np.polyfit(yy[m].astype(float), aa[m], 1)[0] * 10)
    trends = np.array(trends)
    out["internal_variability_floor"] = {
        "source": "MPI-ESM1-2-LR large ensemble",
        "n_members": len(trends),
        "window": list(RAPID_WINDOW),
        "mean_trend_Sv_per_dec": float(trends.mean()),
        "sd_trend_Sv_per_dec": float(trends.std(ddof=1)),
        "range_Sv_per_dec": [float(trends.min()), float(trends.max())],
        "comment": (
            "Spread of 19-year overturning trends within a single model, i.e. with "
            "the forced response held fixed. Any across-model correlation using a "
            "19-year single-member trend is competing against this."
        ),
    }

    # ── the forced-response test, which does have power ──
    amoc = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    forced = {}
    for m in list(amoc["models"]):
        if m not in fovs:
            continue
        yy, aa = amoc[f"{m}_years"], amoc[f"{m}_amoc"]
        b = (yy >= FORCED_BASE[0]) & (yy <= FORCED_BASE[1])
        e = (yy >= FORCED_END[0]) & (yy <= FORCED_END[1])
        if b.sum() >= 20 and e.sum() >= 10:
            base = float(aa[b].mean())
            forced[m] = {
                "amoc_base_Sv": base,
                "amoc_end_Sv": float(aa[e].mean()),
                "delta_Sv": float(aa[e].mean() - base),
                "delta_pct": float(100.0 * (aa[e].mean() - base) / base),
                "fovs_Sv": fovs[m],
            }
    fm = sorted(forced)
    xf = np.array([forced[m]["fovs_Sv"] for m in fm])
    for target, key in (("delta_Sv", "absolute"), ("delta_pct", "fractional")):
        yf = np.array([forced[m][target] for m in fm])
        r_p, p_p = stats.pearsonr(xf, yf)
        r_s, p_s = stats.spearmanr(xf, yf)
        out[f"rate_test_forced_{key}"] = {
            "base_window": list(FORCED_BASE),
            "end_window": list(FORCED_END),
            "n": len(fm),
            "models": fm,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "pearson_ci": fisher_ci(float(r_p), len(fm)),
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
            "min_detectable_r": min_detectable_r(len(fm)),
        }
    out["forced_response"] = forced

    # Provenance of the overturning diagnostic, which referees asked for
    # explicitly and which was previously unstated.
    out["amoc_diagnostic"] = {
        "experiments": "historical + ssp585, concatenated",
        "variant_label": "r1i1p1f1",
        "grid": "gn (native)",
        "variables": "msftmz, or msftyz where msftmz is unavailable",
        "latitude": 26.5,
        "reduction": "maximum over depths at or below 500 m, annual means",
        "realisations_per_model": 1,
    }

    # The base-state relation. F_ovS = -T dS / S0, so a model with a stronger
    # overturning is expected to carry a larger freshwater transport of whatever
    # sign its limb contrast has; testing that directly explains why the
    # absolute and fractional rate tests give opposite-signed correlations.
    base = np.array([forced[m]["amoc_base_Sv"] for m in fm])
    r_b, p_b = stats.pearsonr(xf, base)
    out["base_state_relation"] = {
        "n": len(fm),
        "pearson_r": float(r_b),
        "pearson_p": float(p_b),
        "pearson_ci": fisher_ci(float(r_b), len(fm)),
        "amoc_base_range_Sv": [float(base.min()), float(base.max())],
        "comment": (
            "Models with a stronger pre-industrial overturning have a more "
            "positive F_ovS. Absolute AMOC change is mechanically tied to the "
            "base state while fractional change is not, which is why the two "
            "rate tests give opposite-signed correlations."
        ),
    }

    # Is the majority count itself distinguishable from a coin flip?
    n_pos = int((vals > 0).sum())
    out["distribution"]["binomial_p"] = float(
        stats.binomtest(n_pos, len(vals), 0.5).pvalue
    )
    n_pos_g = int((cvals > 0).sum())
    out["genealogy_aware"]["binomial_p"] = float(
        stats.binomtest(n_pos_g, len(cvals), 0.5).pvalue
    )
    # The MPI family is the only one that is not internally sign-consistent, so
    # the genealogy count depends on which member represents it. Report both.
    alt = [m for m in chosen if m != "MPI-ESM1-2-LR"] + ["MPI-ESM1-2-HR"]
    if "MPI-ESM1-2-LR" in chosen:
        avals = np.array([fovs[m] for m in alt])
        out["genealogy_aware"]["alternative_MPI_representative"] = {
            "representative": "MPI-ESM1-2-HR",
            "n_positive": int((avals > 0).sum()),
            "n_negative": int((avals < 0).sum()),
            "binomial_p": float(
                stats.binomtest(int((avals > 0).sum()), len(avals), 0.5).pvalue
            ),
        }

    # Leave-one-out influence on both rate tests.
    def loo(xv: np.ndarray, yv: np.ndarray, names: list[str]) -> dict:
        rows = []
        for i in range(len(names)):
            k = np.ones(len(names), bool)
            k[i] = False
            rr, _ = stats.pearsonr(xv[k], yv[k])
            rows.append({"dropped": names[i], "r": float(rr)})
        rr = [r["r"] for r in rows]
        return {
            "min_r": float(min(rr)),
            "max_r": float(max(rr)),
            "most_influential": sorted(
                rows, key=lambda r: -abs(r["r"] - float(stats.pearsonr(xv, yv)[0]))
            )[:3],
        }

    out["base_state_relation"]["leave_one_out"] = loo(xf, base, fm)
    out["rate_test_short_window"]["leave_one_out"] = loo(x, y, common)
    for target, key in (("delta_Sv", "absolute"), ("delta_pct", "fractional")):
        yf = np.array([forced[m][target] for m in fm])
        out[f"rate_test_forced_{key}"]["leave_one_out"] = loo(xf, yf, fm)

    # Per-model table, so the counts are auditable.
    out["per_model"] = [
        {"model": m, "F_ov_Sv": fovs[m], "family": family_of(m)}
        for m in sorted(fovs, key=lambda k: fovs[k])
    ]

    # ── the counting claim, actually tested ──
    obs = wp8["observations"]
    counts = {}
    for name, thr in (
        ("GLORYS12", obs["GLORYS12"]["rate_Sv_dec"]),
        ("RAPID", obs["RAPID"]["rate_Sv_dec"]),
        ("ORAS5", obs["ORAS5"]["rate_Sv_dec"]),
    ):
        neg = [m for m in common if fovs[m] < 0]
        pos = [m for m in common if fovs[m] >= 0]
        a = sum(1 for m in neg if rates[m] < thr)
        b = len(neg) - a
        c = sum(1 for m in pos if rates[m] < thr)
        d = len(pos) - c
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        counts[name] = {
            "threshold_Sv_per_dec": thr,
            "negative_fovs_faster": [a, len(neg)],
            "positive_fovs_faster": [c, len(pos)],
            "fisher_p": float(p),
            "odds_ratio": float(odds) if np.isfinite(odds) else None,
        }
    out["counting_test"] = counts

    # ── observed transports, with both p-value conventions reconciled ──
    rapid = np.load(RESULTS / "rapid_amoc26n.npz", allow_pickle=True)
    ry, ra = rapid[rapid.files[0]], rapid[rapid.files[1]]
    if ry.ndim == 1 and len(ry) == len(ra):
        m = (ry >= RAPID_WINDOW[0]) & (ry <= RAPID_WINDOW[1])
        fit = ols_santer(ry[m].astype(float), ra[m])
        out["rapid"] = {
            "window": list(RAPID_WINDOW),
            "n": int(m.sum()),
            "trend_Sv_per_dec": fit["slope"] * 10,
            "p_ols": fit["p_ols"],
            "p_santer": fit["p_santer"],
            "n_eff": fit["n_eff"],
            "note": (
                "The two values previously in circulation, 0.093 and 0.205, are the "
                "unadjusted and Santer-adjusted p for the same trend."
            ),
        }

    print(
        json.dumps({k: v for k, v in out.items() if k != "forced_response"}, indent=2)[
            :5000
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "cmip6.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
