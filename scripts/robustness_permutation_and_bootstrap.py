#!/usr/bin/env python3
"""SI §S2 — robustness of the 12-percentage-point mechanism-conditional gap.

Three independent tests of the headline claim that the salinity-dominant
CMIP6 subset projects a larger 21st-century AMOC weakening than the
velocity-dominant subset:

  1. Permutation test on the mechanism-class label (10 000 draws):
     shuffle the (v, s) labels across the analysed model subset, and
     recompute the class-median weakening gap each draw.

  2. Three effect-size estimators with separate uncertainty estimates:
        (i)   class-resampling bootstrap   — resample within each class
        (ii)  model-resampling bootstrap   — leave-one-model-out style
                                              (the conservative test of
                                               "is one model carrying it?")
        (iii) Hodges-Lehmann robust difference with Wilcoxon CI

  3. SI Fig S5 — histogram of the permutation null distribution with the
     observed gap marked.

Outputs:
  data/results/robustness_permutation.json
  data/results/robustness_table_S2.csv
  figures/paper2/FigureS5_permutation.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
FIG_DIR = REPO / "figures" / "paper2"


def _classify(v_pct: float, s_pct: float) -> str:
    if v_pct > 60:
        return "v"
    if s_pct > 60:
        return "s"
    return "mixed"


def _classify_manuscript(row: pd.Series) -> str:
    """Manuscript's mechanism classification (from plot_paper2_Figure2.py)."""
    if row["delta_total"] >= -0.01:
        return "stable"  # essentially flat/strengthening
    if row["velocity_share_pct"] > 60:
        return "v"
    if row["salinity_share_pct"] > 60:
        return "s"
    return "mixed"


def _load_classes_and_weakening(bistable_only: bool = False) -> pd.DataFrame:
    """For each CMIP6 model in the canonical summary CSV (the same source the
    manuscript Figure 2 uses), return (model, class, weakening_pct) using the
    manuscript baseline 1950-1980 vs endpoint 2081-2100.

    Args:
        bistable_only: if True, retain only models with F_ov_baseline < 0
            (the bistable indicator subset).
    """
    csv = RESULTS / "fovs_decomposition_cmip6_summary.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    cmip6 = pd.read_csv(csv)
    cmip6["class"] = cmip6.apply(_classify_manuscript, axis=1)
    if bistable_only:
        cmip6 = cmip6[cmip6["F_ov_baseline"] < 0].copy()

    amoc_data = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)

    rows = []
    for _, r in cmip6.iterrows():
        if r["class"] in ("mixed", "stable"):
            continue
        m = r["model"]
        amoc_key = f"{m}_amoc"
        years_key = f"{m}_years"
        if amoc_key not in amoc_data.files or years_key not in amoc_data.files:
            continue
        years = amoc_data[years_key]
        amoc = amoc_data[amoc_key]
        mask_baseline = (years >= 1950) & (years <= 1980)
        mask_end = (years >= 2081) & (years <= 2100)
        if not mask_baseline.any() or not mask_end.any():
            continue
        baseline = float(np.nanmean(amoc[mask_baseline]))
        endpoint = float(np.nanmean(amoc[mask_end]))
        if baseline <= 0 or not np.isfinite(baseline) or not np.isfinite(endpoint):
            continue
        weakening_pct = 100.0 * (baseline - endpoint) / baseline
        rows.append({
            "model": m,
            "class": r["class"],
            "weakening_pct": weakening_pct,
            "v_share_pct": r["velocity_share_pct"],
            "s_share_pct": r["salinity_share_pct"],
            "F_ov_baseline": r["F_ov_baseline"],
        })
    return pd.DataFrame(rows)


def permutation_test(df: pd.DataFrame, n_draws: int = 10_000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    # With manuscript convention (positive = weakening), s-class median should
    # exceed v-class median. Define gap = s_median - v_median (positive expected).
    obs_gap = df.loc[df["class"] == "s", "weakening_pct"].median() - \
              df.loc[df["class"] == "v", "weakening_pct"].median()
    labels = df["class"].values.copy()
    null = np.empty(n_draws)
    for i in range(n_draws):
        rng.shuffle(labels)
        v_med = np.median(df.loc[labels == "v", "weakening_pct"])
        s_med = np.median(df.loc[labels == "s", "weakening_pct"])
        null[i] = s_med - v_med
    p_two_sided = float((np.abs(null) >= abs(obs_gap)).mean())
    p_one_sided = float((null >= obs_gap).mean() if obs_gap > 0 else (null <= obs_gap).mean())
    return {
        "observed_gap_pp": float(obs_gap),
        "n_draws": n_draws,
        "p_two_sided": p_two_sided,
        "p_one_sided": p_one_sided,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_pct_5": float(np.percentile(null, 5)),
        "null_pct_95": float(np.percentile(null, 95)),
        "null_distribution": null.tolist(),
    }


def class_resampling_bootstrap(df: pd.DataFrame, n_draws: int = 10_000, seed: int = 17) -> dict:
    """Resample with replacement *within each class*. gap = s_median - v_median."""
    rng = np.random.default_rng(seed)
    v = df.loc[df["class"] == "v", "weakening_pct"].values
    s = df.loc[df["class"] == "s", "weakening_pct"].values
    gaps = np.empty(n_draws)
    for i in range(n_draws):
        bv = rng.choice(v, size=len(v), replace=True)
        bs = rng.choice(s, size=len(s), replace=True)
        gaps[i] = np.median(bs) - np.median(bv)
    return {
        "median_gap_pp": float(np.median(gaps)),
        "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
    }


def model_resampling_bootstrap(df: pd.DataFrame, n_draws: int = 10_000, seed: int = 19) -> dict:
    """Resample with replacement *across the full model list*."""
    rng = np.random.default_rng(seed)
    n = len(df)
    gaps = []
    for _ in range(n_draws):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx]
        v = sub.loc[sub["class"] == "v", "weakening_pct"]
        s = sub.loc[sub["class"] == "s", "weakening_pct"]
        if len(v) < 2 or len(s) < 2:
            continue
        gaps.append(s.median() - v.median())
    gaps = np.asarray(gaps)
    return {
        "median_gap_pp": float(np.median(gaps)),
        "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
        "n_valid_draws": int(len(gaps)),
    }


def hodges_lehmann(df: pd.DataFrame) -> dict:
    """HL = median of pairwise s - v differences (positive expected)."""
    v = df.loc[df["class"] == "v", "weakening_pct"].values
    s = df.loc[df["class"] == "s", "weakening_pct"].values
    pairwise = np.subtract.outer(s, v).flatten()
    hl = float(np.median(pairwise))
    # Mann-Whitney CI (Walker-style) — use scipy
    try:
        w_test = stats.mannwhitneyu(v, s, alternative="two-sided")
        p = float(w_test.pvalue)
    except Exception:
        p = np.nan
    # Bootstrap CI for HL
    rng = np.random.default_rng(31)
    hls = []
    for _ in range(2000):
        bv = rng.choice(v, size=len(v), replace=True)
        bs = rng.choice(s, size=len(s), replace=True)
        hls.append(np.median(np.subtract.outer(bs, bv).flatten()))
    hls = np.asarray(hls)
    return {
        "hl_pp": hl,
        "ci95": [float(np.percentile(hls, 2.5)), float(np.percentile(hls, 97.5))],
        "wilcoxon_mannwhitney_p": p,
    }


def plot_permutation_null(perm: dict, out_path: Path) -> None:
    null = np.asarray(perm["null_distribution"])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.hist(null, bins=80, color="0.7", edgecolor="0.4", linewidth=0.3)
    ax.axvline(perm["observed_gap_pp"], color="#EE6677", linewidth=1.5,
               label=f"Observed gap = {perm['observed_gap_pp']:+.1f} pp")
    ax.axvline(0, color="0.3", linewidth=0.5, linestyle=":")
    ax.set_xlabel("s-class median − v-class median  (pp)")
    ax.set_ylabel("Number of permutations")
    ax.legend(loc="upper right", frameon=False, fontsize=6)
    ax.text(0.02, 0.95,
            f"p (two-sided) = {perm['p_two_sided']:.4f}\n"
            f"p (one-sided) = {perm['p_one_sided']:.4f}\n"
            f"n_draws = {perm['n_draws']}",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.3, pad=0.3))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.pdf')}")


def main() -> int:
    # Run both: full forced-weakening subset AND bistable-only subset.
    for bistable_only in (False, True):
        tag = "bistable" if bistable_only else "all_forced_weakening"
        print(f"\n{'='*60}\nSubset: {tag}\n{'='*60}")
        df = _load_classes_and_weakening(bistable_only=bistable_only)
        print(f"Loaded {len(df)} CMIP6 models (excluding 'mixed' and 'stable')")
        print(df.groupby("class").size())
        print(df.groupby("class")["weakening_pct"].describe())
        if df.empty or df["class"].nunique() < 2:
            print("  Insufficient data — skipping.")
            continue
        perm = permutation_test(df, n_draws=10_000)
        print(f"  Observed gap (s − v median): {perm['observed_gap_pp']:+.2f} pp")
        print(f"  p (two-sided): {perm['p_two_sided']:.4f}")
        print(f"  p (one-sided): {perm['p_one_sided']:.4f}")
        cr = class_resampling_bootstrap(df, n_draws=10_000)
        print(f"  class-resampling 95% CI: {cr['ci95']}")
        mr = model_resampling_bootstrap(df, n_draws=10_000)
        print(f"  model-resampling 95% CI: {mr['ci95']}")
        hl = hodges_lehmann(df)
        print(f"  Hodges-Lehmann: {hl['hl_pp']:+.2f} pp  (MW p={hl['wilcoxon_mannwhitney_p']:.3f})")

    # Re-do the canonical run (all forced-weakening, not bistable-only) for the
    # downstream files SI §S2 references.
    df = _load_classes_and_weakening(bistable_only=False)

    if df.empty or df["class"].nunique() < 2:
        print("Insufficient data — aborting.")
        return 1

    print("\nRunning permutation test (10000 draws) ...")
    perm = permutation_test(df, n_draws=10_000)
    print(f"  Observed gap: {perm['observed_gap_pp']:+.2f} pp")
    print(f"  p (two-sided): {perm['p_two_sided']:.4f}")
    print(f"  p (one-sided): {perm['p_one_sided']:.4f}")

    print("\nRunning class-resampling bootstrap (10000 draws) ...")
    cr = class_resampling_bootstrap(df, n_draws=10_000)
    print(f"  Median gap: {cr['median_gap_pp']:+.2f} pp, 95% CI {cr['ci95']}")

    print("\nRunning model-resampling bootstrap (10000 draws) ...")
    mr = model_resampling_bootstrap(df, n_draws=10_000)
    print(f"  Median gap: {mr['median_gap_pp']:+.2f} pp, 95% CI {mr['ci95']}")

    print("\nRunning Hodges-Lehmann robust difference ...")
    hl = hodges_lehmann(df)
    print(f"  HL estimate: {hl['hl_pp']:+.2f} pp, 95% CI {hl['ci95']}, MW p={hl['wilcoxon_mannwhitney_p']:.4f}")

    # Write JSON (drop the bulky null distribution from the JSON; keep a summary)
    perm_summary = {k: v for k, v in perm.items() if k != "null_distribution"}
    summary = {
        "n_models": int(len(df)),
        "n_v_class": int((df["class"] == "v").sum()),
        "n_s_class": int((df["class"] == "s").sum()),
        "permutation": perm_summary,
        "class_resampling_bootstrap": cr,
        "model_resampling_bootstrap": mr,
        "hodges_lehmann": hl,
    }
    (RESULTS / "robustness_permutation.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {RESULTS / 'robustness_permutation.json'}")

    # Effect-size triangulation table for SI Table S2
    table = pd.DataFrame([
        {"estimator": "Class-resampling bootstrap",
         "value_pp": cr["median_gap_pp"],
         "ci95_low_pp": cr["ci95"][0], "ci95_high_pp": cr["ci95"][1]},
        {"estimator": "Model-resampling bootstrap",
         "value_pp": mr["median_gap_pp"],
         "ci95_low_pp": mr["ci95"][0], "ci95_high_pp": mr["ci95"][1]},
        {"estimator": "Hodges-Lehmann (Mann-Whitney p)",
         "value_pp": hl["hl_pp"],
         "ci95_low_pp": hl["ci95"][0], "ci95_high_pp": hl["ci95"][1]},
    ])
    table.to_csv(RESULTS / "robustness_table_S2.csv", index=False)
    print(f"Wrote {RESULTS / 'robustness_table_S2.csv'}")

    # SI figure S5: permutation null
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_permutation_null(perm, FIG_DIR / "FigureS5_permutation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
