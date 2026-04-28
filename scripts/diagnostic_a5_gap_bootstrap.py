#!/usr/bin/env python3
"""Bootstrap a 95% confidence interval on the v-vs-s AMOC-weakening gap.

Addresses MAJOR M3 from the adversarial review: the manuscript reports a
10-percentage-point class median gap with no formal uncertainty interval.
With n_v=5 and n_s=6 the Mann-Whitney p-value is structurally bounded
above ~0.13, so a non-parametric bootstrap of the median difference is
the most defensible interval to quote.

Procedure:
  1. Apply the headline partition (|Delta F_ov| floor = 10 mSv,
     threshold = 60% on the velocity- or salinity-share).
  2. Drop "increasing" CMIP6 models (Delta F_ov >= -1 mSv).
  3. Within each class, resample with replacement n times; compute
     median_s - median_v at each draw; collect the bootstrap
     distribution.
  4. Report 95% percentile interval, the 1-sided P(gap <= 0), and a
     hierarchical-bootstrap version that pools UKESM1-0-LL +
     HadGEM3-GC31-LL into one cluster (shared NEMO+UM core) so the
     CI is not inflated by treating model genealogy as independent.

Outputs:
  data/results/diagA5_gap_bootstrap.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "data" / "results"

THRESHOLD_PCT = 60.0
FLOOR_MSV = 10.0
N_BOOT = 10_000
RNG_SEED = 20260428

# Models that share the NEMO+UM core. Treated as one bootstrap cluster
# in the hierarchical variant so the CI is not inflated by genealogy.
SHARED_CORE_GROUP = {"UKESM1-0-LL", "HadGEM3-GC31-LL"}


def _pct_weakening(model: str, amoc_models, amoc):
    if model not in amoc_models:
        return float("nan")
    y = amoc[f"{model}_years"]
    a = amoc[f"{model}_amoc"]
    base = float(np.nanmean(a[(y >= 1950) & (y <= 1980)]))
    end = float(np.nanmean(a[(y >= 2081) & (y <= 2100)]))
    if not (np.isfinite(base) and np.isfinite(end)) or base <= 0:
        return float("nan")
    return 100 * (1 - end / base)


def _classify(row, threshold):
    if row["velocity_share_pct"] > threshold:
        return "v"
    if row["salinity_share_pct"] > threshold:
        return "s"
    return "mixed"


def main() -> None:
    df = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")
    df["abs_delta_msv"] = df["delta_total"].abs() * 1e3

    amoc_npz = np.load(RESULTS / "yearly_amoc26n_cmip6.npz",
                       allow_pickle=True)
    amoc_models = [str(m) for m in amoc_npz["models"]]
    df["weakening_pct"] = df["model"].apply(
        lambda m: _pct_weakening(m, amoc_models, amoc_npz)
    )

    weakening = df[(df["delta_total"] < -0.001)
                    & (df["abs_delta_msv"] >= FLOOR_MSV)].copy()
    weakening["cls"] = weakening.apply(
        lambda r: _classify(r, THRESHOLD_PCT), axis=1
    )
    weakening = weakening.dropna(subset=["weakening_pct"])

    v = weakening[weakening["cls"] == "v"]
    s = weakening[weakening["cls"] == "s"]
    obs_gap = float(s["weakening_pct"].median() - v["weakening_pct"].median())
    print(f"v-dominant (n={len(v)}): {sorted(v['model'].tolist())}")
    print(f"  median weakening = {v['weakening_pct'].median():.2f}%")
    print(f"s-dominant (n={len(s)}): {sorted(s['model'].tolist())}")
    print(f"  median weakening = {s['weakening_pct'].median():.2f}%")
    print(f"observed gap = {obs_gap:.2f} pp")
    print()

    rng = np.random.default_rng(RNG_SEED)
    v_arr = v["weakening_pct"].to_numpy()
    s_arr = s["weakening_pct"].to_numpy()

    boot_gap = np.empty(N_BOOT)
    for i in range(N_BOOT):
        v_draw = rng.choice(v_arr, size=len(v_arr), replace=True)
        s_draw = rng.choice(s_arr, size=len(s_arr), replace=True)
        boot_gap[i] = float(np.median(s_draw) - np.median(v_draw))
    ci_lo, ci_hi = np.percentile(boot_gap, [2.5, 97.5])
    p_neg = float(np.mean(boot_gap <= 0))
    print(f"Standard bootstrap (n_v={len(v_arr)}, n_s={len(s_arr)}, "
          f"{N_BOOT} draws):")
    print(f"  95% CI = [{ci_lo:.2f}, {ci_hi:.2f}] pp")
    print(f"  P(gap <= 0) = {p_neg:.3f}")
    print()

    # Hierarchical variant: collapse the NEMO+UM cluster to one entry
    # before resampling so genealogically related models do not double-
    # count.
    def _collapse(frame):
        shared = frame[frame["model"].isin(SHARED_CORE_GROUP)]
        if len(shared) <= 1:
            return frame.copy()
        merged = shared.iloc[[0]].copy()
        merged["model"] = "/".join(sorted(SHARED_CORE_GROUP))
        merged["weakening_pct"] = shared["weakening_pct"].mean()
        rest = frame[~frame["model"].isin(SHARED_CORE_GROUP)]
        return pd.concat([rest, merged], ignore_index=True)

    v_hi = _collapse(v); s_hi = _collapse(s)
    v_arr_h = v_hi["weakening_pct"].to_numpy()
    s_arr_h = s_hi["weakening_pct"].to_numpy()
    obs_gap_h = float(np.median(s_arr_h) - np.median(v_arr_h))

    boot_gap_h = np.empty(N_BOOT)
    for i in range(N_BOOT):
        v_draw = rng.choice(v_arr_h, size=len(v_arr_h), replace=True)
        s_draw = rng.choice(s_arr_h, size=len(s_arr_h), replace=True)
        boot_gap_h[i] = float(np.median(v_draw) * -1 + np.median(s_draw))
    ci_lo_h, ci_hi_h = np.percentile(boot_gap_h, [2.5, 97.5])
    p_neg_h = float(np.mean(boot_gap_h <= 0))
    print(f"Hierarchical-bootstrap (NEMO+UM cluster pooled, "
          f"n_v={len(v_arr_h)}, n_s={len(s_arr_h)}):")
    print(f"  observed gap = {obs_gap_h:.2f} pp")
    print(f"  95% CI = [{ci_lo_h:.2f}, {ci_hi_h:.2f}] pp")
    print(f"  P(gap <= 0) = {p_neg_h:.3f}")

    out = {
        "threshold_pct": THRESHOLD_PCT,
        "abs_delta_msv_floor": FLOOR_MSV,
        "n_boot": N_BOOT,
        "rng_seed": RNG_SEED,
        "standard": {
            "n_v": int(len(v_arr)),
            "n_s": int(len(s_arr)),
            "obs_gap_pp": obs_gap,
            "ci_95_pp": [float(ci_lo), float(ci_hi)],
            "p_gap_le_zero": p_neg,
        },
        "hierarchical_genealogy": {
            "n_v": int(len(v_arr_h)),
            "n_s": int(len(s_arr_h)),
            "shared_cluster": sorted(SHARED_CORE_GROUP),
            "obs_gap_pp": obs_gap_h,
            "ci_95_pp": [float(ci_lo_h), float(ci_hi_h)],
            "p_gap_le_zero": p_neg_h,
        },
    }
    out_path = RESULTS / "diagA5_gap_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
