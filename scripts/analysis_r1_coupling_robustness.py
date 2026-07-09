#!/usr/bin/env python3
"""Robustness of the CMIP6 dFs-vs-weakening coupling (R1.2 follow-up).

Two checks requested by the red-team pass on the revision:

1. Leave-one-out jackknife on the forced-weakening (n=17) coupling
   between the salinity-driven component delta_s and the projected AMOC
   weakening: does the sign survive every single-model drop, and which
   drops (if any) push the Pearson or Spearman p-value above 0.05?

2. Family collapse: both mechanism classes contain sibling models that
   share components (CESM2/CESM2-WACCM and MPI-ESM1-2-HR/LR in the
   velocity class; HadGEM3-GC31-LL/UKESM1-0-LL in the salinity class;
   ACCESS-CM2/ACCESS-ESM1-5 span mixed/velocity). Collapse each family
   to its member mean and recompute (a) the coupling correlations and
   (b) the class-median weakening gap.

Reads:  data/results/fovs_decomposition_cmip6_summary.csv
        data/results/yearly_amoc26n_cmip6.npz
Writes: revision/results/R1_coupling_robustness.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS = Path("data/results")
OUT_R = Path("revision/results")

BASE = (1950, 1980)
FUT = (2081, 2100)

FAMILIES = {
    "CESM2-family": ["CESM2", "CESM2-WACCM"],
    "MPI-ESM1-2-family": ["MPI-ESM1-2-HR", "MPI-ESM1-2-LR"],
    "MOHC-family": ["HadGEM3-GC31-LL", "UKESM1-0-LL"],
    "ACCESS-family": ["ACCESS-CM2", "ACCESS-ESM1-5"],
}


def amoc_weakening_pct(npz) -> dict[str, float]:
    models = [str(m) for m in npz["models"]]
    out = {}
    for m in models:
        yrs = npz[f"{m}_years"].astype(float)
        a = np.asarray(npz[f"{m}_amoc"], dtype=float)
        base = np.nanmean(a[(yrs >= BASE[0]) & (yrs <= BASE[1])])
        fut = np.nanmean(a[(yrs >= FUT[0]) & (yrs <= FUT[1])])
        if np.isfinite(base) and np.isfinite(fut) and base > 0:
            out[m] = 100.0 * (1.0 - fut / base)
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n": int(len(x)),
    }


def _classify(row) -> str:
    if row["delta_total"] >= -0.01:
        return "excluded"
    if row["velocity_share_pct"] > 60:
        return "v-dominant"
    if row["salinity_share_pct"] > 60:
        return "s-dominant"
    return "mixed"


def jackknife(fw: pd.DataFrame) -> dict:
    """Leave-one-out stats for weakening vs delta_s (and delta_v control)."""
    out: dict = {"drops": []}
    for i, dropped in enumerate(fw["model"]):
        sub = fw.drop(fw.index[i])
        w = sub["weakening_pct"].to_numpy()
        cs = _corr(w, sub["delta_s_mSv"].to_numpy())
        cv = _corr(w, sub["delta_v_mSv"].to_numpy())
        out["drops"].append(
            {
                "dropped": dropped,
                "delta_s": cs,
                "delta_v_abs_r": abs(cv["pearson_r"]),
            }
        )
    rs = [d["delta_s"]["pearson_r"] for d in out["drops"]]
    ps = [d["delta_s"]["pearson_p"] for d in out["drops"]]
    rhos = [d["delta_s"]["spearman_r"] for d in out["drops"]]
    sps = [d["delta_s"]["spearman_p"] for d in out["drops"]]
    out["summary"] = {
        "pearson_r_range": [float(min(rs)), float(max(rs))],
        "pearson_p_max": float(max(ps)),
        "pearson_p_gt_05_when_dropping": [
            d["dropped"] for d in out["drops"] if d["delta_s"]["pearson_p"] > 0.05
        ],
        "spearman_rho_range": [float(min(rhos)), float(max(rhos))],
        "spearman_p_max": float(max(sps)),
        "spearman_p_gt_05_when_dropping": [
            d["dropped"] for d in out["drops"] if d["delta_s"]["spearman_p"] > 0.05
        ],
        "sign_preserved_all_drops": bool(max(rs) < 0 and max(rhos) < 0),
        "delta_v_abs_r_max": float(max(d["delta_v_abs_r"] for d in out["drops"])),
    }
    return out


def collapse_families(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Average sibling models onto one row per family; keep singletons."""
    rows = []
    consumed: set[str] = set()
    for fam, members in FAMILIES.items():
        present = [m for m in members if m in set(df["model"])]
        if len(present) >= 2:
            sub = df[df["model"].isin(present)]
            row = {"model": fam}
            for c in cols:
                row[c] = float(sub[c].mean())
            rows.append(row)
            consumed |= set(present)
        # a lone family member stays as itself
    for _, r in df.iterrows():
        if r["model"] not in consumed:
            rows.append({"model": r["model"], **{c: float(r[c]) for c in cols}})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")
    npz = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    weak = amoc_weakening_pct(npz)

    df["weakening_pct"] = df["model"].map(weak)
    df["delta_s_mSv"] = df["delta_s"] * 1000.0
    df["delta_v_mSv"] = df["delta_v"] * 1000.0
    df["class"] = df.apply(_classify, axis=1)

    m = df.dropna(subset=["weakening_pct"]).copy()
    fw = m[m["delta_total"] < -0.01].copy().reset_index(drop=True)
    log.info(f"forced-weakening models with AMOC: n={len(fw)}")
    assert len(fw) == 17, f"expected 17, got {len(fw)}"

    results: dict = {
        "baseline_years": BASE,
        "future_years": FUT,
        "models": fw["model"].tolist(),
        "full_n17": {
            "delta_s": _corr(
                fw["weakening_pct"].to_numpy(), fw["delta_s_mSv"].to_numpy()
            ),
            "delta_v": _corr(
                fw["weakening_pct"].to_numpy(), fw["delta_v_mSv"].to_numpy()
            ),
        },
    }

    # ---- 1. jackknife ----
    results["jackknife"] = jackknife(fw)
    s = results["jackknife"]["summary"]
    log.info("\n=== leave-one-out (delta_s vs weakening) ===")
    log.info(
        f"  Pearson r range  {s['pearson_r_range'][0]:+.2f} .. "
        f"{s['pearson_r_range'][1]:+.2f}, max p = {s['pearson_p_max']:.3f}"
    )
    log.info(f"  p>0.05 when dropping: {s['pearson_p_gt_05_when_dropping']}")
    log.info(
        f"  Spearman rho range {s['spearman_rho_range'][0]:+.2f} .. "
        f"{s['spearman_rho_range'][1]:+.2f}, max p = {s['spearman_p_max']:.3f}"
    )
    log.info(f"  p>0.05 when dropping: {s['spearman_p_gt_05_when_dropping']}")
    log.info(f"  sign preserved in all drops: {s['sign_preserved_all_drops']}")

    # ---- 2a. family-collapsed coupling ----
    cols = ["weakening_pct", "delta_s_mSv", "delta_v_mSv"]
    fam = collapse_families(fw[["model", *cols]], cols)
    results["family_collapsed_coupling"] = {
        "n": int(len(fam)),
        "units": "family-mean weakening_pct / delta mSv",
        "delta_s": _corr(
            fam["weakening_pct"].to_numpy(), fam["delta_s_mSv"].to_numpy()
        ),
        "delta_v": _corr(
            fam["weakening_pct"].to_numpy(), fam["delta_v_mSv"].to_numpy()
        ),
        "members": fam["model"].tolist(),
    }
    c = results["family_collapsed_coupling"]["delta_s"]
    log.info(f"\n=== family-collapsed coupling (n={len(fam)}) ===")
    log.info(
        f"  delta_s: Pearson r={c['pearson_r']:+.2f} (p={c['pearson_p']:.3f}), "
        f"Spearman rho={c['spearman_r']:+.2f} (p={c['spearman_p']:.3f})"
    )

    # ---- 2b. family-collapsed class-median gap ----
    vdom = fw[fw["class"] == "v-dominant"]
    sdom = fw[fw["class"] == "s-dominant"]
    gap_raw = float(sdom["weakening_pct"].median() - vdom["weakening_pct"].median())

    vfam = collapse_families(vdom[["model", "weakening_pct"]], ["weakening_pct"])
    sfam = collapse_families(sdom[["model", "weakening_pct"]], ["weakening_pct"])
    gap_fam = float(sfam["weakening_pct"].median() - vfam["weakening_pct"].median())

    results["class_gap"] = {
        "raw": {
            "n_v": int(len(vdom)),
            "n_s": int(len(sdom)),
            "v_median_pct": float(vdom["weakening_pct"].median()),
            "s_median_pct": float(sdom["weakening_pct"].median()),
            "gap_pp": gap_raw,
            "v_models": vdom["model"].tolist(),
            "s_models": sdom["model"].tolist(),
        },
        "family_collapsed": {
            "n_v": int(len(vfam)),
            "n_s": int(len(sfam)),
            "v_median_pct": float(vfam["weakening_pct"].median()),
            "s_median_pct": float(sfam["weakening_pct"].median()),
            "gap_pp": gap_fam,
            "v_members": vfam["model"].tolist(),
            "s_members": sfam["model"].tolist(),
        },
    }
    g = results["class_gap"]
    log.info("\n=== class-median weakening gap ===")
    log.info(
        f"  raw:              s {g['raw']['s_median_pct']:.1f}% - "
        f"v {g['raw']['v_median_pct']:.1f}% = {g['raw']['gap_pp']:+.1f} pp "
        f"(n={g['raw']['n_s']}/{g['raw']['n_v']})"
    )
    log.info(
        f"  family-collapsed: s {g['family_collapsed']['s_median_pct']:.1f}% - "
        f"v {g['family_collapsed']['v_median_pct']:.1f}% = "
        f"{g['family_collapsed']['gap_pp']:+.1f} pp "
        f"(n={g['family_collapsed']['n_s']}/{g['family_collapsed']['n_v']})"
    )

    OUT_R.mkdir(parents=True, exist_ok=True)
    out = OUT_R / "R1_coupling_robustness.json"
    out.write_text(json.dumps(results, indent=2))
    log.info(f"\nSaved {out}")


if __name__ == "__main__":
    main()
