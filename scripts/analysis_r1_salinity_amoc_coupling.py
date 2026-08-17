#!/usr/bin/env python3
"""Is the CMIP6 salt-redistribution signal coupled to AMOC weakening?

Positive-evidence analysis addressing Reviewer 1's central objection (R1.2):
that a salinity-dominated Delta F_ovS need not reflect the AMOC at all (the
salinity could change for an unrelated, exogenous reason).

If the salt-advection interpretation holds, the salt-redistribution component
of the F_ovS change (delta_s) should scale with the amount of AMOC weakening
ACROSS models: a model that weakens more should pile up more salt in the South
Atlantic, giving a more negative delta_s. The velocity component (delta_v) has
no such prediction. So the discriminating test is:

    corr(AMOC weakening %, delta_s)  -- expected significant & negative
    corr(AMOC weakening %, delta_v)  -- expected weak / non-significant

We do NOT claim this proves AMOC -> salinity causation (a shared forcing, e.g.
warming-driven evaporation, could contribute); we report it as evidence that
the salt-redistribution signal co-varies with AMOC decline, consistent with the
salt-advection reading and inconsistent with a purely exogenous salinity change.

Reads:  data/results/fovs_decomposition_cmip6_summary.csv
        data/results/yearly_amoc26n_cmip6.npz
Writes: revision/results/R1_salinity_amoc_coupling.json
        revision/figures/R1_salinity_amoc_coupling.pdf (+ .png)
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
OUT_F = Path("revision/figures")

BASE = (1950, 1980)
FUT = (2081, 2100)


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
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp), "n": int(len(x)),
    }


def main() -> None:
    df = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")
    npz = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    weak = amoc_weakening_pct(npz)

    df["weakening_pct"] = df["model"].map(weak)
    # mSv for readability
    df["delta_s_mSv"] = df["delta_s"] * 1000.0
    df["delta_v_mSv"] = df["delta_v"] * 1000.0
    df["delta_tot_mSv"] = df["delta_total"] * 1000.0

    m = df.dropna(subset=["weakening_pct"]).copy()
    # forced-weakening subset (delta_total < 0), which is the physically
    # relevant set for the salt-advection prediction
    fw = m[m["delta_total"] < 0].copy()

    results = {"baseline_years": BASE, "future_years": FUT}
    for name, sub in [("all_with_amoc", m), ("forced_weakening", fw)]:
        w = sub["weakening_pct"].to_numpy()
        res = {
            "models": sub["model"].tolist(),
            "n": int(len(sub)),
            "weakening_vs_delta_s": _corr(w, sub["delta_s_mSv"].to_numpy()),
            "weakening_vs_delta_v": _corr(w, sub["delta_v_mSv"].to_numpy()),
            "weakening_vs_delta_total": _corr(w, sub["delta_tot_mSv"].to_numpy()),
        }
        results[name] = res
        log.info(f"\n=== {name} (n={len(sub)}) ===")
        for k in ("weakening_vs_delta_s", "weakening_vs_delta_v",
                  "weakening_vs_delta_total"):
            c = res[k]
            log.info(
                f"  {k:28s} Pearson r={c['pearson_r']:+.2f} (p={c['pearson_p']:.3f})"
                f"  Spearman rho={c['spearman_r']:+.2f} (p={c['spearman_p']:.3f})"
            )

    OUT_R.mkdir(parents=True, exist_ok=True)
    (OUT_R / "R1_salinity_amoc_coupling.json").write_text(json.dumps(results, indent=2))
    log.info(f"\nSaved {OUT_R / 'R1_salinity_amoc_coupling.json'}")

    # ---- figure: weakening vs delta_s and delta_v ----
    _plot(fw)


def _plot(fw: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    w = fw["weakening_pct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.2))
    for ax, col, lab, color, panel in [
        (axes[0], "delta_s_mSv", r"salinity-driven $\Delta F_s$ (mSv)", "#c0392b", "a"),
        (axes[1], "delta_v_mSv", r"velocity-driven $\Delta F_v$ (mSv)", "#2c7fb8", "b"),
    ]:
        ax.text(-0.16, 1.10, panel, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left")
        y = fw[col].to_numpy()
        ax.scatter(w, y, s=22, color=color, edgecolor="0.2", linewidth=0.4, zorder=3)
        pr, pp = stats.pearsonr(w, y)
        b, a0 = np.polyfit(w, y, 1)
        xs = np.array([w.min(), w.max()])
        ax.plot(xs, b * xs + a0, color=color, lw=1.2,
                ls="-" if pp < 0.05 else "--", zorder=2)
        ax.set_xlabel("AMOC weakening by 2100 (%)")
        ax.set_ylabel(lab)
        ax.set_title(f"r = {pr:+.2f}  (p = {pp:.3f}, n = {len(w)})", fontsize=7)
        ax.axhline(0, color="0.7", lw=0.6, zorder=1)
    fig.tight_layout()
    OUT_F.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_F / f"R1_salinity_amoc_coupling.{ext}", bbox_inches="tight")
    log.info(f"Saved {OUT_F / 'R1_salinity_amoc_coupling.pdf'}")


if __name__ == "__main__":
    main()
