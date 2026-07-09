#!/usr/bin/env python3
"""F_ovS-vs-AMOC lead-lag cross-correlation by decomposition class (C3).

Answers reviewer requests R2.16/R3.9: is the F_ovS -> AMOC lag-correlation
different for salinity-dominant vs velocity-dominant CMIP6 models?

Inputs (precomputed, NOT recomputed here)
-----------------------------------------
- data/results/cmip6_fovs_amoc_leadlag.nc
    Per-model cross-correlation ccf(model, lag) with lag in [-50, +50] yr.
    Convention: positive lag => F_ovS LEADS AMOC (attr lag_convention).
- data/results/fovs_decomposition_cmip6_summary.csv
    Velocity/salinity shares of the forced Delta F_ovS, used to classify
    each model (same rule as diagnostic_a3_continuous_correlation.py):
      delta_total >= -0.01 Sv          -> "increasing"  (excluded)
      velocity_share_pct > 60          -> "v-dom"
      salinity_share_pct > 60          -> "s-dom"
      otherwise                        -> "mixed"

Analysis is restricted to forced-weakening AND collapsing models
(amoc_decline_frac > 0.30). Peak lags are located within a bounded window
of +/-20 years (the full +/-50 range is noise-prone at the edges).

Outputs
-------
- revision/results/C3_leadlag_by_class.csv   (per-model)
- revision/results/C3_leadlag_by_class.json  (per-class summary)
- revision/figures/Fig7a_leadlag_by_class.pdf / .png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ardp.viz.style import (  # noqa: E402
    COLORS,
    apply_nature_style,
    save_publication_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADLAG_NC = REPO_ROOT / "data/results/cmip6_fovs_amoc_leadlag.nc"
DECOMP_CSV = REPO_ROOT / "data/results/fovs_decomposition_cmip6_summary.csv"
OUT_CSV = REPO_ROOT / "revision/results/C3_leadlag_by_class.csv"
OUT_JSON = REPO_ROOT / "revision/results/C3_leadlag_by_class.json"
OUT_FIG = REPO_ROOT / "revision/figures/Fig7a_leadlag_by_class"

PEAK_WINDOW = 20  # years: bounded window for peak-lag search
COLLAPSE_THRESHOLD = 0.30

CLASS_COLORS = {"s-dom": COLORS["red"], "v-dom": COLORS["blue"]}
CLASS_LABELS = {"s-dom": "salinity-dominant", "v-dom": "velocity-dominant"}


def classify(row: pd.Series) -> str:
    """Classify a model by its Delta F_ovS decomposition (repo-standard rule)."""
    if row["delta_total"] >= -0.01:
        return "increasing"
    if row["velocity_share_pct"] > 60:
        return "v-dom"
    if row["salinity_share_pct"] > 60:
        return "s-dom"
    return "mixed"


def bounded_peak(lags: np.ndarray, ccf: np.ndarray, window: int) -> tuple[int, float]:
    """Peak (max |r|) lag and r within |lag| <= window. NaN-safe."""
    mask = np.abs(lags) <= window
    sub_lags = lags[mask]
    sub_ccf = ccf[mask]
    if not np.isfinite(sub_ccf).any():
        return 0, float("nan")
    i = int(np.nanargmax(np.abs(sub_ccf)))
    return int(sub_lags[i]), float(sub_ccf[i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leadlag-nc", type=Path, default=LEADLAG_NC)
    parser.add_argument("--decomp-csv", type=Path, default=DECOMP_CSV)
    parser.add_argument("--peak-window", type=int, default=PEAK_WINDOW)
    parser.add_argument("--collapse-threshold", type=float, default=COLLAPSE_THRESHOLD)
    args = parser.parse_args()

    ds = xr.open_dataset(args.leadlag_nc)
    assert ds.attrs["lag_convention"] == "positive_lag=FovS_leads_AMOC"
    lags = ds["lag"].values.astype(int)

    decomp = pd.read_csv(args.decomp_csv)
    decomp["class"] = decomp.apply(classify, axis=1)
    class_map = dict(zip(decomp["model"], decomp["class"], strict=True))

    rows = []
    for model in ds["model"].values:
        model = str(model)
        cls = class_map.get(model)
        if cls is None:
            log.warning(f"  {model}: no decomposition entry, dropped")
            continue
        decline = float(ds["amoc_decline_frac"].sel(model=model))
        ccf = ds["ccf"].sel(model=model).values
        if not np.isfinite(ccf).any():
            log.warning(f"  {model}: CCF all-NaN, dropped")
            continue
        keep = (
            cls in ("s-dom", "v-dom")
            and np.isfinite(decline)
            and decline > args.collapse_threshold
        )
        pk_lag, pk_r = bounded_peak(lags, ccf, args.peak_window)
        rows.append(
            {
                "model": model,
                "class": cls,
                "amoc_decline_frac": decline,
                "included": keep,
                "peak_lag_yr": pk_lag,
                "peak_r": pk_r,
                "r_lag0": float(ccf[lags == 0][0]),
            }
        )
        log.info(
            f"  {model:20s} {cls:11s} decline={decline * 100:5.1f}%  "
            f"peak lag={pk_lag:+3d}y r={pk_r:+.2f}  "
            f"{'INCLUDED' if keep else 'excluded'}"
        )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.info(f"Saved: {OUT_CSV}")

    # ── Per-class ensemble statistics ────────────────────────────────────
    window_mask = np.abs(lags) <= args.peak_window
    win_lags = lags[window_mask]

    summary: dict[str, dict] = {
        "lag_convention": "positive_lag=FovS_leads_AMOC",
        "peak_window_years": args.peak_window,
        "collapse_threshold_frac": args.collapse_threshold,
        "classes": {},
    }
    class_curves: dict[str, np.ndarray] = {}
    class_members: dict[str, np.ndarray] = {}

    for cls in ("s-dom", "v-dom"):
        members = df[(df["class"] == cls) & df["included"]]["model"].tolist()
        if not members:
            log.warning(f"{cls}: no included models")
            continue
        ccf_stack = np.stack([ds["ccf"].sel(model=m).values for m in members])
        mean_ccf = np.nanmean(ccf_stack, axis=0)
        class_curves[cls] = mean_ccf[window_mask]
        class_members[cls] = ccf_stack[:, window_mask]

        pk_lag, pk_r = bounded_peak(lags, mean_ccf, args.peak_window)
        r0 = float(mean_ccf[lags == 0][0])
        pos = float(np.nanmean(mean_ccf[(lags > 0) & window_mask]))
        neg = float(np.nanmean(mean_ccf[(lags < 0) & window_mask]))
        member_peaks = df[df["model"].isin(members)]

        summary["classes"][cls] = {
            "n_models": len(members),
            "models": members,
            "ensemble_mean_ccf_peak_lag_yr": pk_lag,
            "ensemble_mean_ccf_peak_r": round(pk_r, 3),
            "mean_r_lag0": round(r0, 3),
            "mean_r_positive_lags": round(pos, 3),
            "mean_r_negative_lags": round(neg, 3),
            "per_model_peak_lag_mean_yr": round(
                float(member_peaks["peak_lag_yr"].mean()), 1
            ),
            "per_model_peak_lag_median_yr": float(member_peaks["peak_lag_yr"].median()),
        }
        log.info(
            f"{cls}: n={len(members)}  ensemble-mean peak lag={pk_lag:+d}y "
            f"r={pk_r:+.2f}  r(0)={r0:+.2f}  "
            f"<r>_pos={pos:+.2f}  <r>_neg={neg:+.2f}"
        )

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved: {OUT_JSON}")

    # ── Figure ───────────────────────────────────────────────────────────
    apply_nature_style()
    import matplotlib.pyplot as plt  # noqa: E402  (after style)

    fig, ax = plt.subplots(figsize=(3.46, 2.6))

    for cls, curve in class_curves.items():
        color = CLASS_COLORS[cls]
        for member_curve in class_members[cls]:
            ax.plot(win_lags, member_curve, color=color, lw=0.4, alpha=0.25)
        n = summary["classes"][cls]["n_models"]
        ax.plot(
            win_lags,
            curve,
            color=color,
            lw=1.6,
            label=f"{CLASS_LABELS[cls]} (n={n})",
        )
        pk_lag = summary["classes"][cls]["ensemble_mean_ccf_peak_lag_yr"]
        pk_r = summary["classes"][cls]["ensemble_mean_ccf_peak_r"]
        ax.plot(
            pk_lag,
            pk_r,
            marker="o",
            ms=4,
            color=color,
            mec="white",
            mew=0.5,
            zorder=5,
        )

    ax.axvline(0, color="0.6", lw=0.6, ls=":")
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(-args.peak_window, args.peak_window)
    ax.set_xlabel(r"Lag $\tau$ (years)  [positive: $F_\mathrm{ovS}$ leads AMOC]")
    ax.set_ylabel(r"$r\,(F_\mathrm{ovS}(t),\ \mathrm{AMOC}(t+\tau))$")
    ax.legend(loc="lower center", fontsize=7)

    save_publication_figure(fig, OUT_FIG)

    # ── Stdout summary ───────────────────────────────────────────────────
    print()
    for cls in ("s-dom", "v-dom"):
        if cls not in summary["classes"]:
            continue
        s = summary["classes"][cls]
        print(
            f"{CLASS_LABELS[cls]:22s}: n={s['n_models']}  "
            f"peak lag={s['ensemble_mean_ccf_peak_lag_yr']:+d} yr  "
            f"peak r={s['ensemble_mean_ccf_peak_r']:+.2f}  "
            f"r(0)={s['mean_r_lag0']:+.2f}"
        )
    if all(c in summary["classes"] for c in ("s-dom", "v-dom")):
        ls = summary["classes"]["s-dom"]["ensemble_mean_ccf_peak_lag_yr"]
        lv = summary["classes"]["v-dom"]["ensemble_mean_ccf_peak_lag_yr"]
        if ls >= 3 and lv < 3:
            verdict = (
                "s-dominant models show F_ovS leading AMOC while "
                "v-dominant models do not."
            )
        elif abs(ls - lv) < 3 and max(ls, lv) < 3:
            verdict = (
                "s-dominant and v-dominant classes do NOT differ meaningfully: "
                "both ensemble-mean CCFs peak at or near zero lag, so F_ovS "
                "does not robustly lead AMOC in either class."
            )
        else:
            verdict = (
                f"Peak lags differ (s-dom {ls:+d} yr vs v-dom {lv:+d} yr); "
                "see JSON for details."
            )
        print(verdict)


if __name__ == "__main__":
    main()
