#!/usr/bin/env python3
"""Uncertainty and epoch-independence for the F_ovS variation identity.

``analysis_paper3v2_feedback_identity.py`` evaluates

    dF_ovS = -(1/S0) * (DvS dPsi + Psi dDvS + dPsi dDvS)

for four hand-picked epoch pairs and reports point estimates only. Two
objections follow immediately: the partition might be an artefact of the chosen
epochs, and a share of the form ``term / total`` has no confidence interval and
a pathological sampling distribution when the denominator is uncertain.

This script answers both without picking any epoch at all.

1. **Continuous attribution.** Hold one factor at its record mean and let the
   other vary, giving two counterfactual series

       F_Psi(t)  = -(1/S0) * Psi(t) * <DvS>
       F_DvS(t)  = -(1/S0) * <Psi>  * DvS(t)

   whose trends sum, to within the cross term, to the trend of F_ovS. Every
   year of the record is used and the uncertainty is an ordinary trend
   uncertainty (Santer N_eff plus a circular block bootstrap).

2. **Epoch census.** Every non-overlapping early/late epoch pair the record
   supports, for epoch lengths of 10, 13 and 15 years, so that "in three of
   four cases" is replaced by a distribution over hundreds of cases.

3. **Bootstrap intervals on the terms themselves**, in mSv, for the epoch pairs
   used in the paper. Differences, not ratios, are the reported quantity; the
   ratio interval is computed as well and its width is reported honestly.

4. **Argo-era repeat.** The whole calculation restricted to 2004 onwards, so
   the attribution can be judged on the period in which the salinity field at
   the section is actually constrained by floats.

Writes ``PAPER_3_v2/analysis/identity_robustness.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"

S0 = 35.0
RNG = np.random.default_rng(20260805)
N_BOOT = 10_000
BLOCK_YEARS = 5

PRODUCTS = {
    "oras5": {
        "label": "ORAS5",
        "fovs": "oras5_f_ovs.nc",
        "moc": "oras5_moc_34S.nc",
        "epochs": {
            "pre_registered": ((1993, 2005), (2013, 2025)),
            "long_epochs": ((1960, 1989), (1995, 2024)),
        },
        "decomp": {
            "pre_registered": "fovs_decomposition_oras5.nc",
            "long_epochs": "fovs_decomposition_oras5_paper3_epochs.nc",
        },
    },
    "glorys12": {
        "label": "GLORYS12V1",
        "fovs": "glorys12_f_ovs.nc",
        "moc": "glorys12_moc_34S.nc",
        "epochs": {
            "pre_registered": ((1993, 2005), (2013, 2025)),
            "halves": ((1993, 2008), (2009, 2024)),
        },
        "decomp": {
            "pre_registered": "fovs_decomposition_glorys12.nc",
            "halves": "fovs_decomposition_glorys12_paper3_halves.nc",
        },
    },
}


def annual(path: Path, var: str) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        grouped = ds[var].groupby("time.year").mean()
        years = grouped["year"].values.astype(int)
        values = grouped.values.astype(float)
    good = np.isfinite(values)
    return years[good], values[good]


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index vector of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block)) % n for s in starts])
    return idx[:n]


# ──────────────────────────────────────────────────────────────────────
# 1. Continuous attribution
# ──────────────────────────────────────────────────────────────────────


def continuous_attribution(
    years: np.ndarray, psi: np.ndarray, dvs: np.ndarray, fovs: np.ndarray
) -> dict:
    """Trend of F_ovS split by holding each factor at its record mean.

    F_Psi varies only the overturning strength, F_DvS only the salinity
    contrast. Their trends plus the residual reproduce the F_ovS trend.
    """
    psi_bar = float(psi.mean())
    dvs_bar = float(dvs.mean())
    f_psi = -(1.0 / S0) * psi * dvs_bar
    f_dvs = -(1.0 / S0) * psi_bar * dvs

    x = years.astype(float)
    fit_tot = ols_santer(x, fovs)
    fit_psi = ols_santer(x, f_psi)
    fit_dvs = ols_santer(x, f_dvs)

    # Block bootstrap of the regression residuals, not of the values: resampling
    # the values themselves destroys the trend and returns a null distribution
    # rather than a confidence interval. The same block indices are applied to
    # all three residual series so the joint structure, and therefore the ratio,
    # is preserved.
    n = len(years)

    def fit_parts(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coef = np.polyfit(x, v, 1)
        trend = np.polyval(coef, x)
        return trend, v - trend

    tr_tot, rs_tot = fit_parts(fovs)
    tr_psi, rs_psi = fit_parts(f_psi)
    tr_dvs, rs_dvs = fit_parts(f_dvs)

    boot_psi = np.empty(N_BOOT)
    boot_dvs = np.empty(N_BOOT)
    boot_tot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = block_indices(n, BLOCK_YEARS, RNG)
        boot_tot[i] = np.polyfit(x, tr_tot + rs_tot[idx], 1)[0]
        boot_psi[i] = np.polyfit(x, tr_psi + rs_psi[idx], 1)[0]
        boot_dvs[i] = np.polyfit(x, tr_dvs + rs_dvs[idx], 1)[0]

    def ci(a: np.ndarray, scale: float = 1e3) -> list[float]:
        return [
            float(np.percentile(a, 2.5) * scale),
            float(np.percentile(a, 97.5) * scale),
        ]

    ratio = np.abs(boot_psi) / np.abs(boot_tot)
    return {
        "n_years": int(n),
        "window": [int(years[0]), int(years[-1])],
        "psi_mean_Sv": psi_bar,
        "dvs_mean_PSU": dvs_bar,
        "trend_fovs_mSv_per_yr": fit_tot["slope"] * 1e3,
        "trend_fovs_p_santer": fit_tot["p_santer"],
        "trend_fovs_n_eff": fit_tot["n_eff"],
        "trend_fovs_ci_mSv_per_yr": ci(boot_tot),
        "trend_psi_term_mSv_per_yr": fit_psi["slope"] * 1e3,
        "trend_psi_term_p_santer": fit_psi["p_santer"],
        "trend_psi_term_n_eff": fit_psi["n_eff"],
        "trend_psi_term_ci_mSv_per_yr": ci(boot_psi),
        "trend_dvs_term_mSv_per_yr": fit_dvs["slope"] * 1e3,
        "trend_dvs_term_p_santer": fit_dvs["p_santer"],
        "trend_dvs_term_n_eff": fit_dvs["n_eff"],
        "trend_dvs_term_ci_mSv_per_yr": ci(boot_dvs),
        "residual_mSv_per_yr": (fit_tot["slope"] - fit_psi["slope"] - fit_dvs["slope"])
        * 1e3,
        "abs_share_psi_point": float(abs(fit_psi["slope"]) / abs(fit_tot["slope"])),
        "abs_share_psi_ci": [
            float(np.percentile(ratio, 2.5)),
            float(np.percentile(ratio, 97.5)),
        ],
        "prob_psi_term_opposes_total": float(
            np.mean(np.sign(boot_psi) != np.sign(boot_tot))
        ),
        "prob_abs_share_below_0p25": float(np.mean(ratio < 0.25)),
    }


# ──────────────────────────────────────────────────────────────────────
# 2. Epoch census
# ──────────────────────────────────────────────────────────────────────


def identity_terms(
    psi1: float, psi2: float, f1: float, f2: float
) -> tuple[float, float, float, float]:
    dvs1 = -S0 * f1 / psi1
    dvs2 = -S0 * f2 / psi2
    d_psi = psi2 - psi1
    d_dvs = dvs2 - dvs1
    t_psi = -(1.0 / S0) * dvs1 * d_psi
    t_dvs = -(1.0 / S0) * psi1 * d_dvs
    t_cross = -(1.0 / S0) * d_psi * d_dvs
    return t_psi * 1e3, t_dvs * 1e3, t_cross * 1e3, (f2 - f1) * 1e3


def epoch_census(
    years: np.ndarray,
    psi: np.ndarray,
    fovs: np.ndarray,
    lengths: tuple[int, ...] = (10, 13, 15),
) -> dict:
    """Identity terms for every non-overlapping early/late epoch pair."""
    rows = []
    for length in lengths:
        n = len(years)
        for i in range(0, n - 2 * length + 1):
            for j in range(i + length, n - length + 1):
                e = slice(i, i + length)
                lt = slice(j, j + length)
                t_psi, t_dvs, t_cross, total = identity_terms(
                    float(psi[e].mean()),
                    float(psi[lt].mean()),
                    float(fovs[e].mean()),
                    float(fovs[lt].mean()),
                )
                rows.append(
                    {
                        "length": length,
                        "early": [int(years[e][0]), int(years[e][-1])],
                        "late": [int(years[lt][0]), int(years[lt][-1])],
                        "term_psi_mSv": t_psi,
                        "term_dvs_mSv": t_dvs,
                        "term_cross_mSv": t_cross,
                        "total_mSv": total,
                    }
                )
    arr_psi = np.array([r["term_psi_mSv"] for r in rows])
    arr_dvs = np.array([r["term_dvs_mSv"] for r in rows])
    arr_tot = np.array([r["total_mSv"] for r in rows])

    # Only pairs whose total change is not numerically negligible can support a
    # share; report the threshold used rather than silently dropping cases.
    thr = 5.0  # mSv
    usable = np.abs(arr_tot) > thr
    share = np.abs(arr_psi[usable]) / np.abs(arr_tot[usable])
    return {
        "n_pairs": len(rows),
        "epoch_lengths": list(lengths),
        "share_threshold_mSv": thr,
        "n_pairs_usable": int(usable.sum()),
        "frac_total_negative": float(np.mean(arr_tot < 0)),
        "frac_psi_term_opposes_total": float(
            np.mean(np.sign(arr_psi) != np.sign(arr_tot))
        ),
        "frac_dvs_term_same_sign_as_total": float(
            np.mean(np.sign(arr_dvs) == np.sign(arr_tot))
        ),
        "abs_share_psi_median": float(np.median(share)),
        "abs_share_psi_p90": float(np.percentile(share, 90)),
        "abs_share_psi_p95": float(np.percentile(share, 95)),
        "abs_share_psi_max": float(share.max()),
        "frac_abs_share_below_0p25": float(np.mean(share < 0.25)),
        "abs_share_dvs_median": float(
            np.median(np.abs(arr_dvs[usable]) / np.abs(arr_tot[usable]))
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Bootstrap intervals on the paper's epoch pairs
# ──────────────────────────────────────────────────────────────────────


def epoch_bootstrap(
    years: np.ndarray,
    psi: np.ndarray,
    fovs: np.ndarray,
    early: tuple[int, int],
    late: tuple[int, int],
) -> dict:
    m_e = (years >= early[0]) & (years <= early[1])
    m_l = (years >= late[0]) & (years <= late[1])
    point = identity_terms(
        float(psi[m_e].mean()),
        float(psi[m_l].mean()),
        float(fovs[m_e].mean()),
        float(fovs[m_l].mean()),
    )

    b_psi = np.empty(N_BOOT)
    b_dvs = np.empty(N_BOOT)
    b_tot = np.empty(N_BOOT)
    psi_e, fov_e = psi[m_e], fovs[m_e]
    psi_l, fov_l = psi[m_l], fovs[m_l]
    for i in range(N_BOOT):
        ie = block_indices(len(psi_e), BLOCK_YEARS, RNG)
        il = block_indices(len(psi_l), BLOCK_YEARS, RNG)
        t_psi, t_dvs, _, tot = identity_terms(
            float(psi_e[ie].mean()),
            float(psi_l[il].mean()),
            float(fov_e[ie].mean()),
            float(fov_l[il].mean()),
        )
        b_psi[i], b_dvs[i], b_tot[i] = t_psi, t_dvs, tot

    ratio = np.abs(b_psi) / np.abs(b_tot)
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {
        "early": list(early),
        "late": list(late),
        "term_psi_mSv": point[0],
        "term_psi_ci_mSv": pct(b_psi),
        "term_dvs_mSv": point[1],
        "term_dvs_ci_mSv": pct(b_dvs),
        "total_mSv": point[3],
        "total_ci_mSv": pct(b_tot),
        # A difference is well behaved where a ratio is not.
        "dvs_minus_psi_mSv": point[1] - point[0],
        "dvs_minus_psi_ci_mSv": pct(b_dvs - b_psi),
        "prob_dvs_term_larger_in_magnitude": float(
            np.mean(np.abs(b_dvs) > np.abs(b_psi))
        ),
        "prob_psi_term_opposes_total": float(np.mean(np.sign(b_psi) != np.sign(b_tot))),
        "abs_share_psi_point": abs(point[0]) / abs(point[3]),
        "abs_share_psi_ci": pct(ratio),
        "prob_abs_share_below_0p25": float(np.mean(ratio < 0.25)),
    }


# ──────────────────────────────────────────────────────────────────────


def barotropic_correction(cfg: dict) -> dict:
    out = {}
    for tag, fname in cfg["decomp"].items():
        with xr.open_dataset(RESULTS / fname) as ds:
            out[tag] = {
                "V_net_early_Sv": float(ds.attrs["V_net_early_Sv"]),
                "V_net_late_Sv": float(ds.attrs["V_net_late_Sv"]),
            }
    return out


def run_product(key: str, cfg: dict) -> dict:
    yrs_f, fovs = annual(RESULTS / cfg["fovs"], "F_ovS")
    yrs_p, psi = annual(RESULTS / cfg["moc"], "moc_upper")
    assert np.array_equal(yrs_f, yrs_p), f"{key}: year axes differ"
    years = yrs_f
    dvs = -S0 * fovs / psi

    out: dict = {
        "label": cfg["label"],
        "record": [int(years[0]), int(years[-1])],
        "continuous_full": continuous_attribution(years, psi, dvs, fovs),
        "epoch_census": epoch_census(years, psi, fovs),
        "epoch_bootstrap": {
            tag: epoch_bootstrap(years, psi, fovs, e, lt)
            for tag, (e, lt) in cfg["epochs"].items()
        },
        "barotropic_correction": barotropic_correction(cfg),
    }

    # Argo era, and for ORAS5 also the GLORYS12-matched window so the two
    # products are attributed over identical years.
    for tag, y0 in (("continuous_argo", 2004), ("continuous_1993", 1993)):
        m = years >= y0
        if m.sum() >= 15 and int(years[m][0]) != int(years[0]):
            out[tag] = continuous_attribution(years[m], psi[m], dvs[m], fovs[m])
    return out


def report(res: dict) -> None:
    for r in res.values():
        print(
            f"\n{'=' * 72}\n{r['label']}  {r['record'][0]}-{r['record'][1]}\n{'=' * 72}"
        )
        for tag in ("continuous_full", "continuous_1993", "continuous_argo"):
            c = r.get(tag)
            if c is None:
                continue
            print(
                f"\n  -- continuous attribution, {c['window'][0]}-{c['window'][1]} "
                f"(n={c['n_years']})"
            )
            print(
                f"     dF_ovS/dt      {c['trend_fovs_mSv_per_yr']:+7.3f} mSv/yr "
                f"[{c['trend_fovs_ci_mSv_per_yr'][0]:+.3f}, "
                f"{c['trend_fovs_ci_mSv_per_yr'][1]:+.3f}] "
                f"p={c['trend_fovs_p_santer']:.3f} N_eff={c['trend_fovs_n_eff']:.1f}"
            )
            print(
                f"     Psi term       {c['trend_psi_term_mSv_per_yr']:+7.3f} mSv/yr "
                f"[{c['trend_psi_term_ci_mSv_per_yr'][0]:+.3f}, "
                f"{c['trend_psi_term_ci_mSv_per_yr'][1]:+.3f}] "
                f"p={c['trend_psi_term_p_santer']:.3f} "
                f"N_eff={c['trend_psi_term_n_eff']:.1f}"
            )
            print(
                f"     DvS term       {c['trend_dvs_term_mSv_per_yr']:+7.3f} mSv/yr "
                f"[{c['trend_dvs_term_ci_mSv_per_yr'][0]:+.3f}, "
                f"{c['trend_dvs_term_ci_mSv_per_yr'][1]:+.3f}] "
                f"p={c['trend_dvs_term_p_santer']:.3f} "
                f"N_eff={c['trend_dvs_term_n_eff']:.1f}"
            )
            print(
                f"     |Psi|/|total|  {c['abs_share_psi_point']:.3f} "
                f"[{c['abs_share_psi_ci'][0]:.3f}, {c['abs_share_psi_ci'][1]:.3f}]  "
                f"P(share<0.25)={c['prob_abs_share_below_0p25']:.3f}  "
                f"P(opposes)={c['prob_psi_term_opposes_total']:.3f}"
            )
        e = r["epoch_census"]
        print(
            f"\n  -- epoch census: {e['n_pairs']} pairs, {e['n_pairs_usable']} with "
            f"|total| > {e['share_threshold_mSv']:.0f} mSv"
        )
        print(
            f"     total negative in {100 * e['frac_total_negative']:.1f}% of pairs; "
            f"Psi term opposes total in {100 * e['frac_psi_term_opposes_total']:.1f}%"
        )
        print(
            f"     |Psi term|/|total|: median {e['abs_share_psi_median']:.3f}, "
            f"p90 {e['abs_share_psi_p90']:.3f}, p95 {e['abs_share_psi_p95']:.3f}, "
            f"max {e['abs_share_psi_max']:.3f}; below 0.25 in "
            f"{100 * e['frac_abs_share_below_0p25']:.1f}%"
        )
        for tag, b in r["epoch_bootstrap"].items():
            print(
                f"\n  -- {tag}: {b['early'][0]}-{b['early'][1]} vs "
                f"{b['late'][0]}-{b['late'][1]}"
            )
            print(
                f"     Psi term {b['term_psi_mSv']:+7.2f} "
                f"[{b['term_psi_ci_mSv'][0]:+.2f}, {b['term_psi_ci_mSv'][1]:+.2f}] mSv"
            )
            print(
                f"     DvS term {b['term_dvs_mSv']:+7.2f} "
                f"[{b['term_dvs_ci_mSv'][0]:+.2f}, {b['term_dvs_ci_mSv'][1]:+.2f}] mSv"
            )
            print(
                f"     total    {b['total_mSv']:+7.2f} "
                f"[{b['total_ci_mSv'][0]:+.2f}, {b['total_ci_mSv'][1]:+.2f}] mSv"
            )
            print(
                f"     DvS minus Psi {b['dvs_minus_psi_mSv']:+7.2f} "
                f"[{b['dvs_minus_psi_ci_mSv'][0]:+.2f}, "
                f"{b['dvs_minus_psi_ci_mSv'][1]:+.2f}] mSv; "
                f"P(|DvS|>|Psi|)={b['prob_dvs_term_larger_in_magnitude']:.3f}"
            )
            print(
                f"     share {b['abs_share_psi_point']:.3f} "
                f"[{b['abs_share_psi_ci'][0]:.3f}, {b['abs_share_psi_ci'][1]:.3f}]  "
                f"P(share<0.25)={b['prob_abs_share_below_0p25']:.3f}"
            )
        for tag, v in r["barotropic_correction"].items():
            print(
                f"\n  -- barotropic correction, {tag}: net section transport "
                f"{v['V_net_early_Sv']:+.2f} Sv early, {v['V_net_late_Sv']:+.2f} Sv "
                f"late"
            )


def main() -> None:
    res = {k: run_product(k, cfg) for k, cfg in PRODUCTS.items()}
    report(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "identity_robustness.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
