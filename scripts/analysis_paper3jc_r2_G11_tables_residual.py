#!/usr/bin/env python3
"""Residual interval of the upper-cell trend decomposition (Fig. 6c) and the
split of the packed SAMBA row of Table S7(d) (paper 3, revision round 2, G11).

Residual. For the ORAS5 upper (NADW) cell, with annual means F_up, T_up and
dS_up from data/results/paper3r1_cells_oras5.nc, the decomposition of
analysis_paper3v2_attribution.continuous is

    F_up trend = T term + S term + residual,

where the T term freezes dS at its window mean and the S term freezes T at its
window mean. The residual is the total trend minus the two terms (point values
equal those of r1_cells.json). Its interval is a percentile bootstrap from the
joint AR(1) sieve of paper3r1_stats.sieve_coefficients applied to the three
series F_up, the T term and the S term at the same time indices, so that every
term is drawn from the same resample. Seed 20261104, 10,000 resamples. The
intervals of the two terms and of the total are reported from the same draws.

SAMBA split. Table S7(d) packs, for each SAMBA cell, the standard-deviation
ratio with its interval, the correlation and the p value. The values are read
from PAPER_3_v2/analysis/jc_n10_samba.json (pairs, ORAS5), where the standard
deviation ratio and its approximate F interval, the correlation, its p value and
the effective sample size are stored. No new SAMBA computation is made.

Inputs: data/results/paper3r1_cells_oras5.nc; PAPER_3_v2/analysis/jc_n10_samba.json.
Output: PAPER_3_v2/analysis/jc_r2_G11_tables_residual.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_attribution import (  # noqa: E402
    ARGO_START,
    COMMON_WINDOW,
    S0,
    annual,
    continuous,
)
from paper3r1_stats import sieve_coefficients, trend_design  # noqa: E402

SEED = 20261104
N_BOOT = 10_000
RESULTS = REPO / "data" / "results"
ANA = REPO / "PAPER_3_v2" / "analysis"
OUT = ANA / "jc_r2_G11_tables_residual.json"


def interval(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def windows(years: np.ndarray) -> dict:
    w = {"own": (int(years[0]), int(years[-1]))}
    w["argo"] = (ARGO_START, int(years[-1]))
    if years[0] <= COMMON_WINDOW[0] and years[-1] >= COMMON_WINDOW[1]:
        w["common"] = COMMON_WINDOW
    return w


def residual_block(years: np.ndarray, t: np.ndarray, d: np.ndarray, f: np.ndarray,
                   rng: np.random.Generator) -> dict:
    """Point decomposition (as in continuous) and the joint bootstrap of its terms."""
    point = continuous(years, t, d, f)
    t_bar, d_bar = float(t.mean()), float(d.mean())
    f_t = -(1.0 / S0) * t * d_bar
    f_d = -(1.0 / S0) * t_bar * d
    x = trend_design(years.astype(float))
    draws = sieve_coefficients(x, np.column_stack([f, f_t, f_d]), 1, rng, n_boot=N_BOOT)
    resid = (draws[:, 0] - draws[:, 1] - draws[:, 2]) * 1e3
    tot = draws[:, 0] * 1e3
    tt = draws[:, 1] * 1e3
    ss = draws[:, 2] * 1e3
    return {
        "n_years": int(len(years)),
        "window": [int(years[0]), int(years[-1])],
        "trend_total_mSv_per_yr": point["trend_total_mSv_per_yr"],
        "trend_T_term_mSv_per_yr": point["trend_T_mSv_per_yr"],
        "trend_S_term_mSv_per_yr": point["trend_S_mSv_per_yr"],
        "residual_mSv_per_yr": point["residual_mSv_per_yr"],
        "residual_ci95_bootstrap_mSv_per_yr": interval(resid),
        "residual_bootstrap_median_mSv_per_yr": float(np.median(resid)),
        "residual_bootstrap_sd_mSv_per_yr": float(np.std(resid)),
        "residual_ci_contains_zero": bool(
            np.percentile(resid, 2.5) <= 0.0 <= np.percentile(resid, 97.5)
        ),
        "total_ci95_bootstrap_mSv_per_yr": interval(tot),
        "T_term_ci95_bootstrap_mSv_per_yr": interval(tt),
        "S_term_ci95_bootstrap_mSv_per_yr": interval(ss),
        "T_term_ci95_from_test_mSv_per_yr": point["ci_T_mSv_per_yr"],
        "S_term_ci95_from_test_mSv_per_yr": point["ci_S_mSv_per_yr"],
        "total_ci95_from_test_mSv_per_yr": point["ci_total_mSv_per_yr"],
        "n_eff_total": point["n_eff_total"],
        "n_eff_T": point["n_eff_T"],
        "n_eff_S": point["n_eff_S"],
        "residual_share_of_total_point": point["residual_mSv_per_yr"]
        / point["trend_total_mSv_per_yr"],
        "residual_share_of_total_ci95_bootstrap": interval(resid / tot),
        "sieve_resamples": N_BOOT,
    }


def upper_cell_residuals() -> dict:
    ds = xr.open_dataset(RESULTS / "paper3r1_cells_oras5.nc")
    ann = annual(ds)
    years = ann["year"].values
    rng = np.random.default_rng(SEED)
    out = {}
    for wname, (y0, y1) in windows(years).items():
        m = (years >= y0) & (years <= y1)
        y = years[m]
        out[wname] = residual_block(
            y,
            ann["T_up"].values[m],
            ann["dS_up"].values[m],
            ann["F_up"].values[m],
            rng,
        )
    return out


def samba_split() -> dict:
    pj = json.loads((ANA / "jc_n10_samba.json").read_text())["pairs"]["ORAS5"]
    up = pj["up_cell__samba_up"]
    ab_key = [k for k in pj if k.startswith("ab_cell") and "samba" in k]
    if len(ab_key) != 1:
        raise KeyError(f"abyssal pair key not unique: {ab_key}")
    ab = pj[ab_key[0]]

    def row(p: dict) -> dict:
        return {
            "n_months": p["n"],
            "std_ratio_ORAS5_over_SAMBA": p["std_ratio_x_over_y"],
            "std_ratio_ci95_approx_F": p["std_ratio_ci_95"],
            "correlation": p["r"],
            "p_value_bretherton_neff": p["p"],
            "n_eff_bretherton": p["n_eff"],
            "correlation_ci95_fisher_approx": p["r_ci_95_fisher_approx"],
        }

    return {
        "source_key": "pairs/ORAS5/up_cell__samba_up and pairs/ORAS5/" + ab_key[0],
        "period": "Sep 2013 to Jul 2017 (SAMBA record in the repository)",
        "upper_cell_psi_max_vs_SAMBA_upper": row(up),
        "abyssal_cell_vs_SAMBA_abyssal": row(ab),
        "table_S7d_packed_row_check": {
            "upper": f"std ratio {up['std_ratio_x_over_y']:.2f}; correlation "
            f"{up['r']:.2f}; p {up['p']:.3f}",
            "abyssal": f"std ratio {ab['std_ratio_x_over_y']:.2f}; correlation "
            f"{ab['r']:.2f}; p {ab['p']:.3f}",
        },
    }


def main() -> None:
    res = {
        "meta": {
            "id": "jc_r2_G11_tables_residual",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G11_tables_residual.py",
            "inputs": [
                "data/results/paper3r1_cells_oras5.nc",
                "PAPER_3_v2/analysis/jc_n10_samba.json",
            ],
            "seeds": {"residual_sieve": SEED},
            "replicates": {"residual_sieve_resamples": N_BOOT},
            "units": "mSv per year for trends; dimensionless for shares and "
            "correlations; Sv for SAMBA standard deviations",
            "method": "continuous() decomposition of the ORAS5 upper cell; joint "
            "AR(1) sieve (paper3r1_stats.sieve_coefficients) for the bootstrap",
        },
        "upper_cell_residual_ORAS5": upper_cell_residuals(),
        "samba_table_S7d": samba_split(),
    }
    OUT.write_text(json.dumps(res, indent=1))
    for w, b in res["upper_cell_residual_ORAS5"].items():
        print(
            w,
            round(b["residual_mSv_per_yr"], 4),
            [round(v, 4) for v in b["residual_ci95_bootstrap_mSv_per_yr"]],
        )
    print(res["samba_table_S7d"]["table_S7d_packed_row_check"])


if __name__ == "__main__":
    main()
