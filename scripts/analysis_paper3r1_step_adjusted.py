#!/usr/bin/env python3
"""Transport share of the F_ovS trend after fitting an observing-system step.

Each of F_ovS and its two counterfactuals, F_T(t) = -T(t) <dS> / S0 and
F_S(t) = -<T> dS(t) / S0, is fitted as a + b t + c I(t >= t_step), so that the
level shift and the trend are estimated jointly. The step-adjusted transport
share is |b_T| / |b_F|. The step years are the 2004 Argo transition for
GLORYS12V1 and the 2015 join of the consolidated and operational streams for
ORAS5.

Writes ``PAPER_3_v2/analysis/r1_step_adjusted.json``.
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
    N_BOOT,
    PRIMARY_BLOCK,
    S0,
    annual,
    block_idx,
)

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_step_adjusted.json"
STEPS = {"glorys12": 2004, "oras5": 2015}


def fit(x: np.ndarray, y: np.ndarray, step: int) -> np.ndarray:
    a = np.column_stack([np.ones_like(x), x, (x >= step).astype(float)])
    return np.linalg.lstsq(a, y, rcond=None)[0]


def main() -> None:
    out = {}
    for key, step in STEPS.items():
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
        ann = annual(ds)
        x = ann["year"].values.astype(float)
        t = ann["T_limb_Sv"].values
        d = ann["dS_limb"].values
        f = ann["F_ov"].values
        series = {"F": f, "T": -(1 / S0) * t * d.mean(), "S": -(1 / S0) * t.mean() * d}
        coefs = {k: fit(x, v, step) for k, v in series.items()}
        boot = {k: np.empty(N_BOOT) for k in series}
        for k, v in series.items():
            a = np.column_stack([np.ones_like(x), x, (x >= step).astype(float)])
            fitted = a @ coefs[k]
            resid = v - fitted
            for i in range(N_BOOT):
                idx = block_idx(len(x), PRIMARY_BLOCK)
                boot[k][i] = fit(x, fitted + resid[idx], step)[1]
        ratio = np.abs(boot["T"]) / np.abs(boot["F"])
        out[key] = {
            "step_year": step,
            "trend_F_mSv_per_yr": float(coefs["F"][1] * 1e3),
            "step_F_Sv": float(coefs["F"][2]),
            "trend_T_term_mSv_per_yr": float(coefs["T"][1] * 1e3),
            "trend_S_term_mSv_per_yr": float(coefs["S"][1] * 1e3),
            "share_T_point": float(abs(coefs["T"][1] / coefs["F"][1])),
            "share_T_ci": [
                float(np.percentile(ratio, 2.5)),
                float(np.percentile(ratio, 97.5)),
            ],
        }
        print(
            key,
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in out[key].items()
            },
        )
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
