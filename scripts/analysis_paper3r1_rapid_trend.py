#!/usr/bin/env python3
"""Trend of the RAPID overturning at 26.5 N over its complete years.

Reads the monthly RAPID transports (data/external/rapid_moc_monthly.nc,
variable moc_mar_hc10), forms calendar-year means for the years with twelve
months of data, and tests the linear trend with the corrected effective sample
size of ``paper3r1_stats.py``. The annual series written by
``compute_yearly_amoc26n.py`` (data/results/rapid_amoc26n.npz) is tested too,
as a cross-check.

Writes ``PAPER_3_v2/analysis/r1_rapid.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import trend_test  # noqa: E402

OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_rapid.json"


def summary(years: np.ndarray, values: np.ndarray) -> dict:
    r = trend_test(years.astype(float), values)
    return {
        "window": [int(years[0]), int(years[-1])],
        "n_years": int(len(years)),
        "mean_Sv": float(values.mean()),
        "trend_Sv_per_decade": r["value"] * 10,
        "ci_Sv_per_decade": [c * 10 for c in r["ci"]],
        "p": r["p"],
        "p_ols": r["p_ols"],
        "n_eff": r["n_eff"],
        "lag1": r["lag1"],
    }


def main() -> None:
    ds = xr.open_dataset(REPO / "data" / "external" / "rapid_moc_monthly.nc")
    moc = ds["moc_mar_hc10"].to_series()
    moc.index = pd.DatetimeIndex(moc.index)
    counts = moc.groupby(moc.index.year).count()
    full = counts[counts == 12].index
    ann = moc.groupby(moc.index.year).mean().loc[full]
    out = {"monthly_file": summary(ann.index.values, ann.values)}
    npz = REPO / "data" / "results" / "rapid_amoc26n.npz"
    if npz.exists():
        z = np.load(npz)
        out["annual_npz"] = summary(z["years"], z["amoc"])
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
