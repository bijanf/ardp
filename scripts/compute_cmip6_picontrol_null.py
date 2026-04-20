#!/usr/bin/env python3
"""Null-test: Δv / ΔS decomposition from CMIP6 piControl random segments.

For each CMIP6 model with piControl sections available, bootstrap the
mechanism decomposition between two random non-overlapping 30-year
segments. This establishes the null distribution of v-share and s-share
expected from INTERNAL variability alone (no external forcing).

The forced-regime result (ssp585 2080-2100 vs historical 1950-1980)
should lie OUTSIDE the null distribution. If not, the signal is
indistinguishable from internal variability and mechanism attribution is
unreliable.

Reads:  data/cmip6_sections/{model}_piControl_{vo,so}.nc
Writes: data/results/fovs_decomposition_cmip6_null.nc (bootstrap distribution)
        data/results/fovs_decomposition_cmip6_null.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0, SAMBA_LAT
from ardp.physics.fovs_decomposition import decompose_fovs_trend

# Reuse the loading helpers from the forced decomposition script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cmip6_fovs_decomposition import (  # noqa: E402
    CMIP6_DIR, _grid_metrics, _load_section,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
SEGMENT_YEARS = 30  # each bootstrap segment is 30 years (match forced-vs-baseline gap)


def _bootstrap_one(da_v: xr.DataArray, da_s: xr.DataArray,
                   atl: np.ndarray, e1t: np.ndarray, e3t: np.ndarray,
                   rng: np.random.Generator) -> dict | None:
    """Pick two random non-overlapping 30-year windows and decompose."""
    t_year = da_v["time"].dt.year.values
    y_min, y_max = int(t_year.min()), int(t_year.max())
    avail = y_max - y_min + 1
    if avail < 2 * SEGMENT_YEARS + 10:
        return None

    # Pick two random non-overlapping start years
    for _ in range(50):
        a = rng.integers(y_min, y_max - SEGMENT_YEARS + 1)
        b = rng.integers(y_min, y_max - SEGMENT_YEARS + 1)
        if abs(a - b) >= SEGMENT_YEARS:
            break
    else:
        return None

    def _window_mean(da, start):
        mask = (t_year >= start) & (t_year < start + SEGMENT_YEARS)
        return da.isel(time=np.where(mask)[0]).mean(dim="time", skipna=True).values

    v1 = _window_mean(da_v, min(a, b))
    s1 = _window_mean(da_s, min(a, b))
    v2 = _window_mean(da_v, max(a, b))
    s2 = _window_mean(da_s, max(a, b))

    v1_a = v1[:, atl]; s1_a = s1[:, atl]
    v2_a = v2[:, atl]; s2_a = s2[:, atl]
    for arr in (s1_a, s2_a):
        arr[arr <= 0] = np.nan
        arr[arr > 100] = np.nan
    for arr in (v1_a, v2_a):
        np.nan_to_num(arr, copy=False, nan=0.0)

    result = decompose_fovs_trend(v1_a, s1_a, v2_a, s2_a, e1t[atl], e3t, s0=S0)
    dtot = result["delta_total"]
    if abs(dtot) < 1e-6:
        return None
    return {
        "delta_total": float(dtot),
        "delta_v": float(result["delta_v"]),
        "delta_s": float(result["delta_s"]),
        "delta_cross": float(result["delta_cross"]),
        "velocity_share_pct": 100 * result["delta_v"] / dtot,
        "salinity_share_pct": 100 * result["delta_s"] / dtot,
    }


def process_model(model: str, n_bootstrap: int, rng: np.random.Generator) -> list[dict]:
    path_v = CMIP6_DIR / f"{model}_piControl_vo.nc"
    path_s = CMIP6_DIR / f"{model}_piControl_so.nc"
    if not (path_v.exists() and path_s.exists()):
        log.warning(f"  {model}: no piControl sections")
        return []

    try:
        da_v, depth, lon_180, _, _ = _load_section(path_v, "vo")
        da_s, _, _, _, _ = _load_section(path_s, "so")
    except Exception as e:
        log.error(f"  {model}: load failed ({e})")
        return []

    atl, e1t, e3t = _grid_metrics(lon_180, depth, SAMBA_LAT)

    rows = []
    for i in range(n_bootstrap):
        r = _bootstrap_one(da_v, da_s, atl, e1t, e3t, rng)
        if r is None:
            continue
        r["model"] = model
        r["draw"] = i
        rows.append(r)

    if rows:
        v_median = float(np.median([r["velocity_share_pct"] for r in rows]))
        s_median = float(np.median([r["salinity_share_pct"] for r in rows]))
        dtot_med = float(np.median([r["delta_total"] for r in rows]))
        log.info(
            f"  {model:20s}  n={len(rows)}  "
            f"ΔF median = {dtot_med * 1000:+.2f} mSv  "
            f"v_share [p25-p75] ≈ {v_median:+.0f}%  s_share ≈ {s_median:+.0f}%"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="Number of bootstrap draws per model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Find piControl models
    files = [f.name for f in CMIP6_DIR.glob("*_piControl_vo.nc")]
    models = sorted(f[:-len("_piControl_vo.nc")] for f in files)
    log.info(f"Bootstrap Δv/ΔS null distribution: {len(models)} models × {args.n_bootstrap} draws, "
             f"{SEGMENT_YEARS}-year segments")

    all_rows = []
    for m in models:
        all_rows.extend(process_model(m, args.n_bootstrap, rng))

    if not all_rows:
        log.error("No bootstrap results.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "fovs_decomposition_cmip6_null.csv", index=False)
    log.info(f"Saved: {RESULTS_DIR / 'fovs_decomposition_cmip6_null.csv'}  ({len(df)} rows)")

    log.info("")
    log.info("Aggregate null distribution:")
    log.info(f"  |ΔF| median in piControl: {np.median(np.abs(df['delta_total'])) * 1000:.2f} mSv")
    log.info(f"  v_share:  p5 = {np.percentile(df['velocity_share_pct'], 5):+.0f}%, "
             f"p95 = {np.percentile(df['velocity_share_pct'], 95):+.0f}%, "
             f"median = {np.median(df['velocity_share_pct']):+.0f}%")
    log.info(f"  s_share:  p5 = {np.percentile(df['salinity_share_pct'], 5):+.0f}%, "
             f"p95 = {np.percentile(df['salinity_share_pct'], 95):+.0f}%, "
             f"median = {np.median(df['salinity_share_pct']):+.0f}%")


if __name__ == "__main__":
    main()
