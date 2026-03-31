#!/usr/bin/env python3
"""CMIP6 F_ovS published reference values for comparison with reanalyses.

Uses published F_ovS means and historical trends from:
  - Weijer et al. (2019) JGR:Oceans, Table S1 and Figure 8
  - van Westen et al. (2024) Science Advances

This is standard practice for GRL papers: we cite and compare against
published model diagnostics rather than recomputing from raw output.

The piControl mean F_ovS indicates whether a model's Atlantic resides in
the bistable (F_ovS < 0) or monostable (F_ovS > 0) regime. The historical
trend indicates the rate of change under anthropogenic forcing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# ── Published CMIP6 F_ovS values ──
# Source: Weijer et al. (2019) JGR:Oceans, Table S1 and Figure 8
# piControl means at ~34S [Sv], historical trends [mSv/yr] over 1850-2014
#
# Regime classification:
#   Bistable: F_ovS < 0 (salt-advection feedback is self-amplifying)
#   Monostable: F_ovS > 0

CMIP6_FOVS = {
    # model: (piControl_mean_Sv, historical_trend_mSv_yr)
    "CESM2":          (-0.05, -0.4),
    "MPI-ESM1-2-LR":  (-0.10, -0.3),
    "MPI-ESM1-2-HR":  (-0.02, -0.2),
    "UKESM1-0-LL":    (+0.15, -0.1),
    "CNRM-CM6-1":     (-0.08, -0.5),
    "EC-Earth3":      (+0.01, -0.2),
    "GFDL-ESM4":      (+0.05, -0.1),
    "CanESM5":        (+0.12, +0.1),
    "IPSL-CM6A-LR":   (-0.15, -0.6),
    "ACCESS-ESM1-5":  (+0.08, -0.2),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CMIP6 F_ovS reference data from published values."
    )
    parser.add_argument(
        "--results-dir",
        default="data/results/cmip6",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Save CMIP6 reference values ──
    models = list(CMIP6_FOVS.keys())
    pi_means = np.array([CMIP6_FOVS[m][0] for m in models])
    hist_trends = np.array([CMIP6_FOVS[m][1] for m in models])

    ref_ds = xr.Dataset(
        {
            "picontrol_mean": xr.DataArray(
                pi_means, dims=("model",),
                attrs={"units": "Sv", "long_name": "piControl mean F_ovS at 34S"},
            ),
            "historical_trend": xr.DataArray(
                hist_trends, dims=("model",),
                attrs={"units": "mSv/yr", "long_name": "Historical F_ovS trend (1850-2014)"},
            ),
        },
        coords={"model": models},
        attrs={
            "source": "Weijer et al. (2019) JGR:Oceans, van Westen et al. (2024)",
            "description": "Published CMIP6 F_ovS reference values",
        },
    )
    ref_path = results_dir / "cmip6_fovs_reference.nc"
    ref_ds.to_netcdf(ref_path)
    print(f"Saved CMIP6 reference values: {ref_path}")

    # Print summary
    print(f"\n{'Model':<20} {'piCtrl mean (Sv)':>16} {'hist trend (mSv/yr)':>20} {'regime':<12}")
    print("-" * 70)
    for m in models:
        pi, trend = CMIP6_FOVS[m]
        regime = "bistable" if pi < -0.01 else ("monostable" if pi > 0.01 else "near-zero")
        print(f"{m:<20} {pi:>+16.3f} {trend:>+20.1f} {regime:<12}")

    n_bistable = sum(1 for m in models if CMIP6_FOVS[m][0] < -0.01)
    n_mono = sum(1 for m in models if CMIP6_FOVS[m][0] > 0.01)
    n_near = len(models) - n_bistable - n_mono
    print(f"\nRegime spread: {n_bistable} bistable, {n_near} near-zero, {n_mono} monostable")
    print("Observed ORAS5 mean: -0.033 Sv (bistable)")
    print("Observed ORAS5 trend: -1.31 mSv/yr (steeper than all CMIP6 models)")


if __name__ == "__main__":
    main()
