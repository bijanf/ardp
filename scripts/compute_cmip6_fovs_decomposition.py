#!/usr/bin/env python3
"""CMIP6 mechanism decomposition: tie-breaker for the reanalysis disagreement.

For each CMIP6 model with 34.5S sections (vo + so) in historical AND
ssp585, compute the Δv / ΔS decomposition between:
  - Baseline period: 1950-1980  (historical)
  - Forced period  : 2080-2100  (ssp585)

The physical question: in coupled models that actually simulate a forced
AMOC weakening, does the F_ovS decline come primarily from velocity
change (Δv) or salinity change (ΔS)?

This provides the *physical ground truth* against which the observational
reanalyses (ORAS5, GLORYS12V1, SODA, ECCO) can be calibrated: the
reanalysis whose decomposition matches the forced-CMIP6 fingerprint is
the one capturing the true salt-advection feedback.

Reads:  data/cmip6_sections/{model}_{historical,ssp585}_{vo,so}.nc
Writes: data/results/fovs_decomposition_cmip6_{model}.nc
        data/results/fovs_decomposition_cmip6_summary.csv
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CMIP6_DIR = Path("data/cmip6_sections")
RESULTS_DIR = Path("data/results")

# Default windows. Can be overridden on the command line via
# --baseline-years and --forced-years for the timescale-consistency
# diagnostic (decadal 1993-2005 vs 2013-2025).
BASELINE_YEARS = (1950, 1980)
FORCED_YEARS = (2080, 2100)

# CESM-family models use POP with depth in centimeters — convert to m
POP_MODELS = {"CESM2", "CESM2-WACCM"}


def _find_models() -> list[str]:
    """Models with vo+so in both historical and ssp585."""
    files = [f.name for f in CMIP6_DIR.glob("*.nc")]
    have = {}
    for f in files:
        base = f[:-3]
        parts = base.rsplit("_", 1)
        if len(parts) != 2:
            continue
        have.setdefault(parts[0], set()).add(parts[1])

    models = []
    for key in have:
        if "vo" not in have[key] or "so" not in have[key]:
            continue
        if not key.endswith("_historical"):
            continue
        model = key[:-len("_historical")]
        ssp_key = f"{model}_ssp585"
        if ssp_key in have and "vo" in have[ssp_key] and "so" in have[ssp_key]:
            models.append(model)
    return sorted(models)


def _load_section(path: Path, var: str) -> tuple[xr.DataArray, np.ndarray, np.ndarray, str, str]:
    """Load (time, lev, x) section, returning (da, depth_m, lon_180, x_dim, lev_dim).

    Handles CESM POP (depth in cm), varying x-dim names (i, x, nlon).
    """
    ds = xr.open_dataset(path, decode_times=True, use_cftime=True)
    da = ds[var]
    # Determine x dimension
    # Note: some models name the Atlantic-section zonal index 'j' (CMCC)
    # or 'lon' (GISS) after j-index selection.
    x_dim = next((d for d in da.dims if d in ("i", "x", "nlon", "j", "lon")), None)
    lev_dim = next((d for d in da.dims if d in ("lev", "olevel", "depth", "z_t")), None)
    if x_dim is None or lev_dim is None:
        ds.close()
        raise RuntimeError(f"Cannot identify x/lev dims in {path.name}: {da.dims}")

    # Longitude coordinate
    lon = None
    for cand in ("longitude", "lon", "nav_lon"):
        if cand in ds:
            lon = ds[cand].values
            break
    if lon is None or lon.ndim != 1:
        ds.close()
        raise RuntimeError(f"No 1D longitude in {path.name}")
    # Normalize to -180..180
    lon_180 = np.where(lon > 180, lon - 360, lon)

    # Depth values
    depth = ds[lev_dim].values.astype(float)
    # POP: cm → m
    if any(model in path.name for model in POP_MODELS):
        depth = depth / 100.0

    return da, depth, lon_180, x_dim, lev_dim


def _period_mean(da: xr.DataArray, years: tuple[int, int]) -> np.ndarray:
    """Time-mean of (time, lev, x) section restricted to [y0, y1] inclusive."""
    y0, y1 = years
    t_year = da["time"].dt.year.values
    mask = (t_year >= y0) & (t_year <= y1)
    if mask.sum() == 0:
        raise RuntimeError(f"No timesteps in {y0}-{y1}")
    return da.isel(time=np.where(mask)[0]).mean(dim="time", skipna=True).values


def _grid_metrics(lon_180: np.ndarray, depth: np.ndarray, lat: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (atlantic_mask, e1t_all, e3t)."""
    atl = (lon_180 >= ATLANTIC_LON_MIN) & (lon_180 <= ATLANTIC_LON_MAX)
    dlon = np.diff(lon_180)
    # Fix wrap-around
    dlon = np.where(dlon > 180, dlon - 360, dlon)
    dlon = np.where(dlon < -180, dlon + 360, dlon)
    dlon = np.append(dlon, dlon[-1])
    e1t = np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(lat))
    e1t = np.clip(e1t, 1.0, None)

    # Vertical cell thickness
    order = np.argsort(depth)
    z_sorted = depth[order]
    e3t_sorted = np.diff(z_sorted, prepend=0.0)
    e3t = np.empty_like(e3t_sorted)
    e3t[order] = e3t_sorted
    return atl, e1t, e3t


def process_model(
    model: str,
    baseline_years: tuple[int, int] = BASELINE_YEARS,
    forced_years: tuple[int, int] = FORCED_YEARS,
    write_per_model_nc: bool = True,
) -> dict | None:
    """Run decomposition for one model. Returns summary dict or None if error.

    The decadal window pair (1993-2005 vs 2013-2025) requires that BOTH
    sample years come from concatenated historical+ssp585 records — the
    historical experiment only runs to 2014. We therefore concatenate
    the two records when at least one of the requested windows extends
    past 2014.
    """
    try:
        vo_hist_da, depth, lon_180, x_dim, lev_dim = _load_section(
            CMIP6_DIR / f"{model}_historical_vo.nc", "vo"
        )
        so_hist_da, *_ = _load_section(CMIP6_DIR / f"{model}_historical_so.nc", "so")
        vo_ssp_da, *_ = _load_section(CMIP6_DIR / f"{model}_ssp585_vo.nc", "vo")
        so_ssp_da, *_ = _load_section(CMIP6_DIR / f"{model}_ssp585_so.nc", "so")

        def _maybe_concat(hist_da, ssp_da):
            """Concat hist+ssp into one time series, deduplicating.
            Some CMIP6 models publish historical with calendar=gregorian
            but ssp585 with calendar=proleptic_gregorian (CNRM-CM6-1,
            EC-Earth3, MIROC6); xr.concat refuses to mix DatetimeGregorian
            and DatetimeProlepticGregorian objects. Normalize to one
            cftime type via the underlying datetime tuple, but only when
            the calendars actually differ — converting matched calendars
            corrupts xr.concat's dtype recognition (issue with noleap-to-
            proleptic round-trip)."""
            hist_cal = hist_da.time.encoding.get("calendar")
            ssp_cal = ssp_da.time.encoding.get("calendar")
            if hist_cal and ssp_cal and hist_cal != ssp_cal:
                try:
                    hist_da = hist_da.convert_calendar("proleptic_gregorian")
                    ssp_da = ssp_da.convert_calendar("proleptic_gregorian")
                except (TypeError, ValueError):
                    pass
            combined = xr.concat([hist_da, ssp_da], dim="time")
            t_year = combined["time"].dt.year.values
            t_month = combined["time"].dt.month.values
            tag = t_year * 100 + t_month
            _, unique = np.unique(tag, return_index=True)
            return combined.isel(time=sorted(unique))

        # Pick source da per window: historical if window ends ≤2014,
        # else concat(hist+ssp).
        def _choose(hist_da, ssp_da, years):
            return hist_da if years[1] <= 2014 else _maybe_concat(hist_da, ssp_da)

        vo_for_baseline = _choose(vo_hist_da, vo_ssp_da, baseline_years)
        so_for_baseline = _choose(so_hist_da, so_ssp_da, baseline_years)
        vo_for_forced = _choose(vo_hist_da, vo_ssp_da, forced_years)
        so_for_forced = _choose(so_hist_da, so_ssp_da, forced_years)

        v1 = _period_mean(vo_for_baseline, baseline_years)
        s1 = _period_mean(so_for_baseline, baseline_years)
        v2 = _period_mean(vo_for_forced, forced_years)
        s2 = _period_mean(so_for_forced, forced_years)

        # Ensure (lev, x) order
        for name, arr in [("v1", v1), ("s1", s1), ("v2", v2), ("s2", s2)]:
            if arr.ndim != 2:
                raise RuntimeError(f"{name} not 2D: shape={arr.shape}")

        atl, e1t, e3t = _grid_metrics(lon_180, depth, SAMBA_LAT)

        # Restrict to Atlantic
        v1_atl = v1[:, atl]
        s1_atl = s1[:, atl]
        v2_atl = v2[:, atl]
        s2_atl = s2[:, atl]
        e1t_atl = e1t[atl]

        # Handle salinity "land" convention — CMIP6 uses NaN typically but
        # some models use 0 or 1e20
        for arr in (s1_atl, s2_atl):
            arr[arr <= 0] = np.nan
            arr[arr > 100] = np.nan  # sensible upper bound

        # Fill NaN in velocity (land) with zero per decomposition kernel
        for arr in (v1_atl, v2_atl):
            np.nan_to_num(arr, copy=False, nan=0.0)

        result = decompose_fovs_trend(v1_atl, s1_atl, v2_atl, s2_atl, e1t_atl, e3t, s0=S0)

        v_frac = 100 * result["delta_v"] / result["delta_total"] if abs(result["delta_total"]) > 1e-6 else np.nan
        s_frac = 100 * result["delta_s"] / result["delta_total"] if abs(result["delta_total"]) > 1e-6 else np.nan

        log.info(
            f"  {model:20s}  F1={result['F_ov_1']:+.3f}  F2={result['F_ov_2']:+.3f}  "
            f"ΔF={result['delta_total']:+.3f} Sv   v:{v_frac:+.0f}%  s:{s_frac:+.0f}%"
        )

        # Save per-model netCDF (only for the default centennial run;
        # the diagnostic decadal run skips this to avoid clobbering).
        if write_per_model_nc:
            out_path = RESULTS_DIR / f"fovs_decomposition_cmip6_{model}.nc"
            ds_out = xr.Dataset(
                data_vars={
                    "depth_Sv_v": ("depth", result["depth_Sv_v"]),
                    "depth_Sv_s": ("depth", result["depth_Sv_s"]),
                    "depth_Sv_cross": ("depth", result["depth_Sv_cross"]),
                },
                coords={"depth": depth},
                attrs={
                    "model": model,
                    "baseline_period": f"{baseline_years[0]}-{baseline_years[1]}",
                    "forced_period": f"{forced_years[0]}-{forced_years[1]}",
                    "F_ov_baseline_Sv": result["F_ov_1"],
                    "F_ov_forced_Sv": result["F_ov_2"],
                    "delta_total_Sv": result["delta_total"],
                    "delta_v_Sv": result["delta_v"],
                    "delta_s_Sv": result["delta_s"],
                    "delta_cross_Sv": result["delta_cross"],
                    "residual_Sv": result["residual"],
                    "velocity_share_pct": float(v_frac),
                    "salinity_share_pct": float(s_frac),
                },
            )
            ds_out.to_netcdf(out_path)

        return {
            "model": model,
            "F_ov_baseline": result["F_ov_1"],
            "F_ov_forced": result["F_ov_2"],
            "delta_total": result["delta_total"],
            "delta_v": result["delta_v"],
            "delta_s": result["delta_s"],
            "delta_cross": result["delta_cross"],
            "velocity_share_pct": v_frac,
            "salinity_share_pct": s_frac,
            "residual": result["residual"],
        }
    except Exception as e:
        log.error(f"  {model}: FAILED ({e})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models (default: all with both sections)")
    parser.add_argument("--baseline-years", nargs=2, type=int,
                        default=list(BASELINE_YEARS), metavar=("Y0", "Y1"),
                        help="Inclusive baseline window (default: 1950 1980)")
    parser.add_argument("--forced-years", nargs=2, type=int,
                        default=list(FORCED_YEARS), metavar=("Y0", "Y1"),
                        help="Inclusive forced window (default: 2080 2100)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Summary CSV path (default: "
                             "data/results/fovs_decomposition_cmip6_summary.csv "
                             "for the centennial windows; otherwise must be "
                             "supplied explicitly)")
    args = parser.parse_args()

    baseline = tuple(args.baseline_years)
    forced = tuple(args.forced_years)
    is_default_window = baseline == BASELINE_YEARS and forced == FORCED_YEARS

    if args.output is not None:
        out_csv = args.output
    elif is_default_window:
        out_csv = RESULTS_DIR / "fovs_decomposition_cmip6_summary.csv"
    else:
        parser.error("--output must be specified when using non-default "
                     "windows so the centennial CSV is not overwritten")
        return

    models = args.models if args.models else _find_models()
    log.info(f"Decomposing {len(models)} CMIP6 models: "
             f"baseline({baseline[0]}-{baseline[1]}) → "
             f"forced({forced[0]}-{forced[1]})")

    rows = []
    for model in models:
        row = process_model(
            model,
            baseline_years=baseline,
            forced_years=forced,
            write_per_model_nc=is_default_window,
        )
        if row is not None:
            rows.append(row)

    # Summary CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved: {out_csv}")

    if rows:
        # Classify models
        weakening = [r for r in rows if r["delta_total"] < -0.01]  # > 10 mSv decline
        v_dominant = [r for r in weakening if r["velocity_share_pct"] > 60]
        s_dominant = [r for r in weakening if r["salinity_share_pct"] > 60]
        mixed = [r for r in weakening if r not in v_dominant and r not in s_dominant]

        log.info("")
        log.info(f"Summary across {len(weakening)} forced-weakening models:")
        log.info(f"  Velocity-dominant (>60% v): n={len(v_dominant)}")
        log.info(f"  Salinity-dominant (>60% s): n={len(s_dominant)}")
        log.info(f"  Mixed (40-60%)             : n={len(mixed)}")
        if weakening:
            v_pcts = [r["velocity_share_pct"] for r in weakening]
            s_pcts = [r["salinity_share_pct"] for r in weakening]
            log.info(
                f"  Ensemble mean: v = {np.nanmean(v_pcts):+.1f}%, "
                f"s = {np.nanmean(s_pcts):+.1f}%"
            )


if __name__ == "__main__":
    main()
