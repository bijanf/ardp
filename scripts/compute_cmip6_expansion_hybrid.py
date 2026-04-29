#!/usr/bin/env python3
"""Hybrid CMIP6 expansion: Pangeo cloud preferred, ESGF fallback.

For each new model in NEW_MODELS, extract the 34.5 deg S section for
(historical, ssp585) x (vo, so) and save to data/cmip6_sections/.
For each (model, exp, var), the script:

  1. Looks up Pangeo CMIP6 cloud catalog. If a zarr store exists at
     gn (native) grid for r1i1p1f1, opens it lazily, slices to the
     34.5 deg S row, materialises the section, saves to NetCDF, done.
  2. Otherwise falls back to the ESGF HTTP-download path
     (compute_cmip6_expansion_esgf.py). Slow but always works when
     files exist on any ESGF mirror.

Resume-friendly: skips files already present.

Outputs:
    data/cmip6_sections/{model}_{historical,ssp585}_{vo,so}.nc

After completion, run compute_cmip6_fovs_decomposition.py to update
the headline CSV.
"""
from __future__ import annotations

import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_cmip6_expansion_esgf import (
    EXP_YEARS, NEW_MODELS, SCRATCH_BASE, _process_model_exp_var,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
SECTIONS_DIR = REPO / "data" / "cmip6_sections"

TARGET_LAT = -34.5
PANGEO_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


def _find_pangeo_zstore(cat_df, model, exp, var) -> str | None:
    """Find Pangeo zarr URL for (model, exp, var) at native grid r1i1p1f1."""
    sub = cat_df[
        (cat_df.source_id == model)
        & (cat_df.experiment_id == exp)
        & (cat_df.variable_id == var)
        & (cat_df.member_id == "r1i1p1f1")
        & (cat_df.grid_label == "gn")
        & (cat_df.table_id == "Omon")
    ]
    if len(sub) == 0:
        return None
    return sub.iloc[0]["zstore"]


def _find_j(ds: xr.Dataset) -> tuple[int, str, str, float]:
    """Locate j-index nearest TARGET_LAT in 1D or 2D lat coords."""
    lat_name = None
    for n in ["lat", "latitude", "nav_lat"]:
        if n in ds.coords or n in ds.data_vars:
            lat_name = n
            break
    if lat_name is None:
        for n in list(ds.coords) + list(ds.data_vars):
            if "lat" in str(n).lower():
                lat_name = n
                break
    if lat_name is None:
        raise ValueError("no lat coord")
    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 1:
        j = int(np.abs(lat_vals - TARGET_LAT).argmin())
        return j, lat_name, lat_name, float(lat_vals[j])
    else:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j = int(np.abs(lat_1d - TARGET_LAT).argmin())
        return j, ds[lat_name].dims[0], lat_name, float(lat_1d[j])


def _pangeo_extract(zstore: str, model: str, exp: str, var: str,
                    out_path: Path) -> bool:
    """Open zarr, slice to 34.5S, save section."""
    log.info(f"  [Pangeo] opening {zstore.split('/')[-3]}...")
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_zarr(zstore, consolidated=True,
                           decode_times=time_coder)
    except Exception as e:
        log.warning(f"  [Pangeo] open_zarr failed: {e}")
        return False

    # Restrict to year window of interest
    y_lo, y_hi = EXP_YEARS[exp]
    try:
        ds = ds.sel(time=slice(f"{y_lo}-01-01", f"{y_hi}-12-31"))
    except Exception as e:
        log.warning(f"  [Pangeo] time slice failed: {e}")
        return False

    if var not in ds.data_vars:
        log.warning(f"  [Pangeo] {var} not in dataset")
        return False

    if ds.sizes.get("time", 0) == 0:
        log.warning(f"  [Pangeo] empty time dim after slice -> ESGF fallback")
        return False

    try:
        j_idx, j_dim, lat_name, actual_lat = _find_j(ds)
    except ValueError as e:
        log.warning(f"  [Pangeo] {e}")
        return False
    log.info(f"  [Pangeo] section at j={j_idx} (lat={actual_lat:.2f}), "
             f"j_dim='{j_dim}'")

    section = ds.isel({j_dim: j_idx})[[var]]
    section[var].attrs.update({
        "source_id": model, "experiment_id": exp,
        "grid_label": "gn", "member_id": "r1i1p1f1",
        "section_latitude": float(actual_lat),
        "section_j_index": int(j_idx),
    })

    log.info(f"  [Pangeo] loading + saving (typically 30s-3min) ...")
    import time as _t
    t0 = _t.time()
    section.load()
    elapsed = _t.time() - t0
    nbytes = section[var].nbytes / 1e6
    log.info(f"  [Pangeo] transferred {nbytes:.0f} MB in {elapsed:.0f}s "
             f"({nbytes/max(elapsed,1):.1f} MB/s)")
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    section.to_netcdf(out_path, format="NETCDF4")
    log.info(f"  [Pangeo] wrote {out_path.name} "
             f"({out_path.stat().st_size/1e6:.0f} MB)")
    section.close()
    ds.close()
    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated subset of NEW_MODELS to run "
                             "(default: all 9). Use to launch parallel jobs.")
    args = parser.parse_args()
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            if m not in NEW_MODELS:
                log.warning(f"  unknown model '{m}' (not in NEW_MODELS)")
    else:
        models = list(NEW_MODELS)

    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_BASE.mkdir(parents=True, exist_ok=True)

    log.info(f"Processing {len(models)} model(s): {models}")
    log.info("Opening Pangeo CMIP6 cloud catalog ...")
    import intake
    cat = intake.open_esm_datastore(PANGEO_URL)
    cat_df = cat.df

    successes = 0
    pangeo_used = 0
    esgf_used = 0
    failures: list[str] = []
    total = 0
    for i, model in enumerate(models, 1):
        log.info(f"\n=== [{i}/{len(models)}] {model} ===")
        scratch = SCRATCH_BASE / model
        for exp in ("historical", "ssp585"):
            for var in ("vo", "so"):
                total += 1
                out_path = SECTIONS_DIR / f"{model}_{exp}_{var}.nc"
                if out_path.exists() and out_path.stat().st_size > 0:
                    log.info(f"  [cache] {out_path.name}")
                    successes += 1
                    continue
                # Try Pangeo first
                zstore = _find_pangeo_zstore(cat_df, model, exp, var)
                if zstore is not None:
                    ok = _pangeo_extract(zstore, model, exp, var, out_path)
                    if ok:
                        successes += 1
                        pangeo_used += 1
                        continue
                    log.warning(f"  Pangeo failed, falling back to ESGF")
                # Fall back to ESGF download
                log.info(f"  -- ESGF fallback {model}/{exp}/{var} --")
                ok = _process_model_exp_var(model, exp, var, scratch)
                if ok:
                    successes += 1
                    esgf_used += 1
                else:
                    failures.append(f"{model}/{exp}/{var}")
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    log.info(f"\nDone. {successes}/{total} (model,exp,var) combos completed.")
    log.info(f"  Pangeo used: {pangeo_used} | ESGF fallback: {esgf_used}")
    if failures:
        log.warning(f"FAILURES ({len(failures)}):")
        for f in failures:
            log.warning(f"  - {f}")


if __name__ == "__main__":
    main()
