#!/usr/bin/env python3
"""Compute F_ovS at 34.5S from ECCO-V4r4 (NASA JPL, 1992-2017).

Streams both meridional velocity (NVEL) and salinity (SALT) from NASA
Earthdata via earthaccess, extracts the 34.5S section, and computes F_ovS
via the shared kernel in ardp.physics.fovs.

Products used:
- ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4        (NVEL, EVEL, WVEL)
- ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4    (SALT, THETA)

Both are on the same 0.5-degree lat-lon-Z grid (no staggering). Z is
negative (depth below surface); we take |Z| for thickness calculations.

Usage:
    python scripts/compute_ecco_fovs.py
Requires NASA Earthdata credentials in ~/.netrc.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import earthaccess
import numpy as np
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ardp.constants import ATLANTIC_LON_MAX, ATLANTIC_LON_MIN, S0, SAMBA_LAT
from ardp.physics.fovs import compute_fovs_from_section

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

VEL_PRODUCT = "ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4"
SAL_PRODUCT = "ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════════════

class FovsCheckpoint:
    def __init__(self, product: str):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHECKPOINT_DIR / f"fovs_{product}.json"
        self.data: dict[int, float] = {}
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            self.data = {int(k): v for k, v in raw.items()}
            log.info(
                f"Resumed checkpoint: {len(self.data)} years "
                f"({min(self.data)}-{max(self.data)})"
            )

    def has(self, year: int) -> bool:
        return year in self.data

    def add(self, year: int, fovs: float):
        self.data[year] = fovs
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        years = sorted(self.data.keys())
        return np.array(years), np.array([self.data[y] for y in years])


# ═══════════════════════════════════════════════════════════════════════
# Grid setup (shared across the run)
# ═══════════════════════════════════════════════════════════════════════

def build_grid(ds: xr.Dataset) -> dict:
    """Extract j-index, Atlantic mask, and grid metrics from an ECCO dataset."""
    lat = ds["latitude"].values
    lon = ds["longitude"].values  # already in -180..180
    z = np.abs(ds["Z"].values.astype(float))  # positive depth [m]

    j_idx = int(np.abs(lat - SAMBA_LAT).argmin())
    actual_lat = float(lat[j_idx])

    atl = (lon >= ATLANTIC_LON_MIN) & (lon <= ATLANTIC_LON_MAX)

    # Zonal spacing at this latitude
    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    e1t = np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(actual_lat))
    e1t = np.clip(e1t, 1.0, None)

    # Vertical thickness from depth levels (Z is negative; flip to positive)
    order = np.argsort(z)
    z_sorted = z[order]
    e3t_sorted = np.diff(z_sorted, prepend=0.0)
    # Reorder back to original Z order
    e3t = np.empty_like(e3t_sorted)
    e3t[order] = e3t_sorted

    return {
        "j_idx": j_idx,
        "actual_lat": actual_lat,
        "z": z,
        "e1t_atl": e1t[atl],
        "e3t": e3t,
        "atl": atl,
        "n_atl": int(atl.sum()),
        "order": order,
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-year processing
# ═══════════════════════════════════════════════════════════════════════

ECCO_CACHE = Path("data/ecco")


def _download_year(short_name: str, year: int, subdir: str) -> list[Path]:
    """Download monthly granules for one year to local cache (resumable)."""
    r = earthaccess.search_data(short_name=short_name, temporal=(f"{year}-01", f"{year}-12"))
    if not r:
        raise RuntimeError(f"{short_name} {year}: no granules")
    target = ECCO_CACHE / subdir / str(year)
    target.mkdir(parents=True, exist_ok=True)
    # earthaccess.download skips files already in target
    earthaccess.download(r, local_path=str(target), threads=4)
    return sorted(target.glob("*.nc"))


def process_one_year(year: int, grid_ref: dict | None) -> tuple[float, dict]:
    """Download monthly granules locally, compute annual-mean F_ovS."""
    files_v = _download_year(VEL_PRODUCT, year, "vel")
    files_s = _download_year(SAL_PRODUCT, year, "sal")
    if not files_v or not files_s:
        raise RuntimeError(f"{year}: no local files after download ({len(files_v)} v, {len(files_s)} s)")

    ds_v = xr.open_mfdataset(files_v, combine="by_coords", parallel=False)
    ds_s = xr.open_mfdataset(files_s, combine="by_coords", parallel=False)

    grid = grid_ref if grid_ref is not None else build_grid(ds_v)

    v_annual = ds_v["NVEL"].mean(dim="time").isel(latitude=grid["j_idx"]).load()
    s_annual = ds_s["SALT"].mean(dim="time").isel(latitude=grid["j_idx"]).load()
    ds_v.close(); ds_s.close()

    v_atl = v_annual.values[:, grid["atl"]]
    s_atl = s_annual.values[:, grid["atl"]]
    s_atl = np.where(s_atl > 0, s_atl, np.nan)  # ECCO uses 0 for land

    fovs = compute_fovs_from_section(v_atl, s_atl, grid["e1t_atl"], grid["e3t"], s0=S0)
    return fovs, grid


def main() -> None:
    log.info("=" * 50)
    log.info("ECCO-V4r4  F_ovS at 34.5S  (1992-2017)")
    log.info("=" * 50)

    earthaccess.login(strategy="netrc", persist=False)
    log.info("Authenticated with NASA Earthdata")

    ckpt = FovsCheckpoint("ecco")
    grid: dict | None = None

    for year in range(1992, 2018):
        if ckpt.has(year):
            continue
        try:
            f, grid = process_one_year(year, grid)
            if not np.isfinite(f):
                log.warning(f"  {year}: non-finite F_ovS, skipping")
                continue
            ckpt.add(year, float(f))
            log.info(f"  {year}: F_ovS = {f:+.4f} Sv")
            gc.collect()
        except Exception as e:
            log.error(f"  {year}: {e}")
            continue

    years, fovs = ckpt.to_arrays()
    if len(years) == 0:
        log.error("No years processed.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ecco_f_ovs.nc"
    lat_val = grid["actual_lat"] if grid is not None else -34.5
    da = xr.DataArray(
        fovs,
        dims=("year",),
        coords={"year": years},
        name="F_ovS",
        attrs={
            "units": "Sv",
            "long_name": f"Overturning freshwater transport at {lat_val:.1f}S",
            "section_latitude": lat_val,
            "product": "ECCO-V4r4",
            "source": "NASA Earthdata / earthaccess",
        },
    )
    da.to_netcdf(out)
    log.info(f"Saved: {out}  ({len(years)} years)")
    log.info(f"Mean F_ovS = {float(np.mean(fovs)):+.4f} Sv")
    log.info(f"Range      = [{float(np.min(fovs)):+.4f}, {float(np.max(fovs)):+.4f}] Sv")

    if len(years) > 3:
        sl, _, _, pv, _ = stats.linregress(years, fovs)
        log.info(f"Trend      = {sl * 1000:+.2f} mSv/yr (p={pv:.3e})")


if __name__ == "__main__":
    main()
