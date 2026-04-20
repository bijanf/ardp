#!/usr/bin/env python3
"""Compute F_ovS at 34.5S from SODA 3.15.2 (UMD, 1980-2022).

Follows the download+checkpoint pattern of compute_reanalysis_amoc.py but
extracts BOTH velocity (v) and salinity (salt) at the SAMBA latitude and
computes F_ovS via the shared kernel in ardp.physics.fovs.

Grid notes (SODA 3.15.2 regridded 0.5 degree):
- v is on (yu_ocean, xu_ocean) — U-grid
- salt is on (yt_ocean, xt_ocean) — T-grid
- C-grid stagger is ~0.25 degrees; we take nearest j-index on each grid
  independently and use the v grid for zonal spacing. At 34.5S this
  introduces negligible error for a zonally-averaged quantity.
- Longitudes go 0-360; Atlantic at 34.5S spans [290, 360] U [0, 20].

Usage:
    python scripts/compute_soda_fovs.py
"""

from __future__ import annotations

import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

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
SODA_DATA_DIR = Path("data/soda")
BASE_URL = "https://dsrs.atmos.umd.edu/DATA/soda3.15.2/REGRIDED/ocean/"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════════════

class FovsCheckpoint:
    """Per-year F_ovS checkpoint; resumable across runs."""

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
# Grid setup
# ═══════════════════════════════════════════════════════════════════════

def build_section_grid(ref_file: Path) -> dict:
    """Extract j-indices, grid spacings, and Atlantic mask from a reference file."""
    ds = xr.open_dataset(ref_file, decode_timedelta=False)

    # v grid (yu_ocean, xu_ocean)
    yu = ds["yu_ocean"].values
    xu = ds["xu_ocean"].values
    # salt grid (yt_ocean, xt_ocean)
    yt = ds["yt_ocean"].values
    xt = ds["xt_ocean"].values
    # Depth
    depth = ds["st_ocean"].values

    # Nearest j-index to SAMBA_LAT (= -34.5) on each grid
    j_v = int(np.abs(yu - SAMBA_LAT).argmin())
    j_t = int(np.abs(yt - SAMBA_LAT).argmin())
    lat_v = float(yu[j_v])
    lat_t = float(yt[j_t])

    # Convert 0-360 to -180..180 for Atlantic masking
    def to_180(lon: np.ndarray) -> np.ndarray:
        return np.where(lon > 180, lon - 360, lon)

    xu_180 = to_180(xu)
    xt_180 = to_180(xt)

    atl_v = (xu_180 >= ATLANTIC_LON_MIN) & (xu_180 <= ATLANTIC_LON_MAX)
    atl_t = (xt_180 >= ATLANTIC_LON_MIN) & (xt_180 <= ATLANTIC_LON_MAX)

    # Zonal grid spacing at v grid: dlon * 111 km * cos(lat)
    dlon_v = np.diff(xu_180)
    dlon_v = np.where(dlon_v > 180, dlon_v - 360, dlon_v)
    dlon_v = np.where(dlon_v < -180, dlon_v + 360, dlon_v)
    dlon_v = np.append(dlon_v, dlon_v[-1])
    e1t_v = np.abs(dlon_v) * 111000.0 * np.cos(np.deg2rad(lat_v))
    e1t_v = np.clip(e1t_v, 1.0, None)

    # Vertical thickness from depth levels
    e3t = np.diff(depth, prepend=0.0)

    ds.close()

    # The v and salt sections will each be extracted on their own Atlantic
    # mask; for the F_ovS kernel they must be aligned. Strategy: downsample
    # salt to match v's Atlantic x-grid by nearest-neighbor (both grids have
    # the same 720 longitudes, shifted by 0.25 degrees). For the 0.5-degree
    # regridded SODA grid the atl_v and atl_t masks differ by at most one
    # point, so we align by taking the intersection.
    return {
        "j_v": j_v, "j_t": j_t,
        "lat_v": lat_v, "lat_t": lat_t,
        "depth": depth,
        "e1t_v_atl": e1t_v[atl_v],  # zonal spacing at Atlantic V-grid points
        "e3t": e3t,
        "atl_v": atl_v,
        "atl_t": atl_t,
        "n_atl_v": int(atl_v.sum()),
        "n_atl_t": int(atl_t.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-snapshot F_ovS
# ═══════════════════════════════════════════════════════════════════════

def compute_snapshot_fovs(ds: xr.Dataset, grid: dict) -> float:
    """Extract section, align v and salt, compute F_ovS."""
    v = ds["v"].isel(time=0, yu_ocean=grid["j_v"]).values  # (z, x)
    s = ds["salt"].isel(time=0, yt_ocean=grid["j_t"]).values  # (z, x)

    v_atl = v[:, grid["atl_v"]]  # (z, n_atl_v)
    s_atl = s[:, grid["atl_t"]]  # (z, n_atl_t)

    # Align lengths — pick the shorter dimension if they differ by 1
    n = min(v_atl.shape[1], s_atl.shape[1])
    v_atl = v_atl[:, :n]
    s_atl = s_atl[:, :n]
    e1t = grid["e1t_v_atl"][:n]

    return compute_fovs_from_section(v_atl, s_atl, e1t, grid["e3t"], s0=S0)


# ═══════════════════════════════════════════════════════════════════════
# Download + process
# ═══════════════════════════════════════════════════════════════════════

def ensure_ref_file() -> Path:
    ref_file = SODA_DATA_DIR / "test_soda.nc"
    if not ref_file.exists():
        log.info("Downloading reference SODA file...")
        SODA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "wget", "-q", "--timeout=300",
                BASE_URL + "soda3.15.2_5dy_ocean_reg_1990_01_13.nc",
                "-O", str(ref_file),
            ],
            check=True, timeout=600,
        )
    return ref_file


def download_snapshot(year: int, month: int, tmp: Path) -> bool:
    """Try 5-day file candidates for (year, month); return True on success.

    SODA 3.15.2 5-day files start on days aligned with a recurring
    5-day schedule; try sensible candidates for each month.
    """
    # 5-day starts commonly appear on days 1, 6, 11, 16, 21, 26, 28, 31
    for day in (11, 16, 6, 21, 1, 26, 13, 8, 18, 23, 28):
        fname = f"soda3.15.2_5dy_ocean_reg_{year}_{month:02d}_{day:02d}.nc"
        try:
            r = subprocess.run(
                ["wget", "-q", "--timeout=300", "--tries=2", BASE_URL + fname, "-O", str(tmp)],
                capture_output=True, timeout=600,
            )
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1_000_000:
                return True
        except subprocess.TimeoutExpired:
            continue
    tmp.unlink(missing_ok=True)
    return False


def process_soda_fovs() -> None:
    log.info("=" * 50)
    log.info("SODA 3.15.2  F_ovS at 34.5S  (1980-2022)")
    log.info("=" * 50)

    ref = ensure_ref_file()
    grid = build_section_grid(ref)
    log.info(
        f"Grid: v_lat={grid['lat_v']:.2f}, t_lat={grid['lat_t']:.2f}, "
        f"{grid['n_atl_v']} Atl v-pts, {grid['n_atl_t']} Atl t-pts, "
        f"{len(grid['depth'])} depths"
    )

    ckpt = FovsCheckpoint("soda")
    tmp = SODA_DATA_DIR / "_tmp_fovs_download.nc"

    for year in range(1980, 2023):
        if ckpt.has(year):
            continue

        quarterly = []
        for month in (1, 4, 7, 10):
            if not download_snapshot(year, month, tmp):
                continue
            try:
                ds = xr.open_dataset(tmp, decode_timedelta=False)
                f = compute_snapshot_fovs(ds, grid)
                ds.close()
                if np.isfinite(f):
                    quarterly.append(f)
            except Exception as e:
                log.warning(f"  {year}-{month:02d}: {e}")
            finally:
                tmp.unlink(missing_ok=True)
                gc.collect()

        if quarterly:
            yr_mean = float(np.mean(quarterly))
            ckpt.add(year, yr_mean)
            log.info(f"  {year}: F_ovS = {yr_mean:+.4f} Sv ({len(quarterly)}/4 quarters)")

    # Save
    years, fovs = ckpt.to_arrays()
    if len(years) == 0:
        log.error("No years processed. Aborting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "soda_f_ovs.nc"
    da = xr.DataArray(
        fovs,
        dims=("year",),
        coords={"year": years},
        name="F_ovS",
        attrs={
            "units": "Sv",
            "long_name": f"Overturning freshwater transport at {grid['lat_v']:.1f}S",
            "section_latitude_v": grid["lat_v"],
            "section_latitude_t": grid["lat_t"],
            "product": "SODA 3.15.2",
            "source": BASE_URL,
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
    process_soda_fovs()
