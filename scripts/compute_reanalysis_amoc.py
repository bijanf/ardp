#!/usr/bin/env python3
"""Compute AMOC strength at 26.5°N from ocean reanalysis products.

Supports: SODA3.15.2, ECCO-V4r4, ORAS5, GLORYS12.

Design:
  - Incremental: saves after each year → safe to interrupt and resume
  - Unified: same AMOC computation for all products, only data access differs
  - Memory-safe: loads one year at a time, explicit gc after each

Usage:
  python scripts/compute_reanalysis_amoc.py --product soda
  python scripts/compute_reanalysis_amoc.py --product ecco
  python scripts/compute_reanalysis_amoc.py --product all
"""

from __future__ import annotations

import argparse
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
from ardp.constants import AMOC_DEPTH_MAX, AMOC_DEPTH_MIN, RAPID_LAT
from ardp.spatial.regions import atlantic_lon_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint persistence
# ═══════════════════════════════════════════════════════════════════════

class AMOCCheckpoint:
    """Incremental checkpoint: saves after every year, resumes on restart."""

    def __init__(self, product: str):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHECKPOINT_DIR / f"amoc26n_{product}.json"
        self.data: dict[int, float] = {}
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            self.data = {int(k): v for k, v in raw.items()}
            log.info(f"Resumed checkpoint: {len(self.data)} years ({min(self.data)}–{max(self.data)})")

    def has(self, year: int) -> bool:
        return year in self.data

    def add(self, year: int, amoc: float):
        self.data[year] = amoc
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        years = sorted(self.data.keys())
        return np.array(years), np.array([self.data[y] for y in years])

    def save_final(self, product: str):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        years, amoc = self.to_arrays()
        outfile = RESULTS_DIR / f"yearly_amoc26n_{product}.npz"
        np.savez_compressed(outfile, years=years, amoc=amoc)
        log.info(f"Saved: {outfile} ({len(years)} years)")
        return years, amoc


# ═══════════════════════════════════════════════════════════════════════
# Core AMOC computation (shared)
# ═══════════════════════════════════════════════════════════════════════

def compute_amoc_from_section(
    v_section: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
    actual_lat: float,
    atlantic_mask: np.ndarray,
    dx: np.ndarray,
    dz: np.ndarray,
) -> float:
    """Compute AMOC upper-cell strength from a velocity section."""
    v = np.where(np.isfinite(v_section), v_section, 0.0)
    v_transport = np.array([
        (v[k, atlantic_mask] * dx[atlantic_mask]).sum() * dz[k]
        for k in range(len(depth))
    ])
    psi = np.cumsum(v_transport) / 1e6
    z_mask = (depth >= AMOC_DEPTH_MIN) & (depth <= AMOC_DEPTH_MAX)
    return float(np.max(psi[z_mask])) if z_mask.any() else np.nan


def build_grid_metrics(lon: np.ndarray, depth: np.ndarray, actual_lat: float):
    """Compute Atlantic mask, dx, dz from coordinates."""
    lon_wrapped = np.where(lon > 180, lon - 360, lon)
    lo, hi = atlantic_lon_bounds(actual_lat)
    atl = (lon_wrapped >= lo) & (lon_wrapped <= hi)

    dlon = np.diff(lon)
    dlon = np.append(dlon, dlon[-1])
    dlon = (dlon + 180) % 360 - 180
    dx = np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(actual_lat))
    dx = np.clip(dx, 1.0, None)

    dz = np.diff(depth, prepend=0.0)
    return atl, dx, dz


def report_trend(years: np.ndarray, amoc: np.ndarray, label: str):
    """Print summary statistics."""
    sl, _, _, pv, _ = stats.linregress(years, amoc)
    log.info(f"{label}: {int(years[0])}–{int(years[-1])}, "
             f"mean={amoc.mean():.1f} Sv, trend={sl*10:.2f} Sv/decade (p={pv:.2e})")


# ═══════════════════════════════════════════════════════════════════════
# SODA 3.15.2
# ═══════════════════════════════════════════════════════════════════════

def process_soda():
    """Download and compute AMOC from SODA3.15.2 (UMD, 1980–2022)."""
    log.info("=" * 50)
    log.info("SODA 3.15.2 (1980–2022)")
    log.info("=" * 50)

    ckpt = AMOCCheckpoint("soda")

    # Grid info from a reference file
    ref_file = Path("data/soda/test_soda.nc")
    if not ref_file.exists():
        log.info("Downloading reference SODA file...")
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "wget", "-q", "--timeout=300",
            "https://dsrs.atmos.umd.edu/DATA/soda3.15.2/REGRIDED/ocean/"
            "soda3.15.2_5dy_ocean_reg_1990_01_13.nc",
            "-O", str(ref_file),
        ], timeout=600)

    ds = xr.open_dataset(ref_file)
    lat = ds["yu_ocean"].values
    lon = ds["xu_ocean"].values
    depth = ds["st_ocean"].values
    j_idx = int(np.abs(lat - RAPID_LAT).argmin())
    actual_lat = float(lat[j_idx])
    atl, dx, dz = build_grid_metrics(lon, depth, actual_lat)
    ds.close()
    log.info(f"Grid: lat={actual_lat:.2f}, {atl.sum()} Atlantic pts, {len(depth)} depths")

    base_url = "https://dsrs.atmos.umd.edu/DATA/soda3.15.2/REGRIDED/ocean/"
    tmp = Path("data/soda/_tmp_download.nc")

    for year in range(1980, 2023):
        if ckpt.has(year):
            continue

        quarterly_amoc = []
        for month in [1, 4, 7, 10]:
            # Try multiple days (5-day files: 3, 8, 13, 18, 23, 28)
            downloaded = False
            for day in [13, 8, 18, 3, 23, 28]:
                fname = f"soda3.15.2_5dy_ocean_reg_{year}_{month:02d}_{day:02d}.nc"
                try:
                    result = subprocess.run(
                        ["wget", "-q", "--timeout=300", "--tries=2",
                         base_url + fname, "-O", str(tmp)],
                        capture_output=True, timeout=600,
                    )
                    if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1_000_000:
                        downloaded = True
                        break
                except subprocess.TimeoutExpired:
                    continue

            if not downloaded:
                tmp.unlink(missing_ok=True)
                continue

            try:
                ds = xr.open_dataset(tmp)
                v = ds["v"].isel(time=0, yu_ocean=j_idx).values
                ds.close()
                tmp.unlink(missing_ok=True)
                amoc = compute_amoc_from_section(v, lon, depth, actual_lat, atl, dx, dz)
                if np.isfinite(amoc):
                    quarterly_amoc.append(amoc)
            except Exception as e:
                log.warning(f"  {year}-{month:02d}: read error ({e})")
                tmp.unlink(missing_ok=True)

            gc.collect()

        if quarterly_amoc:
            mean_amoc = float(np.mean(quarterly_amoc))
            ckpt.add(year, mean_amoc)
            log.info(f"  {year}: AMOC = {mean_amoc:.1f} Sv ({len(quarterly_amoc)}/4 quarters)")

    years, amoc = ckpt.save_final("soda")
    if len(years) > 2:
        report_trend(years, amoc, "SODA3.15.2")


# ═══════════════════════════════════════════════════════════════════════
# ECCO V4r4
# ═══════════════════════════════════════════════════════════════════════

def process_ecco():
    """Stream and compute AMOC from ECCO-V4r4 (NASA, 1992–2017)."""
    log.info("=" * 50)
    log.info("ECCO V4r4 (1992–2017)")
    log.info("=" * 50)

    import earthaccess
    earthaccess.login(strategy="netrc")

    ckpt = AMOCCheckpoint("ecco")
    grid_cache = {}

    for year in range(1992, 2018):
        if ckpt.has(year):
            continue

        try:
            results = earthaccess.search_data(
                short_name="ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4",
                temporal=(f"{year}-01", f"{year}-12"),
            )
            if not results:
                log.warning(f"  {year}: no granules found")
                continue

            files = earthaccess.open(results)
            ds = xr.open_mfdataset(files, combine="by_coords")

            if not grid_cache:
                v_name = next((n for n in ["NVEL", "VVEL", "vo", "v"] if n in ds), None)
                lat_name = next(c for c in ds.coords if "lat" in c.lower())
                lon_name = next(c for c in ds.coords if "lon" in c.lower())
                z_name = next((c for c in ds.coords if c in ("Z", "depth", "lev")),
                              next(d for d in ds.dims if d in ("Z", "depth", "lev")))

                lat = ds[lat_name].values
                lon = ds[lon_name].values
                depth = np.abs(ds[z_name].values.astype(float))
                j_idx = int(np.abs(lat - RAPID_LAT).argmin())
                actual_lat = float(lat[j_idx])
                atl, dx, dz = build_grid_metrics(lon, depth, actual_lat)

                grid_cache = dict(
                    v_name=v_name, lat_name=lat_name, j_idx=j_idx,
                    lon=lon, depth=depth, actual_lat=actual_lat,
                    atl=atl, dx=dx, dz=dz,
                )
                log.info(f"Grid: v={v_name}, lat={actual_lat:.2f}, "
                         f"{atl.sum()} Atlantic pts, {len(depth)} depths")

            g = grid_cache
            v_annual = ds[g["v_name"]].isel(**{g["lat_name"]: g["j_idx"]}).mean(dim="time").load()
            amoc = compute_amoc_from_section(
                v_annual.values, g["lon"], g["depth"], g["actual_lat"],
                g["atl"], g["dx"], g["dz"],
            )
            ckpt.add(year, amoc)
            log.info(f"  {year}: AMOC = {amoc:.1f} Sv ({len(results)} files)")

            ds.close()
            del ds, v_annual, files
            gc.collect()

        except Exception as e:
            log.error(f"  {year}: FAILED ({e})")
            gc.collect()

    years, amoc = ckpt.save_final("ecco")
    if len(years) > 2:
        report_trend(years, amoc, "ECCO-V4r4")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute AMOC at 26.5°N from ocean reanalysis products.",
    )
    parser.add_argument(
        "--product", required=True,
        choices=["soda", "ecco", "all"],
        help="Which product to process",
    )
    args = parser.parse_args()

    if args.product in ("soda", "all"):
        process_soda()
    if args.product in ("ecco", "all"):
        process_ecco()


if __name__ == "__main__":
    main()
