#!/usr/bin/env python3
"""Download CMIP6 SSP585 extension 34.5S sections (vo + so) from ESGF via OPeNDAP.

Only downloads a single latitude slice at ~34.5S, so each file is tiny (~few MB).
This is much faster than downloading full 3D fields.

Output: data/cmip6_sections/{model}_ssp585ext_{var}_345S.nc
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import requests
import xarray as xr

from ardp.constants import ESGF_NODE_DKRZ as ESGF_NODE
from ardp.models import CMIP6_SSP585_EXT_MODELS as MODELS

MEMBER = "r1i1p1f1"
TARGET_LAT = -34.5


def find_extension_files(model: str, variable: str) -> list[str]:
    """Query ESGF for post-2100 OPeNDAP URLs."""
    r = requests.get(ESGF_NODE,
        params={
            "type": "File",
            "source_id": model,
            "experiment_id": "ssp585",
            "variable_id": variable,
            "table_id": "Omon",
            "member_id": MEMBER,
            "grid_label": "gn",
            "format": "application/solr+json",
            "limit": 200,
            "fields": "title,url",
        },
        timeout=60)
    data = r.json()
    docs = data.get("response", {}).get("docs", [])

    urls = []
    seen = set()
    for d in docs:
        title = d.get("title", "")
        if title in seen:
            continue
        parts = title.replace(".nc", "").split("_")
        for p in parts:
            if "-" in p and len(p) >= 13:
                start_yr = p.split("-")[0][:4]
                if start_yr.isdigit() and int(start_yr) >= 2101:
                    file_urls = d.get("url", [])
                    opendap = [u.split("|")[0].replace(".html", "")
                               for u in file_urls if "OPENDAP" in u]
                    if opendap:
                        # Prefer CEDA, then any
                        ceda = [u for u in opendap if "ceda" in u]
                        urls.append(ceda[0] if ceda else opendap[0])
                        seen.add(title)
                    break
    return sorted(urls)


def find_lat_index(ds: xr.Dataset, variable: str) -> tuple[int, float, str, str]:
    """Find j-index nearest to TARGET_LAT. Returns (j_idx, actual_lat, lat_name, j_dim)."""
    lat_name = None
    for name in ["lat", "latitude", "nav_lat"]:
        if name in ds.coords:
            lat_name = name
            break
    if lat_name is None:
        for name in ds.coords:
            if "lat" in name.lower():
                lat_name = name
                break

    lat_vals = ds[lat_name].values
    if lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_dim = ds[lat_name].dims[0]
    else:
        lat_1d = lat_vals
        j_dim = lat_name

    j_idx = int(np.abs(lat_1d - TARGET_LAT).argmin())
    actual_lat = float(lat_1d[j_idx])
    return j_idx, actual_lat, lat_name, j_dim


def download_sections(model: str, variable: str, output_dir: Path, force: bool = False) -> bool:
    """Download 34.5S section for one model/variable."""
    outfile = output_dir / f"{model}_ssp585ext_{variable}_345S.nc"

    if outfile.exists() and not force:
        print(f"  Already exists: {outfile.name}")
        return True

    print(f"  Querying ESGF for post-2100 {variable} files...")
    file_urls = find_extension_files(model, variable)
    if not file_urls:
        print(f"  No extension files found")
        return False
    print(f"  Found {len(file_urls)} files")

    # Open first file to get grid info
    try:
        ds0 = xr.open_dataset(file_urls[0], decode_times=False)
    except Exception as e:
        print(f"  FAILED to open first file: {e}")
        return False

    j_idx, actual_lat, lat_name, j_dim = find_lat_index(ds0, variable)
    print(f"  Target lat: {TARGET_LAT}, actual: {actual_lat:.2f} (j={j_idx})")
    ds0.close()

    all_sections = []
    all_times = []
    t_start = time.time()

    for fi, url in enumerate(file_urls):
        fname = url.rsplit("/", 1)[1]
        print(f"  File {fi+1}/{len(file_urls)}: {fname}")

        try:
            ds = xr.open_dataset(url, decode_times=False)
            section = ds[variable].isel({j_dim: j_idx}).load()
            t_vals = ds["time"].values
            t_attrs = ds["time"].attrs

            all_sections.append(section)
            all_times.append(t_vals)

            elapsed = time.time() - t_start
            print(f"    {len(t_vals)} timesteps ({elapsed:.0f}s)")
            ds.close()

        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    if not all_sections:
        print(f"  No data downloaded")
        return False

    # Concatenate
    combined = xr.concat(all_sections, dim="time")
    times_all = np.concatenate(all_times)

    # Fix time coordinate
    import cftime as _cftime
    time_nums = []
    for t in times_all:
        time_nums.append(float(t))
    combined["time"] = xr.Variable("time", np.array(time_nums, dtype=np.float64),
                                    attrs={"units": t_attrs.get("units", "days since 1850-01-01"),
                                           "calendar": t_attrs.get("calendar", "proleptic_gregorian")})

    # Add metadata
    combined.attrs["section_latitude"] = actual_lat
    combined.attrs["source_id"] = model
    combined.attrs["experiment_id"] = "ssp585ext"

    combined.to_netcdf(outfile)
    elapsed = time.time() - t_start
    print(f"  Saved: {outfile.name} ({outfile.stat().st_size / 1e6:.1f} MB, {elapsed:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download CMIP6 SSP585 extension 34.5S sections (vo + so)."
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--variables", nargs="+", default=["vo", "so"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/cmip6_sections"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    successes, failures = 0, 0
    for model in args.models:
        for variable in args.variables:
            print(f"\n{'='*60}")
            print(f"=== {model} / {variable} (ssp585ext 34.5S section) ===")
            print(f"{'='*60}")
            if download_sections(model, variable, args.output_dir, args.force):
                successes += 1
            else:
                failures += 1

    print(f"\nDone: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
