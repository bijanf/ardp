#!/usr/bin/env python3
"""Download CMIP5 34.5S latitude slices for F_ovS computation via ESGF.

Downloads vo (meridional velocity) and so (salinity) at the nearest latitude
to 34.5S from ESGF OPeNDAP endpoints for CMIP5 historical simulations.

CMIP5 data is split across many individual netCDF files (5-year chunks).
We use OPeNDAP to extract just the latitude section from each file, then
concatenate them.
"""

from __future__ import annotations

import argparse
import functools
import time
import warnings
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import requests
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# CMIP5 models with published piControl F_ovS (Weijer et al. 2019, Table 2)
TARGET_MODELS = [
    "GFDL-ESM2M",
    "GFDL-ESM2G",
    "CCSM4",
    "MPI-ESM-LR",
    "MIROC5",
    "CNRM-CM5",
    "HadGEM2-ES",
    "EC-EARTH",
]

TARGET_EXPERIMENTS = ["historical"]
TARGET_VARIABLES = ["vo", "so"]
TARGET_LAT = -34.5


def find_esgf_opendap_urls(
    model: str,
    experiment: str,
    variable: str,
    ensemble: str = "r1i1p1",
) -> list[str]:
    """Search ESGF for OPeNDAP URLs for all files of a model/experiment/variable.

    Returns OPeNDAP URLs from a single data node for consistency, sorted by filename.
    """
    url = "https://esgf-node.llnl.gov/esg-search/search"
    file_params = {
        "project": "CMIP5",
        "model": model,
        "experiment": experiment,
        "variable": variable,
        "cmor_table": "Omon",
        "ensemble": ensemble,
        "format": "application/solr+json",
        "limit": 500,
        "type": "File",
        "distrib": "true",
    }
    r = requests.get(url, params=file_params, timeout=60)
    data = r.json()

    if data["response"]["numFound"] == 0:
        return []

    # Collect all OPeNDAP URLs, grouped by data_node
    node_urls: dict[str, list[tuple[str, str]]] = {}  # node -> [(title, url)]
    for fdoc in data["response"]["docs"]:
        title = fdoc.get("title", "")
        for u in fdoc.get("url", []):
            parts = u.split("|")
            if len(parts) >= 3 and parts[2] == "OPENDAP":
                dap_url = parts[0]
                if dap_url.endswith(".html"):
                    dap_url = dap_url[:-5]
                node = dap_url.split("/")[2]
                if node not in node_urls:
                    node_urls[node] = []
                node_urls[node].append((title, dap_url))

    if not node_urls:
        # Fall back: try constructing OPeNDAP URLs from HTTPServer URLs
        for fdoc in data["response"]["docs"]:
            title = fdoc.get("title", "")
            for u in fdoc.get("url", []):
                parts = u.split("|")
                if len(parts) >= 3 and parts[2] == "HTTPServer":
                    http_url = parts[0]
                    # Try converting fileServer to dodsC for THREDDS
                    if "/thredds/fileServer/" in http_url:
                        dap_url = http_url.replace(
                            "/thredds/fileServer/", "/thredds/dodsC/"
                        )
                        node = dap_url.split("/")[2]
                        if node not in node_urls:
                            node_urls[node] = []
                        node_urls[node].append((title, dap_url))

    if not node_urls:
        return []

    # Pick the node with the most files (prefer completeness)
    best_node = max(node_urls, key=lambda n: len(node_urls[n]))
    files = node_urls[best_node]

    # Deduplicate by title (same file may appear multiple times)
    seen = set()
    unique = []
    for title, dap_url in sorted(files, key=lambda x: x[0]):
        if title not in seen:
            seen.add(title)
            unique.append(dap_url)

    return unique


def find_nearest_lat_idx(lat: np.ndarray, target: float) -> int:
    return int(np.abs(lat - target).argmin())


def extract_section_from_opendap(
    model: str,
    experiment: str,
    variable: str,
    output_dir: Path,
) -> bool:
    """Extract 34.5S section from ESGF OPeNDAP for one model/experiment/variable."""
    outfile = output_dir / f"{model}_{experiment}_{variable}.nc"
    if outfile.exists():
        print(f"  Already exists: {outfile.name}")
        return True

    print(f"  Searching ESGF...")
    dap_urls = find_esgf_opendap_urls(model, experiment, variable)

    if not dap_urls:
        print(f"  NOT FOUND (no OPeNDAP URLs)")
        return False

    node = dap_urls[0].split("/")[2]
    print(f"  Found {len(dap_urls)} files on {node}")

    # Open the first file to determine grid structure
    t0 = time.time()
    try:
        ds0 = xr.open_dataset(dap_urls[0])
    except Exception as e:
        print(f"  Failed to open first file: {e}")
        return False

    # Find latitude coordinate
    lat_name = None
    for name in ["lat", "latitude", "nav_lat"]:
        if name in ds0.coords:
            lat_name = name
            break
    if lat_name is None:
        for name in ds0.coords:
            if "lat" in name.lower():
                lat_name = name
                break
    if lat_name is None:
        print(f"  Cannot find latitude in {list(ds0.coords)}")
        ds0.close()
        return False

    lat_vals = ds0[lat_name].values

    if lat_vals.ndim == 1:
        j_idx = find_nearest_lat_idx(lat_vals, TARGET_LAT)
        actual_lat = float(lat_vals[j_idx])
        j_dim = lat_name
    elif lat_vals.ndim == 2:
        lat_1d = np.nanmean(lat_vals, axis=1)
        j_idx = find_nearest_lat_idx(lat_1d, TARGET_LAT)
        actual_lat = float(lat_1d[j_idx])
        j_dim = ds0[lat_name].dims[0]
    else:
        print(f"  Unexpected lat shape {lat_vals.shape}")
        ds0.close()
        return False

    curv = "curvilinear" if lat_vals.ndim == 2 else "regular"
    print(f"  Nearest lat: {actual_lat:.2f} (target: {TARGET_LAT}, {curv})")

    # Save coordinate info from first file
    lon_data = {}
    for name in ds0.coords:
        if "lon" in name.lower():
            lon_vals = ds0[name]
            if lon_vals.ndim == 1:
                lon_data[name] = lon_vals
            elif lon_vals.ndim == 2 and j_dim in lon_vals.dims:
                lon_data[name] = lon_vals.isel({j_dim: j_idx}).load()

    depth_data = {}
    for name in ds0.coords:
        if name in ["lev", "depth", "olevel", "deptht"]:
            depth_data[name] = ds0[name]

    ds0.close()

    # Extract section from each file
    sections = []
    failed = 0
    for i, dap_url in enumerate(dap_urls):
        fname = dap_url.split("/")[-1]
        try:
            ds = xr.open_dataset(dap_url)
            da = ds[variable]
            section = da.isel({j_dim: j_idx}).load()
            sections.append(section)
            ds.close()
            elapsed = time.time() - t0
            print(
                f"    [{i+1}/{len(dap_urls)}] {fname} "
                f"({section.sizes['time']} months, {elapsed:.0f}s)"
            )
        except Exception as e:
            failed += 1
            print(f"    [{i+1}/{len(dap_urls)}] {fname}: FAILED ({e})")
            if failed > 5:
                print(f"  Too many failures, stopping")
                break

    if not sections:
        print(f"  FAILED: no data loaded")
        return False

    # Concatenate all sections
    combined = xr.concat(sections, dim="time")
    combined = combined.sortby("time")

    # Remove duplicate times
    _, unique_idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=sorted(unique_idx))

    combined.attrs["source_id"] = model
    combined.attrs["experiment_id"] = experiment
    combined.attrs["section_latitude"] = actual_lat
    combined.attrs["section_j_index"] = j_idx

    out_ds = combined.to_dataset(name=variable)
    for lon_name, lon_arr in lon_data.items():
        out_ds[lon_name] = lon_arr
    for d_name, d_arr in depth_data.items():
        if d_name not in out_ds.coords:
            out_ds[d_name] = d_arr

    out_ds.to_netcdf(outfile)
    elapsed = time.time() - t0
    size_mb = outfile.stat().st_size / 1e6
    total = combined.sizes["time"]
    print(
        f"  Saved: {outfile.name} ({size_mb:.1f} MB, {total} months, {elapsed:.0f}s)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CMIP5 34.5S sections for F_ovS computation via ESGF."
    )
    parser.add_argument("--output-dir", default="data/cmip5_sections")
    parser.add_argument("--models", nargs="+", default=TARGET_MODELS)
    parser.add_argument("--experiments", nargs="+", default=TARGET_EXPERIMENTS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target models: {args.models}")
    print(f"Target experiments: {args.experiments}")
    print()

    successes = 0
    failures = 0
    t_start = time.time()

    for model in args.models:
        for experiment in args.experiments:
            for variable in TARGET_VARIABLES:
                print(f"[{model}] {experiment}/{variable}:")
                ok = extract_section_from_opendap(
                    model, experiment, variable, output_dir
                )
                if ok:
                    successes += 1
                else:
                    failures += 1
                print()

    total_time = time.time() - t_start
    print(
        f"\nDone in {total_time / 60:.1f} min: {successes} successes, {failures} failures"
    )


if __name__ == "__main__":
    main()
