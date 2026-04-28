#!/usr/bin/env python3
"""Tier-1 discovery: which CMIP6 models beyond our current 15 publish
monthly vo + so for historical + ssp585 on native grid?

Read-only ESGF Solr query against the DKRZ index (distrib=true so we
search the whole federation). For each candidate model we count files
per (variable, experiment, grid_label) and report whether it satisfies
the inclusion criteria of the manuscript:

  - variable_id in {vo, so}
  - experiment_id in {historical, ssp585}
  - table_id = Omon (monthly ocean)
  - variant_label = r1i1p1f1
  - grid_label = gn (native)

Output: data/results/discovery_cmip6_expansion.csv

This is no-commit research: nothing is downloaded.
"""
from __future__ import annotations

from pathlib import Path
from time import time

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "data" / "results"
OUT = RESULTS / "discovery_cmip6_expansion.csv"

# Candidate models -- everything publicly known to exist in CMIP6 with
# hist+ssp585 hist runs that we don't already have.  NorESM2-* are
# excluded a priori (isopycnal coordinates, our Methods explicitly
# disallows the depth-coordinate remap that would be required).
CANDIDATES = [
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",       # FESOM unstructured -- flag for separate handling
    "BCC-CSM2-MR",
    "CESM2-WACCM",
    "CIESM",
    "CMCC-ESM2",
    "CNRM-ESM2-1",
    "E3SM-1-0",
    "E3SM-1-1",
    "E3SM-2-0",
    "EC-Earth3",
    "EC-Earth3-CC",
    "EC-Earth3-Veg",
    "EC-Earth3-Veg-LR",
    "FGOALS-f3-L",
    "GFDL-ESM4",
    "GISS-E2-1-H",
    "GISS-E2-2-G",
    "GISS-E2-2-H",
    "HadGEM3-GC31-MM",
    "IITM-ESM",
    "INM-CM4-8",
    "INM-CM5-0",           # sigma coords -- flag separately
    "KACE-1-0-G",
    "KIOST-ESM",
    "MPI-ESM1-2-HAM",
    "MRI-ESM2-0",
    "NorCPM1",
    "SAM0-UNICON",
    "TaiESM1",
]

# Models we already have (do not re-download).
HAVE = {
    "CESM2", "CMCC-CM2-SR5", "CNRM-CM6-1", "CanESM5", "FGOALS-g3",
    "FIO-ESM-2-0", "GFDL-CM4", "GISS-E2-1-G", "HadGEM3-GC31-LL",
    "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
    "NESM3", "UKESM1-0-LL", "ACCESS-CM2",
}

VARIABLES = ["vo", "so"]
EXPERIMENTS = ["historical", "ssp585"]
SOLR = "https://esgf-data.dkrz.de/esg-search/search"


def _count_files(source_id: str, variable: str, experiment: str,
                 grid_label: str = "gn") -> int:
    params = {
        "format": "application/solr+json",
        "type": "File",
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment,
        "variable": variable,
        "table_id": "Omon",
        "variant_label": "r1i1p1f1",
        "grid_label": grid_label,
        "distrib": "true",
        "limit": 0,
    }
    r = requests.get(SOLR, params=params, timeout=60)
    r.raise_for_status()
    return int(r.json().get("response", {}).get("numFound", 0))


def _check_model(source_id: str) -> dict:
    out = {"model": source_id}
    for var in VARIABLES:
        for exp in EXPERIMENTS:
            out[f"{var}_{exp}_gn"] = _count_files(source_id, var, exp, "gn")
            out[f"{var}_{exp}_gr"] = _count_files(source_id, var, exp, "gr")
    out["have_native"] = all(out[f"{v}_{e}_gn"] > 0 for v in VARIABLES
                              for e in EXPERIMENTS)
    out["have_regridded"] = all(out[f"{v}_{e}_gr"] > 0 for v in VARIABLES
                                 for e in EXPERIMENTS)
    return out


def main() -> None:
    print(f"Discovering {len(CANDIDATES)} CMIP6 candidates against "
          f"federated ESGF (distrib=true) ...")
    print("Required: vo + so, monthly (Omon), historical + ssp585, "
          "r1i1p1f1.\n")
    rows: list[dict] = []
    t0 = time()
    for i, model in enumerate(CANDIDATES, 1):
        print(f"[{i:2d}/{len(CANDIDATES)}] {model:20s}", end="  ", flush=True)
        try:
            row = _check_model(model)
        except requests.HTTPError as e:
            print(f"!! Solr error: {e}")
            continue
        gn_total = sum(row[f"{v}_{e}_gn"] for v in VARIABLES
                        for e in EXPERIMENTS)
        gr_total = sum(row[f"{v}_{e}_gr"] for v in VARIABLES
                        for e in EXPERIMENTS)
        flag = "OK_native" if row["have_native"] else (
            "regridded_only" if row["have_regridded"] else "incomplete"
        )
        print(f"native={gn_total:3d} regridded={gr_total:3d}  -> {flag}")
        rows.append(row)
    elapsed = time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df)} models, {elapsed:.0f}s)")

    # Top-line verdict.
    native_ok = df[df["have_native"]]["model"].tolist()
    regrid_ok = df[(~df["have_native"]) & df["have_regridded"]]["model"].tolist()
    incomplete = df[(~df["have_native"]) & (~df["have_regridded"])]["model"].tolist()

    print("\n" + "=" * 70)
    print(f"VERDICT (n={len(CANDIDATES)} candidates):")
    print(f"  native-grid full coverage : {len(native_ok):2d} -> {native_ok}")
    print(f"  regridded-only coverage   : {len(regrid_ok):2d} -> {regrid_ok}")
    print(f"  incomplete                : {len(incomplete):2d} -> {incomplete}")
    print()
    print(f"Existing ensemble (already on disk): n={len(HAVE)}")
    print(f"Realistic expandable n (existing + native-OK new) : "
          f"{len(HAVE) + len(native_ok)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
