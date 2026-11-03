#!/usr/bin/env python3
"""Water bodies on the 34.5 S grid row that lie outside the Atlantic section.

A fixed longitude box from 70 W to 20 E admits, in GLORYS12V1, grid cells that
are not connected to the open Atlantic along the row. This script lists every
run of contiguous ocean cells on the row, its longitude range and its surface
salinity, so that each run can be matched to a geographical water body.

Writes ``PAPER_3_v2/analysis/r1_water_bodies.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_water_bodies.json"
BOX = (-70.0, 20.0)


def runs(lon: np.ndarray, sss: np.ndarray) -> list[dict]:
    order = np.argsort(lon)
    lon, sss = lon[order], sss[order]
    inbox = (lon >= BOX[0]) & (lon <= BOX[1])
    lon, sss = lon[inbox], sss[inbox]
    wet = ~np.isnan(sss)
    out, start = [], None
    for i, w in enumerate(np.append(wet, False)):
        if w and start is None:
            start = i
        elif not w and start is not None:
            seg = slice(start, i)
            out.append(
                {
                    "lon_west": float(lon[seg][0]),
                    "lon_east": float(lon[seg][-1]),
                    "n_cells": int(i - start),
                    "sss_min": float(np.nanmin(sss[seg])),
                    "sss_max": float(np.nanmax(sss[seg])),
                    "sss_mean": float(np.nanmean(sss[seg])),
                }
            )
            start = None
    return out


def main() -> None:
    res = {}
    for p in ("glorys12", "oras5", "ecco"):
        f = RESULTS / f"paper3r1_section3d_{p}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        lon = ds["row_lon"].values.astype(float)
        lon = np.where(lon > 180, lon - 360, lon)
        res[p] = {
            "row_latitude": float(ds.attrs["section_latitude"]),
            "section_lon_range": [float(ds["lon"].min()), float(ds["lon"].max())],
            "runs_in_box": runs(lon, ds["row_sss_mean"].values.astype(float)),
        }
    OUT.write_text(json.dumps(res, indent=1))
    for p, r in res.items():
        print(p, r["row_latitude"], "section", r["section_lon_range"])
        for x in r["runs_in_box"]:
            print("   ", x)


if __name__ == "__main__":
    main()
