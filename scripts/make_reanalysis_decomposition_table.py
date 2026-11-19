#!/usr/bin/env python3
"""Decomposition of the F_ovS change for the four reanalyses and two window pairs (Table 3).

Reads the stored decomposition files (main windows and Argo-era windows) and
writes the total change, its velocity, salinity and cross contributions in mSv
and the velocity and salinity shares in per cent. Shares are reported where
|dF_ovS| reaches the 10 mSv classification floor.

Run from the repository root:
    python scripts/make_reanalysis_decomposition_table.py --out data/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import xarray as xr

PRODUCTS = [("ORAS5", "oras5"), ("GLORYS12V1", "glorys12"), ("SODA 3.15.2", "soda"), ("ECCO-V4r4", "ecco")]
FLOOR_MSV = 10.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="data/results")
    p.add_argument("--out", default="data/results")
    a = p.parse_args()
    rows = []
    for label, key in PRODUCTS:
        for tag, suffix in (("main", ""), ("Argo era", "_postargo")):
            with xr.open_dataset(Path(a.results) / f"fovs_decomposition_{key}{suffix}.nc") as ds:
                at = dict(ds.attrs)
            d = {k: 1e3 * float(at[f"delta_{k}_Sv"]) for k in ("total", "v", "s", "cross")}
            ok = abs(d["total"]) >= FLOOR_MSV
            rows.append({
                "product": label, "windows": tag,
                "early": at["early_period"], "late": at["late_period"],
                "F_early_mSv": 1e3 * float(at["F_ov_early_Sv"]), "F_late_mSv": 1e3 * float(at["F_ov_late_Sv"]),
                "dF": d["total"], "dFv": d["v"], "dFs": d["s"], "dFx": d["cross"],
                "fv": 100 * d["v"] / d["total"] if ok else None,
                "fs": 100 * d["s"] / d["total"] if ok else None,
            })
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "table3_reanalysis_decomposition.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        sh = "n.c. & n.c." if r["fv"] is None else f"{r['fv']:.0f} & {r['fs']:.0f}"
        print(f"{r['product']} & {r['windows']} & {r['dF']:+.1f} & {r['dFv']:+.1f} & {r['dFs']:+.1f} & {r['dFx']:+.1f} & {sh} \\\\"
              f"   % {r['early']} vs {r['late']}")


if __name__ == "__main__":
    main()
