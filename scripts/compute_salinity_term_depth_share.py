#!/usr/bin/env python3
"""Vertical distribution of the Argo-era salinity-driven contribution at 34.5S.

For ORAS5 and GLORYS12V1, the per-layer salinity-driven contribution of the
Argo-era decomposition (2006-2012 vs 2018-2024) is summed above 300 m and above
800 m and expressed as a share of the full-depth value. The depth of the
northward-flowing upper limb is taken as the depth of the maximum of the
time-mean overturning streamfunction at 34.5S.

Run from the repository root:
    python scripts/compute_salinity_term_depth_share.py --out data/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="data/results")
    p.add_argument("--out", default="data/results")
    a = p.parse_args()
    res = Path(a.results)
    out = {}
    for prod, key in (("ORAS5", "oras5"), ("GLORYS12V1", "glorys12")):
        with xr.open_dataset(res / f"fovs_decomposition_{key}_postargo.nc") as ds:
            z = ds["depth"].values.astype(float)
            s = ds["depth_Sv_s"].values.astype(float)
        tot = np.nansum(s)
        rec = {"dFs_total_mSv": 1e3 * tot,
               "share_above_300m": float(np.nansum(s[z <= 300]) / tot),
               "share_above_800m": float(np.nansum(s[z <= 800]) / tot),
               "layers_negative_above_900m": bool(np.all(s[(z <= 900) & np.isfinite(s)] <= 0))}
        sf = {"oras5": "moc_streamfunction_oras5_1993_2024.npz",
              "glorys12": "moc_streamfunction_glorys12_2005_2024.npz"}[key]
        d = np.load(res / sf)
        j = int(np.nanargmin(np.abs(d["lat"] - (-34.5))))
        prof = d["psi"][:, j]
        k = int(np.nanargmax(prof))
        rec["streamfunction_file"] = sf
        rec["depth_of_overturning_max_m"] = float(d["depth"][k])
        rec["overturning_max_Sv"] = float(prof[k])
        out[prod] = rec
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "salinity_term_depth_share.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
