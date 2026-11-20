#!/usr/bin/env python3
"""Overturning freshwater transport at a northern section and the divergence indicator, ECCO-V4r4.

The overturning freshwater transport F_ov is computed with the kernel used for
F_ovS at 34.5S (ardp.physics.fovs.compute_fovs_from_section: baroclinic,
section-mean-subtracted zonally integrated velocity, reference salinity S0) on
the ECCO-V4r4 0.5-degree latitude-longitude fields, at 34.5S and at the northern
sections listed in SECTIONS. Annual means are formed from the monthly files;
the time mean is taken over the complete years 1992-2017. The divergence
indicator is dF_ov = F_ovS - F_ovN (Liu and Liu, 2013).

Run from the repository root:
    python scripts/compute_ecco_divergence_indicator.py --out data/results
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path.cwd()))
from ardp.constants import S0  # noqa: E402
from ardp.physics.fovs import compute_fovs_from_section  # noqa: E402

# name: (latitude, western longitude, eastern longitude) of the Atlantic section
SECTIONS = {
    "34.5S": (-34.5, -70.0, 20.0),
    "26.5N": (26.5, -81.0, -13.0),
    "45N": (45.0, -67.0, -1.0),
    "55N": (55.0, -62.0, 9.0),
    "60N": (60.0, -66.0, 10.0),
    "65N": (65.0, -70.0, 15.0),
    # Northern boundary of Liu and Liu (2013), near 80N: Nares Strait, Fram Strait
    # and the western Barents shelf, taken along one latitude row.
    "80N": (80.0, -75.0, 20.0),
}
YEARS = range(1992, 2018)


def month_key(path: Path) -> str:
    m = re.search(r"_(\d{4}-\d{2})_", path.name)
    return m.group(1) if m else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/ecco")
    p.add_argument("--out", default="data/results")
    a = p.parse_args()
    root = Path(a.data)
    vel = {month_key(f): f for f in root.glob("vel/*/*.nc")}
    sal = {month_key(f): f for f in root.glob("sal/*/*.nc")}
    months = sorted(k for k in vel if k in sal and int(k[:4]) in YEARS)

    with xr.open_dataset(vel[months[0]]) as g:
        lat = g["latitude"].values
        lon = g["longitude"].values
        z = np.abs(g["Z"].values.astype(float))
    order = np.argsort(z)
    e3t_sorted = np.diff(z[order], prepend=0.0)
    e3t = np.empty_like(e3t_sorted)
    e3t[order] = e3t_sorted
    dlon = np.append(np.diff(lon), np.diff(lon)[-1])

    spec = {}
    for name, (phi, w, e) in SECTIONS.items():
        j = int(np.abs(lat - phi).argmin())
        atl = (lon >= w) & (lon <= e)
        e1t = np.clip(np.abs(dlon) * 111000.0 * np.cos(np.deg2rad(lat[j])), 1.0, None)[atl]
        spec[name] = (j, atl, e1t, float(lat[j]))
    rows = sorted({s[0] for s in spec.values()})

    acc_v: dict[int, list] = defaultdict(list)
    acc_s: dict[int, list] = defaultdict(list)
    for k in months:
        with xr.open_dataset(vel[k]) as dv, xr.open_dataset(sal[k]) as ds:
            v = dv["NVEL"].isel(time=0, latitude=rows).values.astype(float)
            s = ds["SALT"].isel(time=0, latitude=rows).values.astype(float)
        acc_v[int(k[:4])].append(v)
        acc_s[int(k[:4])].append(s)
    years = [y for y in sorted(acc_v) if len(acc_v[y]) == 12]
    va = {y: np.mean(acc_v[y], axis=0) for y in years}
    sa = {y: np.mean(acc_s[y], axis=0) for y in years}

    def fov(v_rows: np.ndarray, s_rows: np.ndarray, name: str) -> float:
        j, atl, e1t, _ = spec[name]
        r = rows.index(j)
        vs = np.nan_to_num(v_rows[:, r, :][:, atl], nan=0.0)
        ss = s_rows[:, r, :][:, atl]
        ss = np.where(ss > 0, ss, np.nan)
        return float(compute_fovs_from_section(vs, ss, e1t, e3t, s0=S0))

    def vnet(v_rows: np.ndarray, name: str) -> float:
        j, atl, e1t, _ = spec[name]
        vs = np.nan_to_num(v_rows[:, rows.index(j), :][:, atl], nan=0.0)
        return float((vs * e1t[None, :] * e3t[:, None]).sum() / 1e6)

    out = {"years": [years[0], years[-1]], "n_years": len(years), "sections": {}}
    for name in SECTIONS:
        annual = np.array([fov(va[y], sa[y], name) for y in years])
        vbar = np.mean([va[y] for y in years], axis=0)
        sbar = np.mean([sa[y] for y in years], axis=0)
        out["sections"][name] = {
            "latitude_used": spec[name][3],
            "lon_window": SECTIONS[name][1:],
            "Fov_mean_of_annual_Sv": float(annual.mean()),
            "Fov_annual_std_Sv": float(annual.std(ddof=1)),
            "Fov_of_time_mean_section_Sv": fov(vbar, sbar, name),
            "net_volume_transport_Sv": vnet(vbar, name),
            "annual_Sv": dict(zip(map(str, years), annual.tolist())),
        }
    fs = out["sections"]["34.5S"]["Fov_mean_of_annual_Sv"]
    out["divergence_Sv"] = {n: fs - out["sections"][n]["Fov_mean_of_annual_Sv"] for n in SECTIONS if n != "34.5S"}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ecco_divergence_indicator.json").write_text(json.dumps(out, indent=1))
    for n, r in out["sections"].items():
        print(f"{n:6s} lat {r['latitude_used']:+.2f}  F_ov {r['Fov_mean_of_annual_Sv']:+.4f} Sv "
              f"(sd {r['Fov_annual_std_Sv']:.4f}; time-mean section {r['Fov_of_time_mean_section_Sv']:+.4f}; "
              f"net volume {r['net_volume_transport_Sv']:+.2f} Sv)")
    print("divergence F_ovS - F_ovN:", {k: round(v, 4) for k, v in out["divergence_Sv"].items()})


if __name__ == "__main__":
    main()
