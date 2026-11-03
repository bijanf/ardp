#!/usr/bin/env python3
"""Where along the 34.5 S section, and at which depths, the northward limb gains salt.

With the circulation frozen at its record-mean profile, the transport-weighted
northward-limb salinity is a fixed linear combination of the cell salinities,

    S_N(t) = sum_k w_k S_k(t),          w_k = max(q_k, 0) / T,
    S_k(t) = sum_x a_kx s_kx(t) / a_k,  a_kx = dx_x dz_k,

with q_k the record-mean level transport after removal of the net section
transport. Its trend therefore splits exactly into cell contributions

    C_kx = w_k (a_kx / a_k) d s_kx / dt,

which are summed over longitude bands and depth layers. The same is done for
the southward limb. The bands separate the Brazil Current and the
Brazil-Malvinas confluence region at the western boundary, the western and
eastern interior of the subtropical gyre, and the Cape Basin, where Agulhas
leakage enters the Atlantic; the layers separate the surface mixed layer, the
thermocline (South Atlantic Central Water) and the intermediate water (Subantarctic
Mode Water and Antarctic Intermediate Water).

Writes ``PAPER_3_v2/analysis/r1_salinification_structure.json`` and
``data/results/paper3r1_salinity_trend_section_<product>_<window>.nc``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compute_paper3r1_section_quantities import thickness  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_salinification_structure.json"

BANDS = {
    "western boundary (west of 45 W)": (-60.0, -45.0),
    "western interior (45 W to 20 W)": (-45.0, -20.0),
    "eastern interior (20 W to 0)": (-20.0, 0.0),
    "Cape Basin (east of 0)": (0.0, 25.0),
}
LAYERS = {
    "0-100 m": (0.0, 100.0),
    "100-500 m": (100.0, 500.0),
    "500-1300 m": (500.0, 1300.0),
    "below 1300 m": (1300.0, 7000.0),
}
WINDOWS = {
    "oras5": {
        "own": (1958, 2025),
        "1993-2025": (1993, 2025),
        "2005-2024": (2005, 2024),
    },
    "glorys12": {"own": (1993, 2025), "2005-2024": (2005, 2024)},
    "ecco": {"own": (1992, 2017)},
}


def santer_trend(years: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised OLS trend along axis 0 (per decade) and its p-value, with the
    corrected effective sample size of paper3r1_stats.py."""
    from paper3r1_stats import trend_test_many

    shape = y.shape[1:]
    r = trend_test_many(years, y.reshape(len(years), -1))
    return r["slope"].reshape(shape) * 10.0, r["p"].reshape(shape)


def run(product: str) -> dict:
    f3 = xr.open_dataset(RESULTS / f"paper3r1_section3d_{product}.nc")
    sec = xr.open_dataset(RESULTS / f"paper3r1_section_{product}.nc")
    depth = f3["depth"].values
    dz = thickness(depth, product, f3["e3t"].values, "bounds")
    lon = f3["lon"].values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    e1t = f3["e1t"].values

    s = f3["s"]
    wet = np.isfinite(s.isel(time=0).values)
    area = np.where(wet, e1t[None, :] * dz[:, None], 0.0)
    a_k = area.sum(axis=1)
    frac = np.where(a_k[:, None] > 0, area / np.where(a_k > 0, a_k, 1)[:, None], 0.0)

    q = sec["V_bc"].values.mean(axis=0) * sec["e3t"].values
    t_n = np.clip(q, 0, None).sum()
    w_n = np.clip(q, 0, None) / t_n
    w_s = np.clip(-q, 0, None) / np.clip(-q, 0, None).sum()

    ann = s.groupby("time.year").mean().load()
    years_all = ann["year"].values
    res: dict = {"product": f3.attrs.get("product", product), "windows": {}}
    for wname, (y0, y1) in WINDOWS[product].items():
        m = (years_all >= y0) & (years_all <= y1)
        yrs = years_all[m]
        vals = np.where(wet[None], ann.values[m], 0.0)
        tr, p = santer_trend(yrs, vals)
        tr = np.where(wet, tr, np.nan)
        p = np.where(wet, p, np.nan)
        c_n = np.where(wet, w_n[:, None] * frac * np.nan_to_num(tr), 0.0)
        c_s = np.where(wet, w_s[:, None] * frac * np.nan_to_num(tr), 0.0)

        # Direct check: trend of the frozen-velocity limb salinity itself.
        sk = (np.where(wet[None], ann.values[m], 0.0) * frac[None]).sum(axis=2)
        sn_series = (sk * w_n[None]).sum(axis=1)
        ss_series = (sk * w_s[None]).sum(axis=1)
        sn_direct = float(np.polyfit(yrs.astype(float), sn_series, 1)[0] * 10.0)
        ss_direct = float(np.polyfit(yrs.astype(float), ss_series, 1)[0] * 10.0)

        table_n, table_s = {}, {}
        for bname, (lo, hi) in BANDS.items():
            bx = (lon >= lo) & (lon < hi)
            for lname, (z0, z1) in LAYERS.items():
                lz = (depth >= z0) & (depth < z1)
                key = f"{bname} | {lname}"
                table_n[key] = float(c_n[np.ix_(lz, bx)].sum())
                table_s[key] = float(c_s[np.ix_(lz, bx)].sum())
        band_n = {
            b: float(sum(v for k, v in table_n.items() if k.startswith(b)))
            for b in BANDS
        }
        layer_n = {
            ly: float(sum(v for k, v in table_n.items() if k.endswith(ly)))
            for ly in LAYERS
        }
        band_s = {
            b: float(sum(v for k, v in table_s.items() if k.startswith(b)))
            for b in BANDS
        }
        res["windows"][wname] = {
            "window": [int(yrs[0]), int(yrs[-1])],
            "S_north_frozen_trend_sum_of_cells": float(c_n.sum()),
            "S_north_frozen_trend_direct": sn_direct,
            "S_south_frozen_trend_sum_of_cells": float(c_s.sum()),
            "S_south_frozen_trend_direct": ss_direct,
            "northward_by_band": band_n,
            "northward_by_layer": layer_n,
            "northward_by_band_and_layer": table_n,
            "southward_by_band": band_s,
            "band_width_fraction": {
                b: float(e1t[(lon >= lo) & (lon < hi)].sum() / e1t.sum())
                for b, (lo, hi) in BANDS.items()
            },
        }
        xr.Dataset(
            {
                "trend": (("depth", "lon"), tr),
                "p": (("depth", "lon"), p),
                "contribution_north": (("depth", "lon"), c_n),
                "mean_salinity": (
                    ("depth", "lon"),
                    np.where(wet, ann.values[m].mean(axis=0), np.nan),
                ),
                "v_mean": (("depth", "lon"), f3["v"].values.mean(axis=0)),
                "limb_weight_north": ("depth", w_n),
            },
            coords={"depth": depth, "lon": lon},
            attrs={
                "product": res["product"],
                "window": f"{yrs[0]}-{yrs[-1]}",
                "units": "PSU per decade",
            },
        ).to_netcdf(RESULTS / f"paper3r1_salinity_trend_section_{product}_{wname}.nc")
    return res


def en4_section(path: Path):
    from analysis_paper3v2_en4 import TARGET_LAT, atlantic_segment

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        j = int(np.abs(lat - TARGET_LAT).argmin())
        sal = ds["salinity"].isel(time=0, lat=j).values
        wobs = ds["salinity_observation_weights"].isel(time=0, lat=j).values
        year = int(pd.Timestamp(ds["time"].values[0]).year)
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    sel, e1t = atlantic_segment(
        lon180, np.full(lon.shape, float(lat[j])), np.isfinite(sal[0])
    )
    return year, sal[:, sel], wobs[:, sel], lon180[sel], e1t


def run_en4() -> dict:
    """The same decomposition in the EN4 objective analysis (34 S row, 2005-2024)."""
    from concurrent.futures import ProcessPoolExecutor

    from analysis_paper3v2_en4 import EN4_DIR, frozen_velocity

    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(en4_section, files, chunksize=8))
    lon, e1t = rows[0][3], rows[0][4]
    with xr.open_dataset(files[0], decode_timedelta=False) as ds:
        depth = ds["depth"].values
        bnds = ds["depth_bnds"].values
    dz = np.abs(bnds[:, 1] - bnds[:, 0])
    years = np.array([r[0] for r in rows])
    sal = np.array([r[1] for r in rows])
    wobs = np.array([r[2] for r in rows]).mean(axis=0)
    uy = np.unique(years)
    ann = np.array([np.nanmean(sal[years == y], axis=0) for y in uy])
    wet = np.isfinite(ann).all(axis=0)
    area = np.where(wet, e1t[None, :] * dz[:, None], 0.0)
    a_k = area.sum(axis=1)
    frac = np.where(a_k[:, None] > 0, area / np.where(a_k > 0, a_k, 1)[:, None], 0.0)
    tr, p = santer_trend(uy, np.where(wet[None], ann, 0.0))
    tr = np.where(wet, tr, np.nan)
    out = {"window": [int(uy[0]), int(uy[-1])], "frozen": {}}
    for key in ("oras5", "glorys12"):
        v = frozen_velocity(f"paper3r1_section_{key}.nc", depth, dz)
        ok = a_k > 0
        v = np.where(ok, v, 0.0)
        v = v - (v * dz)[ok].sum() / dz[ok].sum() * ok
        q = v * dz
        w_n = np.clip(q, 0, None) / np.clip(q, 0, None).sum()
        c_n = np.where(wet, w_n[:, None] * frac * np.nan_to_num(tr), 0.0)
        out["frozen"][key] = {
            "S_north_frozen_trend_sum_of_cells": float(c_n.sum()),
            "northward_by_band": {
                b: float(c_n[:, (lon >= lo) & (lon < hi)].sum())
                for b, (lo, hi) in BANDS.items()
            },
            "northward_by_layer": {
                ly: float(c_n[(depth >= z0) & (depth < z1)].sum())
                for ly, (z0, z1) in LAYERS.items()
            },
            "northward_by_band_and_layer": {
                f"{b} | {ly}": float(
                    c_n[
                        np.ix_((depth >= z0) & (depth < z1), (lon >= lo) & (lon < hi))
                    ].sum()
                )
                for b, (lo, hi) in BANDS.items()
                for ly, (z0, z1) in LAYERS.items()
            },
        }
        if key == "oras5":
            xr.Dataset(
                {
                    "trend": (("depth", "lon"), tr),
                    "p": (("depth", "lon"), np.where(wet, p, np.nan)),
                    "contribution_north": (("depth", "lon"), c_n),
                    "obs_weight": (("depth", "lon"), wobs),
                },
                coords={"depth": depth, "lon": lon},
                attrs={
                    "product": "EN4.2.2",
                    "window": f"{uy[0]}-{uy[-1]}",
                    "units": "PSU per decade",
                },
            ).to_netcdf(RESULTS / "paper3r1_salinity_trend_section_en4_2005-2024.nc")
    out["band_width_fraction"] = {
        b: float(e1t[(lon >= lo) & (lon < hi)].sum() / e1t.sum())
        for b, (lo, hi) in BANDS.items()
    }
    return out


def main() -> None:
    out = {"en4": run_en4()}
    print("== EN4", out["en4"]["window"])
    for k, d in out["en4"]["frozen"].items():
        print(f"   frozen {k}: S_N {d['S_north_frozen_trend_sum_of_cells']:+.4f}")
        print(
            "   by band :",
            {b.split(" (")[0]: round(v, 4) for b, v in d["northward_by_band"].items()},
        )
        print(
            "   by layer:", {b: round(v, 4) for b, v in d["northward_by_layer"].items()}
        )
    for p in WINDOWS:
        if (RESULTS / f"paper3r1_section3d_{p}.nc").exists():
            out[p] = run(p)
            for w, d in out[p]["windows"].items():
                sn = d["S_north_frozen_trend_direct"]
                sn_cells = d["S_north_frozen_trend_sum_of_cells"]
                ss = d["S_south_frozen_trend_direct"]
                print(
                    f"== {p} {w} {d['window']}: S_N frozen {sn:+.4f} "
                    f"(cells {sn_cells:+.4f}); S_S {ss:+.4f}"
                )
                print(
                    "   by band :",
                    {
                        k.split(" (")[0]: round(v, 4)
                        for k, v in d["northward_by_band"].items()
                    },
                )
                print(
                    "   by layer:",
                    {k: round(v, 4) for k, v in d["northward_by_layer"].items()},
                )
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
