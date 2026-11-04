#!/usr/bin/env python3
"""Limb salinity levels at 34.5 S frozen at each circulation, with an Argo check.

For 2005 to 2024 this script reports the northward and southward limb
salinities S_N and S_S and their contrast dS at the 34.5 S section, for four
salinity sources:

* the reanalyses' own salinity (ORAS5, GLORYS12V1; section fields of
  ``paper3r1_section_<product>.nc``),
* the EN4.2.2 observation-only analysis (row 34 S, as in
  ``analysis_paper3v2_en4.py``, since EN4 is on whole degrees), and
* the Roemmich and Gilson Argo climatology (RG09, row 34.5 S, which is the
  section latitude itself; its grid is at half degrees).

Each salinity source is combined with the velocity structure frozen at a chosen
circulation: the record-mean barotropic-corrected profile of ORAS5 or GLORYS12V1
(``frozen_velocity``). The limb split is that of ``limbs`` in
``analysis_paper3v2_en4_depth_test.py``: the net transport is removed, the levels
are split by sign, and the salinities are transport weighted, so that
F_ovS = -(1/S0) T dS holds exactly.

Two layers are reported: the full column of each source, and the upper 2000 m,
where the net transport removal is re-applied within the layer. RG09 spans
0 to 2000 dbar only (pressure is used as depth), so its full column is the
upper 2000 m layer.

Limb quantities are formed month by month and averaged to calendar years. The
trends are OLS trends of the 2005 to 2024 annual values, tested with the
bias-corrected effective sample size and the intervals of
``paper3r1_stats.trend_test``; window means use ``paper3r1_stats.mean_test``.

Writes ``PAPER_3_v2/analysis/jc_n05_limb_levels_rg09.json``.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_en4 import (  # noqa: E402
    EN4_DIR,
    TARGET_LAT,
    atlantic_segment,
    en4_month,
    frozen_velocity,
)
from analysis_paper3v2_en4_depth_test import en4_month_full, limbs  # noqa: E402
from paper3r1_stats import mean_test, trend_test  # noqa: E402

RESULTS = REPO / "data" / "results"
RG_DIR = REPO / "data" / "argo_rg09"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n05_limb_levels_rg09.json"

WINDOW = (2005, 2024)
ARGO_FLOOR = 2000.0
RG_ROW_LAT = -34.5
CIRC_FILES = {
    "ORAS5": "paper3r1_section_oras5.nc",
    "GLORYS12V1": "paper3r1_section_glorys12.nc",
}
FIELDS = ("S_north", "S_south", "dS", "T_Sv", "F_ov_Sv")
UNITS = {
    "S_north": "PSU",
    "S_south": "PSU",
    "dS": "PSU",
    "T_Sv": "Sv",
    "F_ov_Sv": "Sv",
}
YEARS = np.arange(WINDOW[0], WINDOW[1] + 1)
LAYERS = (("full_column", None), ("upper_2000m", ARGO_FLOOR))


def clean(a: np.ndarray) -> np.ndarray:
    """Replace the RG09 missing value (-999) by NaN."""
    a = np.asarray(a, dtype=float)
    return np.where(a < -900.0, np.nan, a)


def zonal_mean(field: np.ndarray, sel: np.ndarray, e1t: np.ndarray) -> np.ndarray:
    """Width-weighted zonal mean over the section points sel, per level."""
    f = field[:, sel]
    w = np.where(np.isfinite(f), e1t, 0.0)
    ws = w.sum(axis=1)
    return np.where(
        ws > 0, (np.nan_to_num(f) * w).sum(axis=1) / np.where(ws > 0, ws, 1), np.nan
    )


# ── Argo RG09 ─────────────────────────────────────────────────────────


def rg_thickness(pres: np.ndarray) -> np.ndarray:
    """Layer thicknesses from the midpoints between levels; top 0, bottom 2000 m."""
    mid = 0.5 * (pres[1:] + pres[:-1])
    edges = np.concatenate([[0.0], mid, [ARGO_FLOOR]])
    return np.diff(edges)


def load_rg09(row_lat: float) -> dict:
    """Absolute RG09 salinity along one latitude row, 2005 to 2024 monthly.

    The base file holds the 2004 to 2018 climatology and its monthly anomalies
    (TIME = months since 2004-01 plus 0.5). The monthly files for 2019 to 2024
    carry anomalies from the same climatology. Absolute salinity is the mean
    plus the anomaly.
    """
    base_path = RG_DIR / "RG_ArgoClim_Salinity_2019.nc"
    with xr.open_dataset(base_path, decode_times=False) as ds:
        lat = ds["LATITUDE"].values.astype(float)
        j = int(np.argmin(np.abs(lat - row_lat)))
        lat_row = float(lat[j])
        lon = ds["LONGITUDE"].values.astype(float)
        pres = ds["PRESSURE"].values.astype(float)
        tbase = ds["TIME"].values.astype(float)
        mean = ds["ARGO_SALINITY_MEAN"].isel(LATITUDE=j).values.astype(float)
        anom_base = ds["ARGO_SALINITY_ANOMALY"].isel(LATITUDE=j).values.astype(float)
        long_names = {
            "mean": ds["ARGO_SALINITY_MEAN"].attrs.get("long_name", ""),
            "anomaly_base": ds["ARGO_SALINITY_ANOMALY"].attrs.get("long_name", ""),
        }
    if not np.allclose(tbase, np.arange(anom_base.shape[0]) + 0.5):
        raise RuntimeError("RG09 base file TIME is not monthly from 2004-01")

    anomalies: dict[tuple[int, int], np.ndarray] = {}
    for k in range(anom_base.shape[0]):
        anomalies[(2004 + k // 12, k % 12 + 1)] = anom_base[k]
    base_keys = sorted(anomalies)

    pat = re.compile(r"RG_ArgoClim_(\d{4})(\d{2})_2019\.nc")
    ext_files = []
    ext_long_name = ""
    for f in sorted(RG_DIR.glob("RG_ArgoClim_*_2019.nc")):
        mt = pat.fullmatch(f.name)
        if mt is None:
            continue
        yr, mo = int(mt.group(1)), int(mt.group(2))
        with xr.open_dataset(f, decode_times=False) as ds:
            t = float(ds["TIME"].values[0])
            lat_f = float(ds["LATITUDE"].values[j])
            anom = ds["ARGO_SALINITY_ANOMALY"].isel(TIME=0, LATITUDE=j).values.astype(float)
            ext_long_name = ds["ARGO_SALINITY_ANOMALY"].attrs.get("long_name", "")
        expected = 12 * (yr - 2004) + (mo - 1) + 0.5
        if abs(t - expected) > 1e-3 or abs(lat_f - lat_row) > 1e-3:
            raise RuntimeError(f"RG09 file {f.name} time or latitude does not match its name")
        anomalies[(yr, mo)] = anom
        ext_files.append(f.name)

    mean_c = clean(mean)
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    ocean = np.isfinite(mean_c[0])
    lat_arr = np.full(lon.shape, lat_row)
    sel, e1t = atlantic_segment(lon180, lat_arr, ocean)

    keys = [(y, m) for y in YEARS for m in range(1, 13)]
    missing = [k for k in keys if k not in anomalies]
    if missing:
        raise RuntimeError(f"RG09 months missing in the window: {missing[:5]}")
    sbar = []
    for key in keys:
        s = mean_c + clean(anomalies[key])
        sbar.append(zonal_mean(s, sel, e1t))
    sbar = np.array(sbar)

    e3 = rg_thickness(pres)
    all_keys = sorted(anomalies)
    info = {
        "file_base": "RG_ArgoClim_Salinity_2019.nc",
        "base_months": len(base_keys),
        "base_first": list(base_keys[0]),
        "base_last": list(base_keys[-1]),
        "monthly_files": len(ext_files),
        "monthly_first": list(all_keys[len(base_keys)]),
        "monthly_last": list(all_keys[-1]),
        "total_months": len(all_keys),
        "months_in_window": len(keys),
        "row_latitude_used": lat_row,
        "section_latitude": row_lat,
        "pressure_levels_dbar": pres.tolist(),
        "layer_thickness_m": e3.tolist(),
        "layer_thickness_sum_m": float(e3.sum()),
        "mean_long_name": long_names["mean"],
        "anomaly_long_name_base": long_names["anomaly_base"],
        "anomaly_long_name_monthly": ext_long_name,
        "segment_lon_range": [float(lon180[sel].min()), float(lon180[sel].max())],
        "segment_n_points": int(len(sel)),
        "level_fraction_finite_over_window": np.isfinite(sbar).mean(axis=0).tolist(),
    }
    return {
        "sbar": sbar,
        "years": np.array([k[0] for k in keys]),
        "depth": pres,
        "e3": e3,
        "info": info,
    }


# ── EN4 and reanalysis sections ───────────────────────────────────────


def _en4_one(path):
    return en4_month_full(path)


def read_en4() -> dict:
    files = []
    for f in sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc")):
        yr = int(f.stem.split(".")[-1][:4])
        if WINDOW[0] <= yr <= WINDOW[1]:
            files.append(f)
    rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for fut in as_completed([ex.submit(_en4_one, f) for f in files]):
            r = fut.result()
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["time"])
    return {
        "sbar": np.array([r["S_bar"] for r in rows]),
        "years": np.array([r["time"].year for r in rows]),
        "depth": rows[0]["depth"].astype(float),
        "e3": rows[0]["e3t"].astype(float),
        "n_months": len(rows),
        "first": str(rows[0]["time"].date()),
        "last": str(rows[-1]["time"].date()),
    }


def section_source(product_file: str) -> dict:
    """Reanalysis section salinity S_bar(z, t) and its own monthly limb series."""
    with xr.open_dataset(RESULTS / product_file) as ds:
        time = pd.DatetimeIndex(ds["time"].values)
        yrs = np.asarray(time.year)
        m = (yrs >= WINDOW[0]) & (yrs <= WINDOW[1])
        own = {
            "S_north": ds["S_north"].values.astype(float)[m],
            "S_south": ds["S_south"].values.astype(float)[m],
            "dS": ds["dS_limb"].values.astype(float)[m],
            "T_Sv": ds["T_limb_Sv"].values.astype(float)[m],
            "F_ov_Sv": ds["F_ov"].values.astype(float)[m],
        }
        return {
            "sbar": ds["S_bar"].values.astype(float)[m],
            "years": yrs[m],
            "depth": ds["depth"].values.astype(float),
            "e3": ds["e3t"].values.astype(float),
            "own": own,
            "section_latitude": float(ds.attrs["section_latitude"]),
            "product": ds.attrs.get("product", ""),
        }


# ── annual means and tests ────────────────────────────────────────────


def annual_from_monthly(monthly: list, years_m: np.ndarray) -> dict:
    out = {}
    for f in FIELDS:
        vals = np.array([m[f] if m is not None else np.nan for m in monthly], float)
        out[f] = annual_from_values(vals, years_m)
    return out


def annual_from_values(vals: np.ndarray, years_m: np.ndarray) -> np.ndarray:
    arr = np.full(len(YEARS), np.nan)
    for k, y in enumerate(YEARS):
        v = np.asarray(vals, float)[years_m == y]
        v = v[np.isfinite(v)]
        if v.size:
            arr[k] = v.mean()
    return arr


def summarise(ann: dict) -> dict:
    out = {}
    for f in FIELDS:
        y = np.asarray(ann[f], float)
        ok = np.isfinite(y)
        tt = trend_test(YEARS[ok], y[ok])
        mt = mean_test(y[ok])
        out[f] = {
            "unit": UNITS[f],
            "n_years": int(ok.sum()),
            "window": list(WINDOW),
            "mean": float(mt["value"]),
            "mean_ci95": [float(mt["ci"][0]), float(mt["ci"][1])],
            "trend_per_decade": float(10.0 * tt["value"]),
            "trend_ci95_per_decade": [float(10.0 * tt["ci"][0]), float(10.0 * tt["ci"][1])],
            "p": float(tt["p"]),
            "n_eff": float(tt["n_eff"]),
            "n_eff_at_floor": bool(tt["n_eff_at_floor"]),
            "lag1": float(tt["lag1"]),
            "phi_corrected": float(tt["phi_corrected"]),
        }
    return out


def en4_crosscheck(ann_or: dict, key: str) -> dict | None:
    """Compare the EN4 annual dS with the series stored in en4.json."""
    ref_path = OUT.parent / "en4.json"
    if not ref_path.exists():
        return None
    ref = json.loads(ref_path.read_text())
    ref_years = np.array(ref["years"])
    ref_dS = np.array(ref[key]["dS"]["series"], float)
    diffs = []
    for k, y in enumerate(YEARS):
        idx = np.where(ref_years == y)[0]
        if idx.size:
            diffs.append(abs(ann_or["dS"][k] - ref_dS[idx[0]]))
    return {
        "compared_with": "en4.json",
        "key": key,
        "years": int(len(diffs)),
        "max_abs_dS_difference_PSU": float(np.max(diffs)) if diffs else None,
    }


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    rg = load_rg09(RG_ROW_LAT)
    en4 = read_en4()
    sources = {
        "EN4": {
            "sbar": en4["sbar"],
            "years": en4["years"],
            "depth": en4["depth"],
            "e3": en4["e3"],
        },
        "ORAS5": section_source(CIRC_FILES["ORAS5"]),
        "GLORYS12V1": section_source(CIRC_FILES["GLORYS12V1"]),
        "RG09": {
            "sbar": rg["sbar"],
            "years": rg["years"],
            "depth": rg["depth"],
            "e3": rg["e3"],
        },
    }

    summary: dict = {}
    annual: dict = {}
    for layer, zmax in LAYERS:
        summary[layer] = {}
        annual[layer] = {}
        for circ, cfile in CIRC_FILES.items():
            summary[layer][circ] = {}
            annual[layer][circ] = {}
            for src, s in sources.items():
                if layer == "full_column" and src == "RG09":
                    continue  # RG09 spans 0 to 2000 dbar only
                v = frozen_velocity(cfile, s["depth"], s["e3"])
                monthly = [
                    limbs(v, s["sbar"][i], s["e3"], zmax=zmax, depth=s["depth"])
                    for i in range(len(s["sbar"]))
                ]
                ann = annual_from_monthly(monthly, s["years"])
                annual[layer][circ][src] = ann
                summary[layer][circ][src] = summarise(ann)

            # the reanalysis's own monthly limbs (own circulation and salinity)
            if layer == "full_column":
                prod = "ORAS5" if circ == "ORAS5" else "GLORYS12V1"
                own = sources[prod]["own"]
                ann_own = {
                    f: annual_from_values(own[f], sources[prod]["years"]) for f in FIELDS
                }
                annual[layer][circ][f"{prod}_own_monthly"] = ann_own
                summary[layer][circ][f"{prod}_own_monthly"] = summarise(ann_own)

        # RG09 minus EN4 over the same upper 2000 m layer
        if layer == "upper_2000m":
            for circ in CIRC_FILES:
                a_rg = annual[layer][circ]["RG09"]
                a_en = annual[layer][circ]["EN4"]
                diff = {f: a_rg[f] - a_en[f] for f in FIELDS}
                annual[layer][circ]["RG09_minus_EN4"] = diff
                summary[layer][circ]["RG09_minus_EN4"] = summarise(diff)

    checks = {
        "en4_vs_en4json_ORAS5_circulation_full_column": en4_crosscheck(
            annual["full_column"]["ORAS5"]["EN4"], "oras5"
        ),
        "en4_vs_en4json_GLORYS12V1_circulation_full_column": en4_crosscheck(
            annual["full_column"]["GLORYS12V1"]["EN4"], "glorys12"
        ),
        "en4_months_in_window": en4["n_months"],
        "en4_first_last_month": [en4["first"], en4["last"]],
        "en4_section_row_latitude": TARGET_LAT,
        "rg09_section_row_latitude": rg["info"]["row_latitude_used"],
        "rg09_segment_lon_range": rg["info"]["segment_lon_range"],
        "rg09_segment_n_points": rg["info"]["segment_n_points"],
        "rg09_layer_thickness_sum_m": rg["info"]["layer_thickness_sum_m"],
    }
    # Largest residual lag-1 autocorrelation the calibration can invert for
    # this window (trend design, n = 20). Larger values give phi = 0.98 and the
    # effective sample size at its floor of 3.
    from paper3r1_stats import PHI_GRID, bias_table, trend_design

    bt = bias_table(trend_design(YEARS))
    checks["trend_test_lag1_ceiling_n20"] = {
        "max_mean_lag1_at_phi_0p98": float(bt[-1]),
        "phi_grid_max": float(PHI_GRID[-1]),
        "note": "residual lag-1 above this value saturates phi at 0.98 and the N_eff at its floor",
    }
    en4_first = next(
        f
        for f in sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
        if WINDOW[0] <= int(f.stem.split(".")[-1][:4]) <= WINDOW[1]
    )
    en4_row = en4_month(en4_first)
    checks["en4_segment_lon_range"] = [float(en4_row["lon_min"]), float(en4_row["lon_max"])]
    checks["en4_segment_n_points"] = int(en4_row["n_points"])

    table = []
    for layer in summary:
        for circ in summary[layer]:
            for src, quants in summary[layer][circ].items():
                for f in FIELDS:
                    q = quants[f]
                    table.append(
                        {
                            "layer": layer,
                            "circulation": circ,
                            "source": src,
                            "quantity": f,
                            "unit": q["unit"],
                            "mean": q["mean"],
                            "mean_ci95": q["mean_ci95"],
                            "trend_per_decade": q["trend_per_decade"],
                            "trend_ci95_per_decade": q["trend_ci95_per_decade"],
                            "p": q["p"],
                            "n_eff": q["n_eff"],
                            "n_years": q["n_years"],
                        }
                    )

    out = {
        "id": "n05",
        "analysis": "Limb salinity levels frozen at each circulation, EN4, RG09 and reanalysis salinity",
        "window": list(WINDOW),
        "section": {
            "reanalysis_sections": "34.5 S (ORAS5 -34.56, GLORYS12V1 -34.5), section fields of paper3r1_section_<product>.nc",
            "EN4_row_latitude": TARGET_LAT,
            "RG09_row_latitude": rg["info"]["row_latitude_used"],
        },
        "method": {
            "velocity": "record-mean barotropic-corrected profile V_bc of ORAS5 or GLORYS12V1, interpolated onto each source's levels (frozen_velocity)",
            "limbs": "net transport removed, split by sign, transport-weighted S_N and S_S; F_ovS = -(1/S0) T dS with S0 = 35 PSU (limbs of analysis_paper3v2_en4_depth_test.py)",
            "layers": {
                "full_column": "all levels of each source (EN4 and reanalyses); not available for RG09",
                "upper_2000m": "levels with centre depth <= 2000 m; transport removal re-applied in the layer; RG09 uses all of its 0 to 2000 dbar levels (pressure as depth)",
            },
            "averaging": "monthly limb quantities, then calendar-year means over 2005 to 2024",
            "tests": "trend_test (OLS trend of annual means, bias-corrected N_eff, interval from the test inversion); mean_test for window means",
            "reanalysis_sources_note": "the reanalyses' own salinity with the frozen circulation of the other product is a cross-product combination, not the product's own limb series",
        },
        "rg09": rg["info"],
        "checks": checks,
        "table": table,
        "summary": summary,
        "annual_series": {
            layer: {
                circ: {
                    src: {f: np.asarray(a[f], float).tolist() for f in FIELDS}
                    for src, a in d.items()
                }
                for circ, d in per_c.items()
            }
            for layer, per_c in annual.items()
        },
        "years": YEARS.tolist(),
    }

    def clean_json(o):
        if isinstance(o, dict):
            return {k: clean_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean_json(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(float(o)) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return clean_json(o.tolist())
        return o

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean_json(out), indent=1))

    print(f"wrote {OUT}")
    print(f"RG09: base {rg['info']['base_first']}..{rg['info']['base_last']} "
          f"({rg['info']['base_months']} months), {rg['info']['monthly_files']} monthly files "
          f"{rg['info']['monthly_first']}..{rg['info']['monthly_last']}")
    for row in table:
        if row["quantity"] in ("S_north", "S_south", "dS"):
            print(
                f"{row['layer']:12s} {row['circulation']:10s} {row['source']:20s} "
                f"{row['quantity']:8s} mean {row['mean']:8.4f}  "
                f"trend {row['trend_per_decade']:+.4f} [{row['trend_ci95_per_decade'][0]:+.4f}, "
                f"{row['trend_ci95_per_decade'][1]:+.4f}] /dec p={row['p']:.3f} n={row['n_years']}"
            )


if __name__ == "__main__":
    main()
