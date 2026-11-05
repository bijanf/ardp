#!/usr/bin/env python3
"""Trend per unit transport weight by layer and band, and the EN4 observation share.

For each product (ORAS5, GLORYS12V1, EN4.2.2 on its 34.0 S row) and each frozen
circulation (ORAS5 or GLORYS12V1 record-mean V_bc), the northward-limb salinity
is decomposed into section cells as in ``analysis_paper3r1_salinification_structure.py``.
With the level transport weights w_k and the cell area fractions f_kx = a_kx / a_k,
the transport weight of a cell is

    W_kx = w_k f_kx,     sum over all cells = 1,

and the cell contribution to the trend of the northward-limb salinity is
C_kx = W_kx ds_kx/dt. For every group G of cells (a band, a layer, or a band
and a layer) the script computes

* the transport-weighted contribution C_G (the Fig. 4e bars) and its share of
  the total, with an interval from the joint AR(1) sieve bootstrap of
  ``paper3r1_stats``;
* the trend per unit transport weight, the trend of the group's
  transport-weighted mean salinity, equal to C_G / W_G, with the corrected
  interval of ``trend_test``;
* the uniform-area share: the same decomposition with area weights
  A_kx = a_kx / a_total in place of W_kx (the Fig. 4e black ticks are the
  band-level area fractions times the total);
* the uniform-cell share: each wet cell carries equal weight, so the share is
  the sum of the cell trends in G over the sum over all cells, and the mean
  cell trend is that sum divided by the number of cells in G.

Layers follow the brief: upper 500 m (depth < 500 m), 500 to 1000 m, and below
1000 m. Bands are the four of Fig. 4e (western boundary west of 45 W, western
interior 45 to 20 W, eastern interior 20 W to 0, Cape Basin east of 0) and the
three-band grouping used for the text: western boundary, interior (WI and EI
together), and eastern boundary (Cape Basin).

Windows: 2005 to 2024 and 2008 to 2024. Trends are per decade.

Observation share (EN4 only): for each EN4 cell the observation weight
(``salinity_observation_weights``) is averaged over the window. The script
reports the per-cell mean observation weight of each group, the share of the
section area-weighted observation weight, and the share of the
transport-weighted observation weight under each frozen circulation. Core-Argo-only
profile selection is not computed: the repository holds only the gridded EN4
monthly fields, with no profile-level files.

Validation: the own-circulation and EN4 totals and the band totals of the
2005 to 2024 window are compared with the stored
``PAPER_3_v2/analysis/r1_salinification_structure.json``.

Seeds: each sieve call uses numpy default_rng(20261104), with 10,000 resamples.

Inputs:  data/results/paper3r1_section3d_{oras5,glorys12}.nc,
         data/results/paper3r1_section_{oras5,glorys12}.nc (record-mean V_bc),
         data/en4/EN.4.2.2.f.analysis.g10.*.nc,
         PAPER_3_v2/analysis/r1_salinification_structure.json (validation only).
Output:  PAPER_3_v2/analysis/jc_r2_G06_en4rows_layerband.json
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_en4 import (  # noqa: E402
    EN4_DIR,
    RESULTS,
    TARGET_LAT,
    atlantic_segment,
    frozen_velocity,
)
from compute_paper3r1_section_quantities import thickness  # noqa: E402
from paper3r1_stats import N_BOOT, sieve_coefficients, trend_design, trend_test  # noqa: E402

OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G06_en4rows_layerband.json"
STORED = REPO / "PAPER_3_v2" / "analysis" / "r1_salinification_structure.json"
SEED = 20261104
WINDOWS = {"2005-2024": (2005, 2024), "2008-2024": (2008, 2024)}
CIRCS = {"ORAS5": "paper3r1_section_oras5.nc", "GLORYS12V1": "paper3r1_section_glorys12.nc"}
PRODUCTS = ("ORAS5", "GLORYS12V1", "EN4.2.2")

BAND4 = {"WB": (-60.0, -45.0), "WI": (-45.0, -20.0), "EI": (-20.0, 0.0), "CB": (0.0, 25.0)}
BAND3 = {"WB": ("WB",), "INT": ("WI", "EI"), "EB": ("CB",)}
LAYER3 = {"upper 500 m": (0.0, 500.0), "500-1000 m": (500.0, 1000.0), "below 1000 m": (1000.0, np.inf)}


# ── section cells ─────────────────────────────────────────────────────


def reanalysis_cells(key: str) -> dict:
    f3 = xr.open_dataset(RESULTS / f"paper3r1_section3d_{key}.nc")
    depth = f3["depth"].values.astype(float)
    dz = np.asarray(thickness(depth, key, f3["e3t"].values, "bounds"), dtype=float)
    lon = f3["lon"].values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    e1t = f3["e1t"].values.astype(float)
    s = f3["s"]
    wet0 = np.isfinite(s.isel(time=0).values)
    ann = s.groupby("time.year").mean().load()
    out = {
        "depth": depth,
        "dz": dz,
        "lon": lon,
        "e1t": e1t,
        "wet0": wet0,
        "years": ann["year"].values.astype(int),
        "ann": ann.values.astype(float),
        "grid": f"{key} section, {depth.size} levels, {lon.size} longitudes",
    }
    f3.close()
    return out


def _en4_month(path: Path):
    import xarray as xr_local

    with xr_local.open_dataset(path, decode_timedelta=False) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        j = int(np.abs(lat - TARGET_LAT).argmin())
        sal = ds["salinity"].isel(time=0, lat=j).values
        wobs = ds["salinity_observation_weights"].isel(time=0, lat=j).values
        year = int(pd.Timestamp(ds["time"].values[0]).year)
        depth = ds["depth"].values.astype(float)
        bnds = ds["depth_bnds"].values
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    sel, e1t = atlantic_segment(lon180, np.full(lon.shape, float(lat[j])), np.isfinite(sal[0]))
    return year, sal[:, sel], wobs[:, sel], lon180[sel], e1t, depth, np.abs(bnds[:, 1] - bnds[:, 0])


def en4_cells() -> dict:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(_en4_month, files, chunksize=8))
    years = np.array([r[0] for r in rows])
    sal = np.array([r[1] for r in rows])
    wob = np.array([r[2] for r in rows])
    lon, e1t, depth, dz = rows[0][3], rows[0][4], rows[0][5], rows[0][6]
    uy = np.unique(years)
    ann = np.array([np.nanmean(sal[years == y], axis=0) for y in uy])
    ann_w = np.array([np.nanmean(wob[years == y], axis=0) for y in uy])
    return {
        "depth": depth,
        "dz": dz,
        "lon": lon,
        "e1t": e1t,
        "years": uy.astype(int),
        "ann": ann,
        "ann_w": ann_w,
        "grid": f"EN4 34.0 S row, {depth.size} levels, {lon.size} longitudes",
    }


# ── weights and masks ─────────────────────────────────────────────────


def cell_weights(v: np.ndarray, dz: np.ndarray, e1t: np.ndarray, wet: np.ndarray):
    """Transport weights W_kx, area weights A_kx and the cell area a_kx."""
    area = np.where(wet, e1t[None, :] * dz[:, None], 0.0)
    a_k = area.sum(axis=1)
    ok = a_k > 0
    vv = np.where(ok, v, 0.0)
    vv = vv - (vv * dz)[ok].sum() / dz[ok].sum() * ok
    q = vv * dz
    qn = np.clip(q, 0.0, None)
    wn = qn / qn.sum()
    frac = np.where(ok[:, None], area / np.where(ok, a_k, 1.0)[:, None], 0.0)
    W = wn[:, None] * frac
    A = area / area.sum()
    return W, A, area


def cell_masks(lon: np.ndarray, depth: np.ndarray, wet: np.ndarray) -> dict:
    """Boolean masks of the groups used in the tables, keyed by label."""
    lonm4 = {b: (lon >= lo) & (lon < hi) for b, (lo, hi) in BAND4.items()}
    dm = {
        l: (depth >= lo) & (depth < hi) for l, (lo, hi) in LAYER3.items()
    }
    masks = {}
    allz = np.ones_like(wet, dtype=bool)
    for b in BAND4:
        for l in LAYER3:
            masks[f"{b} | {l}"] = dm[l][:, None] & lonm4[b][None, :] & wet
    for b3, members in BAND3.items():
        lonm = np.zeros(lon.shape, dtype=bool)
        for m in members:
            lonm |= lonm4[m]
        for l in LAYER3:
            masks[f"{b3}3 | {l}"] = dm[l][:, None] & lonm[None, :] & wet
    for b in BAND4:
        masks[f"{b} | all layers"] = lonm4[b][None, :] & wet & allz
    for b3, members in BAND3.items():
        lonm = np.zeros(lon.shape, dtype=bool)
        for m in members:
            lonm |= lonm4[m]
        masks[f"{b3}3 | all layers"] = lonm[None, :] & wet
    for l in LAYER3:
        masks[f"all bands | {l}"] = dm[l][:, None] & wet
    masks["all bands | all layers"] = wet.copy()
    return masks


def group_label_parts(key: str) -> tuple[str, str]:
    band, layer = key.split(" | ")
    return band, layer


# ── one product, one circulation, one window ─────────────────────────


def run_combo(cells: dict, W: np.ndarray, A: np.ndarray, area: np.ndarray, wet_w: np.ndarray,
              ann_w_field: np.ndarray, years: np.ndarray, masks: dict) -> dict:
    keys = list(masks.keys())
    yrs = years.astype(float)
    x_design = trend_design(yrs)
    fields = {
        "transport": W,
        "area": A,
        "cell": wet_w.astype(float),
    }
    series = {}
    for fname, F in fields.items():
        cols = []
        for k in keys:
            Fm = np.where(masks[k], F, 0.0)
            cols.append(np.einsum("tzx,zx->t", ann_w_field, Fm))
        series[fname] = np.column_stack(cols)  # (n, m)

    # Weight totals of each group
    weight_tot = {
        fname: {k: float(np.where(masks[k], F, 0.0).sum()) for k in keys}
        for fname, F in fields.items()
    }
    n_cells = {k: int(masks[k].sum()) for k in keys}
    tot_key = "all bands | all layers"
    j_tot = keys.index(tot_key)

    out = {"n_years": int(len(years)), "years": [int(v) for v in years], "n_wet_cells": int(wet_w.sum()),
           "groups": {}}
    trend_tot = {}
    sieve = {}
    for fname in fields:
        ys = series[fname]
        rng = np.random.default_rng(SEED)
        sieve[fname] = sieve_coefficients(x_design, ys, 1, rng, n_boot=N_BOOT)
        trend_tot[fname] = trend_test(yrs, ys[:, j_tot])

    for j, k in enumerate(keys):
        band, layer = group_label_parts(k)
        g = {"band": band, "layer": layer, "n_cells": n_cells[k]}
        # transport-weighted contribution and per unit weight
        tw = weight_tot["transport"][k]
        t_c = trend_test(yrs, series["transport"][:, j])
        g["transport_weight_fraction"] = tw
        g["transport_contribution"] = {
            "trend_per_decade": t_c["value"] * 10,
            "ci_per_decade": [t_c["ci"][0] * 10, t_c["ci"][1] * 10],
            "n_eff": t_c["n_eff"],
            "p": t_c["p"],
        }
        if tw > 0:
            t_u = trend_test(yrs, series["transport"][:, j] / tw)
            g["trend_per_unit_transport_weight"] = {
                "trend_per_decade": t_u["value"] * 10,
                "ci_per_decade": [t_u["ci"][0] * 10, t_u["ci"][1] * 10],
                "n_eff": t_u["n_eff"],
                "p": t_u["p"],
            }
        else:
            g["trend_per_unit_transport_weight"] = None
        den_t = sieve["transport"][:, j_tot]
        num_t = sieve["transport"][:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_t = num_t / den_t
        g["share_transport"] = {
            "point": float(t_c["value"] / trend_tot["transport"]["value"]),
            "ci95": [float(np.nanpercentile(r_t, 2.5)), float(np.nanpercentile(r_t, 97.5))],
        }
        # uniform area weights
        a_w = weight_tot["area"][k]
        t_a = trend_test(yrs, series["area"][:, j])
        g["uniform_area"] = {
            "area_fraction": a_w,
            "share": {
                "point": float(t_a["value"] / trend_tot["area"]["value"]),
                "ci95": [
                    float(np.nanpercentile(sieve["area"][:, j] / sieve["area"][:, j_tot], 2.5)),
                    float(np.nanpercentile(sieve["area"][:, j] / sieve["area"][:, j_tot], 97.5)),
                ],
            },
            "trend_per_unit_area_fraction": {
                "trend_per_decade": (t_a["value"] * 10) / a_w if a_w > 0 else None,
                "ci_per_decade": [t_a["ci"][0] * 10 / a_w, t_a["ci"][1] * 10 / a_w] if a_w > 0 else None,
            },
        }
        # uniform cell weights
        nc = n_cells[k]
        t_cell = trend_test(yrs, series["cell"][:, j])
        g["uniform_cell"] = {
            "share": {
                "point": float(t_cell["value"] / trend_tot["cell"]["value"]),
                "ci95": [
                    float(np.nanpercentile(sieve["cell"][:, j] / sieve["cell"][:, j_tot], 2.5)),
                    float(np.nanpercentile(sieve["cell"][:, j] / sieve["cell"][:, j_tot], 97.5)),
                ],
            },
            "mean_cell_trend_per_decade": {
                "trend_per_decade": (t_cell["value"] * 10) / nc if nc > 0 else None,
                "ci_per_decade": [t_cell["ci"][0] * 10 / nc, t_cell["ci"][1] * 10 / nc] if nc > 0 else None,
                "n_eff": t_cell["n_eff"],
            },
        }
        out["groups"][k] = g
    out["total"] = {
        "transport_trend_per_decade": trend_tot["transport"]["value"] * 10,
        "transport_ci_per_decade": [trend_tot["transport"]["ci"][0] * 10, trend_tot["transport"]["ci"][1] * 10],
        "transport_n_eff": trend_tot["transport"]["n_eff"],
        "transport_p": trend_tot["transport"]["p"],
    }
    return out


def obs_share(cells_w: np.ndarray, W_by_circ: dict, area: np.ndarray, masks: dict) -> dict:
    """Mean observation weight per cell and area and transport shares of each group."""
    keys = list(masks.keys())
    wbar = cells_w
    tot_area = float((area * wbar).sum())
    tot_t = {c: float((W * wbar).sum()) for c, W in W_by_circ.items()}
    out = {}
    for k in keys:
        m = masks[k]
        n = int(m.sum())
        out[k] = {
            "n_cells": n,
            "mean_obs_weight_per_cell": float(wbar[m].mean()) if n else None,
            "obs_share_area_weighted": float((area * wbar)[m].sum() / tot_area) if tot_area > 0 else None,
            "obs_share_transport_weighted": {
                c: float((W * wbar)[m].sum() / tot_t[c]) for c, W in W_by_circ.items()
            },
        }
    return out


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    stored = json.loads(STORED.read_text())
    results: dict = {}
    validation: dict = {}

    en4 = en4_cells()
    reana = {"ORAS5": reanalysis_cells("oras5"), "GLORYS12V1": reanalysis_cells("glorys12")}
    circ_v = {}
    for cname, fname in CIRCS.items():
        with xr.open_dataset(RESULTS / fname) as ds:
            circ_v[cname] = (ds["depth"].values.astype(float), ds["V_bc"].values.astype(float).mean(axis=0))

    for prod in PRODUCTS:
        results[prod] = {}
        for cname in CIRCS:
            results[prod][cname] = {}
            for wname, (y0, y1) in WINDOWS.items():
                if prod == "EN4.2.2":
                    C = en4
                    sel = (C["years"] >= y0) & (C["years"] <= y1)
                    yrs = C["years"][sel]
                    ann = C["ann"][sel]
                    wet = np.isfinite(ann).all(axis=0)
                    dz = C["dz"]
                    e1t = C["e1t"]
                    depth = C["depth"]
                    lon = C["lon"]
                    v = frozen_velocity(CIRCS[cname], depth, dz)
                else:
                    C = reana["ORAS5" if prod == "ORAS5" else "GLORYS12V1"]
                    sel = (C["years"] >= y0) & (C["years"] <= y1)
                    yrs = C["years"][sel]
                    ann = C["ann"][sel]
                    wet = C["wet0"] & np.isfinite(ann).all(axis=0)
                    dz = C["dz"]
                    e1t = C["e1t"]
                    depth = C["depth"]
                    lon = C["lon"]
                    dph, vm = circ_v[cname]
                    v = np.interp(depth, dph, vm, left=vm[0], right=0.0)
                ann_w = np.where(wet[None], ann, 0.0)
                W, A, area = cell_weights(v, dz, e1t, wet)
                masks = cell_masks(lon, depth, wet)
                res = run_combo(C, W, A, area, wet, ann_w, yrs, masks)
                res["grid"] = C["grid"]
                # Longitude-width share of each band, as the uniform ticks of Fig. 4e.
                width4 = {
                    b: float(C["e1t"][(lon >= lo) & (lon < hi)].sum() / C["e1t"].sum())
                    for b, (lo, hi) in BAND4.items()
                }
                width3 = {
                    b3: float(sum(width4[m] for m in members)) for b3, members in BAND3.items()
                }
                for gk, gv in res["groups"].items():
                    if gk.endswith("| all layers") and gv["band"] in width4:
                        gv["uniform_width_share"] = width4[gv["band"]]
                    elif gk.endswith("| all layers") and gv["band"].endswith("3") and gv["band"][:-1] in width3:
                        gv["uniform_width_share"] = width3[gv["band"][:-1]]
                    elif gk == "all bands | all layers":
                        gv["uniform_width_share"] = 1.0
                if prod == "EN4.2.2":
                    res["window_note"] = "EN4 monthly fields averaged to calendar years; wet cells require finite annual means in the window"
                results[prod][cname][wname] = res

                # Validation against the stored decomposition (2005-2024 windows).
                if wname == "2005-2024" and (prod == "EN4.2.2" or prod == cname):
                    if prod == "EN4.2.2":
                        st = stored["en4"]["frozen"]["oras5" if cname == "ORAS5" else "glorys12"]
                        ref_total = st["S_north_frozen_trend_sum_of_cells"]
                        ref_bands = st["northward_by_band"]
                    else:
                        st = stored["oras5" if prod == "ORAS5" else "glorys12"]["windows"]["2005-2024"]
                        ref_total = st["S_north_frozen_trend_sum_of_cells"]
                        ref_bands = st["northward_by_band"]
                    key = f"{prod}|circulation {cname}|2005-2024"
                    rec = results[prod][cname][wname]["groups"]
                    band_diff = {}
                    for b_full, b_key in (
                        ("western boundary", "WB"),
                        ("western interior", "WI"),
                        ("eastern interior", "EI"),
                        ("Cape Basin", "CB"),
                    ):
                        stored_val = [v for k2, v in ref_bands.items() if k2.startswith(b_full)][0]
                        rec_val = sum(
                            rec[f"{b_key} | {l}"]["transport_contribution"]["trend_per_decade"]
                            for l in LAYER3
                        )
                        band_diff[b_key] = float(rec_val - stored_val)
                    validation[key] = {
                        "total_recomputed_PSU_per_decade": res["total"]["transport_trend_per_decade"],
                        "total_stored_PSU_per_decade": float(ref_total),
                        "total_recomputed_minus_stored": float(res["total"]["transport_trend_per_decade"] - ref_total),
                        "band_totals_recomputed_minus_stored_PSU_per_decade": band_diff,
                    }

        if prod == "EN4.2.2":
            # Observation share per cell (window mean of the EN4 observation weights).
            obs = {}
            for wname, (y0, y1) in WINDOWS.items():
                sel = (en4["years"] >= y0) & (en4["years"] <= y1)
                wbar = np.nanmean(en4["ann_w"][sel], axis=0)
                ann = en4["ann"][sel]
                wet = np.isfinite(ann).all(axis=0)
                wbar = np.where(wet, wbar, 0.0)
                Wt = {}
                for cname, fname in CIRCS.items():
                    v = frozen_velocity(fname, en4["depth"], en4["dz"])
                    W, _, _ = cell_weights(v, en4["dz"], en4["e1t"], wet)
                    Wt[cname] = W
                area = np.where(wet, en4["e1t"][None, :] * en4["dz"][:, None], 0.0)
                masks = cell_masks(en4["lon"], en4["depth"], wet)
                obs[wname] = {
                    "window": list(WINDOWS[wname]),
                    "groups": obs_share(wbar, Wt, area, masks),
                    "note": (
                        "mean_obs_weight_per_cell is the unweighted mean of the window-mean "
                        "salinity_observation_weights over the cells of the group; "
                        "obs_share_area_weighted is the area-weighted share of the section "
                        "observation weight; obs_share_transport_weighted uses the transport "
                        "weights under each frozen circulation."
                    ),
                }
            results["EN4.2.2"]["observation_share"] = obs

    out = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G06_en4rows_layerband.py",
            "action_ids": ["A105"],
            "cluster": "C53",
            "inputs": [
                "data/results/paper3r1_section3d_oras5.nc, paper3r1_section3d_glorys12.nc",
                "data/results/paper3r1_section_oras5.nc, paper3r1_section_glorys12.nc (record-mean V_bc)",
                "data/en4/EN.4.2.2.f.analysis.g10.*.nc (34.0 S row)",
                "PAPER_3_v2/analysis/r1_salinification_structure.json (validation only)",
            ],
            "windows_years": {k: list(v) for k, v in WINDOWS.items()},
            "layers": {"upper 500 m": "depth < 500 m", "500-1000 m": "500 <= depth < 1000 m", "below 1000 m": "depth >= 1000 m"},
            "bands_fig4e": {
                "WB": "western boundary, 60 W to 45 W",
                "WI": "western interior, 45 W to 20 W",
                "EI": "eastern interior, 20 W to 0",
                "CB": "Cape Basin, 0 to 25 E",
            },
            "bands_three": {"WB": "western boundary (WB)", "INT": "interior (WI and EI)", "EB": "eastern boundary (CB)"},
            "groups_keys": "'<band> | <layer>', '<band>3 | <layer>' (three-band grouping), '<band> | all layers', '<band>3 | all layers', 'all bands | <layer>', 'all bands | all layers'",
            "weights": {
                "transport": "W_kx = w_k a_kx/a_k, w_k = positive part of the record-mean level transport normalised over the section; sums to 1",
                "area": "A_kx = a_kx / a_total",
                "cell": "equal weight per wet cell",
            },
            "circulation_frozen": "record-mean V_bc of the named reanalysis, interpolated to the product levels; renormalised over the levels with ocean",
            "trend_units": "PSU per decade (salinity); transport-weight fraction is dimensionless",
            "intervals": "95 per cent; trend intervals from trend_test (bias-corrected lag-one effective sample size); share intervals from the joint AR(1) sieve bootstrap",
            "seed": SEED,
            "bootstrap_resamples": N_BOOT,
            "limitations": {
                "core_argo_only": "not computed: no profile-level EN4 files in data/en4 (gridded monthly g10 fields only)",
            },
        },
        "results": results,
        "validation_against_r1_salinification_structure": validation,
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT}")
    for k, v in validation.items():
        print(k, "total diff", v["total_recomputed_minus_stored"], "band diffs", v["band_totals_recomputed_minus_stored_PSU_per_decade"])


if __name__ == "__main__":
    main()
