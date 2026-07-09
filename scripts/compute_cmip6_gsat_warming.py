#!/usr/bin/env python3
"""Global-mean surface-air warming per CMIP6 model + warming-partialled coupling.

Addresses the shared-driver objection to the dFs-vs-weakening coupling
(R1.2): warming-enhanced evaporative salinification could load onto the
salinity term by construction while also scaling with AMOC weakening.
The discriminating check is the partial correlation of delta_s vs AMOC
weakening with each model's global-mean warming (GSAT change) held fixed.

Downloads Amon tas from the Pangeo CMIP6 cloud catalogue for the 17
forced-weakening models, computes
    dT = mean(tas, ssp585 2081-2100) - mean(tas, historical 1950-1980)
with cos(lat) area weights, then computes partial Pearson and Spearman
correlations.

Writes: data/results/cmip6_gsat_warming.csv
        revision/results/R1_coupling_partial_warming.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RESULTS = Path("data/results")
OUT_R = Path("revision/results")
CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"

BASE = (1950, 1980)
FUT = (2081, 2100)

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CESM2",
    "CESM2-WACCM",
    "CNRM-CM6-1",
    "FGOALS-f3-L",
    "FGOALS-g3",
    "FIO-ESM-2-0",
    "GFDL-CM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NESM3",
    "UKESM1-0-LL",
]


def _member_sort_key(member: str) -> tuple[int, ...]:
    import re

    m = re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)", member)
    if not m:
        return (999, 999, 999, 999)
    r, i, p, f = (int(g) for g in m.groups())
    return (f, p, i, r)  # prefer low forcing/physics index first, then r


def pick_member(cat: pd.DataFrame, model: str) -> str | None:
    """Member available for both historical and ssp585 tas/Amon."""
    sub = cat[
        (cat.source_id == model) & (cat.table_id == "Amon") & (cat.variable_id == "tas")
    ]
    hist = set(sub[sub.experiment_id == "historical"].member_id)
    ssp = set(sub[sub.experiment_id == "ssp585"].member_id)
    common = hist & ssp
    if not common:
        return None
    if "r1i1p1f1" in common:
        return "r1i1p1f1"
    return sorted(common, key=_member_sort_key)[0]


def global_annual_mean(zstore: str) -> pd.Series:
    ds = xr.open_zarr(fsspec.get_mapper(zstore, token="anon"), consolidated=True)
    tas = ds["tas"]
    lat_name = "lat" if "lat" in tas.dims else "latitude"
    lon_name = "lon" if "lon" in tas.dims else "longitude"
    w = np.cos(np.deg2rad(ds[lat_name]))
    gm = tas.weighted(w).mean((lat_name, lon_name))
    annual = gm.groupby("time.year").mean().compute()
    return pd.Series(annual.values, index=annual["year"].values)


def compute_dt(cat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        member = pick_member(cat, model)
        if member is None:
            log.warning(f"{model}: no common tas member; skipped")
            continue
        try:
            sub = cat[
                (cat.source_id == model)
                & (cat.table_id == "Amon")
                & (cat.variable_id == "tas")
                & (cat.member_id == member)
            ]
            z_hist = sub[sub.experiment_id == "historical"].zstore.iloc[0]
            z_ssp = sub[sub.experiment_id == "ssp585"].zstore.iloc[0]
            hist = global_annual_mean(z_hist)
            ssp = global_annual_mean(z_ssp)
            base = hist[(hist.index >= BASE[0]) & (hist.index <= BASE[1])].mean()
            fut = ssp[(ssp.index >= FUT[0]) & (ssp.index <= FUT[1])].mean()
            dt = float(fut - base)
            rows.append(
                {
                    "model": model,
                    "member": member,
                    "dT_K": dt,
                    "base_K": float(base),
                    "fut_K": float(fut),
                }
            )
            log.info(f"{model:18s} {member:10s} dT = {dt:+.2f} K")
        except Exception as e:  # noqa: BLE001 - log and continue per model
            log.warning(f"{model}: FAILED ({e})")
    return pd.DataFrame(rows)


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial Pearson correlation r_xy.z with two-sided p (t, n-3 dof)."""
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    rp = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    n = len(x)
    t = rp * np.sqrt((n - 3) / (1 - rp**2))
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return float(rp), float(p)


def _rank(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a)


def main() -> None:
    log.info("loading Pangeo catalogue ...")
    cat = pd.read_csv(CATALOG_URL)
    log.info(f"catalogue rows: {len(cat)}")

    dt = compute_dt(cat)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dt.to_csv(RESULTS / "cmip6_gsat_warming.csv", index=False)
    log.info(f"saved {RESULTS / 'cmip6_gsat_warming.csv'} (n={len(dt)})")

    # ---- partial correlations against the coupling inputs ----
    df = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")
    npz = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    weak = {}
    for m in [str(x) for x in npz["models"]]:
        yrs = npz[f"{m}_years"].astype(float)
        a = np.asarray(npz[f"{m}_amoc"], dtype=float)
        b = np.nanmean(a[(yrs >= BASE[0]) & (yrs <= BASE[1])])
        f = np.nanmean(a[(yrs >= FUT[0]) & (yrs <= FUT[1])])
        if np.isfinite(b) and np.isfinite(f) and b > 0:
            weak[m] = 100.0 * (1.0 - f / b)
    df["weakening_pct"] = df["model"].map(weak)
    df["dT_K"] = df["model"].map(dict(zip(dt["model"], dt["dT_K"], strict=True)))
    fw = df[(df["delta_total"] < -0.01)].dropna(subset=["weakening_pct", "dT_K"]).copy()
    log.info(f"partial-correlation sample: n={len(fw)}")

    w = fw["weakening_pct"].to_numpy()
    ds_ = (fw["delta_s"] * 1000).to_numpy()
    dv_ = (fw["delta_v"] * 1000).to_numpy()
    z = fw["dT_K"].to_numpy()

    res: dict = {
        "n": int(len(fw)),
        "models": fw["model"].tolist(),
        "windows": {"base": BASE, "fut": FUT},
    }

    def _rp(a: np.ndarray, b: np.ndarray) -> dict:
        r, p = stats.pearsonr(a, b)
        return {"r": float(r), "p": float(p)}

    res["raw"] = {
        "weakening_vs_dT": _rp(w, z),
        "delta_s_vs_dT": _rp(ds_, z),
        "delta_v_vs_dT": _rp(dv_, z),
    }
    rp, pp = _partial_corr(w, ds_, z)
    res["partial_pearson_weakening_vs_delta_s_given_dT"] = {"r": rp, "p": pp}
    rp_v, pp_v = _partial_corr(w, dv_, z)
    res["partial_pearson_weakening_vs_delta_v_given_dT"] = {"r": rp_v, "p": pp_v}
    srp, spp = _partial_corr(_rank(w), _rank(ds_), _rank(z))
    res["partial_spearman_weakening_vs_delta_s_given_dT"] = {"rho": srp, "p": spp}
    srp_v, spp_v = _partial_corr(_rank(w), _rank(dv_), _rank(z))
    res["partial_spearman_weakening_vs_delta_v_given_dT"] = {"rho": srp_v, "p": spp_v}

    OUT_R.mkdir(parents=True, exist_ok=True)
    out = OUT_R / "R1_coupling_partial_warming.json"
    out.write_text(json.dumps(res, indent=2))
    log.info(json.dumps(res, indent=2))
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
