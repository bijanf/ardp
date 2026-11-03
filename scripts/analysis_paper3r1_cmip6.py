#!/usr/bin/env python3
"""CMIP6 at 34.5 S on the same basis as the reanalyses.

For every model of the 25-model set, the overturning freshwater transport is
computed month by month from the 34.5 S sections of meridional velocity and
salinity, exactly as for the reanalyses: the contiguous Atlantic run of ocean
points containing 25 W, zonal cell widths at each point's own latitude,
vertical cell thicknesses from level bounds reconstructed as midpoints between
level centres, removal of the net section transport as a uniform velocity over
the section area, and the limb factorisation F_ovS = -(1/S0) T dS. Monthly
values are averaged afterwards, so the within-window covariance of velocity and
salinity is retained.

Windows: 1850-1900 and 1950-1980 (historical), 1993-2014 (historical, the
years all three reanalyses cover), and 2081-2100 (ssp585). The overturning
strength at 34.5 S is the maximum over depths of at least 500 m of the
streamfunction of the annual-mean uncorrected velocity, the definition used at
26.5 N.

Cross-model tests: the overlap-window comparison with the reanalyses; the sign
of F_ovS against the sign of the limb salinity contrast; the relation of the
mean F_ovS to the forced change of the overturning at 26.5 N and at 34.5 S; the
relation between the forced changes at the two latitudes; and, within each
model, the correlation of annual overturning at the two latitudes.

Writes ``PAPER_3_v2/analysis/r1_cmip6.json`` and
``data/results/paper3r1_cmip6_<model>.nc``.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compute_paper3r1_section_quantities import section_series  # noqa: E402
from compute_paper3v2_section_profiles import atlantic_segment  # noqa: E402
from paper3r1_stats import corr_test  # noqa: E402

CMIP6_DIR = REPO / "data" / "cmip6_sections"
RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "r1_cmip6.json"
POP_MODELS = {"CESM2", "CESM2-WACCM"}
WINDOWS = {
    "1850-1900": (1850, 1900),
    "1950-1980": (1950, 1980),
    "1993-2014": (1993, 2014),
    "2081-2100": (2081, 2100),
}
CHUNK = 120


def load(model: str, exp: str):
    pv, ps = CMIP6_DIR / f"{model}_{exp}_vo.nc", CMIP6_DIR / f"{model}_{exp}_so.nc"
    if not (pv.exists() and ps.exists()):
        return None
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    dv = xr.open_dataset(pv, decode_times=coder)
    dsal = xr.open_dataset(ps, decode_times=coder)
    v, s = dv["vo"], dsal["so"]
    x_dim = next(d for d in v.dims if d in ("i", "x", "nlon", "j", "lon"))
    lev_dim = next(d for d in v.dims if d in ("lev", "olevel", "depth", "z_t"))
    lon = next(dv[c].values for c in ("longitude", "lon", "nav_lon") if c in dv)
    lat = next((dv[c].values for c in ("latitude", "lat", "nav_lat") if c in dv), None)
    depth = dv[lev_dim].values.astype(float)
    if model in POP_MODELS:
        depth = depth / 100.0
    lon = np.where(lon > 180, lon - 360, lon).astype(float)
    lat = (
        np.full(lon.shape, -34.5)
        if lat is None or np.ndim(lat) != 1
        else lat.astype(float)
    )
    years = v["time"].dt.year.values
    months = v["time"].dt.month.values
    return {
        "v": v.transpose("time", lev_dim, x_dim),
        "s": s.transpose("time", lev_dim, x_dim),
        "lon": lon,
        "lat": lat,
        "depth": depth,
        "years": years,
        "months": months,
    }


def bounds_thickness(depth: np.ndarray) -> np.ndarray:
    order = np.argsort(depth)
    z = depth[order]
    zw = np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1])])
    zw = np.append(zw, z[-1] + 0.5 * (z[-1] - z[-2]))
    e3 = np.empty_like(z)
    e3[order] = np.diff(zw)
    return e3


def monthly_series(d: dict) -> pd.DataFrame:
    order = np.argsort(d["depth"])
    s0 = d["s"].isel(time=0).values
    s0 = np.where((s0 > 0) & (s0 < 100), s0, np.nan)
    lon180 = ((d["lon"] + 180.0) % 360.0) - 180.0
    # Contiguous ocean run containing 25 W, bounded by 70 W and 20 E: on coarse
    # grids Africa may not reach the section row, and the walk would otherwise
    # continue into the Indian Ocean; 20 E is the meridian of Cape Agulhas.
    in_box = (lon180 >= -70.0) & (lon180 <= 20.0)
    sel, e1t = atlantic_segment(lon180, d["lat"], np.isfinite(s0[order[0]]) & in_box)
    d["lon"] = lon180
    e3t = bounds_thickness(d["depth"])
    rows = []
    nt = d["v"].sizes["time"]
    for i0 in range(0, nt, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, nt))
        v = d["v"].isel(time=sl).values[:, :, sel][:, order, :].astype(float)
        s = d["s"].isel(time=sl).values[:, :, sel][:, order, :].astype(float)
        s = np.where((s > 0) & (s < 100), s, np.nan)
        v = np.where(np.isfinite(s), v, np.nan)
        v = np.where(np.abs(v) < 1e3, v, np.nan)
        out = section_series(v, s, e1t, e3t[order], np.full(len(sel), -1))
        q = out["V_int"] * e3t[order][None, :]
        for k in range(v.shape[0]):
            rows.append(
                {
                    "year": int(d["years"][sl][k]),
                    "month": int(d["months"][sl][k]),
                    "F_ov": out["F_ov"][k],
                    "T": out["T_limb_Sv"][k],
                    "dS": out["dS_limb"][k],
                    "S_north": out["S_north"][k],
                    "S_south": out["S_south"][k],
                    "F_az": out["F_az"][k],
                    "q": q[k],
                }
            )
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(["year", "month"])
        .sort_values(["year", "month"])
    )
    meta = {
        "n_points": int(len(sel)),
        "lon_range": [float(d["lon"][sel].min()), float(d["lon"][sel].max())],
        "lat_range": [float(np.min(d["lat"][sel])), float(np.max(d["lat"][sel]))],
    }
    return df, d["depth"][order], meta


def annual(df: pd.DataFrame, depth: np.ndarray) -> pd.DataFrame:
    g = df.groupby("year")
    ann = g[["F_ov", "T", "dS", "S_north", "S_south", "F_az"]].mean()
    deep = depth >= 500.0
    psi = []
    for _, grp in g:
        qm = np.mean(np.stack(grp["q"].values), axis=0)
        cum = np.cumsum(qm) / 1e6
        psi.append(float(np.nanmax(np.where(deep, cum, np.nan))))
    ann["psi34"] = psi
    return ann


def run_model(model: str) -> dict | None:
    try:
        parts = []
        meta = {}
        for exp in ("historical", "ssp585"):
            d = load(model, exp)
            if d is None:
                continue
            df, depth_sorted, meta[exp] = monthly_series(d)
            parts.append((df, depth_sorted))
        if not parts:
            return None
        depth = parts[0][1]
        df = pd.concat([p[0] for p in parts]).drop_duplicates(
            ["year", "month"], keep="first"
        )
        ann = annual(df, depth)
        ann.to_xarray().to_netcdf(RESULTS / f"paper3r1_cmip6_{model}.nc")
        res = {"model": model, "meta": meta, "windows": {}}
        for name, (y0, y1) in WINDOWS.items():
            sub = ann.loc[(ann.index >= y0) & (ann.index <= y1)]
            if len(sub) < 0.8 * (y1 - y0 + 1):
                continue
            res["windows"][name] = {k: float(sub[k].mean()) for k in ann.columns}
            res["windows"][name]["n_years"] = int(len(sub))
        return res
    except Exception as exc:  # report and continue with the other models
        return {"model": model, "error": repr(exc)}


def pearson(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[ok], y[ok])
    n = int(ok.sum())
    r_crit = float(
        stats.t.ppf(0.975, n - 2) / np.sqrt(n - 2 + stats.t.ppf(0.975, n - 2) ** 2)
    )
    return {"r": float(r), "p": float(p), "n": n, "smallest_detectable_abs_r": r_crit}


def reanalysis_windows() -> dict:
    out = {}
    for key in ("oras5", "glorys12", "ecco"):
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
        a = ds.groupby("time.year").mean()
        y = a["year"].values
        for name, (y0, y1) in WINDOWS.items():
            m = (y >= y0) & (y <= y1)
            if m.sum() >= 0.8 * (y1 - y0 + 1):
                out.setdefault(key, {})[name] = {
                    "F_ov": float(a["F_ov"].values[m].mean()),
                    "T": float(a["T_limb_Sv"].values[m].mean()),
                    "dS": float(a["dS_limb"].values[m].mean()),
                }
    return out


def reanalysis_two_latitudes() -> dict:
    """Annual overturning at 34.5 S against 26.5 N in the reanalyses."""
    out = {}
    for key in ("oras5", "glorys12", "ecco"):
        f26 = RESULTS / f"yearly_amoc26n_{key}.npz"
        if not f26.exists():
            continue
        d = np.load(f26)
        y26 = pd.Series(d["amoc"], index=d["years"].astype(int))
        ds = xr.open_dataset(RESULTS / f"paper3r1_section_{key}.nc")
        a = ds.groupby("time.year").mean()
        s34 = pd.Series(
            a["psi_max_surface_Sv"].values, index=a["year"].values.astype(int)
        )
        t34 = pd.Series(a["T_limb_Sv"].values, index=a["year"].values.astype(int))
        common = sorted(set(y26.index) & set(s34.index))
        x = y26.loc[common].values
        res = {"years": [common[0], common[-1]], "n": len(common)}
        for name, ser in (("psi_max", s34), ("T", t34)):
            yv = ser.loc[common].values
            yr = np.array(common, float)
            res[f"{name}_raw"] = pearson(x, yv)
            # Detrended correlation with the effective sample size of
            # Bretherton et al. (1999) from the corrected lag-1 coefficients.
            res[f"{name}_detrended"] = corr_test(x, yv, detrend=True)
            res[f"{name}_trend_Sv_per_decade"] = float(np.polyfit(yr, yv, 1)[0] * 10)
        res["amoc26_trend_Sv_per_decade"] = float(
            np.polyfit(np.array(common, float), x, 1)[0] * 10
        )
        out[key] = res
    return out


def main() -> None:
    models = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")[
        "model"
    ].tolist()
    old = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv").set_index(
        "model"
    )
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(run_model, models))
    ok = [r for r in results if r and "error" not in r]
    errors = {r["model"]: r["error"] for r in results if r and "error" in r}

    amoc26 = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)
    m26 = set(amoc26["models"].tolist())

    def amoc26_change(m):
        yrs, am = amoc26[f"{m}_years"].astype(int), amoc26[f"{m}_amoc"]
        a = am[(yrs >= 1850) & (yrs <= 1900)].mean()
        b = am[(yrs >= 2081) & (yrs <= 2100)].mean()
        return float(a), float(b - a)

    table = []
    within = []
    for r in ok:
        w = r["windows"]
        row = {"model": r["model"]}
        for name in WINDOWS:
            if name in w:
                row[f"F_{name}"] = w[name]["F_ov"]
                row[f"T_{name}"] = w[name]["T"]
                row[f"dS_{name}"] = w[name]["dS"]
                row[f"psi34_{name}"] = w[name]["psi34"]
        row["F_old_1950-1980"] = float(old.loc[r["model"], "F_ov_baseline"])
        hist = r["meta"].get("historical", {})
        row["n_points"] = hist.get("n_points")
        row["lon_west"], row["lon_east"] = hist.get("lon_range", [np.nan, np.nan])
        row["lat_south"], row["lat_north"] = hist.get("lat_range", [np.nan, np.nan])
        if "1850-1900" in w and "2081-2100" in w:
            row["dpsi34"] = w["2081-2100"]["psi34"] - w["1850-1900"]["psi34"]
        if r["model"] in m26:
            row["amoc26_1850-1900"], row["damoc26"] = amoc26_change(r["model"])
            ann = xr.open_dataset(
                RESULTS / f"paper3r1_cmip6_{r['model']}.nc"
            ).to_dataframe()
            yrs = amoc26[f"{r['model']}_years"].astype(int)
            s26 = pd.Series(amoc26[f"{r['model']}_amoc"], index=yrs)
            common = sorted(set(s26.index) & set(ann.index) & set(range(1850, 2015)))
            if len(common) > 30:
                x = s26.loc[common].values
                yv = ann.loc[common, "psi34"].values
                yr = np.array(common, float)
                dx = x - np.polyval(np.polyfit(yr, x, 1), yr)
                dy = yv - np.polyval(np.polyfit(yr, yv, 1), yr)
                within.append(float(np.corrcoef(dx, dy)[0, 1]))
        table.append(row)
    df = pd.DataFrame(table).set_index("model")

    rean = reanalysis_windows()
    out: dict = {
        "n_models": int(len(df)),
        "errors": errors,
        "per_model": df.reset_index().to_dict(orient="records"),
        "reanalysis_windows": rean,
        "reanalysis_two_latitudes": reanalysis_two_latitudes(),
        "summary": {},
    }
    s = out["summary"]
    for name in ("1950-1980", "1993-2014"):
        col = df[f"F_{name}"].dropna()
        s[f"F_{name}"] = {
            "n": int(col.size),
            "median": float(col.median()),
            "min": float(col.min()),
            "max": float(col.max()),
            "n_positive": int((col > 0).sum()),
            "binomial_p_positive": float(
                stats.binomtest(int((col > 0).sum()), int(col.size)).pvalue
            ),
            "percentile_of_reanalyses": {
                k: float(100 * (col > v[name]["F_ov"]).mean())
                for k, v in rean.items()
                if name in v
            },
        }
        both = df[[f"F_{name}", f"dS_{name}"]].dropna()
        s[f"sign_F_equals_minus_sign_dS_{name}"] = bool(
            (np.sign(both[f"F_{name}"]) == -np.sign(both[f"dS_{name}"])).all()
        )
        s[f"n_negative_dS_{name}"] = int((both[f"dS_{name}"] < 0).sum())
        s[f"T_median_{name}"] = float(df[f"T_{name}"].median())
        s[f"dS_median_{name}"] = float(df[f"dS_{name}"].median())
    s["change_vs_old_1950-1980"] = {
        "max_abs_Sv": float((df["F_1950-1980"] - df["F_old_1950-1980"]).abs().max()),
        "rms_Sv": float(
            np.sqrt(((df["F_1950-1980"] - df["F_old_1950-1980"]) ** 2).mean())
        ),
        "sign_flips": int(
            (np.sign(df["F_1950-1980"]) != np.sign(df["F_old_1950-1980"])).sum()
        ),
    }

    def partial_r(x, y, z) -> float:
        a = np.column_stack([np.ones_like(z), z])
        rx = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]
        ry = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])

    def partial(x, y, z, names=None) -> dict:
        """Partial correlation of x and y given z, with n - 3 degrees of
        freedom, its Fisher interval and its leave-one-out range."""
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        n = len(z)
        r = partial_r(x, y, z)
        dof = n - 3
        tval = r * np.sqrt(dof / (1 - r**2))
        t975 = stats.t.ppf(0.975, dof)
        zr, se = np.arctanh(r), 1.0 / np.sqrt(n - 4)
        loo = [
            partial_r(np.delete(x, i), np.delete(y, i), np.delete(z, i))
            for i in range(n)
        ]
        out = {
            "r": r,
            "n": n,
            "p": float(2 * stats.t.sf(abs(tval), dof)),
            "smallest_detectable_abs_r": float(t975 / np.sqrt(dof + t975**2)),
            "fisher_ci": [
                float(np.tanh(zr - 1.96 * se)),
                float(np.tanh(zr + 1.96 * se)),
            ],
            "leave_one_out_range": [float(min(loo)), float(max(loo))],
        }
        if names is not None:
            i_min = int(np.argmin(loo))
            out["leave_one_out_min_without"] = str(names[i_min])
        return out

    # Genealogy: one representative per family; every choice is enumerated,
    # because the result depends on which member stands for a family.
    families = {
        "CESM2": ["CESM2", "CESM2-WACCM", "TaiESM1"],
        "MPI": ["MPI-ESM1-2-LR", "MPI-ESM1-2-HR"],
        "ACCESS": ["ACCESS-CM2", "ACCESS-ESM1-5"],
        "FGOALS": ["FGOALS-f3-L", "FGOALS-g3"],
        "CMCC": ["CMCC-CM2-SR5", "CMCC-ESM2"],
        "UKMO": ["UKESM1-0-LL", "HadGEM3-GC31-LL"],
    }
    import itertools

    for name in ("1950-1980", "1993-2014"):
        col = df[f"F_{name}"].dropna()
        fams = {k: [m for m in v if m in col.index] for k, v in families.items()}
        fams = {k: v for k, v in fams.items() if v}
        singles = [m for m in col.index if not any(m in v for v in fams.values())]
        counts = []
        for choice in itertools.product(*fams.values()):
            members = singles + list(choice)
            counts.append(int((col.loc[members] > 0).sum()))
        n_ind = len(singles) + len(fams)
        s[f"genealogy_{name}"] = {
            "n_independent": n_ind,
            "families": fams,
            "n_positive_range": [min(counts), max(counts)],
            "binomial_p_range": [
                float(stats.binomtest(c, n_ind).pvalue)
                for c in (min(counts), max(counts))
            ],
        }
    # Models whose section row spans more than 1 degree of latitude.
    span = (df["lat_north"] - df["lat_south"]).abs()
    tilted = sorted(span[span > 1.0].index)
    s["tilted_rows"] = {
        m: [float(df.loc[m, "lat_south"]), float(df.loc[m, "lat_north"])]
        for m in tilted
    }
    for name in ("1950-1980", "1993-2014"):
        col = df[f"F_{name}"].drop(index=tilted, errors="ignore").dropna()
        s[f"F_{name}_without_tilted_rows"] = {
            "n": int(col.size),
            "median": float(col.median()),
            "n_positive": int((col > 0).sum()),
            "percentile_of_reanalyses": {
                k: float(100 * (col > v[name]["F_ov"]).mean())
                for k, v in rean.items()
                if name in v
            },
        }

    if "damoc26" in df:
        sub = df.dropna(subset=["damoc26"])
        s["F1950-1980_vs_damoc26"] = pearson(sub["F_1950-1980"], sub["damoc26"])
        s["F1950-1980_vs_base_amoc26"] = pearson(
            sub["F_1950-1980"], sub["amoc26_1850-1900"]
        )
        frac26 = sub["damoc26"] / sub["amoc26_1850-1900"]
        s["F1950-1980_vs_fractional_damoc26"] = pearson(sub["F_1950-1980"], frac26)
        s["base_amoc26_vs_damoc26"] = pearson(sub["amoc26_1850-1900"], sub["damoc26"])
        s["partial_F_vs_damoc26_given_base"] = partial(
            sub["F_1950-1980"].values,
            sub["damoc26"].values,
            sub["amoc26_1850-1900"].values,
            names=list(sub.index),
        )
        s["partial_F_vs_fractional_damoc26_given_base"] = partial(
            sub["F_1950-1980"].values,
            frac26.values,
            sub["amoc26_1850-1900"].values,
            names=list(sub.index),
        )
        s["F1950-1980_vs_damoc26_leave_one_out_range"] = [
            float(
                min(
                    pearson(sub["F_1950-1980"].drop(m), sub["damoc26"].drop(m))["r"]
                    for m in sub.index
                )
            ),
            float(
                max(
                    pearson(sub["F_1950-1980"].drop(m), sub["damoc26"].drop(m))["r"]
                    for m in sub.index
                )
            ),
        ]
        s["F1950-1980_vs_base_amoc26_leave_one_out_range"] = [
            float(
                min(
                    pearson(
                        sub["F_1950-1980"].drop(m), sub["amoc26_1850-1900"].drop(m)
                    )["r"]
                    for m in sub.index
                )
            ),
            float(
                max(
                    pearson(
                        sub["F_1950-1980"].drop(m), sub["amoc26_1850-1900"].drop(m)
                    )["r"]
                    for m in sub.index
                )
            ),
        ]
    if "dpsi34" in df:
        sub = df.dropna(subset=["dpsi34"])
        s["F1950-1980_vs_dpsi34"] = pearson(sub["F_1950-1980"], sub["dpsi34"])
        sub2 = df.dropna(subset=["dpsi34", "damoc26"])
        s["dpsi34_vs_damoc26"] = pearson(sub2["dpsi34"], sub2["damoc26"])
        s["dpsi34_median"] = float(sub["dpsi34"].median())
        s["damoc26_median"] = float(sub2["damoc26"].median())
        frac = sub["dpsi34"] / sub["psi34_1850-1900"]
        s["F1950-1980_vs_fractional_dpsi34"] = pearson(sub["F_1950-1980"], frac)
        s["F1950-1980_vs_base_psi34"] = pearson(
            sub["F_1950-1980"], sub["psi34_1850-1900"]
        )
        s["base_psi34_vs_dpsi34"] = pearson(sub["psi34_1850-1900"], sub["dpsi34"])
        s["partial_F_vs_dpsi34_given_base_psi34"] = partial(
            sub["F_1950-1980"].values,
            sub["dpsi34"].values,
            sub["psi34_1850-1900"].values,
            names=list(sub.index),
        )
        s["partial_F_vs_fractional_dpsi34_given_base_psi34"] = partial(
            sub["F_1950-1980"].values, frac.values, sub["psi34_1850-1900"].values
        )
    if within:
        s["within_model_detrended_corr_psi34_amoc26"] = {
            "n_models": len(within),
            "median": float(np.median(within)),
            "min": float(np.min(within)),
            "max": float(np.max(within)),
        }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(s, indent=1, default=float))
    if errors:
        print("errors:", errors)


if __name__ == "__main__":
    main()
