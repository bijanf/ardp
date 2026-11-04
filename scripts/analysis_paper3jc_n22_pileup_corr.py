#!/usr/bin/env python3
"""Correlations of the salinity pile-up index with F_ovS and with dS_limb.

For each pile-up index (the legacy series and the wrapped Indo-Pacific box),
each product (ORAS5, GLORYS12V1) and each window, this script reports the raw
and the linearly detrended Pearson correlation of annual means with

    F_ovS at 34.5 S   (data/results/<product>_f_ovs.nc)
    dS_limb           (northward minus southward limb salinity, section file)

The effective sample size and the p-value come from ``corr_test`` in
``paper3r1_stats`` (Bretherton et al., 1999, with the bias-corrected lag-1
coefficients of the paper's tests). A 95 per cent interval for r is given by
the Fisher transformation with the effective sample size, N_eff - 3 degrees of
freedom; it is undefined when N_eff sits at its floor of 3.

The legacy series (``salinity_pileup.nc``, ``salinity_pileup_glorys12.nc``)
are the values behind the Supplement's pile-up correlations. The wrapped series
(``pileup_wrapped_*.nc``) use the full Indo-Pacific box, from 20 E eastward to
70 W. A plain ``20 <= lon <= 290`` test on a grid running from -180 to 180 would
keep only 20 E to 180 E; ``box_check`` quantifies that difference for one month,
and both stored series are compared with the correlations reported here.

Writes ``PAPER_3_v2/analysis/jc_n22_pileup_corr.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from paper3r1_stats import corr_test  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_n22_pileup_corr.json"

INDEX_FILES = {
    "legacy": {
        "oras5": ("salinity_pileup.nc", "salinity_pileup"),
        "glorys12": ("salinity_pileup_glorys12.nc", "salinity_pileup"),
    },
    "wrapped": {
        "oras5": ("pileup_wrapped_oras5.nc", "pileup"),
        "glorys12": ("pileup_wrapped_glorys12.nc", "pileup"),
    },
}
FOVS_FILES = {
    "oras5": ("oras5_f_ovs.nc", "F_ovS"),
    "glorys12": ("glorys12_f_ovs.nc", "F_ovS"),
}
SECTION_FILES = {
    "oras5": "paper3r1_section_oras5.nc",
    "glorys12": "paper3r1_section_glorys12.nc",
}

# Windows (inclusive years) for each product. The GLORYS12V1 pile-up index
# ends in 2024, so 1993-2025 is not available for that product.
WINDOWS = {
    "oras5": [(1958, 2025), (1993, 2024), (1993, 2025)],
    "glorys12": [(1993, 2024)],
}

# Supplement values (Text S3 and the caption of Fig. S3) for comparison.
SUPPLEMENT_CLAIMS = {
    "oras5_detrended_F_ovS_1958_2025": {"r": -0.19, "p": 0.25},
    "glorys12_detrended_F_ovS_1993_2024": {"r": -0.10, "p": 0.82},
    "oras5_raw_F_ovS_1958_2025": {"r": -0.76},
    "glorys12_raw_F_ovS_1993_2024": {"r": -0.65},
}


def monthly(path: Path, var: str) -> pd.Series:
    """Monthly series indexed by calendar month (pandas Period)."""
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        values = np.asarray(ds[var].values, dtype=float)
    s = pd.Series(values, index=times.to_period("M"))
    return s[~s.index.duplicated()]


def dS_monthly(product: str) -> pd.Series:
    """Northward minus southward limb salinity (PSU), monthly."""
    path = RESULTS / SECTION_FILES[product]
    with xr.open_dataset(path) as ds:
        if "dS_limb" in ds.data_vars:
            values = np.asarray(ds["dS_limb"].values, dtype=float)
        else:
            values = np.asarray(ds["S_north"].values - ds["S_south"].values, dtype=float)
        times = pd.DatetimeIndex(ds["time"].values)
    s = pd.Series(values, index=times.to_period("M"))
    return s[~s.index.duplicated()]


def annual_complete(s: pd.Series) -> pd.Series:
    """Calendar-year means of years with all 12 months present."""
    g = s.groupby(s.index.year)
    mean = g.mean()
    count = g.count()
    return mean[count == 12]


def paired(series: dict, lo: int, hi: int) -> tuple[np.ndarray, dict]:
    """Years common to all named annual series within [lo, hi]."""
    common = None
    for s in series.values():
        yrs = set(s.index[(s.index >= lo) & (s.index <= hi)])
        common = yrs if common is None else common & yrs
    years = np.array(sorted(common), dtype=float)
    return years, {k: s.loc[years.astype(int)].values for k, s in series.items()}


def fisher_ci(r: float, n_eff: float) -> list | None:
    """95 per cent interval for r from Fisher's z with N_eff - 3 degrees of freedom."""
    if not np.isfinite(r) or n_eff <= 3.0 or abs(r) >= 1.0:
        return None
    z = np.arctanh(r)
    half = 1.959964 * 1.0 / np.sqrt(n_eff - 3.0)
    return [float(np.tanh(z - half)), float(np.tanh(z + half))]


def summarise(test: dict) -> dict:
    return {
        "r": test["r"],
        "n_eff": test["n_eff"],
        "p_neff": test["p"],
        "p_nominal_n": test["p_ols"],
        "phi_a": test["phi_a"],
        "phi_b": test["phi_b"],
        "ci95_fisher_neff": fisher_ci(test["r"], test["n_eff"]),
        "n_eff_at_floor": bool(test["n_eff"] <= 3.0 + 1e-9),
    }


def box_check() -> dict:
    """Indo-Pacific box for January 1993 (ORAS5), plain bounds against wrapped.

    The stored index files are compared with both box definitions for one
    month, so the note on the legacy box can be checked against the files.
    """
    f = REPO / "data" / "oras5" / "sosaline_control_monthly_highres_2D_199301_CONS_v0.1.nc"
    with xr.open_dataset(f) as ds:
        lon = np.asarray(ds["nav_lon"].values, dtype=float)
        lat = np.asarray(ds["nav_lat"].values, dtype=float)
        sss = np.asarray(ds["sosaline"].isel(time_counter=0).values, dtype=float)
    weights = np.cos(np.deg2rad(lat))
    band = (lat >= -35.0) & (lat <= -15.0)
    lon360 = np.mod(lon, 360.0)

    def wmean(mask: np.ndarray) -> float:
        valid = mask & np.isfinite(sss)
        return float(np.sum(sss[valid] * weights[valid]) / np.sum(weights[valid]))

    atl = band & ((lon360 >= 300.0) | (lon360 <= 20.0))
    plain = band & (lon >= 20.0) & (lon <= 290.0)
    wrapped = band & (lon360 >= 20.0) & (lon360 <= 290.0)

    stored = {}
    for name, fname, var in (
        ("legacy_file", "salinity_pileup.nc", "salinity_pileup"),
        ("wrapped_file", "pileup_wrapped_oras5.nc", "pileup"),
    ):
        with xr.open_dataset(RESULTS / fname) as ds:
            months = pd.DatetimeIndex(ds["time"].values).to_period("M")
            k = list(months).index(pd.Period("1993-01", freq="M"))
            stored[name] = float(ds[var].values[k])

    return {
        "month": "1993-01",
        "atlantic_mean_PSU": wmean(atl),
        "plain_bounds_box_index_PSU": wmean(atl) - wmean(plain),
        "wrapped_box_index_PSU": wmean(atl) - wmean(wrapped),
        "plain_bounds_box_points": int((plain & np.isfinite(sss)).sum()),
        "wrapped_box_points": int((wrapped & np.isfinite(sss)).sum()),
        "stored_legacy_file_PSU": stored["legacy_file"],
        "stored_wrapped_file_PSU": stored["wrapped_file"],
    }


def main() -> None:
    fovs_monthly = {p: monthly(RESULTS / f, v) for p, (f, v) in FOVS_FILES.items()}
    fovs_annual = {p: annual_complete(s) for p, s in fovs_monthly.items()}
    dS_annual = {p: annual_complete(dS_monthly(p)) for p in SECTION_FILES}

    index_annual: dict = {}
    for variant, files in INDEX_FILES.items():
        index_annual[variant] = {}
        for product, (fname, var) in files.items():
            s = monthly(RESULTS / fname, var)
            index_annual[variant][product] = annual_complete(s)

    rows = []
    for variant in INDEX_FILES:
        for product in ("oras5", "glorys12"):
            for lo, hi in WINDOWS[product]:
                series = {
                    "index": index_annual[variant][product],
                    "F_ovS": fovs_annual[product],
                    "dS": dS_annual[product],
                }
                years, vals = paired(series, lo, hi)
                if len(years) < 10:
                    rows.append({"index_variant": variant, "product": product,
                                 "window": [lo, hi], "note": "insufficient years"})
                    continue
                idx = vals["index"]
                row = {
                    "index_variant": variant,
                    "product": product,
                    "window": [lo, hi],
                    "years_used": [int(years[0]), int(years[-1])],
                    "n_years": int(len(years)),
                    "index_mean_PSU": float(idx.mean()),
                }
                for target, key in (("F_ovS", "F_ovS"), ("dS_limb", "dS")):
                    tgt = vals[key]
                    row[f"{target}_raw"] = summarise(corr_test(idx, tgt, detrend=False))
                    row[f"{target}_detrended"] = summarise(corr_test(idx, tgt, detrend=True))
                rows.append(row)

    # How much the index definition matters: legacy against wrapped, same years.
    definition = {}
    for product in ("oras5", "glorys12"):
        lo, hi = (1993, 2024)
        series = {
            "legacy": index_annual["legacy"][product],
            "wrapped": index_annual["wrapped"][product],
        }
        years, vals = paired(series, lo, hi)
        definition[product] = {
            "window": [lo, hi],
            "n_years": int(len(years)),
            "r_legacy_vs_wrapped_raw": summarise(
                corr_test(vals["legacy"], vals["wrapped"], detrend=False)),
            "r_legacy_vs_wrapped_detrended": summarise(
                corr_test(vals["legacy"], vals["wrapped"], detrend=True)),
            "legacy_mean_PSU": float(vals["legacy"].mean()),
            "wrapped_mean_PSU": float(vals["wrapped"].mean()),
        }

    out = {
        "analysis": "N22 pile-up index correlations with F_ovS and dS_limb",
        "method": "paper3r1_stats.corr_test (Bretherton N_eff, bias-corrected lag-1); "
                  "annual means of complete years; detrended = projection onto a linear trend",
        "inputs": {
            "index_legacy": {k: f"data/results/{v[0]}" for k, v in INDEX_FILES["legacy"].items()},
            "index_wrapped": {k: f"data/results/{v[0]}" for k, v in INDEX_FILES["wrapped"].items()},
            "F_ovS": {k: f"data/results/{v[0]}" for k, v in FOVS_FILES.items()},
            "dS_limb": {k: f"data/results/{v}" for k, v in SECTION_FILES.items()},
        },
        "units": {"index": "PSU", "F_ovS": "Sv", "dS_limb": "PSU", "r": "dimensionless"},
        "rows": rows,
        "index_definition_comparison": definition,
        "supplement_claims_for_comparison": SUPPLEMENT_CLAIMS,
        "box_check_oras5_1993_01": box_check(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    for row in rows:
        if "note" in row:
            print(row)
            continue
        print(f"{row['index_variant']:8s} {row['product']:9s} "
              f"{row['window'][0]}-{row['window'][1]} n={row['n_years']}")
        for tgt in ("F_ovS", "dS_limb"):
            for kind in ("raw", "detrended"):
                s = row[f"{tgt}_{kind}"]
                print(f"    {tgt:7s} {kind:9s} r={s['r']:+.3f} n_eff={s['n_eff']:.1f} "
                      f"p={s['p_neff']:.3f} (nominal-n p={s['p_nominal_n']:.3g}) "
                      f"CI={s['ci95_fisher_neff']}")
    for product, d in definition.items():
        print(f"legacy vs wrapped {product}: raw r={d['r_legacy_vs_wrapped_raw']['r']:+.3f} "
              f"detrended r={d['r_legacy_vs_wrapped_detrended']['r']:+.3f}")
    print("box check", out["box_check_oras5_1993_01"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
