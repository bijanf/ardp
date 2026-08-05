#!/usr/bin/env python3
"""PAPER_3 round-2 WP11: salinity pile-up index trends for ORAS5 and GLORYS12.

`data/results/salinity_pileup.nc` carries 816 monthly values spanning
1958-01-16 to 2025-12-16 on mid-month timestamps, which is exactly the ORAS5
record and cadence; GLORYS12 only begins in 1993. It is therefore already the
ORAS5 index and is used as such rather than being recomputed. Nothing is
overwritten.

Trends are OLS with the Santer et al. (2000) N_eff-adjusted p-value, reusing
`ols_santer` from scripts/analysis_paper3_r2_gate_checks.py. Annual means are
the headline (they carry no seasonal cycle); monthly is reported alongside.

Output: revision/rev_papaer3_02/results/WP11_pileup_trends.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

from ardp.constants import (  # noqa: E402
    SUBTROPICAL_SOUTH_ATLANTIC,
    SUBTROPICAL_SOUTH_INDOPACIFIC,
)

RESULTS = REPO / "data" / "results"
OUT = REPO / "revision" / "rev_papaer3_02" / "results"

SOURCES = {
    "oras5": "salinity_pileup.nc",
    "glorys12": "salinity_pileup_glorys12.nc",
}


def annual(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    g = da.groupby("time.year").mean()
    return g["year"].values.astype(float), g.values.astype(float)


def window(
    years: np.ndarray, vals: np.ndarray, y0: int, y1: int
) -> tuple[np.ndarray, np.ndarray]:
    m = (years >= y0) & (years <= y1) & np.isfinite(vals)
    return years[m], vals[m]


def describe(da: xr.DataArray) -> dict:
    t = da["time"].values
    return {
        "n_months": int(len(t)),
        "first": str(t[0])[:10],
        "last": str(t[-1])[:10],
        "day_of_month_first": int(str(t[0])[8:10]),
        "mean_PSU": float(da.mean()),
        "std_PSU": float(da.std()),
    }


def region_coverage() -> dict:
    """How much of each pile-up region each product's grid actually contains."""
    out = {}

    ds = xr.open_dataset(RESULTS.parent / "glorys12" / "glorys12_1993.nc")
    lon2, lat2 = np.meshgrid(ds.longitude.values, ds.latitude.values)
    ds.close()
    out["glorys12"] = _cover(lon2, lat2)

    d2 = xr.open_dataset(
        RESULTS.parent
        / "oras5"
        / "sosaline_control_monthly_highres_2D_199301_CONS_v0.1.nc"
    )
    out["oras5"] = _cover(d2.nav_lon.values, d2.nav_lat.values)
    d2.close()
    return out


def _cover(lon: np.ndarray, lat: np.ndarray) -> dict:
    res = {"grid_lon_min": float(lon.min()), "grid_lon_max": float(lon.max())}
    for name, (lo0, lo1, la0, la1) in (
        ("STSA", SUBTROPICAL_SOUTH_ATLANTIC),
        ("STSIP", SUBTROPICAL_SOUTH_INDOPACIFIC),
    ):
        m = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
        res[name] = {
            "box_lon": [lo0, lo1],
            "n_pixels": int(m.sum()),
            "lon_span_covered": (
                [float(lon[m].min()), float(lon[m].max())] if m.any() else None
            ),
        }
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series, meta = {}, {}
    for prod, fname in SOURCES.items():
        da = xr.open_dataarray(RESULTS / fname)
        meta[prod] = {"file": f"data/results/{fname}", **describe(da)}
        series[prod] = da

    # Matched window: GLORYS12's index stops at 2024-12, so 1993-2025 is not a
    # like-for-like comparison. Both windows are reported.
    windows = {"1993-2025": (1993, 2025), "1993-2024": (1993, 2024)}

    trends: dict = {}
    for prod, da in series.items():
        yr_a, v_a = annual(da)
        tobj = da["time"].values.astype("datetime64[us]").astype("object")
        yr_m = np.array([t.year + (t.month - 1) / 12.0 for t in tobj])
        v_m = da.values.astype(float)

        trends[prod] = {}
        for wlabel, (y0, y1) in windows.items():
            ya, va = window(yr_a, v_a, y0, y1)
            ym, vm = window(yr_m, v_m, y0, y1 + 0.999)
            if len(ya) < 10:
                trends[prod][wlabel] = {"note": "insufficient years"}
                continue
            a = ols_santer(ya, va)
            m = ols_santer(ym, vm)
            trends[prod][wlabel] = {
                "years_used": [int(ya[0]), int(ya[-1])],
                "annual": {
                    "n_years": a["n_years"],
                    "trend_PSU_per_decade": a["slope"] * 10.0,
                    "p_ols": a["p_ols"],
                    "p_santer": a["p_santer"],
                    "lag1_autocorr": a["lag1_autocorr"],
                    "n_eff": a["n_eff"],
                },
                "monthly": {
                    "n_months": m["n_years"],
                    "trend_PSU_per_decade": m["slope"] * 10.0,
                    "p_ols": m["p_ols"],
                    "p_santer": m["p_santer"],
                    "lag1_autocorr": m["lag1_autocorr"],
                    "n_eff": m["n_eff"],
                },
            }
            aa = trends[prod][wlabel]["annual"]
            print(
                f"{prod:9s} {wlabel}: {aa['trend_PSU_per_decade']:+.4f} PSU/dec "
                f"(annual, n={aa['n_years']}, Santer p={aa['p_santer']:.4f}, "
                f"N_eff={aa['n_eff']:.1f})"
            )

    payload = {
        "note": (
            "salinity_pileup.nc is the ORAS5 index: 816 months, 1958-01-16 to "
            "2025-12-16, mid-month stamps, matching the ORAS5 record. It was "
            "not recomputed and not overwritten."
        ),
        "index_definition": (
            "area-weighted SSS(subtropical South Atlantic) minus "
            "SSS(subtropical South Indo-Pacific), "
            "ardp/fingerprints/salinity_pileup.py"
        ),
        "sources": meta,
        "region_coverage": region_coverage(),
        "trends": trends,
    }
    (OUT / "WP11_pileup_trends.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUT / 'WP11_pileup_trends.json'}")


if __name__ == "__main__":
    main()
