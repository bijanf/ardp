#!/usr/bin/env python3
"""Interbasin salinity contrast index, recomputed with a wrapped Indo-Pacific box.

The index is the area-weighted surface salinity of the subtropical South
Atlantic minus that of the subtropical South Indo-Pacific, both over
15 S to 35 S:

    Atlantic      60 W to 20 E
    Indo-Pacific  20 E eastward to 70 W (that is, 20 E to 290 E)

The earlier version of this index applied the Indo-Pacific bounds as a plain
``20 <= lon <= 290`` test on grids that run from -180 to 180, which silently
truncated the box at the date line and dropped the entire Pacific sector. This
script wraps the box properly and recomputes both products from the surface
salinity fields.

It then relates the index to F_ovS at 34.5 S both raw and after removing a
linear trend from each series, because two series that both trend will
correlate whether or not they share any physics.

Writes ``data/results/pileup_wrapped_<product>.nc`` and
``PAPER_3_v2/analysis/pileup.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"

LAT_BAND = (-35.0, -15.0)
ATLANTIC_LON = (-60.0, 20.0)
# Eastward from 20 E all the way to 70 W, i.e. 20 E to 290 E.
INDOPACIFIC_LON_E = (20.0, 290.0)

WINDOW = (1993, 2024)


def to_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon, 360.0)


def region_means(
    sss: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> tuple[float, float]:
    """Area-weighted (cos-latitude) means of the two boxes for one field.

    ``sss``, ``lon`` and ``lat`` are 2D arrays of the same shape.
    """
    lon360 = to_360(lon)
    in_band = (lat >= LAT_BAND[0]) & (lat <= LAT_BAND[1])
    # The Atlantic box straddles the prime meridian: 300 E to 360 E, then 0 to
    # 20 E. The Indo-Pacific box runs eastward from 20 E to 290 E.
    atl = in_band & ((lon360 >= to_360(ATLANTIC_LON[0])) | (lon360 <= ATLANTIC_LON[1]))
    ind = in_band & (lon360 >= INDOPACIFIC_LON_E[0]) & (lon360 <= INDOPACIFIC_LON_E[1])

    weights = np.cos(np.deg2rad(lat))
    out = []
    for mask in (atl, ind):
        valid = mask & np.isfinite(sss)
        w = weights[valid]
        out.append(float(np.sum(sss[valid] * w) / np.sum(w)))
    return out[0], out[1]


def oras5_index() -> xr.DataArray:
    files = sorted((REPO / "data" / "oras5").glob("sosaline_*_2D_*.nc"))
    if not files:
        raise FileNotFoundError("no ORAS5 2D salinity files")
    with xr.open_dataset(files[0]) as ds0:
        lon = ds0["nav_lon"].values
        lat = ds0["nav_lat"].values
    # The tripolar fold sits far north of this band, so a plain lat/lon box is
    # unambiguous here.
    rows = np.any((lat >= LAT_BAND[0] - 1.0) & (lat <= LAT_BAND[1] + 1.0), axis=1)
    j0, j1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    lon_s, lat_s = lon[j0:j1], lat[j0:j1]

    times: list[np.datetime64] = []
    values: list[float] = []
    for path in files:
        with xr.open_dataset(path) as ds:
            field = ds["sosaline"].isel(time_counter=0).values[j0:j1]
            times.append(ds["time_counter"].values[0])
            a, i = region_means(field, lon_s, lat_s)
            values.append(a - i)
    order = np.argsort(np.asarray(times))
    return xr.DataArray(
        np.asarray(values)[order],
        dims="time",
        coords={"time": np.asarray(times)[order]},
        name="pileup",
        attrs={"units": "PSU", "product": "ORAS5"},
    )


def glorys12_index() -> xr.DataArray:
    path = REPO / "data" / "glorys12_global_sss" / "glorys12_global_sss_1993_2024.nc"
    with xr.open_dataset(path) as ds:
        lon1d = ds["longitude"].values
        lat1d = ds["latitude"].values
        lon, lat = np.meshgrid(lon1d, lat1d)
        values = []
        times = ds["time"].values
        for t in range(ds.sizes["time"]):
            field = ds["so"].isel(time=t, depth=0).values
            a, i = region_means(field, lon, lat)
            values.append(a - i)
    return xr.DataArray(
        np.asarray(values),
        dims="time",
        coords={"time": times},
        name="pileup",
        attrs={"units": "PSU", "product": "GLORYS12V1"},
    )


def annual(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    grouped = da.groupby("time.year").mean()
    return grouped["year"].values.astype(int), grouped.values.astype(float)


def annual_fovs(product: str) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(RESULTS / f"{product}_f_ovs.nc") as ds:
        grouped = ds["F_ovS"].groupby("time.year").mean()
        return grouped["year"].values.astype(int), grouped.values.astype(float)


def detrend(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - np.polyval(np.polyfit(x, y, 1), x)


def analyse(product: str, index: xr.DataArray) -> dict:
    years, vals = annual(index)
    fy, fv = annual_fovs(product)
    lo = max(WINDOW[0], years.min(), fy.min())
    hi = min(WINDOW[1], years.max(), fy.max())
    m_i = (years >= lo) & (years <= hi)
    m_f = (fy >= lo) & (fy <= hi)
    yr = years[m_i].astype(float)
    idx = vals[m_i]
    fov = fv[m_f]
    assert len(idx) == len(fov)

    fit = ols_santer(yr, idx)
    r_raw = float(np.corrcoef(idx, fov)[0, 1])
    r_det = float(np.corrcoef(detrend(yr, idx), detrend(yr, fov))[0, 1])
    n = len(yr)
    # Two-sided p for a Pearson r under an effective sample size that accounts
    # for residual autocorrelation in the detrended pair.
    from scipy import stats

    def pearson_p(r: float, nn: int) -> float:
        if nn <= 3:
            return float("nan")
        t = r * np.sqrt((nn - 2) / max(1e-12, 1 - r**2))
        return float(2 * stats.t.sf(abs(t), nn - 2))

    return {
        "product": product,
        "window": [int(lo), int(hi)],
        "n_years": int(n),
        "index_mean_PSU": float(idx.mean()),
        "trend_PSU_per_decade": fit["slope"] * 10.0,
        "p_santer": fit["p_santer"],
        "significant": bool(fit["p_santer"] < 0.05),
        "n_eff": fit["n_eff"],
        "corr_with_fovs_raw": r_raw,
        "p_raw": pearson_p(r_raw, n),
        "corr_with_fovs_detrended": r_det,
        "p_detrended": pearson_p(r_det, n),
        "years": years[m_i].tolist(),
        "index": idx.tolist(),
        "fovs": fov.tolist(),
    }


def main() -> None:
    out: dict = {}
    for product, builder in (("oras5", oras5_index), ("glorys12", glorys12_index)):
        print(f"building {product} index ...", flush=True)
        index = builder()
        index.to_netcdf(RESULTS / f"pileup_wrapped_{product}.nc")
        out[product] = analyse(product, index)
        r = out[product]
        print(
            f"  {product}: {r['window'][0]}-{r['window'][1]} n={r['n_years']}  "
            f"mean {r['index_mean_PSU']:.3f} PSU  "
            f"trend {r['trend_PSU_per_decade']:+.4f} PSU/dec "
            f"(p={r['p_santer']:.4f}, N_eff={r['n_eff']:.1f})"
        )
        print(
            f"    corr with F_ovS: raw r={r['corr_with_fovs_raw']:+.3f} "
            f"(p={r['p_raw']:.4f})  detrended r="
            f"{r['corr_with_fovs_detrended']:+.3f} (p={r['p_detrended']:.4f})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "pileup.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
