#!/usr/bin/env python3
"""GLORYS12 upper-cell overturning strength at the 34.5 S F_ovS section.

The section (latitude row, Atlantic longitude bounds, cell metrics) is the one
used by ``compute_glorys12_fovs.py``, so the resulting Psi pairs exactly with
``glorys12_f_ovs.nc`` and the two can be combined in the F_ovS variation
identity without any grid mismatch.

Psi is the maximum of the depth-space overturning streamfunction, cumulated
from the surface downward, matching ``compute_oras5_moc.py``.

Writes ``data/results/glorys12_moc_34S.nc``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ardp.constants import SAMBA_LAT
from scripts.compute_glorys12_fovs import get_grid_metrics


def moc_one_year(
    path: Path,
    j_idx: int,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    atlantic_mask: np.ndarray,
) -> tuple[list[np.datetime64], list[float]]:
    """Monthly upper-cell Psi [Sv] for one yearly GLORYS12 file."""
    times: list[np.datetime64] = []
    values: list[float] = []
    with xr.open_dataset(path) as ds:
        for t in range(ds.sizes["time"]):
            v = ds["vo"].isel(time=t, latitude=j_idx).values[:, atlantic_mask]
            # Zonal transport per depth level [m^3/s], then cumulate downward.
            v_int = np.nansum(np.nan_to_num(v) * e1t_atl[None, :], axis=1) * e3t
            psi = np.cumsum(v_int) / 1e6
            times.append(ds["time"].values[t])
            values.append(float(np.nanmax(psi)))
    return times, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/glorys12")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--lat", type=float, default=SAMBA_LAT)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("glorys12_*.nc"))
    if not files:
        raise FileNotFoundError(f"No GLORYS12 files in {data_dir}")

    j_idx, e1t_atl, e3t, atlantic_mask, actual_lat = get_grid_metrics(
        data_dir, target_lat=args.lat
    )
    print(f"{len(files)} yearly files")
    print(f"  section j={j_idx}, lat={actual_lat:.4f} (target {args.lat})")
    print(f"  Atlantic x-points: {int(atlantic_mask.sum())}")

    times: list[np.datetime64] = []
    values: list[float] = []
    for path in files:
        t, v = moc_one_year(path, j_idx, e1t_atl, e3t, atlantic_mask)
        times.extend(t)
        values.extend(v)
        print(f"  {path.name}: {len(v)} months, mean {np.mean(v):.2f} Sv", flush=True)

    order = np.argsort(pd.to_datetime(times))
    ds_out = xr.Dataset(
        {
            "moc_upper": (
                "time",
                np.asarray(values)[order],
                {
                    "units": "Sv",
                    "long_name": f"Upper-cell MOC transport at {actual_lat:.2f}",
                    "section_latitude": actual_lat,
                    "method": "max of depth-space overturning streamfunction",
                    "source": "GLORYS12V1 meridional velocity",
                },
            )
        },
        coords={"time": np.asarray(times)[order]},
    )
    out = Path(args.results_dir) / "glorys12_moc_34S.nc"
    ds_out.to_netcdf(out)
    print(f"wrote {out}: {ds_out.sizes['time']} months")


if __name__ == "__main__":
    main()
