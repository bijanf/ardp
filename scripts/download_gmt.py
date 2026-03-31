#!/usr/bin/env python3
"""Download NASA GISTEMP v4 global mean temperature anomaly and save as NetCDF.

Source: NASA GISS Surface Temperature Analysis (GISTEMP v4)
        Land-Ocean Temperature Index, global mean, annual.
        Baseline: 1951-1980 average.
        Public domain data.

Output: data/external/gmt_gistemp.nc
"""

from __future__ import annotations

import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
OUTPUT = Path("data/external/gmt_gistemp.nc")


def main() -> None:
    print(f"Downloading GISTEMP v4 from {URL} ...")
    response = urllib.request.urlopen(URL)
    text = response.read().decode("utf-8")

    # Parse CSV — skip the first row (header), use second row as column names
    df = pd.read_csv(StringIO(text), skiprows=1, na_values="***")

    # 'J-D' column is the annual (Jan-Dec) mean anomaly
    years = df["Year"].values.astype(int)
    annual_mean = pd.to_numeric(df["J-D"], errors="coerce").values

    # Drop rows with missing annual mean
    valid = np.isfinite(annual_mean)
    years = years[valid]
    annual_mean = annual_mean[valid]

    # Create mid-year timestamps
    times = pd.to_datetime([f"{y}-07-01" for y in years])

    da = xr.DataArray(
        annual_mean,
        dims=["time"],
        coords={"time": times},
        name="gmt_anomaly",
        attrs={
            "long_name": "Global Mean Temperature Anomaly",
            "units": "degC",
            "baseline": "1951-1980",
            "source": "NASA GISTEMP v4 (GLB.Ts+dSST)",
        },
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(OUTPUT)
    print(f"Saved {len(da)} years ({years[0]}-{years[-1]}) to {OUTPUT}")


if __name__ == "__main__":
    main()
