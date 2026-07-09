#!/usr/bin/env python3
"""Annual SST anomaly time series inside the Caesar 2018 Cold Blob box,
plus the global-mean SST anomaly companion series for reference.  Source:
HadISST monthly SST (Rayner et al. 2003).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
HADISST = Path(
    "/home/bijanf/Documents/NEW_Theory/data/external/hadisst/HadISST_sst.nc"
)
OUT = REPO / "data" / "results" / "cold_blob_timeseries_hadisst.nc"

# Tightened Cold Blob core box (matches the ORAS5 panel-d/e box so the two
# products can be plotted on the same axis)
LON_W, LON_E = -45.0, -25.0
LAT_S, LAT_N = 50.0, 56.0

CLIM_Y0, CLIM_Y1 = 1958, 1988  # baseline period (matches ORAS5 script)


def main() -> int:
    print(f"Reading {HADISST}")
    ds = xr.open_dataset(HADISST)
    sst = ds["sst"]
    sst = sst.where(sst > -100)  # drop sentinel for land/ice

    # Annual means globally (so we can extract the global-mean companion line)
    ann = sst.groupby("time.year").mean("time", skipna=True)

    # Caesar box mean (HadISST has decreasing latitude)
    caesar = ann.sel(latitude=slice(LAT_N, LAT_S),
                     longitude=slice(LON_W, LON_E))
    weights = np.cos(np.deg2rad(caesar["latitude"]))
    caesar_mean = caesar.weighted(weights).mean(("latitude", "longitude"))

    # Global mean
    weights_g = np.cos(np.deg2rad(ann["latitude"]))
    global_mean = ann.weighted(weights_g).mean(("latitude", "longitude"))

    # Climatology baseline 1900-1930
    clim_mask = (caesar_mean["year"] >= CLIM_Y0) & (caesar_mean["year"] <= CLIM_Y1)
    caesar_clim = float(caesar_mean.where(clim_mask).mean("year"))
    global_clim = float(global_mean.where(clim_mask).mean("year"))

    caesar_anom = caesar_mean - caesar_clim
    global_anom = global_mean - global_clim

    out_ds = xr.Dataset(
        data_vars={
            "caesar_box_anomaly": (("year",), caesar_anom.values,
                                    {"units": "degC",
                                     "long_name": f"Cold Blob core box SST anomaly vs {CLIM_Y0}-{CLIM_Y1}",
                                     "box_lat": f"{LAT_S}-{LAT_N}",
                                     "box_lon": f"{LON_W}-{LON_E}"}),
            "global_anomaly": (("year",), global_anom.values,
                               {"units": "degC",
                                "long_name": f"Global-mean SST anomaly vs {CLIM_Y0}-{CLIM_Y1}"}),
            "caesar_box_sst": (("year",), caesar_mean.values,
                                {"units": "degC",
                                 "long_name": "Cold Blob core box absolute SST (annual mean)"}),
            "global_sst": (("year",), global_mean.values,
                            {"units": "degC",
                             "long_name": "Global absolute SST (annual mean)"}),
        },
        coords={"year": caesar_mean["year"].values},
        attrs={
            "source": str(HADISST),
            "data_product": "HadISST (Rayner et al. 2003)",
            "climatology_period": f"{CLIM_Y0}-{CLIM_Y1}",
        },
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(OUT)
    # Print useful headline numbers for caption / abstract
    modern_mask = ((caesar_mean["year"] >= 2005)
                   & (caesar_mean["year"] <= 2024))
    caesar_modern = float(caesar_anom.where(modern_mask).mean("year"))
    global_modern = float(global_anom.where(modern_mask).mean("year"))
    print(f"Wrote {OUT}")
    print(f"  Caesar box 2005-2024 anomaly vs 1900-1930: {caesar_modern:+.3f} degC")
    print(f"  Global    2005-2024 anomaly vs 1900-1930: {global_modern:+.3f} degC")
    print(f"  Differential (Caesar - global)           : {caesar_modern - global_modern:+.3f} degC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
