"""North Atlantic Warming Hole (NAWH) index."""

from __future__ import annotations

import xarray as xr

from ardp.constants import NORTH_ATLANTIC_WARMING_HOLE
from ardp.spatial.regions import create_region_mask


def compute_nawh_index(
    sst: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    e1: xr.DataArray,
    e2: xr.DataArray,
    nawh_bounds: tuple[float, float, float, float] = NORTH_ATLANTIC_WARMING_HOLE,
) -> xr.DataArray:
    """Compute the North Atlantic Warming Hole index.

    NAWH = area-mean SST anomaly over the NAWH region minus
           global-mean SST anomaly. A negative value indicates
           relative cooling (the "warming hole").

    Parameters
    ----------
    sst : xr.DataArray
        Sea surface temperature [degC], dims (time, y, x).
    lon, lat : xr.DataArray
        2D longitude and latitude arrays.
    e1, e2 : xr.DataArray
        Grid spacings [m] for area weighting.
    nawh_bounds : tuple
        (lon_min, lon_max, lat_min, lat_max) for the NAWH region.

    Returns
    -------
    xr.DataArray
        NAWH index [degC]. Negative = relative cooling.
    """
    area = e1 * e2

    # Global mean SST
    global_mean = (sst * area).sum(dim=["y", "x"]) / area.sum(dim=["y", "x"])

    # Regional mean SST over NAWH region
    mask_nawh = create_region_mask(lon, lat, *nawh_bounds)
    nawh_mean = (sst * area).where(mask_nawh).sum(dim=["y", "x"]) / area.where(mask_nawh).sum(dim=["y", "x"])

    # NAWH index: remove global trend
    nawh_idx = nawh_mean - global_mean
    nawh_idx.name = "NAWH_index"
    nawh_idx.attrs["units"] = "degC"
    nawh_idx.attrs["long_name"] = "North Atlantic Warming Hole index"
    return nawh_idx
