"""Region mask factory and Atlantic basin masking."""

from __future__ import annotations

import numpy as np
import xarray as xr


def create_region_mask(
    lon: xr.DataArray,
    lat: xr.DataArray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.DataArray:
    """Create a boolean mask for a rectangular lon/lat region.

    Parameters
    ----------
    lon, lat : xr.DataArray
        Longitude and latitude arrays (can be 1D or 2D).
    lon_min, lon_max : float
        Longitude bounds [degrees_east].
    lat_min, lat_max : float
        Latitude bounds [degrees_north].

    Returns
    -------
    xr.DataArray
        Boolean mask (True = inside region).
    """
    mask: xr.DataArray = (
        (lon >= lon_min)
        & (lon <= lon_max)
        & (lat >= lat_min)
        & (lat <= lat_max)
    )
    return mask


def atlantic_basin_mask(
    lon: xr.DataArray,
    lat: xr.DataArray,
) -> xr.DataArray:
    """Create an approximate Atlantic basin mask.

    This is a simplified mask using longitude bounds that vary with latitude.
    For production use with real data, a proper basin mask from the mesh file
    should be preferred.

    Parameters
    ----------
    lon, lat : xr.DataArray
        2D longitude and latitude arrays.

    Returns
    -------
    xr.DataArray
        Boolean mask (True = Atlantic Ocean).
    """
    # Simplified Atlantic boundaries
    # South Atlantic: roughly -70 to 20 E
    # North Atlantic: roughly -80 to 0 E (narrowing northward)
    in_south = (lat >= -60) & (lat < 0) & (lon >= -70) & (lon <= 20)
    in_equatorial = (lat >= 0) & (lat < 10) & (lon >= -80) & (lon <= 0)
    in_north = (lat >= 10) & (lat <= 70) & (lon >= -80) & (lon <= 0)

    mask: xr.DataArray = in_south | in_equatorial | in_north
    return mask


def extract_section_at_latitude(
    ds: xr.Dataset,
    target_lat: float,
    lat_var: str = "nav_lat",
    y_dim: str = "y",
) -> xr.Dataset:
    """Extract a zonal section at the nearest j-index to target latitude.

    For NEMO quasi-regular grids (ORCA), latitude variation across x
    at a given j-index is typically < 0.1 degrees, so nearest-j is appropriate.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 2D lat field.
    target_lat : float
        Target latitude [degrees_north].
    lat_var : str
        Name of the 2D latitude variable.
    y_dim : str
        Name of the y dimension.

    Returns
    -------
    xr.Dataset
        Dataset sliced at the nearest j-index.
    """
    lat = ds[lat_var]

    if lat.ndim == 2:
        # Take the mean latitude at each j-index
        lat_1d = lat.mean(dim=[d for d in lat.dims if d != y_dim])
    else:
        lat_1d = lat

    j_idx = int(np.abs(lat_1d - target_lat).argmin().values)
    return ds.isel({y_dim: j_idx})
