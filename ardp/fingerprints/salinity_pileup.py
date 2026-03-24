"""Salinity pile-up fingerprint: SSS(STSA) - SSS(STSIP)."""

from __future__ import annotations

import xarray as xr

from ardp.constants import SUBTROPICAL_SOUTH_ATLANTIC, SUBTROPICAL_SOUTH_INDOPACIFIC
from ardp.spatial.regions import create_region_mask


def compute_salinity_pileup(
    sss: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    e1: xr.DataArray,
    e2: xr.DataArray,
    stsa_bounds: tuple[float, float, float, float] = SUBTROPICAL_SOUTH_ATLANTIC,
    stsip_bounds: tuple[float, float, float, float] = SUBTROPICAL_SOUTH_INDOPACIFIC,
) -> xr.DataArray:
    r"""Compute the salinity pile-up index.

    \Delta S = \overline{SSS}_{STSA} - \overline{SSS}_{STSIP}

    where the overbar is an area-weighted spatial mean.

    Parameters
    ----------
    sss : xr.DataArray
        Sea surface salinity [PSU], dims (time, y, x) or (y, x).
    lon, lat : xr.DataArray
        2D longitude and latitude arrays.
    e1, e2 : xr.DataArray
        Grid spacings [m] for area weighting.
    stsa_bounds, stsip_bounds : tuple
        (lon_min, lon_max, lat_min, lat_max) for each region.

    Returns
    -------
    xr.DataArray
        Salinity pile-up index [PSU].
    """
    area = e1 * e2

    # Subtropical South Atlantic
    mask_stsa = create_region_mask(lon, lat, *stsa_bounds)
    valid_stsa = mask_stsa & sss.notnull()
    sss_stsa = (sss * area).where(valid_stsa).sum(dim=["y", "x"]) / area.where(valid_stsa).sum(dim=["y", "x"])

    # Subtropical South Indo-Pacific
    mask_stsip = create_region_mask(lon, lat, *stsip_bounds)
    valid_stsip = mask_stsip & sss.notnull()
    sss_stsip = (sss * area).where(valid_stsip).sum(dim=["y", "x"]) / area.where(valid_stsip).sum(dim=["y", "x"])

    delta_s = sss_stsa - sss_stsip
    delta_s.name = "salinity_pileup"
    delta_s.attrs["units"] = "PSU"
    delta_s.attrs["long_name"] = "Salinity pile-up index (STSA - STSIP)"
    return delta_s
