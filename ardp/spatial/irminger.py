"""Irminger Sea diagnostics: upper-ocean density, MLD, tip jet index."""

from __future__ import annotations

import xarray as xr

from ardp.constants import IRMINGER_SEA
from ardp.spatial.regions import create_region_mask


def irminger_mask(
    lon: xr.DataArray,
    lat: xr.DataArray,
    bounds: tuple[float, float, float, float] = IRMINGER_SEA,
) -> xr.DataArray:
    """Create an Irminger Sea region mask."""
    return create_region_mask(lon, lat, *bounds)


def upper_ocean_density(
    sigma0: xr.DataArray,
    depth: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    e1: xr.DataArray,
    e2: xr.DataArray,
    e3: xr.DataArray,
    max_depth: float = 1000.0,
    bounds: tuple[float, float, float, float] = IRMINGER_SEA,
    z_dim: str = "z",
) -> xr.DataArray:
    """Compute area-weighted upper-ocean mean density in the Irminger Sea.

    This is a key predictor of AMOC variability (R=0.73 with AMOC at 45N).

    Parameters
    ----------
    sigma0 : xr.DataArray
        Potential density anomaly [kg/m^3].
    depth : xr.DataArray
        Depth coordinate [m].
    lon, lat : xr.DataArray
        2D coordinate arrays.
    e1, e2, e3 : xr.DataArray
        Grid metrics [m].
    max_depth : float
        Maximum depth for integration [m].
    bounds : tuple
        Irminger Sea region bounds.
    z_dim : str
        Depth dimension name.

    Returns
    -------
    xr.DataArray
        Volume-weighted mean sigma-0 in upper Irminger Sea [kg/m^3].
    """
    mask_region = irminger_mask(lon, lat, bounds)
    mask_depth = depth <= max_depth

    area = e1 * e2
    volume = area * e3

    weighted = (sigma0 * volume).where(mask_region & mask_depth)
    total_vol = volume.where(mask_region & mask_depth)

    mean_density = weighted.sum(dim=[z_dim, "y", "x"]) / total_vol.sum(dim=[z_dim, "y", "x"])
    mean_density.name = "irminger_upper_density"
    mean_density.attrs["units"] = "kg/m^3"
    mean_density.attrs["long_name"] = "Irminger Sea upper-ocean mean density (sigma-0)"
    return mean_density


def mixed_layer_depth(
    sigma0: xr.DataArray,
    depth: xr.DataArray,
    density_threshold: float = 0.03,
    z_dim: str = "z",
) -> xr.DataArray:
    """Compute mixed layer depth using a density criterion.

    MLD is the depth where sigma-0 exceeds the surface value by
    `density_threshold` kg/m^3.

    Parameters
    ----------
    sigma0 : xr.DataArray
        Potential density anomaly [kg/m^3], with depth dimension.
    depth : xr.DataArray
        Depth values [m].
    density_threshold : float
        Density increase threshold [kg/m^3].
    z_dim : str
        Depth dimension name.

    Returns
    -------
    xr.DataArray
        Mixed layer depth [m].
    """
    sigma_surface = sigma0.isel({z_dim: 0})
    exceeds = sigma0 > (sigma_surface + density_threshold)

    # First depth index where threshold is exceeded
    first_exceed = exceeds.argmax(dim=z_dim)

    # Where threshold is never exceeded, set to max depth
    never_exceeded = ~exceeds.any(dim=z_dim)

    mld = depth.isel({z_dim: first_exceed})
    mld = mld.where(~never_exceeded, depth.isel({z_dim: -1}))

    mld.name = "MLD"
    mld.attrs["units"] = "m"
    mld.attrs["long_name"] = "Mixed layer depth (density criterion)"
    return mld
