"""Seawater density computations wrapping GSW via xr.apply_ufunc."""

from __future__ import annotations

import gsw
import numpy as np
import xarray as xr


def potential_density(
    salinity: xr.DataArray,
    temperature: xr.DataArray,
    pressure: xr.DataArray | None = None,
    p_ref: float = 0.0,
) -> xr.DataArray:
    """Compute potential density anomaly (sigma-0 by default).

    Parameters
    ----------
    salinity : xr.DataArray
        Absolute salinity [g/kg] or practical salinity [PSU].
    temperature : xr.DataArray
        Conservative temperature [degC] or potential temperature.
    pressure : xr.DataArray or None
        Sea pressure [dbar]. If None, uses depth coordinate converted to pressure.
    p_ref : float
        Reference pressure [dbar] for potential density (0 = sigma-0).

    Returns
    -------
    xr.DataArray
        Potential density anomaly [kg/m^3] (i.e., rho - 1000).
    """
    result: xr.DataArray = xr.apply_ufunc(
        gsw.sigma0,
        salinity,
        temperature,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    result.name = "sigma0"
    result.attrs["units"] = "kg/m^3"
    result.attrs["long_name"] = "Potential density anomaly (sigma-0)"
    return result


def in_situ_density(
    salinity: xr.DataArray,
    temperature: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Compute in-situ density using GSW.

    Parameters
    ----------
    salinity : xr.DataArray
        Absolute salinity [g/kg].
    temperature : xr.DataArray
        Conservative temperature [degC].
    pressure : xr.DataArray
        Sea pressure [dbar].

    Returns
    -------
    xr.DataArray
        In-situ density [kg/m^3].
    """
    result: xr.DataArray = xr.apply_ufunc(
        gsw.rho,
        salinity,
        temperature,
        pressure,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    result.name = "rho"
    result.attrs["units"] = "kg/m^3"
    result.attrs["long_name"] = "In-situ density"
    return result


def depth_to_pressure(
    depth: xr.DataArray,
    latitude: xr.DataArray,
) -> xr.DataArray:
    """Convert depth [m] to sea pressure [dbar] using GSW.

    Parameters
    ----------
    depth : xr.DataArray
        Depth below sea surface [m, positive down].
    latitude : xr.DataArray
        Latitude [degrees_north].

    Returns
    -------
    xr.DataArray
        Sea pressure [dbar].
    """
    result: xr.DataArray = xr.apply_ufunc(
        gsw.p_from_z,
        -depth,  # gsw expects negative z (height)
        latitude,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    result.name = "pressure"
    result.attrs["units"] = "dbar"
    return result
