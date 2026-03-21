"""Freshwater transport decomposition: overturning (F_ov) and gyre (F_az)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.constants import S0


def freshwater_transport_overturning(
    v: xr.DataArray,
    salinity: xr.DataArray,
    e1: xr.DataArray,
    e3: xr.DataArray,
    mask: xr.DataArray | None = None,
    x_dim: str = "x",
    z_dim: str = "z",
    s0: float = S0,
) -> xr.DataArray:
    r"""Compute overturning freshwater transport F_ov.

    F_ov(y) = -(1/S_0) \int V_int(z) * (<S>(z) - S_0) dz

    where V_int = sum(v * e1, x) is the zonally INTEGRATED velocity [m²/s]
    and <S> is the zonally averaged salinity, following de Vries & Weber (2005).

    Parameters
    ----------
    v : xr.DataArray
        Meridional velocity [m/s], dims including (z, y, x).
    salinity : xr.DataArray
        Salinity [PSU], dims including (z, y, x).
    e1 : xr.DataArray
        Zonal grid spacing [m] at the section.
    e3 : xr.DataArray
        Vertical cell thickness [m].
    mask : xr.DataArray or None
        Ocean mask (1=ocean, 0=land).
    x_dim, z_dim : str
        Dimension names.
    s0 : float
        Reference salinity.

    Returns
    -------
    xr.DataArray
        F_ov in Sverdrups [Sv].
    """
    if mask is not None:
        v = v.where(mask == 1, 0.0)
        salinity = salinity.where(mask == 1, np.nan)

    # Zonally INTEGRATED velocity: V_int(z) = sum(v * e1, x) [m²/s]
    v_zonal_int = (v * e1).sum(dim=x_dim)

    # Zonally averaged salinity: <S>(z) = sum(S * e1, x) / sum(e1, x)
    s_zonal_mean = (salinity * e1).sum(dim=x_dim) / e1.sum(dim=x_dim)

    # Vertical integration: F_ov = -(1/S0) * integral(V_int * (<S> - S0) * dz)
    integrand = v_zonal_int * (s_zonal_mean - s0) * e3
    f_ov = -(1.0 / s0) * integrand.sum(dim=z_dim)

    # Convert m^3/s to Sv
    f_ov = f_ov / 1e6
    f_ov.name = "F_ov"
    f_ov.attrs["units"] = "Sv"
    f_ov.attrs["long_name"] = "Overturning freshwater transport"
    return f_ov


def freshwater_transport_gyre(
    v: xr.DataArray,
    salinity: xr.DataArray,
    e1: xr.DataArray,
    e3: xr.DataArray,
    mask: xr.DataArray | None = None,
    x_dim: str = "x",
    z_dim: str = "z",
    s0: float = S0,
) -> xr.DataArray:
    r"""Compute gyre (azonal) freshwater transport F_az.

    F_az(y) = -(1/S_0) \int\int v' * S' dx dz

    where v' = v - <v> and S' = S - <S> are deviations from zonal mean.

    Parameters
    ----------
    v, salinity, e1, e3, mask, x_dim, z_dim, s0
        Same as freshwater_transport_overturning.

    Returns
    -------
    xr.DataArray
        F_az in Sverdrups [Sv].
    """
    if mask is not None:
        v = v.where(mask == 1, 0.0)
        salinity = salinity.where(mask == 1, np.nan)

    # Zonal means
    v_zonal_mean = (v * e1).sum(dim=x_dim) / e1.sum(dim=x_dim)
    s_zonal_mean = (salinity * e1).sum(dim=x_dim) / e1.sum(dim=x_dim)

    # Deviations
    v_prime = v - v_zonal_mean
    s_prime = salinity - s_zonal_mean

    # Double integration: F_az = -(1/S0) * integral(v' * S' * e1 * dz, x, z)
    integrand = v_prime * s_prime * e1 * e3
    f_az = -(1.0 / s0) * integrand.sum(dim=[x_dim, z_dim])

    f_az = f_az / 1e6
    f_az.name = "F_az"
    f_az.attrs["units"] = "Sv"
    f_az.attrs["long_name"] = "Gyre freshwater transport"
    return f_az
