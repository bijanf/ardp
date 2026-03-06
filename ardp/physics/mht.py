"""Meridional heat transport (MHT)."""

from __future__ import annotations

import xarray as xr

from ardp.constants import CP, RHO_0


def meridional_heat_transport(
    v: xr.DataArray,
    temperature: xr.DataArray,
    e1: xr.DataArray,
    e3: xr.DataArray,
    mask: xr.DataArray | None = None,
    x_dim: str = "x",
    z_dim: str = "z",
    rho0: float = RHO_0,
    cp: float = CP,
) -> xr.DataArray:
    r"""Compute meridional heat transport.

    MHT(y) = \rho_0 \cdot c_p \int\int v \cdot T \, dx \, dz

    Parameters
    ----------
    v : xr.DataArray
        Meridional velocity [m/s]. Should be collocated to T-points
        (or vice versa) before calling.
    temperature : xr.DataArray
        Temperature [degC].
    e1 : xr.DataArray
        Zonal grid spacing [m].
    e3 : xr.DataArray
        Vertical cell thickness [m].
    mask : xr.DataArray or None
        Ocean mask (1=ocean, 0=land).
    x_dim, z_dim : str
        Dimension names.
    rho0 : float
        Reference density [kg/m^3].
    cp : float
        Specific heat capacity [J/(kg·K)].

    Returns
    -------
    xr.DataArray
        MHT in Petawatts [PW].
    """
    if mask is not None:
        v = v.where(mask == 1, 0.0)
        temperature = temperature.where(mask == 1, 0.0)

    integrand = rho0 * cp * v * temperature * e1 * e3
    mht = integrand.sum(dim=[x_dim, z_dim])

    mht = mht / 1e15  # Convert W to PW
    mht.name = "MHT"
    mht.attrs["units"] = "PW"
    mht.attrs["long_name"] = "Meridional heat transport"
    return mht
