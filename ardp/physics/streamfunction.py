"""MOC streamfunction in depth-space and density-space."""

from __future__ import annotations

import numpy as np
import xarray as xr


def moc_depth_space(
    v: xr.DataArray,
    e1: xr.DataArray,
    e3: xr.DataArray,
    mask: xr.DataArray | None = None,
    x_dim: str = "x",
    z_dim: str = "z",
) -> xr.DataArray:
    r"""Compute MOC streamfunction in depth space.

    \psi(y, z) = \int_{bottom}^{z} \int v \, dx \, dz'

    Integrated bottom-up so that psi=0 at the bottom.

    Parameters
    ----------
    v : xr.DataArray
        Meridional velocity [m/s].
    e1 : xr.DataArray
        Zonal grid spacing [m].
    e3 : xr.DataArray
        Vertical cell thickness [m].
    mask : xr.DataArray or None
        Ocean mask (1=ocean, 0=land).
    x_dim, z_dim : str
        Dimension names.

    Returns
    -------
    xr.DataArray
        Streamfunction [Sv], dims (time, z, y) or subset.
    """
    if mask is not None:
        v = v.where(mask == 1, 0.0)

    # Zonal integration of volume transport
    v_transport = (v * e1 * e3).sum(dim=x_dim)

    # Cumulative sum from bottom (reverse z, cumsum, reverse back)
    psi = v_transport.isel({z_dim: slice(None, None, -1)}).cumsum(dim=z_dim)
    psi = psi.isel({z_dim: slice(None, None, -1)})

    psi = psi / 1e6  # Convert to Sv
    psi.name = "moc_psi"
    psi.attrs["units"] = "Sv"
    psi.attrs["long_name"] = "MOC streamfunction (depth space)"
    return psi


def moc_density_space(
    v: xr.DataArray,
    sigma0: xr.DataArray,
    e1: xr.DataArray,
    e3: xr.DataArray,
    sigma_bins: np.ndarray | None = None,
    mask: xr.DataArray | None = None,
    x_dim: str = "x",
    z_dim: str = "z",
) -> xr.DataArray:
    r"""Compute MOC streamfunction in density (sigma-0) space.

    Uses histogram binning of v*dx*dz by sigma-0 classes,
    then cumulative sum from light to dense.

    Parameters
    ----------
    v : xr.DataArray
        Meridional velocity [m/s].
    sigma0 : xr.DataArray
        Potential density anomaly [kg/m^3].
    e1 : xr.DataArray
        Zonal grid spacing [m].
    e3 : xr.DataArray
        Vertical cell thickness [m].
    sigma_bins : np.ndarray or None
        Bin edges for sigma-0. Default: 20 to 28.5 by 0.1.
    mask : xr.DataArray or None
        Ocean mask.
    x_dim, z_dim : str
        Dimension names.

    Returns
    -------
    xr.DataArray
        Streamfunction [Sv], dims (time, sigma_bin, y) or subset.
    """
    from ardp.constants import SIGMA0_BINS_MAX, SIGMA0_BINS_MIN, SIGMA0_BINS_STEP

    if sigma_bins is None:
        sigma_bins = np.arange(SIGMA0_BINS_MIN, SIGMA0_BINS_MAX + SIGMA0_BINS_STEP, SIGMA0_BINS_STEP)

    if mask is not None:
        v = v.where(mask == 1, 0.0)

    transport = v * e1 * e3  # volume transport per cell [m^3/s]

    # Bin transport by sigma-0 class using xarray groupby_bins
    # For dask compatibility, use a manual histogram approach
    bin_centers = (sigma_bins[:-1] + sigma_bins[1:]) / 2.0
    n_bins = len(bin_centers)

    # Flatten spatial dims and bin
    remaining_dims = [d for d in transport.dims if d not in [x_dim, z_dim]]

    binned_list = []
    for i in range(n_bins):
        lo, hi = sigma_bins[i], sigma_bins[i + 1]
        in_bin = (sigma0 >= lo) & (sigma0 < hi)
        bin_transport = transport.where(in_bin, 0.0).sum(dim=[x_dim, z_dim])
        binned_list.append(bin_transport)

    binned = xr.concat(binned_list, dim="sigma_bin")
    binned = binned.assign_coords(sigma_bin=bin_centers)

    # Cumulative sum from lightest (top) to densest (bottom)
    psi = binned.cumsum(dim="sigma_bin")
    psi = psi / 1e6

    psi.name = "moc_psi_sigma"
    psi.attrs["units"] = "Sv"
    psi.attrs["long_name"] = "MOC streamfunction (density space)"
    return psi
