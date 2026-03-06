"""Gulf Stream destabilization point tracking from SSH gradients."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.constants import GULF_STREAM_REGION
from ardp.spatial.regions import create_region_mask


def compute_ssh_gradient(
    ssh: xr.DataArray,
    e1: xr.DataArray,
    e2: xr.DataArray,
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.DataArray:
    """Compute the magnitude of the SSH gradient.

    |grad(SSH)| = sqrt((dSSH/dx)^2 + (dSSH/dy)^2)

    Uses centered differences with grid metrics.

    Parameters
    ----------
    ssh : xr.DataArray
        Sea surface height [m].
    e1, e2 : xr.DataArray
        Zonal and meridional grid spacings [m].
    x_dim, y_dim : str
        Dimension names.

    Returns
    -------
    xr.DataArray
        SSH gradient magnitude [m/m].
    """
    # Centered differences (using xarray's diff with shift)
    dssh_dx = ssh.diff(dim=x_dim) / e1.isel({x_dim: slice(1, None)})
    dssh_dy = ssh.diff(dim=y_dim) / e2.isel({y_dim: slice(1, None)})

    # Align to common grid (interior points)
    dssh_dx = dssh_dx.isel({y_dim: slice(1, None)})
    dssh_dy = dssh_dy.isel({x_dim: slice(1, None)})

    grad_mag: xr.DataArray = np.sqrt(dssh_dx**2 + dssh_dy**2)
    grad_mag.name = "ssh_gradient"
    grad_mag.attrs["units"] = "m/m"
    return grad_mag


def find_jet_axis(
    ssh_gradient: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    region_bounds: tuple[float, float, float, float] = GULF_STREAM_REGION,
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.Dataset:
    """Identify the Gulf Stream jet axis as the max SSH gradient at each longitude.

    Parameters
    ----------
    ssh_gradient : xr.DataArray
        SSH gradient magnitude.
    lon, lat : xr.DataArray
        2D coordinate arrays matching ssh_gradient's spatial dims.
    region_bounds : tuple
        Region to search within.
    x_dim, y_dim : str
        Dimension names.

    Returns
    -------
    xr.Dataset
        Variables 'jet_lat' and 'jet_grad' along the x dimension.
    """
    mask = create_region_mask(lon, lat, *region_bounds)
    grad_masked = ssh_gradient.where(mask, drop=False)

    # For each x position, find the y index of maximum gradient
    # .compute() needed when working with dask-backed arrays
    if hasattr(grad_masked, "chunks") and grad_masked.chunks:
        grad_masked = grad_masked.compute()
    # Fill NaN with 0 so argmax doesn't fail on all-NaN slices
    grad_filled = grad_masked.fillna(0.0)
    j_max = grad_filled.argmax(dim=y_dim)
    jet_lat = lat.isel({y_dim: j_max})
    jet_grad = grad_masked.isel({y_dim: j_max})

    return xr.Dataset({"jet_lat": jet_lat, "jet_grad": jet_grad})


def find_destabilization_point(
    jet_grad: xr.DataArray,
    jet_lon: xr.DataArray,
    threshold_fraction: float = 0.5,
    x_dim: str = "x",
) -> xr.DataArray:
    """Detect the destabilization point where the jet gradient drops below threshold.

    The destabilization point is the easternmost location along the jet axis
    where the gradient exceeds a fraction of the maximum gradient. Beyond this
    point, the Gulf Stream loses coherence.

    Parameters
    ----------
    jet_grad : xr.DataArray
        Gradient magnitude along the jet axis.
    jet_lon : xr.DataArray
        Longitude along the jet axis.
    threshold_fraction : float
        Fraction of peak gradient used as threshold.
    x_dim : str
        Dimension name for the along-jet axis.

    Returns
    -------
    xr.DataArray
        Longitude of the destabilization point.
    """
    peak = jet_grad.max(dim=x_dim)
    threshold = peak * threshold_fraction

    # Find where gradient drops below threshold (scanning eastward)
    above = jet_grad >= threshold

    # Last x-index where gradient is above threshold
    # Use cumulative sum trick: last True = max index where cumsum increases
    cumsum = above.astype(int).cumsum(dim=x_dim)
    last_above_idx = cumsum.argmax(dim=x_dim)

    destab_lon = jet_lon.isel({x_dim: last_above_idx})
    destab_lon.name = "destabilization_lon"
    destab_lon.attrs["units"] = "degrees_east"
    return destab_lon


def track_destabilization_timeseries(
    ssh: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    e1: xr.DataArray,
    e2: xr.DataArray,
    region_bounds: tuple[float, float, float, float] = GULF_STREAM_REGION,
    threshold_fraction: float = 0.5,
    time_dim: str = "time",
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.DataArray:
    """Track Gulf Stream destabilization point over time.

    Parameters
    ----------
    ssh : xr.DataArray
        SSH field (time, y, x).
    lon, lat, e1, e2 : xr.DataArray
        Grid coordinates and metrics.
    region_bounds : tuple
        Search region.
    threshold_fraction : float
        Gradient threshold for destabilization detection.
    time_dim, x_dim, y_dim : str
        Dimension names.

    Returns
    -------
    xr.DataArray
        Destabilization longitude time series.
    """
    grad = compute_ssh_gradient(ssh, e1, e2, x_dim=x_dim, y_dim=y_dim)

    # Trim lon/lat to match gradient's interior grid
    lon_int = lon.isel({x_dim: slice(1, None), y_dim: slice(1, None)})
    lat_int = lat.isel({x_dim: slice(1, None), y_dim: slice(1, None)})

    jet = find_jet_axis(grad, lon_int, lat_int, region_bounds, x_dim, y_dim)

    destab = find_destabilization_point(
        jet["jet_grad"], lon_int.isel({y_dim: 0}), threshold_fraction, x_dim
    )
    return destab
