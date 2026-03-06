"""F_ovS: overturning freshwater transport at 34.5S — the critical AMOC fingerprint."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ardp.constants import FOVS_EXPECTED_TREND, FOVS_TREND_TOLERANCE, S0, SAMBA_LAT
from ardp.physics.freshwater import (
    freshwater_transport_gyre,
    freshwater_transport_overturning,
)
from ardp.spatial.regions import extract_section_at_latitude


def compute_f_ovs(
    ds: xr.Dataset,
    lat: float = SAMBA_LAT,
    v_var: str = "v_velocity",
    s_var: str = "salinity",
    e1_var: str = "e1t",
    e3_var: str = "e3t",
    mask_var: str | None = "vmask",
    lat_var: str = "nav_lat",
    y_dim: str = "y",
    x_dim: str = "x",
    z_dim: str = "z",
    s0: float = S0,
) -> xr.DataArray:
    """Compute F_ovS (overturning freshwater transport) at a given latitude.

    Parameters
    ----------
    ds : xr.Dataset
        Full 3D+time dataset with velocity, salinity, grid metrics, and mask.
    lat : float
        Target latitude for the section [degrees_north].
    v_var, s_var, e1_var, e3_var, mask_var : str
        Variable names in the dataset.
    lat_var : str
        Name of the 2D latitude field.
    y_dim, x_dim, z_dim : str
        Dimension names.
    s0 : float
        Reference salinity.

    Returns
    -------
    xr.DataArray
        F_ovS time series [Sv].
    """
    section = extract_section_at_latitude(ds, lat, lat_var=lat_var, y_dim=y_dim)

    mask = section[mask_var] if mask_var and mask_var in section else None

    f_ov = freshwater_transport_overturning(
        section[v_var],
        section[s_var],
        section[e1_var],
        section[e3_var],
        mask=mask,
        x_dim=x_dim,
        z_dim=z_dim,
        s0=s0,
    )
    return f_ov


def compute_f_azs(
    ds: xr.Dataset,
    lat: float = SAMBA_LAT,
    v_var: str = "v_velocity",
    s_var: str = "salinity",
    e1_var: str = "e1t",
    e3_var: str = "e3t",
    mask_var: str | None = "vmask",
    lat_var: str = "nav_lat",
    y_dim: str = "y",
    x_dim: str = "x",
    z_dim: str = "z",
    s0: float = S0,
) -> xr.DataArray:
    """Compute F_azS (gyre freshwater transport) at a given latitude.

    Same parameters as compute_f_ovs.
    """
    section = extract_section_at_latitude(ds, lat, lat_var=lat_var, y_dim=y_dim)
    mask = section[mask_var] if mask_var and mask_var in section else None

    f_az = freshwater_transport_gyre(
        section[v_var],
        section[s_var],
        section[e1_var],
        section[e3_var],
        mask=mask,
        x_dim=x_dim,
        z_dim=z_dim,
        s0=s0,
    )
    return f_az


def compute_f_ovs_timeseries(
    ds: xr.Dataset,
    time_dim: str = "time",
    **kwargs: object,
) -> xr.DataArray:
    """Compute F_ovS for each timestep.

    Parameters
    ----------
    ds : xr.Dataset
        Full dataset with time dimension.
    time_dim : str
        Name of the time dimension.
    **kwargs
        Passed to compute_f_ovs.

    Returns
    -------
    xr.DataArray
        F_ovS time series [Sv] with time coordinate.
    """
    f_ovs = compute_f_ovs(ds, **kwargs)  # type: ignore[arg-type]
    f_ovs.name = "F_ovS"
    return f_ovs


def validate_trend(
    f_ovs_ts: xr.DataArray,
    expected_trend: float = FOVS_EXPECTED_TREND,
    tolerance: float = FOVS_TREND_TOLERANCE,
    time_dim: str = "time",
) -> dict[str, float | bool]:
    """Validate F_ovS trend against expected value.

    Parameters
    ----------
    f_ovs_ts : xr.DataArray
        F_ovS time series [Sv].
    expected_trend : float
        Expected trend [Sv/year].
    tolerance : float
        Acceptable deviation from expected trend [Sv/year].
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    dict
        Keys: 'trend_sv_per_year', 'trend_msv_per_year', 'expected', 'is_valid'.
    """
    time = f_ovs_ts[time_dim]

    # Convert time to fractional years
    import pandas as pd
    try:
        # Works for both numpy datetime64 and cftime
        timestamps = pd.DatetimeIndex(time.values)
        years = np.array([
            t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
            for t in timestamps
        ])
    except Exception:
        if hasattr(time.values[0], "year"):
            years = np.array([
                t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
                for t in time.values
            ])
        else:
            years = time.values / 365.25

    values = f_ovs_ts.values.ravel()
    valid = np.isfinite(values)

    if valid.sum() < 2:
        return {
            "trend_sv_per_year": float("nan"),
            "trend_msv_per_year": float("nan"),
            "expected": expected_trend * 1e3,
            "is_valid": False,
        }

    # Linear regression
    coeffs = np.polyfit(years[valid], values[valid], 1)
    trend = coeffs[0]  # Sv/year

    return {
        "trend_sv_per_year": trend,
        "trend_msv_per_year": trend * 1e3,
        "expected": expected_trend * 1e3,
        "is_valid": abs(trend - expected_trend) <= tolerance,
    }
