"""Shared F_ovS computation kernel (de Vries & Weber, 2005).

Extracted from compute_oras5_fovs.py and compute_glorys12_fovs.py so the
identical physics can be applied to reanalysis sections, CMIP6 sections,
or any other (depth × x) velocity/salinity slices.
"""

from __future__ import annotations

import numpy as np

from ardp.constants import S0


def compute_fovs_from_section(
    v_section: np.ndarray,
    s_section: np.ndarray,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    s0: float = S0,
) -> float:
    """Compute F_ovS from a single timestep section at one latitude.

    Parameters
    ----------
    v_section : ndarray, shape (nz, n_atlantic)
        Meridional velocity [m/s] at Atlantic grid points.
    s_section : ndarray, shape (nz, n_atlantic)
        Salinity [PSU] at Atlantic grid points. NaN = land.
    e1t_atl : ndarray, shape (n_atlantic,)
        Zonal grid spacing [m] at Atlantic grid points.
    e3t : ndarray, shape (nz,)
        Vertical cell thickness [m].
    s0 : float
        Reference salinity [PSU].

    Returns
    -------
    float
        F_ovS in Sverdrups [Sv].
    """
    nz = v_section.shape[0]
    total = 0.0

    for k in range(nz):
        ocean = ~np.isnan(s_section[k, :])
        if ocean.sum() == 0:
            continue

        # V_int: zonally INTEGRATED velocity over ocean points [m²/s]
        v_k = np.where(ocean, np.nan_to_num(v_section[k, :], nan=0.0), 0.0)
        v_int = (v_k * e1t_atl).sum()

        # S_mean: zonally AVERAGED salinity over ocean points [PSU]
        e1t_ocean = np.where(ocean, e1t_atl, 0.0)
        s_k = np.nan_to_num(s_section[k, :], nan=0.0)
        s_mean = (s_k * e1t_ocean).sum() / e1t_ocean.sum()

        total += v_int * (s_mean - s0) * e3t[k]

    # F_ov = -(1/S0) * total  [m³/s -> Sv]
    return -(1.0 / s0) * total / 1e6
