"""Shared F_ovS computation kernel (de Vries & Weber, 2005).

Computes the overturning freshwater transport

    F_ov = -(1/S0) ∫ V_int^bc(z) · [S̄(z) − S0] dz

where V_int^bc(z) is the *baroclinic* zonally-integrated meridional
velocity at depth z — i.e. the raw zonally-integrated velocity
V_int(z) with the section-mean (barotropic) velocity subtracted so
that ∫ V_int^bc(z) dz = 0 by construction. The barotropic
subtraction is mandatory in data-assimilating Boussinesq products,
where the net volume transport across a zonal section is generally
not zero and drifts over time; without it, a net transport drift of
a few Sv injects a spurious ~10 mSv signal into F_ov because the
mean column salinity differs from S0. See de Vries & Weber (2005),
Mecking et al. (2017), Weijer et al. (2019).
"""

from __future__ import annotations

import numpy as np

from ardp.constants import S0


def _section_profiles(
    v_section: np.ndarray,
    s_section: np.ndarray,
    e1t_atl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-depth zonal integrals for a single section.

    Returns
    -------
    v_int : ndarray, shape (nz,)
        Zonally-integrated velocity [m²/s] at each depth.
    a_xy : ndarray, shape (nz,)
        Zonally-summed ocean-cell width [m] at each depth (land excluded).
    s_mean : ndarray, shape (nz,)
        Zonally-averaged salinity [PSU] over ocean points at each depth.
    """
    nz = v_section.shape[0]
    v_int = np.zeros(nz)
    a_xy = np.zeros(nz)
    s_mean = np.zeros(nz)
    for k in range(nz):
        ocean = ~np.isnan(s_section[k, :])
        if ocean.sum() == 0:
            continue
        v_k = np.where(ocean, np.nan_to_num(v_section[k, :], nan=0.0), 0.0)
        s_k = np.nan_to_num(s_section[k, :], nan=0.0)
        e1t_ocean = np.where(ocean, e1t_atl, 0.0)
        v_int[k] = (v_k * e1t_atl).sum()
        a_xy[k] = e1t_ocean.sum()
        if a_xy[k] > 0:
            s_mean[k] = (s_k * e1t_ocean).sum() / a_xy[k]
    return v_int, a_xy, s_mean


def compute_fovs_from_section(
    v_section: np.ndarray,
    s_section: np.ndarray,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    s0: float = S0,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict]:
    """Compute F_ovS from a single-timestep section with barotropic subtraction.

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
    return_diagnostics : bool
        If True, also return a dict with the section-mean velocity, net
        volume transport, and the raw (uncorrected) F_ov for
        DA-artefact diagnostics.

    Returns
    -------
    float or (float, dict)
        F_ovS in Sverdrups. If return_diagnostics, also a dict with
        keys: 'v_bar' [m/s], 'V_net' [Sv], 'F_ov_raw' [Sv] (without
        barotropic subtraction — reported purely for comparison).
    """
    v_int, a_xy, s_mean = _section_profiles(v_section, s_section, e1t_atl)

    # Barotropic (section-mean) velocity
    v_net = float(np.sum(v_int * e3t))        # m³/s
    a_total = float(np.sum(a_xy * e3t))       # m²
    v_bar = v_net / a_total if a_total > 0 else 0.0

    # Overturning (baroclinic) component of V_int
    v_int_bc = v_int - v_bar * a_xy

    # F_ov uses only the baroclinic component (sums to zero in depth)
    total = float(np.sum(v_int_bc * (s_mean - s0) * e3t))
    f_ov = -(1.0 / s0) * total / 1e6

    if not return_diagnostics:
        return f_ov

    # Diagnostics (uncorrected F_ov for comparison with legacy results)
    raw_total = float(np.sum(v_int * (s_mean - s0) * e3t))
    f_ov_raw = -(1.0 / s0) * raw_total / 1e6
    return f_ov, {
        "v_bar": v_bar,
        "V_net_Sv": v_net / 1e6,
        "F_ov_raw_Sv": f_ov_raw,
    }
