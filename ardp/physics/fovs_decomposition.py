"""F_ovS trend decomposition into velocity- and salinity-driven components.

Given velocity and salinity sections at two time periods (e.g. an early
stable period and a recent period showing the trend), split the change
in F_ovS into three parts:

    ΔF_ovS = ΔF_v + ΔF_s + ΔF_cross

where
    ΔF_v     = -(1/S0) ∫ [v(t2) - v(t1)] · [S(t1) - S0] dz     (velocity change
                                                                 at old salinity)
    ΔF_s     = -(1/S0) ∫ v(t1)           · [S(t2) - S(t1)] dz  (salinity change
                                                                 at old velocity)
    ΔF_cross = -(1/S0) ∫ [v(t2) - v(t1)] · [S(t2) - S(t1)] dz  (covariance term)

The decomposition is exact: ΔF_v + ΔF_s + ΔF_cross = F_ov(t2) - F_ov(t1).

**Physical interpretation.**
- ΔF_v dominant ⇒ trend driven by circulation structure change (could be wind
  or overturning).
- ΔF_s dominant ⇒ trend driven by salinity change (tracer/thermohaline signature,
  consistent with salt-advection feedback).
- If ΔF_s >> ΔF_v, the trend cannot be explained by wind-driven changes alone,
  directly rebutting the Chemke/Keil critique.

References
----------
de Vries & Weber (2005), Geophys. Res. Lett., 32, L09606
van Westen & Dijkstra (2023), Sci. Adv., 9, eadi7066 (similar decomposition)
"""

from __future__ import annotations

import numpy as np

from ardp.constants import S0


def _zonal_integrals(
    v_section: np.ndarray,
    s_section: np.ndarray,
    e1t_atl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-depth zonal-integrated velocity [m²/s] and zonal-averaged salinity [PSU].

    Uses the ocean mask from non-NaN salinity values (land is NaN).
    Returns (v_int(z), s_mean(z)) both of shape (nz,).
    """
    nz = v_section.shape[0]
    v_int = np.zeros(nz)
    s_mean = np.zeros(nz)
    for k in range(nz):
        ocean = ~np.isnan(s_section[k, :])
        if ocean.sum() == 0:
            continue
        v_k = np.where(ocean, np.nan_to_num(v_section[k, :], nan=0.0), 0.0)
        s_k = np.nan_to_num(s_section[k, :], nan=0.0)
        e1t_ocean = np.where(ocean, e1t_atl, 0.0)
        v_int[k] = (v_k * e1t_atl).sum()
        s_mean[k] = (s_k * e1t_ocean).sum() / e1t_ocean.sum() if e1t_ocean.sum() > 0 else 0.0
    return v_int, s_mean


def decompose_fovs_trend(
    v1: np.ndarray,
    s1: np.ndarray,
    v2: np.ndarray,
    s2: np.ndarray,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    s0: float = S0,
) -> dict:
    """Decompose ΔF_ovS = F_ov(t2) - F_ov(t1) into v-, s-, and cross components.

    Parameters
    ----------
    v1, v2 : ndarray, shape (nz, n_atlantic)
        Period-mean meridional velocity sections [m/s] at early and late period.
    s1, s2 : ndarray, shape (nz, n_atlantic)
        Period-mean salinity sections [PSU] at early and late period.
    e1t_atl : ndarray, shape (n_atlantic,)
        Zonal grid spacing [m] at Atlantic grid points.
    e3t : ndarray, shape (nz,)
        Vertical cell thickness [m].
    s0 : float
        Reference salinity [PSU].

    Returns
    -------
    dict with keys:
        - 'F_ov_1', 'F_ov_2'            : F_ovS at each period [Sv]
        - 'delta_total'                 : F_ov_2 - F_ov_1 [Sv]
        - 'delta_v', 'delta_s', 'delta_cross' : integrated components [Sv]
        - 'profile_v', 'profile_s', 'profile_cross' : per-depth contributions [Sv/m]
        - 'depth_Sv_v', 'depth_Sv_s', 'depth_Sv_cross' : integrated-in-depth [Sv]
          so that sum(depth_Sv_*) == delta_* (exact)
        - 'residual'                    : delta_total - (delta_v + delta_s + delta_cross)
          (should be ~0 to machine precision)
    """
    v_int_1, s_mean_1 = _zonal_integrals(v1, s1, e1t_atl)
    v_int_2, s_mean_2 = _zonal_integrals(v2, s2, e1t_atl)

    # Period-total F_ovS at each epoch
    f1 = -(1.0 / s0) * np.sum(v_int_1 * (s_mean_1 - s0) * e3t) / 1e6
    f2 = -(1.0 / s0) * np.sum(v_int_2 * (s_mean_2 - s0) * e3t) / 1e6

    # Decomposition (Reynolds-like): v2 = v1 + dv,  s2 = s1 + ds
    dv = v_int_2 - v_int_1
    ds = s_mean_2 - s_mean_1

    # Per-depth contributions [integrand before dz multiplication]
    # Note: F_ov_2 - F_ov_1 =
    #   -(1/S0) ∫ [v1+dv][s1+ds - S0] dz + (1/S0) ∫ v1 (s1 - S0) dz
    # = -(1/S0) ∫ [dv · (s1 - S0) + v1 · ds + dv · ds] dz
    integrand_v = dv * (s_mean_1 - s0)         # velocity-driven
    integrand_s = v_int_1 * ds                 # salinity-driven
    integrand_c = dv * ds                      # covariance

    depth_Sv_v = -(1.0 / s0) * integrand_v * e3t / 1e6
    depth_Sv_s = -(1.0 / s0) * integrand_s * e3t / 1e6
    depth_Sv_c = -(1.0 / s0) * integrand_c * e3t / 1e6

    delta_v = float(np.sum(depth_Sv_v))
    delta_s = float(np.sum(depth_Sv_s))
    delta_c = float(np.sum(depth_Sv_c))

    delta_total = float(f2 - f1)
    residual = delta_total - (delta_v + delta_s + delta_c)

    return {
        "F_ov_1": float(f1),
        "F_ov_2": float(f2),
        "delta_total": delta_total,
        "delta_v": delta_v,
        "delta_s": delta_s,
        "delta_cross": delta_c,
        "profile_v": integrand_v,
        "profile_s": integrand_s,
        "profile_cross": integrand_c,
        "depth_Sv_v": depth_Sv_v,
        "depth_Sv_s": depth_Sv_s,
        "depth_Sv_cross": depth_Sv_c,
        "residual": float(residual),
    }
