"""F_ovS trend decomposition into velocity- and salinity-driven components.

Given velocity and salinity sections at two time periods (e.g. an early
stable period and a recent period showing the trend), split the change
in F_ovS into three parts:

    ΔF_ovS = ΔF_v + ΔF_s + ΔF_cross

where all three components are defined using the *baroclinic*
(section-mean-subtracted) zonally-integrated velocity V_int^bc(z) so
that the decomposition is free of the net-volume-transport drift that
afflicts data-assimilating Boussinesq products (see ardp.physics.fovs
for the barotropic-subtraction motivation). Concretely, for each
period t ∈ {1, 2} we compute

    V_int^bc_t(z) = V_int_t(z) − v̄_t · A_xy(z)

where v̄_t is the section-mean velocity and A_xy(z) is the wet-cell
width at depth z. The three components are then

    ΔF_v     = -(1/S0) ∫ ΔV_int^bc(z) · [S̄_1(z) − S0] dz
    ΔF_s     = -(1/S0) ∫ V_int^bc_1(z) · ΔS̄(z)        dz
    ΔF_cross = -(1/S0) ∫ ΔV_int^bc(z) · ΔS̄(z)         dz

and the identity ΔF_v + ΔF_s + ΔF_cross = F_ov(t2) − F_ov(t1) holds
exactly (to floating-point round-off).

References
----------
de Vries & Weber (2005), Geophys. Res. Lett., 32, L09606.
Mecking et al. (2017), Clim. Dyn., 49, 2025--2043.
Weijer et al. (2019), JGR Oceans, 124, 5336--5375.
van Westen & Dijkstra (2023), Sci. Adv., 9, eadi7066.
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

    Returns (v_int [m²/s], a_xy [m], s_mean [PSU]), each shape (nz,).
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


def _barotropic_correct(
    v_int: np.ndarray,
    a_xy: np.ndarray,
    e3t: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Subtract the section-mean (barotropic) velocity from V_int(z).

    Returns (v_int_bc, v_bar, v_net) where v_int_bc satisfies
    ∫ v_int_bc(z) dz = 0 exactly.
    """
    v_net = float(np.sum(v_int * e3t))          # m³/s
    a_total = float(np.sum(a_xy * e3t))         # m²
    v_bar = v_net / a_total if a_total > 0 else 0.0
    return v_int - v_bar * a_xy, v_bar, v_net


def decompose_fovs_trend(
    v1: np.ndarray,
    s1: np.ndarray,
    v2: np.ndarray,
    s2: np.ndarray,
    e1t_atl: np.ndarray,
    e3t: np.ndarray,
    s0: float = S0,
) -> dict:
    """Decompose ΔF_ovS = F_ov(t2) − F_ov(t1) into v-, s-, and cross components.

    The decomposition is performed on the *baroclinic* component of the
    zonally-integrated velocity (see module docstring) so that net
    volume-transport drift does not contaminate the mechanism split.

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
        - 'delta_total'                 : F_ov_2 − F_ov_1 [Sv]
        - 'delta_v', 'delta_s', 'delta_cross' : integrated components [Sv]
        - 'profile_v', 'profile_s', 'profile_cross' : per-depth integrand
          (pre-dz, in m²/s·PSU) — for diagnostic plotting of vertical
          structure.
        - 'depth_Sv_v', 'depth_Sv_s', 'depth_Sv_cross' : per-layer
          contribution [Sv] satisfying sum(depth_Sv_*) == delta_* exactly.
        - 'residual' : delta_total − (delta_v + delta_s + delta_cross),
          expected to be O(1e-15) Sv.
        - 'v_bar_1', 'v_bar_2'          : section-mean velocity [m/s]
        - 'V_net_1_Sv', 'V_net_2_Sv'    : net volume transport [Sv]
          (diagnostic: these are the quantities the barotropic subtraction
          removes; their magnitude indicates how strongly the product
          violates mass conservation between periods).
    """
    v_int_1, a_xy_1, s_mean_1 = _section_profiles(v1, s1, e1t_atl)
    v_int_2, a_xy_2, s_mean_2 = _section_profiles(v2, s2, e1t_atl)

    # Per-period barotropic correction
    v_bc_1, v_bar_1, v_net_1 = _barotropic_correct(v_int_1, a_xy_1, e3t)
    v_bc_2, v_bar_2, v_net_2 = _barotropic_correct(v_int_2, a_xy_2, e3t)

    # Period-total F_ovS using the overturning (baroclinic) transport
    f1 = -(1.0 / s0) * np.sum(v_bc_1 * (s_mean_1 - s0) * e3t) / 1e6
    f2 = -(1.0 / s0) * np.sum(v_bc_2 * (s_mean_2 - s0) * e3t) / 1e6

    # Decomposition on the baroclinic velocity
    dv = v_bc_2 - v_bc_1
    ds = s_mean_2 - s_mean_1

    integrand_v = dv * (s_mean_1 - s0)         # velocity-driven
    integrand_s = v_bc_1 * ds                  # salinity-driven
    integrand_c = dv * ds                      # cross

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
        "v_bar_1": float(v_bar_1),
        "v_bar_2": float(v_bar_2),
        "V_net_1_Sv": float(v_net_1 / 1e6),
        "V_net_2_Sv": float(v_net_2 / 1e6),
    }
