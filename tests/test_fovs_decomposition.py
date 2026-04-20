"""Tests for the F_ovS trend decomposition kernel."""

from __future__ import annotations

import numpy as np
import pytest

from ardp.physics.fovs import compute_fovs_from_section
from ardp.physics.fovs_decomposition import decompose_fovs_trend


def _make_section(nz=20, nx=100, seed=0, v_scale=0.01, s_mean=35.0, s_amp=0.5):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, v_scale, size=(nz, nx))
    s = s_mean + rng.normal(0, s_amp, size=(nz, nx))
    return v, s


def _grid(nz=20, nx=100, dx=45e3, dz=50.0):
    return np.full(nx, dx), np.full(nz, dz)


def test_decomposition_sums_to_total():
    """ΔF_v + ΔF_s + ΔF_cross must equal ΔF_total exactly."""
    v1, s1 = _make_section(seed=1)
    v2, s2 = _make_section(seed=2, v_scale=0.02)  # Different velocity
    e1t, e3t = _grid()

    result = decompose_fovs_trend(v1, s1, v2, s2, e1t, e3t)

    computed_total = result["delta_v"] + result["delta_s"] + result["delta_cross"]
    assert abs(result["residual"]) < 1e-12, f"Residual too large: {result['residual']}"
    assert np.isclose(computed_total, result["delta_total"], atol=1e-10), (
        f"Components sum to {computed_total}, expected {result['delta_total']}"
    )


def test_delta_total_matches_direct_kernel():
    """delta_total = F_ov(t2) - F_ov(t1) from the independent F_ovS kernel."""
    v1, s1 = _make_section(seed=3)
    v2, s2 = _make_section(seed=4, s_amp=0.7)
    e1t, e3t = _grid()

    f1_direct = compute_fovs_from_section(v1, s1, e1t, e3t)
    f2_direct = compute_fovs_from_section(v2, s2, e1t, e3t)
    delta_direct = f2_direct - f1_direct

    result = decompose_fovs_trend(v1, s1, v2, s2, e1t, e3t)
    assert np.isclose(result["delta_total"], delta_direct, atol=1e-10)


def test_velocity_only_change():
    """If salinity is identical between periods, delta_s = delta_cross = 0."""
    v1, s = _make_section(seed=5)
    v2 = v1 * 0.5  # scaled velocity, same salinity
    e1t, e3t = _grid()

    result = decompose_fovs_trend(v1, s, v2, s, e1t, e3t)
    assert abs(result["delta_s"]) < 1e-12
    assert abs(result["delta_cross"]) < 1e-12
    assert abs(result["delta_v"] - result["delta_total"]) < 1e-12


def test_salinity_only_change():
    """If velocity is identical, delta_v = delta_cross = 0."""
    v, s1 = _make_section(seed=6)
    s2 = s1 + 0.05  # uniform salinity bump, same velocity
    e1t, e3t = _grid()

    result = decompose_fovs_trend(v, s1, v, s2, e1t, e3t)
    assert abs(result["delta_v"]) < 1e-12
    assert abs(result["delta_cross"]) < 1e-12
    assert abs(result["delta_s"] - result["delta_total"]) < 1e-12


def test_land_mask_respected():
    """NaNs in salinity (= land) must not poison the decomposition."""
    v1, s1 = _make_section(seed=7)
    v2, s2 = _make_section(seed=8, v_scale=0.02)
    # Mark first 10 x-points as land
    s1[:, :10] = np.nan
    s2[:, :10] = np.nan
    e1t, e3t = _grid()

    result = decompose_fovs_trend(v1, s1, v2, s2, e1t, e3t)
    assert np.isfinite(result["delta_total"])
    assert np.isfinite(result["delta_v"])
    assert np.isfinite(result["delta_s"])
    # Consistency still holds
    assert abs(result["residual"]) < 1e-10


def test_profile_shapes():
    """per-depth profiles have the expected shape (nz,)."""
    nz = 15
    v1, s1 = _make_section(nz=nz, nx=50, seed=9)
    v2, s2 = _make_section(nz=nz, nx=50, seed=10)
    e1t, e3t = _grid(nz=nz, nx=50)

    result = decompose_fovs_trend(v1, s1, v2, s2, e1t, e3t)
    assert result["profile_v"].shape == (nz,)
    assert result["profile_s"].shape == (nz,)
    assert result["profile_cross"].shape == (nz,)
    assert result["depth_Sv_v"].shape == (nz,)
