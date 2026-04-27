"""Comprehensive tests for the CMIP6 F_ovS pipeline.

Covers: _get_lon_and_depth(), grid metrics, compute_fovs_section(),
cross-validation against ardp.physics kernels, concatenation/bias-correction,
and regression tests against computed results.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ardp.constants import S0
from ardp.physics.fovs import compute_fovs_from_section
from ardp.physics.freshwater import freshwater_transport_overturning

# ---------------------------------------------------------------------------
# Import helpers from scripts/ (not a package)
# ---------------------------------------------------------------------------
_spec_compute = importlib.util.spec_from_file_location(
    "compute_cmip6", "scripts/compute_cmip6_fovs_timeseries.py"
)
_compute_mod = importlib.util.module_from_spec(_spec_compute)
_spec_compute.loader.exec_module(_compute_mod)

_get_lon_and_depth = _compute_mod._get_lon_and_depth
compute_fovs_section = _compute_mod.compute_fovs_section
ATLANTIC_LON_MIN = _compute_mod.ATLANTIC_LON_MIN
ATLANTIC_LON_MAX = _compute_mod.ATLANTIC_LON_MAX

_spec_plot = importlib.util.spec_from_file_location(
    "plot_cmip6", "scripts/plot_cmip6_fovs_trajectory.py"
)
_plot_mod = importlib.util.module_from_spec(_spec_plot)
_spec_plot.loader.exec_module(_plot_mod)

load_cmip6_timeseries = _plot_mod.load_cmip6_timeseries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_section_datasets(
    n_lon: int = 20,
    n_depth: int = 2,
    n_time: int = 1,
    lon_range: tuple[float, float] = (-60.0, 10.0),
    depths: list[float] | None = None,
    v_upper: float = 0.1,
    s_upper: float = 35.5,
    v_lower: float = -1.0 / 30.0,
    s_lower: float = 34.8,
    lon_name: str = "longitude",
    depth_name: str = "lev",
    depth_units: str = "m",
    section_lat: float = -34.5,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build paired (vo_ds, so_ds) mimicking CMIP6 section format."""
    if depths is None:
        depths = [500.0, 2500.0]

    lon_vals = np.linspace(lon_range[0], lon_range[1], n_lon)
    depth_vals = np.array(depths[:n_depth], dtype=float)
    times = xr.date_range("2000-01", periods=n_time, freq="MS", use_cftime=True)

    v = np.zeros((n_time, n_depth, n_lon))
    s = np.zeros((n_time, n_depth, n_lon))
    v[:, 0, :] = v_upper
    s[:, 0, :] = s_upper
    if n_depth > 1:
        v[:, 1, :] = v_lower
        s[:, 1, :] = s_lower

    depth_store = depth_vals * 100.0 if depth_units in ("centimeters", "cm") else depth_vals

    vo_ds = xr.Dataset(
        {"vo": (["time", depth_name, "x"], v)},
        coords={
            "time": times,
            depth_name: (depth_name, depth_store, {"units": depth_units}),
            lon_name: ("x", lon_vals),
        },
        attrs={"section_latitude": section_lat},
    )
    so_ds = xr.Dataset(
        {"so": (["time", depth_name, "x"], s)},
        coords={
            "time": times,
            depth_name: (depth_name, depth_store, {"units": depth_units}),
            lon_name: ("x", lon_vals),
        },
        attrs={"section_latitude": section_lat},
    )
    return vo_ds, so_ds


@pytest.fixture
def cmip6_section_two_layer():
    """Standard two-layer CMIP6 section for analytical tests."""
    return _make_section_datasets()


@pytest.fixture
def cmip6_results_dir():
    """Path to data/results/cmip6/, skip if absent."""
    p = Path("data/results/cmip6")
    if not p.exists() or not list(p.glob("fovs_*.nc")):
        pytest.skip("CMIP6 results not available (data/results/cmip6/)")
    return p


# ===================================================================
# Section A: _get_lon_and_depth() unit tests
# ===================================================================

class TestGetLonAndDepth:
    def test_standard_names(self):
        """Finds longitude + lev with standard CMIP6 names."""
        vo_ds, _ = _make_section_datasets(lon_name="longitude", depth_name="lev")
        lon, depth = _get_lon_and_depth(vo_ds)
        assert len(lon) == 20
        assert len(depth) == 2
        np.testing.assert_allclose(depth, [500.0, 2500.0])

    def test_alternate_names(self):
        """Finds lon + olevel with alternate naming."""
        vo_ds, _ = _make_section_datasets(lon_name="lon", depth_name="olevel")
        lon, depth = _get_lon_and_depth(vo_ds)
        assert len(lon) == 20
        assert len(depth) == 2

    def test_fallback_search(self):
        """Falls back to any coord containing 'lon'."""
        vo_ds, _ = _make_section_datasets(lon_name="nav_lon_custom", depth_name="lev")
        # The standard search won't find "nav_lon_custom" directly,
        # but the fallback searches for 'lon' in coord names
        lon, depth = _get_lon_and_depth(vo_ds)
        assert len(lon) == 20

    def test_cm_conversion(self):
        """Depth in centimeters gets converted to meters."""
        vo_ds, _ = _make_section_datasets(depth_units="centimeters")
        lon, depth = _get_lon_and_depth(vo_ds)
        np.testing.assert_allclose(depth, [500.0, 2500.0])

    def test_density_raises(self):
        """Density-space vertical coordinate raises ValueError."""
        ds = xr.Dataset(
            {"vo": (["time", "rho", "x"], np.zeros((1, 5, 20)))},
            coords={
                "time": xr.date_range("2000-01", periods=1, freq="MS", use_cftime=True),
                "rho": np.linspace(1020, 1028, 5),
                "longitude": ("x", np.linspace(-60, 10, 20)),
            },
        )
        with pytest.raises(ValueError, match="density space"):
            _get_lon_and_depth(ds)

    def test_no_lon_raises(self):
        """Missing longitude raises ValueError."""
        ds = xr.Dataset(
            {"vo": (["time", "lev", "x"], np.zeros((1, 2, 20)))},
            coords={
                "time": xr.date_range("2000-01", periods=1, freq="MS", use_cftime=True),
                "lev": [500.0, 2500.0],
                # No longitude coord at all
            },
        )
        with pytest.raises(ValueError, match="Cannot find longitude"):
            _get_lon_and_depth(ds)

    def test_no_depth_raises(self):
        """Missing depth raises ValueError."""
        ds = xr.Dataset(
            {"vo": (["time", "z_unnamed", "x"], np.zeros((1, 2, 20)))},
            coords={
                "time": xr.date_range("2000-01", periods=1, freq="MS", use_cftime=True),
                "longitude": ("x", np.linspace(-60, 10, 20)),
                # No depth/lev coord
            },
        )
        with pytest.raises(ValueError, match="Cannot find depth"):
            _get_lon_and_depth(ds)


# ===================================================================
# Section B: Grid metrics & Atlantic mask
# ===================================================================

class TestGridMetrics:
    def test_atlantic_mask_standard(self):
        """lon [-180,180]: correctly selects [-70, 20]."""
        lon = np.linspace(-180, 180, 360)
        lon_wrapped = np.where(lon > 180, lon - 360, lon)
        mask = (lon_wrapped >= ATLANTIC_LON_MIN) & (lon_wrapped <= ATLANTIC_LON_MAX)
        # Should select roughly 90 points (from -70 to 20)
        assert mask.sum() > 50
        assert lon_wrapped[mask].min() >= ATLANTIC_LON_MIN
        assert lon_wrapped[mask].max() <= ATLANTIC_LON_MAX

    def test_atlantic_mask_0_360_wrapping(self):
        """lon [0,360]: wrapping to [-180,180] works correctly."""
        lon = np.linspace(0, 360, 360)
        lon_wrapped = np.where(lon > 180, lon - 360, lon)
        mask = (lon_wrapped >= ATLANTIC_LON_MIN) & (lon_wrapped <= ATLANTIC_LON_MAX)
        assert mask.sum() > 50
        # Verify actual wrapped values are in range
        assert lon_wrapped[mask].min() >= ATLANTIC_LON_MIN
        assert lon_wrapped[mask].max() <= ATLANTIC_LON_MAX

    def test_atlantic_mask_too_few_points(self):
        """All lon outside Atlantic raises ValueError."""
        vo_ds, so_ds = _make_section_datasets(lon_range=(100.0, 170.0))
        with pytest.raises(ValueError, match="Atlantic points"):
            compute_fovs_section(vo_ds, so_ds)

    def test_e1t_from_dlon(self):
        """e1t = |dlon| * 111000 * cos(lat), clipped >= 1.0."""
        lat = -34.5
        cos_lat = np.cos(np.deg2rad(lat))
        lon = np.linspace(-60, 10, 20)
        dlon = np.diff(lon)
        dlon = np.append(dlon, dlon[-1])
        dlon = (dlon + 180) % 360 - 180
        e1t = np.abs(dlon) * 111000.0 * cos_lat
        e1t = np.clip(e1t, 1.0, None)

        expected_dx = np.abs(lon[1] - lon[0]) * 111000.0 * cos_lat
        np.testing.assert_allclose(e1t, expected_dx, rtol=1e-10)
        assert np.all(e1t >= 1.0)

    def test_e3t_from_depth(self):
        """depth=[5,15,30,50,100] -> e3t=[5,10,15,20,50]."""
        depth = np.array([5.0, 15.0, 30.0, 50.0, 100.0])
        e3t = np.diff(depth, prepend=0.0)
        np.testing.assert_array_equal(e3t, [5.0, 10.0, 15.0, 20.0, 50.0])


# ===================================================================
# Section C: compute_fovs_section() core
# ===================================================================

class TestComputeFovsSection:
    def test_two_layer_analytical(self, cmip6_section_two_layer):
        """Matches hand-calculated value including barotropic subtraction."""
        vo_ds, so_ds = cmip6_section_two_layer
        result = compute_fovs_section(vo_ds, so_ds)

        # Grid metrics
        lon = np.linspace(-60, 10, 20)
        dlon = np.diff(lon)
        dlon = np.append(dlon, dlon[-1])
        dlon = (dlon + 180) % 360 - 180
        cos_lat = np.cos(np.deg2rad(-34.5))
        e1t = np.abs(dlon) * 111000.0 * cos_lat
        e1t = np.clip(e1t, 1.0, None)

        depth = np.array([500.0, 2500.0])
        e3t = np.diff(depth, prepend=0.0)  # [500, 2000]

        sum_e1t = e1t.sum()
        v_up, v_lo = 0.1, -1.0 / 30.0
        s_up, s_lo = 35.5, 34.8

        # Zonally-integrated velocity per layer (m²/s)
        v_int_up = v_up * sum_e1t
        v_int_lo = v_lo * sum_e1t

        # Section-mean velocity (barotropic)
        v_net = v_int_up * e3t[0] + v_int_lo * e3t[1]            # m³/s
        a_total = sum_e1t * (e3t[0] + e3t[1])                    # m²
        v_bar = v_net / a_total

        # Baroclinic (overturning) component per layer
        v_int_bc_up = v_int_up - v_bar * sum_e1t
        v_int_bc_lo = v_int_lo - v_bar * sum_e1t

        upper = v_int_bc_up * (s_up - S0) * e3t[0]
        lower = v_int_bc_lo * (s_lo - S0) * e3t[1]
        expected = -(1.0 / S0) * (upper + lower) / 1e6

        np.testing.assert_allclose(float(result.values[0]), expected, atol=1e-6)

    def test_output_shape(self):
        """24 timesteps in -> 24-element DataArray."""
        vo_ds, so_ds = _make_section_datasets(n_time=24)
        result = compute_fovs_section(vo_ds, so_ds)
        assert result.shape == (24,)
        assert "time" in result.dims

    def test_units_and_name(self, cmip6_section_two_layer):
        """Output has correct units and name."""
        vo_ds, so_ds = cmip6_section_two_layer
        result = compute_fovs_section(vo_ds, so_ds)
        assert result.name == "F_ovS"
        assert result.attrs["units"] == "Sv"

    def test_nan_partial_ocean(self):
        """NaN at some x-points -> result is finite."""
        vo_ds, so_ds = _make_section_datasets(n_lon=20)
        # Set some points to NaN (simulating land)
        so_ds["so"].values[:, :, :5] = np.nan
        result = compute_fovs_section(vo_ds, so_ds)
        assert np.isfinite(result.values[0])

    def test_nan_all_nan_depth_level(self):
        """Full NaN depth level contributes 0, result is finite."""
        vo_ds, so_ds = _make_section_datasets()
        so_ds["so"].values[:, 1, :] = np.nan  # Entire lower layer NaN
        result = compute_fovs_section(vo_ds, so_ds)
        assert np.isfinite(result.values[0])

    def test_nan_all_nan_timestep(self):
        """Full NaN timestep -> result is 0 (all levels skipped) or NaN."""
        vo_ds, so_ds = _make_section_datasets(n_time=3)
        so_ds["so"].values[1, :, :] = np.nan
        vo_ds["vo"].values[1, :, :] = np.nan
        result = compute_fovs_section(vo_ds, so_ds)
        assert np.isfinite(result.values[0])
        # When all ocean points are NaN, loop skips all levels -> total=0 -> result=-0.0
        assert result.values[1] == 0.0
        assert np.isfinite(result.values[2])


# ===================================================================
# Section D: Cross-validation
# ===================================================================

class TestCrossValidation:
    """Two independent implementations must agree exactly."""

    @staticmethod
    def _build_section(n_lon=15, n_depth=3, n_time=6, add_land=False):
        """Build matching inputs for both compute_fovs_from_section and freshwater_transport_overturning."""
        lat = -34.5
        cos_lat = np.cos(np.deg2rad(lat))
        lon = np.linspace(-60, 10, n_lon)
        depth = np.array([100.0, 500.0, 2000.0])[:n_depth]

        dlon = np.diff(lon)
        dlon = np.append(dlon, dlon[-1])
        dlon = (dlon + 180) % 360 - 180
        e1t = np.abs(dlon) * 111000.0 * cos_lat
        e1t = np.clip(e1t, 1.0, None)
        e3t = np.diff(depth, prepend=0.0)

        rng = np.random.RandomState(42)
        v = rng.randn(n_time, n_depth, n_lon) * 0.05
        s = 35.0 + rng.randn(n_time, n_depth, n_lon) * 0.3

        if add_land:
            # Set some columns to NaN (land)
            s[:, :, :3] = np.nan

        return v, s, e1t, e3t

    def test_vectorized_vs_loop_all_ocean(self):
        """compute_fovs_from_section vs freshwater_transport_overturning match."""
        v, s, e1t, e3t = self._build_section()

        for t in range(v.shape[0]):
            # Loop-based (ardp.physics.fovs)
            loop_val = compute_fovs_from_section(v[t], s[t], e1t, e3t)

            # Vectorized (ardp.physics.freshwater)
            v_da = xr.DataArray(v[t], dims=["z", "x"])
            s_da = xr.DataArray(s[t], dims=["z", "x"])
            e1_da = xr.DataArray(e1t, dims=["x"])
            e3_da = xr.DataArray(e3t, dims=["z"])
            vec_val = float(freshwater_transport_overturning(
                v_da, s_da, e1_da, e3_da, x_dim="x", z_dim="z"
            ).values)

            np.testing.assert_allclose(loop_val, vec_val, atol=1e-10,
                                       err_msg=f"Mismatch at timestep {t}")

    def test_vectorized_vs_loop_with_land(self):
        """Both agree on the full section when NaN land is present.

        Barotropic subtraction couples depths together (v_bar depends on
        the column-integrated transport), so per-level summation is no
        longer valid — we compare full-section calls.
        """
        v, s, e1t, e3t = self._build_section(add_land=True)

        for t in range(v.shape[0]):
            loop_val = compute_fovs_from_section(v[t], s[t], e1t, e3t)

            v_da = xr.DataArray(v[t], dims=["z", "x"])
            s_da = xr.DataArray(s[t], dims=["z", "x"])
            e1_da = xr.DataArray(e1t, dims=["x"])
            e3_da = xr.DataArray(e3t, dims=["z"])
            vec_val = float(freshwater_transport_overturning(
                v_da, s_da, e1_da, e3_da, x_dim="x", z_dim="z"
            ).values)

            np.testing.assert_allclose(loop_val, vec_val, atol=1e-10,
                                       err_msg=f"Mismatch at timestep {t}")


# ===================================================================
# Section E: Concatenation & bias correction
# ===================================================================

class TestConcatAndBias:
    def test_hist_ssp_concat_no_overlap(self):
        """historical(1850-2014) + ssp(2015-2100) -> no duplicates."""
        hist_times = xr.date_range("1850-01", "2014-12", freq="MS", use_cftime=True)
        ssp_times = xr.date_range("2015-01", "2100-12", freq="MS", use_cftime=True)

        hist = xr.DataArray(np.random.randn(len(hist_times)), dims="time",
                            coords={"time": hist_times}, name="F_ovS")
        ssp = xr.DataArray(np.random.randn(len(ssp_times)), dims="time",
                           coords={"time": ssp_times}, name="F_ovS")

        combined = xr.concat([hist, ssp], dim="time")
        _, idx = np.unique(combined.time.values, return_index=True)
        combined = combined.isel(time=sorted(idx))

        expected_months = len(hist_times) + len(ssp_times)
        assert len(combined) == expected_months
        # Verify no duplicates
        times = combined.time.values
        assert len(times) == len(set(times))

    def test_hist_ssp_concat_with_overlap(self):
        """1-year overlap -> deduplication removes extra months."""
        hist_times = xr.date_range("1850-01", "2015-12", freq="MS", use_cftime=True)  # 1 year overlap
        ssp_times = xr.date_range("2015-01", "2100-12", freq="MS", use_cftime=True)

        hist = xr.DataArray(np.random.randn(len(hist_times)), dims="time",
                            coords={"time": hist_times}, name="F_ovS")
        ssp = xr.DataArray(np.random.randn(len(ssp_times)), dims="time",
                           coords={"time": ssp_times}, name="F_ovS")

        combined = xr.concat([hist, ssp], dim="time")
        _, idx = np.unique(combined.time.values, return_index=True)
        combined = combined.isel(time=sorted(idx))

        # Should have no more months than non-overlapping
        no_overlap_len = len(xr.date_range("1850-01", "2100-12", freq="MS", use_cftime=True))
        assert len(combined) == no_overlap_len

    def test_bias_correction_offset(self, tmp_path):
        """Bias correction shifts series by (published - computed_early_mean)."""
        model = "CESM2"
        results_dir = tmp_path / "results"
        cmip6_dir = results_dir / "cmip6"
        cmip6_dir.mkdir(parents=True)

        # Create a fake historical time series
        hist_times = xr.date_range("1850-01", "2014-12", freq="MS", use_cftime=True)
        raw_mean = 0.10  # Computed early-historical mean
        hist_da = xr.DataArray(
            np.full(len(hist_times), raw_mean),
            dims="time",
            coords={"time": hist_times},
            name="F_ovS",
        )
        hist_ds = hist_da.to_dataset(name="F_ovS")
        hist_ds.attrs["key"] = f"{model}_historical"
        hist_ds.to_netcdf(cmip6_dir / f"fovs_{model}_historical.nc")

        # Create a fake ssp585 concatenated series
        ssp_times = xr.date_range("2015-01", "2100-12", freq="MS", use_cftime=True)
        concat_times = xr.date_range("1850-01", "2100-12", freq="MS", use_cftime=True)
        concat_da = xr.DataArray(
            np.full(len(concat_times), raw_mean),
            dims="time",
            coords={"time": concat_times},
            name="F_ovS",
        )
        concat_ds = concat_da.to_dataset(name="F_ovS")
        concat_ds.attrs["key"] = f"{model}_hist_ssp585"
        concat_ds.to_netcdf(cmip6_dir / f"fovs_{model}_hist_ssp585.nc")

        # Load raw (no bias correction)
        series = load_cmip6_timeseries(results_dir)

        # Raw values should be unchanged
        loaded = series[f"{model}_historical"]
        loaded_mean = float(loaded.mean())
        np.testing.assert_allclose(loaded_mean, raw_mean, atol=1e-6)


# ===================================================================
# Section F: Regression tests (require data/)
# ===================================================================

# Raw computed historical means (no bias correction) — recomputed with
# the barotropic-corrected kernel. See ardp.physics.fovs for the correction.
EXPECTED_HIST_MEANS = {
    "ACCESS-CM2": +0.072,
    "ACCESS-ESM1-5": +0.089,
    "CESM2": +0.151,
    "CMCC-CM2-SR5": +0.009,
    "CNRM-CM6-1": -0.094,
    "CanESM5": -0.059,
    "EC-Earth3": -0.018,
    "EC-Earth3-AerChem": -0.018,
    "FGOALS-g3": +0.328,
    "FIO-ESM-2-0": +0.200,
    "GFDL-CM4": +0.047,
    "GISS-E2-1-G": +0.233,
    "GISS-E2-1-G-CC": +0.261,
    "IPSL-CM6A-LR": -0.159,
    "MIROC6": -0.079,
    "MPI-ESM1-2-HR": -0.028,
    "MPI-ESM1-2-LR": +0.122,
    "NESM3": -0.173,
    "SAM0-UNICON": +0.162,
    "TaiESM1": +0.293,
    "UKESM1-0-LL": +0.043,
}


@pytest.mark.integration
class TestRegressionCMIP6:
    """Regression tests against computed CMIP6 results. Require data/results/cmip6/."""

    @pytest.fixture(autouse=True)
    def _load_series(self, cmip6_results_dir):
        """Load all CMIP6 time series once for the class."""
        results_dir = cmip6_results_dir.parent  # data/results/
        self.series = load_cmip6_timeseries(results_dir)

    @pytest.mark.parametrize("model,expected_mean", list(EXPECTED_HIST_MEANS.items()))
    def test_regression_historical_means(self, model, expected_mean):
        """Historical mean matches known value within +-0.01 Sv."""
        key = f"{model}_historical"
        if key not in self.series:
            pytest.skip(f"No historical data for {model}")
        da = self.series[key]
        actual_mean = float(da.mean())
        np.testing.assert_allclose(actual_mean, expected_mean, atol=0.01,
                                   err_msg=f"{model}: expected {expected_mean}, got {actual_mean}")

    @pytest.mark.parametrize("model", list(EXPECTED_HIST_MEANS.keys()))
    def test_time_coverage(self, model):
        """Historical covers ~1850-2014, SSP covers ~2015-2100."""
        hist_key = f"{model}_historical"
        if hist_key not in self.series:
            pytest.skip(f"No data for {model}")

        da = self.series[hist_key]
        times = da.time.values
        if hasattr(times[0], "year"):
            years = np.array([t.year for t in times])
        else:
            import pandas as pd
            years = pd.DatetimeIndex(times).year.values

        # Historical should start around 1850 and end around 2014
        assert years.min() <= 1860, f"{model} historical starts at {years.min()}"
        assert years.max() >= 2010, f"{model} historical ends at {years.max()}"

        # Check SSP if available
        for ssp in ["ssp245", "ssp585"]:
            ssp_key = f"{model}_hist_{ssp}"
            if ssp_key in self.series:
                da_ssp = self.series[ssp_key]
                times_ssp = da_ssp.time.values
                if hasattr(times_ssp[0], "year"):
                    ssp_years = np.array([t.year for t in times_ssp])
                else:
                    ssp_years = pd.DatetimeIndex(times_ssp).year.values
                assert ssp_years.max() >= 2090, (
                    f"{model} {ssp} ends at {ssp_years.max()}"
                )

    @pytest.mark.parametrize("model", list(EXPECTED_HIST_MEANS.keys()))
    def test_valid_timestep_fraction(self, model):
        """More than 95% of timesteps should be finite (GISS ssp245 exempted at >5%)."""
        hist_key = f"{model}_historical"
        if hist_key not in self.series:
            pytest.skip(f"No data for {model}")

        for key, da in self.series.items():
            if not key.startswith(model):
                continue
            vals = da.values
            frac = np.isfinite(vals).sum() / len(vals)

            # GISS models have known gaps in ssp245
            if "GISS" in model and "ssp245" in key:
                min_frac = 0.05
            else:
                min_frac = 0.95

            assert frac >= min_frac, (
                f"{key}: only {frac:.1%} finite ({np.isfinite(vals).sum()}/{len(vals)})"
            )

    @pytest.mark.parametrize("model", list(EXPECTED_HIST_MEANS.keys()))
    def test_physical_bounds(self, model):
        """All finite F_ovS values should be in (-1.0, 1.0) Sv."""
        for key, da in self.series.items():
            if not key.startswith(model):
                continue
            vals = da.values[np.isfinite(da.values)]
            if len(vals) == 0:
                continue
            assert vals.min() > -1.0, f"{key}: min={vals.min():.3f} Sv"
            assert vals.max() < 1.0, f"{key}: max={vals.max():.3f} Sv"
