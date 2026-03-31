"""CMIP6 model registry, regime classification, and associated metadata.

Centralises model lists, piControl F_ovS means, colour palettes, and
published reference estimates so that every script imports from one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CMIP6 model registry
# ---------------------------------------------------------------------------
# key = source_id, value = dict with piControl F_ovS mean [Sv] and colour.
# piControl means come from Weijer et al. (2019), van Westen & Dijkstra (2024),
# or the 1850-1900 historical mean where no piControl estimate is published.
CMIP6_REGISTRY: dict[str, dict[str, Any]] = {
    # --- Published piControl (Weijer et al. 2019) ---
    #   pi_fovs  = published piControl mean F_ovS [Sv]
    #   fovs     = computed historical-mean F_ovS [Sv] (where available)
    #   color    = hex colour for plotting
    "CESM2":             {"pi_fovs": -0.05, "fovs": +0.162, "color": "#d62728"},
    "MPI-ESM1-2-LR":     {"pi_fovs": -0.10, "fovs": +0.093, "color": "#e377c2"},
    "MPI-ESM1-2-HR":     {"pi_fovs": -0.02, "fovs": -0.044, "color": "#999999"},
    "UKESM1-0-LL":       {"pi_fovs": +0.15, "fovs": +0.051, "color": "#17becf"},
    "CNRM-CM6-1":        {"pi_fovs": -0.08, "fovs": -0.119, "color": "#d62728"},
    "EC-Earth3":          {"pi_fovs": +0.01, "color": "#7f7f7f"},
    "GFDL-ESM4":         {"pi_fovs": +0.05, "color": "#9467bd"},
    "CanESM5":           {"pi_fovs": +0.12, "fovs": -0.040, "color": "#1f77b4"},
    "IPSL-CM6A-LR":      {"pi_fovs": -0.15, "fovs": -0.171, "color": "#ff7f0e"},
    "ACCESS-ESM1-5":     {"pi_fovs": +0.08, "color": "#2ca02c"},
    # --- Published (van Westen & Dijkstra 2024, Ocean Science) ---
    "MIROC6":            {"pi_fovs": -0.10, "fovs": -0.093, "color": "#8c564b"},
    "GFDL-CM4":          {"pi_fovs": +0.06, "fovs": +0.062, "color": "#aec7e8"},
    "ACCESS-CM2":        {"pi_fovs": +0.08, "fovs": +0.072, "color": "#98df8a"},
    "CMCC-CM2-SR5":      {"pi_fovs": +0.09, "fovs": +0.052, "color": "#c5b0d5"},
    "HadGEM3-GC31-LL":   {"pi_fovs": +0.11, "fovs": +0.095, "color": "#c49c94"},
    "CESM2-WACCM":       {"pi_fovs": +0.17, "color": "#f7b6d2"},
    "NorESM2-LM":        {"pi_fovs": +0.23, "color": "#c7c7c7"},
    "GISS-E2-1-G":       {"pi_fovs": +0.24, "fovs": +0.240, "color": "#dbdb8d"},
    "MRI-ESM2-0":        {"pi_fovs": -0.05, "color": "#bcbd22"},
    # --- New models (historical mean as proxy) ---
    "NESM3":             {"pi_fovs": -0.17, "fovs": -0.187, "color": "#e6550d"},
    "CNRM-ESM2-1":       {"pi_fovs": -0.10, "color": "#e45756"},
    "CanESM5-CanOE":     {"pi_fovs": +0.10, "color": "#6baed6"},
    "EC-Earth3-AerChem":  {"pi_fovs": -0.03, "color": "#636363"},
    "FGOALS-g3":         {"pi_fovs": +0.36, "fovs": +0.347, "color": "#e7cb94"},
    "FIO-ESM-2-0":       {"pi_fovs": +0.19, "fovs": +0.186, "color": "#fdae6b"},
    "GISS-E2-1-G-CC":    {"pi_fovs": +0.25, "color": "#b5cf6b"},
    "SAM0-UNICON":       {"pi_fovs": +0.15, "color": "#74c476"},
    "TaiESM1":           {"pi_fovs": +0.28, "color": "#9e9ac8"},
    "NorESM2-MM":        {"pi_fovs": +0.22, "color": "#bdbdbd"},
}

# ---------------------------------------------------------------------------
# Pre-built model subsets
# ---------------------------------------------------------------------------

#: 16 models with full-field vo downloaded for streamfunction analysis
CMIP6_FULLFIELD_MODELS: list[str] = [
    "NESM3", "IPSL-CM6A-LR", "CNRM-CM6-1", "MIROC6",
    "MPI-ESM1-2-HR", "CanESM5",
    "UKESM1-0-LL", "CMCC-CM2-SR5", "GFDL-CM4", "ACCESS-CM2",
    "MPI-ESM1-2-LR", "HadGEM3-GC31-LL", "CESM2", "FIO-ESM-2-0",
    "GISS-E2-1-G", "FGOALS-g3",
]

#: 18 models for cloud-based F_ovS computation (Pangeo direct)
CMIP6_CLOUD_MODELS: list[str] = [
    # Original 9 (already computed locally)
    "CESM2", "MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "IPSL-CM6A-LR",
    "CNRM-CM6-1", "UKESM1-0-LL", "CanESM5", "EC-Earth3", "ACCESS-ESM1-5",
    # New models
    "NorESM2-LM", "NorESM2-MM", "MIROC6", "GISS-E2-1-G",
    "HadGEM3-GC31-LL", "CMCC-CM2-SR5", "ACCESS-CM2", "CESM2-WACCM", "GFDL-ESM4",
]

#: 5 models with SSP585 extension data to 2300
CMIP6_SSP585_EXT_MODELS: list[str] = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "MRI-ESM2-0", "CESM2-WACCM", "IPSL-CM6A-LR",
]

#: CMIP5 predecessor model names (for reference / comparison figures)
CMIP5_MODELS: list[str] = [
    "CESM1-CAM5", "HadGEM2-ES", "IPSL-CM5A-LR", "MPI-ESM-LR",
    "GFDL-ESM2M", "CNRM-CM5", "NorESM1-M", "CanESM2",
]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def models_sorted_by_fovs() -> list[tuple[str, float]]:
    """Return (model, computed_FovS) pairs sorted from most bistable (negative)
    to most monostable (positive), restricted to the 16 fullfield models.

    Uses the pipeline-computed historical-mean F_ovS (``fovs`` field in the
    registry), which differs slightly from published piControl values.
    """
    return sorted(
        [(m, CMIP6_REGISTRY[m]["fovs"]) for m in CMIP6_FULLFIELD_MODELS],
        key=lambda t: t[1],
    )


def picontrol_means() -> dict[str, float]:
    """Return {model: piControl_FovS} for every model in the registry."""
    return {m: v["pi_fovs"] for m, v in CMIP6_REGISTRY.items()}


def model_colors() -> dict[str, str]:
    """Return {model: hex_color} for every model in the registry."""
    return {m: v["color"] for m, v in CMIP6_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Published reference F_ovS estimates
# ---------------------------------------------------------------------------

#: Published reanalysis F_ovS (mean Sv, start_year, end_year, source)
PUBLISHED_REANALYSIS_FOVS: dict[str, tuple[float, int, int, str]] = {
    "SODA 2.2.4":  (+0.02, 1980, 2010, "Weijer2019"),
    "GECCO2":      (-0.16, 1952, 2001, "Weijer2019"),
    "NCEP GODAS":  (-0.11, 1980, 2020, "Weijer2019"),
    "ECDA (GFDL)": (-0.20, 1961, 2010, "Weijer2019"),
}

#: Published hydrographic point estimates at ~34.5S (F_ovS Sv, error, year, source)
HYDRO_ESTIMATES: dict[str, tuple[float, float | None, int, str]] = {
    "Garzoli et al. 2011": (-0.10, 0.10, 2005, "GarzoliMatano2011"),
    "Meinen et al. 2018":  (-0.09, None, 2015, "Meinen2018"),
}

#: Human-readable SSP scenario labels
SSP_LABELS: dict[str, str] = {
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}


# ---------------------------------------------------------------------------
# Dynamic discovery
# ---------------------------------------------------------------------------

def discover_available_models(results_dir: str | Path, pattern: str = "fovs_*_hist_ssp585.nc") -> list[str]:
    """Scan a results directory and return model names with computed F_ovS files."""
    p = Path(results_dir)
    models = []
    for f in sorted(p.glob(pattern)):
        # Extract model name from filenames like fovs_CESM2_hist_ssp585.nc
        parts = f.stem.split("_")
        if len(parts) >= 3:
            model = "_".join(parts[1:-2])  # handles hyphenated names
            if model in CMIP6_REGISTRY:
                models.append(model)
    return models
