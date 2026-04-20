# ARDP — AMOC Reanalysis Diagnostic Pipeline

[![CI](https://github.com/bijanf/ardp/actions/workflows/ci.yml/badge.svg)](https://github.com/bijanf/ardp/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python framework for analyzing Atlantic Meridional Overturning Circulation (AMOC) weakening indicators using ocean reanalysis products (ORAS5, GLORYS12V1, SODA3.15.2, ECCO-V4r4) and CMIP6 model output.

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: xarray, numpy, scipy, matplotlib, cartopy, xgcm, gsw, copernicusmarine, intake

## Reproduction: AMOC Strength at 26.5°N (Figure)

### Step 1: Download data

```bash
# ORAS5 3D meridional velocity (1958-2025, ~800 monthly files)
python scripts/download_oras5_parallel.py

# GLORYS12V1 (1993-2025, from Copernicus Marine Service)
# Requires copernicusmarine login: copernicusmarine login
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1M-m \
  --variable vo --minimum-longitude -80 --maximum-longitude 30 \
  --minimum-latitude -60 --maximum-latitude 70 \
  --start-datetime 1993-01-01 --end-datetime 2025-12-31 \
  --output-directory data/glorys12/

# RAPID array observations at 26.5°N
python scripts/download_rapid.py

# CMIP6 full-field vo (16 models, historical + SSP585)
python scripts/download_cmip6_vo_fullfield.py --experiments historical ssp585
```

### Step 2: Compute yearly AMOC(26.5°N)

```bash
python scripts/compute_yearly_amoc26n.py --product oras5
python scripts/compute_yearly_amoc26n.py --product glorys12
python scripts/compute_yearly_amoc26n.py --product cmip6
```

Output: `data/results/yearly_amoc26n_{oras5,glorys12,cmip6}.npz`

### Step 3: Plot

```bash
python scripts/plot_amoc_strength_comparison.py
```

Output: `figures/grl/fig_amoc_strength_26N.{png,pdf}`

## Reproduction: F_ovS at 34.5°S

### Step 1: Download data

```bash
# ORAS5 3D velocity + salinity (same download as above covers both)
python scripts/download_oras5_parallel.py

# CMIP6 34.5°S sections (vo + so) from Pangeo
python scripts/download_cmip6_fovs_sections.py
```

### Step 2: Compute F_ovS

```bash
# ORAS5 F_ovS (parallel, ~2 min)
python scripts/compute_oras5_fovs.py

# CMIP6 F_ovS from sections
python scripts/compute_cmip6_fovs_timeseries.py
```

### Step 3: Plot

```bash
# F_ovS trajectory with CMIP6 boxplots
python scripts/plot_cmip6_fovs_trajectory.py

# AMOC weakening rate comparison (bar chart)
python scripts/plot_amoc_rate_comparison.py
```

## Reproduction: SSS Trend Map & Salinity Pile-up

```bash
# SSS trend map with pile-up panel (GLORYS12)
python scripts/plot_sss_trend_map.py --product glorys12

# SSS trend map (ORAS5, full 1958-2025)
python scripts/plot_sss_trend_map.py --product oras5
```

## Reproduction: AMOC Anomalies (Multi-Reanalysis, Rahmstorf-style)

```bash
# Compute AMOC at 26.5°N from SODA3.15.2 and ECCO-V4r4
python scripts/compute_reanalysis_amoc.py --product all

# Plot anomalies relative to 1950-2009 climatology (ORAS5 anchor)
# Trend significance uses Santer et al. (2000) N_eff autocorrelation correction
python scripts/plot_amoc_reanalysis_anomalies.py
```

Output: `figures/grl/fig_amoc_reanalysis_anomalies_santer.{png,pdf}`

## Reproduction: Paper 2 (F_ovS mechanism decomposition, Nature Comms target)

Novel contribution: decompose the F_ovS trend into velocity-driven
(Δv·S̄) and salinity-driven (v̄·ΔS) components across four reanalyses
and 18 CMIP6 models to identify which mechanism corresponds to forced
AMOC weakening. Headline finding: CMIP6 models with salinity-dominant
mechanism (matching GLORYS12) project 52% AMOC weakening by 2100,
recovering Portmann et al. (2026, Sci Adv) — while velocity-dominant
models (matching ORAS5) project only 37%.

```bash
# Step 1: compute F_ovS time series from each reanalysis
python scripts/compute_oras5_fovs.py
python scripts/compute_glorys12_fovs.py
python scripts/compute_soda_fovs.py     # uses UMD 5-day server catalogue
python scripts/compute_ecco_fovs.py     # uses NASA Earthdata earthaccess

# Step 2: mechanism decomposition of the trend (ΔF_v + ΔF_s + ΔF_cross)
python scripts/compute_fovs_decomposition.py --product all
python scripts/compute_cmip6_fovs_decomposition.py          # forced hist→ssp585
python scripts/compute_cmip6_picontrol_null.py --n-bootstrap 200  # null test

# Step 3: emergent-constraint regression and mechanism-conditional projections
python scripts/compute_emergent_constraint.py --predictor mean --forecast-end 2100
python scripts/compute_cmip6_fovs_amoc_leadlag.py             # CMIP6 CCF

# Step 4: four publication figures
python scripts/plot_paper2_fig1_multiprod_fovs.py              # 4-product F_ovS
python scripts/plot_paper2_fig2_decomposition.py               # v/s decomposition
python scripts/plot_paper2_fig3_tiebreaker.py                  # CMIP6 tie-breaker
python scripts/plot_paper2_fig4_mechanism_conditional.py       # headline finding
```

Outputs: `figures/paper2/fig{1,2,3,4}_*.{png,pdf}`

## GRL Figure Set

```bash
# All 4 GRL figures (F_ovS, RAPID validation, pile-up, CMIP6)
python scripts/plot_grl_figures.py
```

## Data Directory Structure

```
data/
├── oras5/              # ORAS5 monthly NetCDF (3D velocity + salinity + 2D SSS)
├── glorys12/           # GLORYS12V1 monthly NetCDF (Atlantic domain)
├── glorys12_global_sss/ # GLORYS12V1 global SSS (for pile-up computation)
├── soda/               # SODA3.15.2 5-day NetCDF (1980-2022)
├── external/           # RAPID array observations
├── cmip6_fullfield/    # CMIP6 zonally-integrated vo (16 models)
├── cmip6_sections/     # CMIP6 34.5°S sections (vo + so)
└── results/            # Computed time series and cached fields
```

All data files are gitignored. Total data volume: ~3 TB (ORAS5 + GLORYS12).

## Development

```bash
ruff check .          # Lint
ruff format .         # Format
mypy ardp/            # Type check
pytest tests/ -v      # Run tests (142 tests)
```

## Package Structure

```
ardp/
├── ingestion/      # Data download & loaders for ORAS5, GLORYS12
├── physics/        # Density, freshwater transport, MHT, streamfunction
├── fingerprints/   # F_ovS, salinity pile-up, NAWH
├── spatial/        # Region masks, Gulf Stream tracking
├── constants.py    # Physical constants (S0, SAMBA_LAT, region bounds)
└── viz/            # Publication-style plotting utilities
scripts/
├── download_*.py            # Data acquisition scripts
├── compute_*.py             # Computation pipelines
└── plot_*.py                # Figure generation
```
