# ARDP — AMOC Reanalysis Diagnostic Pipeline

A Python framework for analyzing Atlantic Meridional Overturning Circulation (AMOC) weakening indicators using ocean reanalysis products (ORAS5, GLORYS12V1) and CMIP6 model output.

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
