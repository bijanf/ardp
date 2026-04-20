# Paper 2 — code and reduced-data archive

**Title:** Mechanistic ambiguity in the Atlantic freshwater
fingerprint: reconciling observational constraints on AMOC weakening

**Author:** B. Fallah

**DOI:** [to be assigned at archive time]

## Scope

This archive contains the analysis code and reduced-result NetCDF
files that reproduce every main-text and supplementary figure of the
above manuscript. Raw reanalysis and CMIP6 data are **not** included;
they are available from their primary providers (see citations in the
manuscript).

## Repository layout

```
├── ardp/                       — Python package with shared modules
│   ├── physics/
│   │   ├── fovs.py             — F_ovS kernel (de Vries & Weber 2005)
│   │   └── fovs_decomposition.py — exact v/s/cross decomposition
│   ├── viz/style.py            — figure styling
│   └── constants.py
├── scripts/
│   ├── compute_*_fovs.py       — per-product F_ovS time-series
│   ├── compute_fovs_decomposition.py       — reanalysis mechanism
│   ├── compute_cmip6_fovs_decomposition.py — CMIP6 forced mechanism
│   ├── compute_cmip6_picontrol_null.py     — internal-variability null
│   ├── compute_cmip6_fovs_amoc_leadlag.py  — CCF analysis
│   ├── compute_paper2_tableS1.py           — trend table
│   ├── plot_paper2_fig{1..4}_*.py          — four main figures
│   ├── plot_paper2_figS{1..4}_*.py         — four supplementary figures
│   └── make_paper2.py          — one-shot orchestrator
├── tests/                      — unit tests (ruff+pytest green)
└── paper2/
    ├── manuscript.tex          — this paper
    └── references.bib
```

## Reduced data

The `data/results/` directory contains the NetCDF and CSV outputs of
every computational step (F_ovS time-series per product, reanalysis
and CMIP6 mechanism decompositions, piControl null distribution,
CMIP6-AMOC lead-lag, emergent-constraint regression, period-sensitivity
grid, trend table). Each file is self-describing through CF-convention
attributes.

## Reproducing the figures

```bash
# One-shot regeneration from saved reduced data (no downloads):
python scripts/make_paper2.py --force

# Individual figures:
python scripts/plot_paper2_fig1_multiprod_fovs.py
python scripts/plot_paper2_fig2_decomposition.py
python scripts/plot_paper2_fig3_tiebreaker.py
python scripts/plot_paper2_fig4_mechanism_conditional.py
# ... (see scripts/make_paper2.py for the full list)
```

## Full pipeline from raw data

```bash
# 1) F_ovS time series (requires raw reanalysis data)
python scripts/compute_oras5_fovs.py
python scripts/compute_glorys12_fovs.py
python scripts/compute_soda_fovs.py       # downloads from UMD mirror
python scripts/compute_ecco_fovs.py       # downloads via earthaccess

# 2) Mechanism decompositions
python scripts/compute_fovs_decomposition.py --product all
python scripts/compute_cmip6_fovs_decomposition.py

# 3) Null tests and supplementary analyses
python scripts/compute_cmip6_picontrol_null.py --n-bootstrap 200
python scripts/compute_fovs_decomposition_sensitivity.py \
  --products oras5 glorys12
python scripts/compute_cmip6_fovs_amoc_leadlag.py
python scripts/compute_emergent_constraint.py --predictor mean \
  --forecast-end 2100

# 4) Figures and table
python scripts/make_paper2.py --force
```

Total end-to-end compute (from raw sections to final figures) is
approximately 4--6 hours on a single workstation, dominated by the
SODA 5-day snapshot downloads. The reduced-data pathway
(step~4 alone from archived NetCDFs) runs in under 5 minutes.

## Data dependencies (user must obtain separately)

- **ORAS5**: ECMWF Copernicus Climate Data Store
  (<https://cds.climate.copernicus.eu/datasets/reanalysis-oras5>).
- **GLORYS12V1**: Copernicus Marine Service
  (<https://marine.copernicus.eu/>).
- **SODA 3.15.2**: UMD mirror
  (<https://dsrs.atmos.umd.edu/DATA/soda3.15.2/REGRIDED/ocean/>).
- **ECCO-V4r4**: NASA Earthdata, dataset short-names
  `ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4` and
  `ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4`.
- **CMIP6 34.5°S sections**: Pangeo cloud catalogue
  (<https://storage.googleapis.com/cmip6/pangeo-cmip6.json>).
- **RAPID MOCHA**: <https://rapid.ac.uk/rapidmoc/>.

## Pre-registration

Design, hypotheses, classification thresholds, and locked-in period
choices are pre-registered on OSF at [DOI to be added]. The pre-reg
was submitted before the SODA decomposition was run (the fourth
reanalysis) to lock in the mechanism-classification rule in advance.

## License

Code: MIT License (see `LICENSE` in the repository root).

Reduced data products: CC-BY-4.0.

## Citation

If you use this code or the reduced data, please cite:

> Fallah, B. (2026). Mechanistic ambiguity in the Atlantic freshwater
> fingerprint: reconciling observational constraints on AMOC weakening.
> *Nature Communications* [vol, pages], doi:[TBD].
> Code archived at Zenodo: doi:[TBD].
