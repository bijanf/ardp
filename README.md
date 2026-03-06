# ARDP — AMOC Reanalysis Diagnostic Pipeline

A Python framework for analyzing Atlantic Meridional Overturning Circulation (AMOC) weakening indicators using ocean reanalysis products (ORAS5, GLORYS12V1, C-GLORS).

## AMOC Collapse Fingerprints

1. **F_ovS** — Overturning freshwater transport at 34.5S (SAMBA array latitude)
2. **Salinity pile-up** — SSS differential between subtropical South Atlantic and South Indo-Pacific
3. **North Atlantic Warming Hole** — Regional SST anomaly relative to global mean
4. **Gulf Stream destabilization** — Jet coherence loss tracked via SSH gradients

## Installation

```bash
pip install -e ".[dev]"
```

## Workflow: Download, Compute, Plot

### 1. Download reanalysis data

```bash
# Download GLORYS12 monthly data (small test slice)
python scripts/download_data.py --product glorys12 --start 1993-01 --end 1993-12

# Download ORAS5 (year-by-year)
python scripts/download_data.py --product oras5 --start 1993 --end 2023

# Include Indo-Pacific for salinity pile-up comparison
python scripts/download_data.py --product glorys12 --start 1993-01 --end 2023-12 --include-indopacific
```

### 2. Compute fingerprints

```bash
python scripts/compute_fingerprints.py --product glorys12
```

This computes F_ovS, salinity pile-up, NAWH index, and Gulf Stream destabilization, saving results to `data/results/*.nc`.

### 3. Plot results

```bash
# F_ovS time series with trend
python scripts/plot_f_ovs.py

# 2x2 summary of all fingerprints
python scripts/plot_fingerprints.py

# Gulf Stream analysis (region map + destabilization time series)
python scripts/plot_gulf_stream.py
```

Figures are saved to `figures/`.

## Development

```bash
ruff check .          # Lint
ruff format .         # Format
mypy ardp/            # Type check
pytest tests/ -v      # Run tests
```

## Project Structure

```
ardp/
├── ingestion/      # Data download & loaders for ORAS5, GLORYS12, C-GLORS
├── physics/        # Density, freshwater transport, MHT, streamfunction
├── fingerprints/   # F_ovS, salinity pile-up, NAWH
├── spatial/        # Region masks, Gulf Stream tracking, Irminger Sea
└── viz/            # Cartopy plotting utilities
scripts/
├── download_data.py         # CLI for downloading reanalysis data
├── compute_fingerprints.py  # Compute all 4 AMOC fingerprints
├── plot_f_ovs.py            # F_ovS time series plot
├── plot_fingerprints.py     # 2x2 summary panel
└── plot_gulf_stream.py      # Gulf Stream analysis plots
```
