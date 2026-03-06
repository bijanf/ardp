# 2. Data and Methods

## 2.1 Ocean Reanalysis Products

We analyze three global ocean reanalyses, all based on the NEMO (Nucleus for European Modelling of the Ocean) framework but employing distinct data assimilation strategies and atmospheric forcings (Table 1).

**ORAS5** (Ocean Reanalysis System 5) is produced by ECMWF at eddy-permitting 0.25 deg horizontal resolution (~25 km tropics, ~9 km Arctic) with 75 vertical levels (1 m surface resolution). It employs the NEMOVAR 3D-Var FGAT assimilation, ingesting sub-surface T/S profiles, sea-ice concentration, and altimetric sea-level anomalies. A 5-member ensemble with perturbed forcings and observations enables uncertainty quantification. ORAS5 spans 1958-present, forced by ERA-40 (1958-1978) and ERA-Interim/ERA5 (1979-present), providing the longest high-resolution ocean reanalysis record available.

**GLORYS12V1** is produced by Mercator Ocean at eddy-resolving 1/12 deg (~8 km) resolution with 50 vertical levels. A reduced-order Kalman filter jointly assimilates along-track altimetry, satellite SST, sea-ice concentration, and in-situ T/S profiles, with a supplementary 3D-Var bias correction scheme. GLORYS12V1 spans 1993-present (satellite altimetry era) and is distributed via the Copernicus Marine Service.

**C-GLORS** (CMCC Global Ocean Physical Reanalysis) operates at 0.25 deg with the OceanVar variational assimilation system. It spans 1993-present and provides an independent reanalysis framework for cross-validation.

| Property | ORAS5 | GLORYS12V1 | C-GLORS |
|----------|-------|------------|---------|
| Resolution | 0.25 deg | 1/12 deg | 0.25 deg |
| Vertical levels | 75 | 50 | 50 |
| Period | 1958-2023 | 1993-2023 | 1993-2023 |
| Assimilation | NEMOVAR 3D-Var | Kalman + 3D-Var | OceanVar |
| Atmospheric forcing | ERA-40/ERA5 | ERA-Interim/ERA5 | ERA-Interim/ERA5 |
| Temporal resolution | Monthly | Monthly | Monthly |

## 2.2 Fingerprint Computation

### 2.2.1 Overturning Freshwater Transport (F_ovS)

Following van Westen et al. (2024), the overturning freshwater transport at latitude phi is computed as:

F_ov(phi) = -(1/S_0) * integral_W^E integral_{-H}^0 v_bar(z) * [S_bar(z) - S_0] dz dx

where v_bar(z) is the zonally averaged meridional velocity, S_bar(z) is the zonally averaged salinity, S_0 = 35 PSU is the reference salinity, and the integral spans the full basin width W to E and from the ocean bottom -H to the surface. We compute F_ovS at 34.5S (SAMBA array latitude) for each monthly timestep.

The trend is evaluated against the expected value of -1.20 mSv/yr established by the multi-reanalysis mean in van Westen et al. (2024).

### 2.2.2 Salinity Pile-Up Index

Following Latif et al. (2022), the salinity pile-up index is the area-weighted SSS differential:

Delta_S = <SSS>_STSA - <SSS>_STSIP

where <.>_R denotes area-weighted spatial mean over region R. The Subtropical South Atlantic (STSA: 60W-20E, 35S-15S) and Subtropical South Indo-Pacific (STSIP: 20E-290E, 35S-15S) regions follow the original definitions. The index captures the accumulation of salt in the South Atlantic when the AMOC's northward salt transport weakens.

### 2.2.3 North Atlantic Warming Hole (NAWH) Index

We define the NAWH index as:

NAWH = <SST>_NAWH - <SST>_global

where the NAWH region spans 50W-15W, 45N-60N. Following Keil et al. (2020), we note that up to 50% of the observed NAWH may be attributable to atmospheric forcing (strengthened westerlies) rather than reduced ocean heat transport. The NAWH is therefore treated as a complementary, rather than primary, indicator.

### 2.2.4 Gulf Stream Destabilization Point

Following the Copernicus Ocean State Report methodology, we identify the Gulf Stream jet axis as the latitude of maximum SSH gradient magnitude at each longitude within the Gulf Stream region (80W-45W, 30N-45N). The destabilization point is defined as the easternmost longitude where the jet gradient exceeds 50% of its peak value. Beyond this point, the Gulf Stream loses coherence and transitions to a meandering regime. We track this longitude monthly.

## 2.3 Statistical Methods

### Trend Detection

We apply both ordinary least squares (OLS) regression and the nonparametric Mann-Kendall test for trend significance. The Mann-Kendall test is preferred for its robustness to non-normality and serial correlation. Confidence intervals (95%) on trend slopes are computed via block bootstrap (10,000 resamples) to account for autocorrelation in monthly data.

### Sliding-Window Analysis

To assess the sensitivity of detected trends to the choice of analysis period, we compute trends in sliding windows of 15 years (or the longest available sub-window). This reveals whether detected trends are robust to start/end point selection or are dominated by a single anomalous period.

### Cross-Fingerprint Coherence

Pairwise Pearson correlations between the four fingerprint time series are computed after removing the seasonal cycle (12-month running mean). Significant (p < 0.05) correlations between physically independent metrics (e.g., freshwater transport at 34.5S and SST anomalies at 50N) would support a common underlying AMOC signal rather than independent regional variability.

### Multi-Product Agreement

For each fingerprint, we compare trend slopes across the three reanalyses. Multi-product agreement is assessed by requiring: (1) consistent sign of the trend across all available products, and (2) overlapping 95% confidence intervals. Results where all three products agree are designated "robust"; those with two products agreeing are "likely"; single-product results are flagged as "uncertain."

## 2.4 Reproducibility

All computations are performed using the AMOC Reanalysis Diagnostic Pipeline (ARDP), an open-source Python framework. Code, analysis scripts, and figure generation are publicly available at https://github.com/bijanf/ardp. Raw reanalysis data are accessed via the Copernicus Marine Service (GLORYS12V1, C-GLORS) and Climate Data Store (ORAS5) APIs.
