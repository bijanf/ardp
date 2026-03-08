# GRL Figure Notes

## fig_sss_trend_map (panels a, b)

**Script:** `scripts/plot_sss_trend_map.py`
**Data:** GLORYS12 reanalysis, surface salinity (`so`, depth index 0 = 0.49m), 372 monthly files (1993-01 to 2023-12), ~1561×1321 grid (1/12° resolution).

### Panel (a): SSS trend map

- **Method:** Per-pixel OLS linear regression of monthly SSS against fractional year. Vectorized via normal equations for all-valid pixels; scipy `linregress` fallback for partial-NaN coastal pixels.
- **Units:** PSU/decade (slope × 10).
- **Colormap:** RdBu_r (diverging), symmetric range from 98th percentile of |trend|, centered on zero.
- **Contour lines:**
  - **Red solid (±0.08 PSU/decade):** Data-driven salinification hotspot boundary. Gaussian-smoothed (σ=8 grid points) trend field, then contoured. Organic shapes that follow the ocean signal.
  - **Blue solid (−0.08 PSU/decade):** Freshening hotspot boundary. Same smoothing.
  - **Gray dashed (0):** Zero-trend contour separating freshening from salinification.
- **Hatching:** Cross-hatching over non-significant areas (p ≥ 0.05, two-sided t-test on the OLS slope). Significant areas are left clean.
- **Land:** Dark gray (#a09e99), cartopy Natural Earth features.
- **Extent:** Atlantic basin, [-80°, 30°] lon, [-55°, 70°] lat.
- **Atlantic mask for zonal mean excludes:** Mediterranean (lon > -6°, 30°–46°N), Baltic (lon > 10°, > 54°N), Hudson Bay (lon < -75°, 50°–66°N), Gulf of Mexico (lon < -82°, 18°–31°N).

### Panel (b): Zonal-mean SSS trend with N_eff-adjusted 95% CI

- **Zonal mean:** `nanmean` of trend across Atlantic-masked longitudes at each latitude.
- **Uncertainty:** 95% confidence interval on the zonal mean, corrected for spatial autocorrelation.
  - Effective degrees of freedom via **Bretherton et al. (1999)**: `N_eff = N × (1 − r₁) / (1 + r₁)`, where `r₁` is the lag-1 zonal autocorrelation of trend values along longitude at each latitude.
  - Standard error: `SE = σ / √N_eff`
  - 95% CI: `mean ± 1.96 × SE`
- **Line style:** Solid bold where 95% CI excludes zero (statistically significant zonal-mean trend); dashed where not significant.
- **Background tint:** Subtle red/blue fill between the line and zero for visual emphasis.

### Key results
- STSA mean trend: +0.085 PSU/decade (salinification, consistent with pile-up index).
- 48.2% of ocean pixels have significant trends (p < 0.05).
- Land fraction: 33.5%.

### Suggested caption
> **Figure X.** Geographic pattern of sea surface salinity (SSS) trends from GLORYS12 reanalysis (1993–2023). **(a)** Per-pixel linear SSS trend (PSU decade⁻¹) from OLS regression of 372 monthly fields. Red (blue) contours enclose regions with trends exceeding +0.08 (−0.08) PSU decade⁻¹ after Gaussian smoothing (σ = 8 grid cells). Cross-hatching marks areas where the trend is not significant at the 95% level. The dashed gray line marks the zero-trend contour. **(b)** Zonal-mean SSS trend across Atlantic longitudes (excluding Mediterranean, Baltic, Hudson Bay, and Gulf of Mexico). Shading shows the 95% confidence interval on the mean, adjusted for spatial autocorrelation using effective degrees of freedom (Bretherton et al., 1999). Solid (dashed) segments indicate latitudes where the zonal-mean trend is (is not) significantly different from zero.

### References
- Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., & Bladé, I. (1999). The effective number of spatial degrees of freedom of a time-varying field. *J. Climate*, 12, 1990–2009.
- Zhu, C., & Liu, Z. (2020). Weakening Atlantic overturning circulation causes South Atlantic salinity pile-up. *Nature Climate Change*, 10, 998–1003.
- Zhu, C., Liu, Z., Zhang, S., & Wu, L. (2023). Likely accelerated weakening of Atlantic overturning circulation emerges in optimal salinity fingerprint. *Nature Communications*, 14, 1536.

---

## fig1_fovs_multiproduct (panels a, b)
**Script:** `scripts/plot_grl_figures.py` → `figure1_fovs_multiproduct()`
**Data:** `data/results/oras5_f_ovs.nc` (330 months, non-uniform: monthly 1958-61 & 2004-23, June-only 1962-2003).

## fig2_rapid_validation
**Script:** `scripts/plot_grl_figures.py` → `figure2_rapid_validation()`
**Data:** ORAS5 MOC at 26.5°N vs RAPID monthly (237 pairs, r=0.745).

## fig3_salinity_pileup
**Script:** `scripts/plot_grl_figures.py` → `figure3_salinity_pileup()`
**Data:** ORAS5 + GLORYS12 SSS pile-up index (STSA minus STSIP area-mean SSS).

## fig4_assessment
**Script:** `scripts/plot_grl_figures.py` → `figure4_assessment()`
**Data:** Summary assessment of reanalysis skill.
