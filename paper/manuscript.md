# Multi-Reanalysis Assessment of AMOC Weakening Fingerprints Over 1958–2025

## Authors
[To be filled]

## Abstract

Whether the Atlantic Meridional Overturning Circulation (AMOC) is approaching a critical transition remains a central question in climate science. We present a multi-reanalysis assessment of established AMOC weakening fingerprints using ORAS5 (1958–2025, 816 months) and GLORYS12V1 (1993–2025, 396 months), validated against RAPID observations at 26.5°N. The overturning freshwater transport at 34.5°S (F_ovS) shows a statistically significant decline of −1.31 mSv yr⁻¹ over the full ORAS5 record (p < 0.001), with a mean value of −0.033 Sv placing the Atlantic in the theoretically bistable regime. Both reanalyses reproduce observed MOC variability at 26.5°N (r = 0.75 and 0.71 for ORAS5 and GLORYS12, respectively). A decomposition of GLORYS12 sea surface salinity trends reveals that the amplification hypothesis—whereby trends mirror the mean salinity pattern under hydrological cycle intensification—fails in the Atlantic (R² = 0.06, p = 0.24). Instead, residual trends show an overturning-dipole structure: excess salinification in the South Atlantic and freshening in the subpolar region, consistent with reduced meridional salt export. These results, while subject to the limitations of shared atmospheric forcing across reanalysis products, indicate that the salt-advection feedback identified in theoretical tipping studies is active in the contemporary ocean.

---

## 1. Introduction

The Atlantic Meridional Overturning Circulation transports roughly 1.25 PW of heat northward across the equator, maintaining the mild climate of Northern Europe, modulating the partitioning of CO₂ between ocean and atmosphere, and influencing the position of the Intertropical Convergence Zone (Buckley & Marshall, 2016; Ganachaud & Wunsch, 2000). Paleoclimatic records from Dansgaard-Oeschger events and Heinrich stadials demonstrate that this circulation has undergone rapid, quasi-irreversible transitions in the past, with consequences that extended across hemispheres (Rahmstorf, 2002).

Anthropogenic forcing now raises the question of whether the modern AMOC is approaching such a transition. Accelerating Greenland Ice Sheet melt and intensification of the hydrological cycle both act to freshen the high-latitude North Atlantic, potentially weakening the density-driven formation of North Atlantic Deep Water (Boers, 2021). Yet direct observational constraints remain limited: the RAPID array at 26.5°N began continuous monitoring only in 2004, the SAMBA array at 34.5°S in 2009, and OSNAP at subpolar latitudes in 2014 (Cunningham et al., 2007; Meinen et al., 2018). These records, while transformative, are too short to separate forced trends from the substantial internal variability of the AMOC on decadal to multi-decadal timescales (Wunsch, 2018).

This observational gap has produced divergent assessments. Fu et al. (2025), analyzing basin-integrated heat fluxes from ERA5 and CMIP6 ensembles, concluded that the AMOC has not undergone statistically significant decline over the past six decades. Thompson et al. (2025) projected 18–43% weakening by 2100, well short of collapse. Meanwhile, van Westen et al. (2024) identified a declining trend in the overturning freshwater transport at 34.5°S (F_ovS) across multiple reanalyses, interpreting this as a physics-based early warning of tipping. Latif et al. (2022) reported an accelerating South Atlantic salinity pile-up consistent with reduced northward salt advection.

These studies differ not only in their conclusions but in what they measure. Basin-integrated heat fluxes average over vast spatial scales and conflate atmospheric and oceanic contributions. The fingerprint approach, by contrast, targets specific dynamical processes: F_ovS isolates the overturning component of freshwater transport at a single latitude; the salinity pile-up tracks cumulative salt redistribution; the North Atlantic Warming Hole reflects anomalous subpolar cooling. Whether these fingerprints tell a consistent story when computed from multiple independent reanalyses has not been systematically tested.

Here we address this gap using two ocean reanalyses—ORAS5 (1958–2025) and GLORYS12V1 (1993–2025)—both constrained by observations but employing different resolutions, assimilation schemes, and atmospheric forcings. We compute four AMOC weakening fingerprints, validate the reanalysis MOC against RAPID observations, and decompose Atlantic sea surface salinity trends to distinguish hydrological cycle from circulation-driven components. Our aim is not to resolve the stability-versus-weakening debate, but to evaluate what the reanalysis record can and cannot tell us about the state of the AMOC's salt-advection feedback.

---

## 2. Data and Methods

### 2.1 Ocean Reanalysis Products

We analyze two global ocean reanalyses, both based on the NEMO framework but differing in resolution, data assimilation, and temporal coverage (Table 1).

**ORAS5** (Ocean Reanalysis System 5; ECMWF) operates at 0.25° horizontal resolution (~25 km at the equator) with 75 vertical levels and 1-m surface resolution. Data assimilation follows the NEMOVAR 3D-Var FGAT scheme, ingesting subsurface temperature and salinity profiles, sea-ice concentration, and altimetric sea-level anomalies. ORAS5 spans 1958–2025, forced by ERA-40 (1958–1978) and ERA-Interim/ERA5 (1979–present) (Zuo et al., 2019).

**GLORYS12V1** (Copernicus Marine Service/Mercator Océan) operates at 1/12° (~8 km, eddy-resolving) with 50 vertical levels. A reduced-order Kalman filter jointly assimilates along-track altimetry, satellite SST, sea-ice concentration, and in-situ profiles, with supplementary 3D-Var bias correction. GLORYS12V1 spans 1993–2025 (satellite altimetry era) (Jean-Michel et al., 2021).

| Property | ORAS5 | GLORYS12V1 |
|----------|-------|------------|
| Resolution | 0.25° | 1/12° |
| Vertical levels | 75 | 50 |
| Period | 1958–2025 | 1993–2025 |
| Assimilation | NEMOVAR 3D-Var | Kalman + 3D-Var |
| Atmospheric forcing | ERA-40/ERA5 | ERA-Interim/ERA5 |
| Monthly fields | 816 | 396 |

### 2.2 Observational Validation Data

We use monthly MOC transports from the RAPID-MOCHA array at 26.5°N (Cunningham et al., 2007), available from April 2004 to March 2024 (240 months), to validate reanalysis MOC estimates.

### 2.3 Fingerprint Computation

**Overturning freshwater transport (F_ovS).** Following van Westen et al. (2024), we compute:

F_ov(φ) = −(1/S₀) ∫∫ v̄(z) [S̄(z) − S₀] dz dx

where v̄ and S̄ are zonally averaged meridional velocity and salinity, S₀ = 35 PSU is the reference salinity, and the integration spans the Atlantic basin width and full depth. F_ovS is evaluated at 34.5°S (SAMBA latitude) from ORAS5 monthly fields.

**Salinity pile-up index.** Following Latif et al. (2022), we compute the area-weighted SSS differential ΔS = ⟨SSS⟩_STSA − ⟨SSS⟩_STSIP, where the Subtropical South Atlantic (STSA: 60°W–20°E, 35°S–15°S) and Subtropical South Indo-Pacific (STSIP: 20°E–290°E, 35°S–15°S) regions follow the original definitions.

**North Atlantic Warming Hole (NAWH) index.** Defined as ⟨SST⟩_NAWH − ⟨SST⟩_global, where the NAWH region spans 50°W–15°W, 45°N–60°N.

**Gulf Stream destabilization point.** The longitude where the Gulf Stream loses jet coherence, identified from the SSH gradient field following Copernicus Ocean State Report methodology.

### 2.4 SSS Trend Decomposition

To assess whether Atlantic SSS trends are driven by the hydrological cycle or ocean circulation, we test the amplification hypothesis (Held & Soden, 2006; Durack & Wijffels, 2010). Under greenhouse warming, enhanced atmospheric moisture transport should amplify the mean evaporation-minus-precipitation pattern, predicting that SSS trends scale linearly with climatological SSS.

We regress zonal-mean SSS trends against zonal-mean climatological SSS across 25 latitude bands (5° width, 55°S to 70°N) in the Atlantic. The pixel-level residual field (observed trend minus amplification-predicted trend) isolates the component attributable to ocean dynamics.

Monthly SSS fields are deseasonalized by removing the pixel-level monthly climatology before computing OLS trends. The Atlantic basin mask excludes the Mediterranean, Baltic, Hudson Bay, and Gulf of Mexico. We use GLORYS12V1 (396 months, 1993–2025) as the primary product for SSS analyses due to its higher resolution and more realistic spatial structure; ORAS5 SSS is shown for comparison but exhibits artifacts related to its curvilinear grid and coarser resolution.

### 2.5 Statistical Methods

Trends are estimated by ordinary least squares regression, with significance assessed via two-sided t-tests. For spatial fields, we account for spatial autocorrelation in zonal-mean uncertainty estimates using the effective sample size N_eff = N(1−r₁)/(1+r₁), where r₁ is the lag-1 autocorrelation of zonal pixel values (Bretherton et al., 1999).

### 2.6 Code Availability

All analyses are performed using the AMOC Reanalysis Diagnostic Pipeline (ARDP), available at https://github.com/bijanf/ardp.

---

## 3. Results

### 3.1 Overturning Freshwater Transport (F_ovS)

The ORAS5 F_ovS time series at 34.5°S spans 816 months (January 1958 to December 2025) and reveals a sustained negative mean of −0.033 Sv (Fig. 1a). This negative value places the Atlantic in the regime where the salt-advection feedback amplifies perturbations, as established by Rahmstorf (1996) and confirmed in the tipping framework of van Westen et al. (2024). The linear trend is −1.31 mSv yr⁻¹ (p < 0.001), consistent with the multi-reanalysis mean of −1.20 mSv yr⁻¹ reported by van Westen et al. (2024) from a shorter record.

The 12-month running mean shows substantial interannual to decadal variability superimposed on the long-term decline, with F_ovS ranging from approximately −0.15 to +0.10 Sv. A period of relatively high F_ovS values during the 1960s–1980s transitions to persistently lower values after the mid-1990s.

When restricted to the satellite era (1993–2025, Fig. 1b), the trend steepens, though the shorter record makes it more sensitive to the choice of start and end dates. This period dependence underscores the importance of the extended ORAS5 record for separating forced trends from multi-decadal variability.

### 3.2 Reanalysis Validation Against RAPID

The RAPID array provides the most direct test of whether reanalyses capture MOC variability. Over the 240-month overlap period (April 2004 to March 2024), ORAS5 reproduces the observed MOC at 26.5°N with a correlation of r = 0.75 (p < 0.001), while GLORYS12V1 achieves r = 0.71 (p < 0.001) (Fig. 2). Both products capture the seasonal cycle and the major interannual excursions, including the pronounced 2009–2010 minimum when the observed MOC dropped below 10 Sv.

The RAPID mean transport is 17.0 Sv, compared to 14.3 Sv for ORAS5 and 16.0 Sv for GLORYS12 over the same period. The low bias in ORAS5 may reflect its coarser resolution, which underestimates the Florida Straits contribution. GLORYS12's higher resolution yields a mean closer to observations.

ORAS5 MOC at both 26.5°N and 34.5°S shows long-term declining trends (Fig. 4), with the 34.5°S decline (−0.055 Sv yr⁻¹, p < 0.001) steeper than at 26.5°N. However, the Southern Hemisphere observing network is substantially sparser, and reanalysis skill at 34.5°S is likely lower than at 26.5°N. This limitation should be borne in mind when interpreting the F_ovS trend, which is computed at this poorly observed latitude.

### 3.3 Salinity Pile-Up

The salinity pile-up index shows a highly significant positive trend of +0.0062 PSU yr⁻¹ (p < 0.001) over 1993–2025 (Fig. 3a), indicating progressive salt accumulation in the subtropical South Atlantic relative to the Indo-Pacific. This is consistent with reduced northward salt advection by a weakening upper AMOC limb (Latif et al., 2022).

The annual-mean F_ovS and salinity pile-up are correlated (Fig. 3b), as expected from the underlying physical mechanism: when F_ovS becomes more negative (more freshwater exported, less salt), salt accumulates in the source region. The correlation supports a common dynamical origin rather than independent regional processes.

### 3.4 North Atlantic Warming Hole and Gulf Stream Destabilization

The NAWH index, defined as the subpolar SST anomaly relative to the global mean, averages −1.05°C over 1993–2025 but does not show a statistically significant trend (p = 0.81). This is consistent with Keil et al. (2020), who attributed roughly half the observed NAWH to strengthened westerlies rather than reduced ocean heat transport.

The Gulf Stream destabilization longitude averages −46.7°E with no significant linear trend (p = 0.15) over the satellite era. The large interannual variability (standard deviation 3.2°) complicates trend detection in this relatively short record.

Neither the NAWH nor the Gulf Stream destabilization point provides strong evidence for or against AMOC weakening over the available period. This does not mean these quantities are uninformative—they may require longer records or may respond nonlinearly to AMOC changes—but it does mean that our confidence in the weakening signal rests primarily on the F_ovS trend and the salinity pile-up.

### 3.5 SSS Trend Decomposition

#### 3.5.1 The amplification hypothesis fails in the Atlantic

The amplification model predicts that SSS trends should scale with climatological SSS—regions that are already salty should get saltier under greenhouse warming, and fresh regions fresher (Held & Soden, 2006). This prediction holds globally and particularly in the Pacific and Indian Oceans (Durack et al., 2012).

In the Atlantic, however, the relationship breaks down. Regressing zonal-mean SSS trends against zonal-mean climatological SSS across 25 latitude bands yields R² = 0.06 (p = 0.24) for GLORYS12V1 over 1993–2025 (Fig. 5a). The hydrological cycle intensification—a real and well-documented phenomenon—explains less than 10% of the variance in Atlantic SSS trends. The remaining 90% must be attributed to other processes.

#### 3.5.2 The residual reveals an overturning-dipole pattern

The pixel-level residual (observed trend minus amplification-predicted trend) shows organized spatial structure (Fig. 5b). The subtropical South Atlantic displays strong positive residuals: observed salinification of +0.084 PSU decade⁻¹ exceeds the amplification prediction of +0.030 PSU decade⁻¹, leaving a residual of +0.054 PSU decade⁻¹. This means 64% of the observed South Atlantic salinification cannot be explained by changes in the hydrological cycle.

North of approximately 40°N, the residuals become negative, indicating freshening beyond what the amplification model predicts. The zero-residual contour roughly follows the inter-gyre boundary, separating the accumulation zone to the south from the depletion zone to the north.

This north-south dipole is the spatial fingerprint of reduced meridional overturning: when the AMOC weakens, salt that would normally be advected northward accumulates in the South Atlantic, while the subpolar region receives less salty water from the subtropics (Zhu & Liu, 2020).

#### 3.5.3 Implications for attribution

The failure of the amplification test constrains the attribution of Atlantic SSS trends. If these trends were a passive response to the water cycle, the regression would be strong. Instead, the dominant signal requires active ocean circulation changes. At the same time, the positive (though weak) regression slope confirms that the hydrological cycle contributes a baseline anthropogenic signal. The Atlantic SSS trends thus contain two superimposed anthropogenic components: a thermodynamic one from the water cycle and a dynamical one from changing circulation.

#### 3.5.4 Product sensitivity

We note that ORAS5 SSS trends over the common 1993–2025 period show a qualitatively different spatial pattern from GLORYS12, with horizontal banding artifacts and a slight freshening (−0.014 PSU decade⁻¹) in the subtropical South Atlantic where GLORYS12 shows salinification (+0.084 PSU decade⁻¹). The GLORYS12 pattern is more consistent with in-situ Argo observations and with the salinity pile-up independently computed from area-averaged indices. We therefore present the GLORYS12 decomposition as the primary result, while acknowledging that SSS trend fields are sensitive to reanalysis product choice.

---

## 4. Discussion

### 4.1 What the Reanalysis Record Supports

The most robust finding of this study is the long-term decline of F_ovS, which has persisted across the full ORAS5 record from 1958 to 2025 with a trend of −1.31 mSv yr⁻¹. The negative mean (−0.033 Sv) confirms that the Atlantic currently resides in the bistable regime where the salt-advection feedback is self-amplifying (Rahmstorf, 1996). This is corroborated by the salinity pile-up, which shows a coherent and significant accumulation trend over the satellite era.

The SSS decomposition adds a complementary line of evidence. By showing that Atlantic salinity redistribution is not a passive response to the water cycle but requires active circulation changes, it links the observed salinity anomalies to reduced meridional overturning. The spatial structure of the residual—salt accumulation in the South Atlantic and freshening in the subpolar region—matches the theoretical expectation for AMOC weakening (Zhu & Liu, 2020).

### 4.2 What the Record Does Not Support

Not all fingerprints point in the same direction. The NAWH index shows no significant trend over 1993–2025, and the Gulf Stream destabilization point lacks a detectable linear trend. While neither result contradicts the weakening hypothesis (the NAWH is confounded by atmospheric forcing, and the Gulf Stream metric may respond nonlinearly), the lack of supporting evidence from these indicators means that the "multi-fingerprint convergence" argument is weaker than it would be if all four metrics agreed.

We also stress that reanalysis skill is latitude-dependent. ORAS5 reproduces MOC variability well at 26.5°N (r = 0.75), where the observing network is dense. At 34.5°S—the latitude where F_ovS is computed—the observing network is substantially sparser, and reanalysis skill is likely lower. The F_ovS trend is therefore computed at a latitude where independent validation remains limited.

### 4.3 Product Disagreement on SSS Trends

The divergence between ORAS5 and GLORYS12 SSS trends in the South Atlantic deserves particular attention. GLORYS12 shows salinification consistent with the salinity pile-up and with Argo float observations, while ORAS5 shows slight freshening with spatial artifacts. This disagreement likely reflects differences in how the two systems assimilate near-surface salinity data: GLORYS12's higher resolution and Kalman-filter-based scheme may better represent the mesoscale salinity field.

This sensitivity is a cautionary finding. Previous studies that diagnosed AMOC weakening from SSS trends in a single reanalysis product may be less robust than assumed. Our use of two products reveals this uncertainty and leads us to present conclusions about SSS trends as product-dependent rather than universal.

### 4.4 Shared Forcing and the Independence Problem

Both reanalyses are forced by ERA-family atmospheric products and assimilate overlapping observation sets. Agreement between ORAS5 and GLORYS12 therefore provides only partial independence. A systematic bias in ERA5 wind stress, for instance, could produce correlated ocean circulation trends in both products. Truly independent verification requires comparison with observational arrays (RAPID, SAMBA, OSNAP) or with reanalyses forced by non-ERA atmospheric products.

### 4.5 Attribution: Forced Signal Versus Internal Variability

The most fundamental challenge to interpreting F_ovS trends is whether the observed decline reflects a forced response to anthropogenic freshwater input, or simply the downswing of the Atlantic Multidecadal Variability (AMV). We address this through three complementary tests.

First, the AMV index (Kaplan SST-based AMO; Enfield et al., 2001) transitioned from a negative phase during 1965–1994 (mean −0.19°C) to a positive phase during 1998–2022 (mean +0.16°C). If F_ovS were tracking the AMV, it should have recovered. Instead, F_ovS continued its decline, falling from a mean of −7.8 mSv during the negative AMV phase to −59.7 mSv during the positive phase (Fig. 5a). The annual correlation between F_ovS and the AMO is r = −0.63 over the full 1958–2022 period, but drops to r = 0.10 (p = 0.60) after 1995. The two quantities have decoupled.

Second, regressing out the AMO-correlated component of F_ovS leaves a residual trend of −0.74 mSv yr⁻¹ (p < 10⁻⁷), representing 57% of the raw trend. This AMO-independent decline cannot be attributed to internal Atlantic variability.

Third, a block bootstrap test (10,000 iterations, 5-year blocks to preserve autocorrelation) yields a 95% null range of [−0.73, +0.72] mSv yr⁻¹. The observed trend of −1.31 mSv yr⁻¹ falls outside the 99.99th percentile of the null distribution (p = 0.0001). The record's own variability structure cannot produce a trend this steep by chance.

Together, these tests indicate that the F_ovS decline contains a substantial forced component that persists after accounting for AMV influence. The SSS decomposition (Section 3.5) provides an independent line of evidence for the same conclusion: the Atlantic salinity redistribution pattern requires both hydrological cycle intensification (anthropogenic) and circulation changes (AMOC weakening), neither of which is consistent with purely internal variability.

### 4.6 Implications for Tipping Risk

Van Westen et al. (2024) demonstrated in a 4,400-year CESM simulation that F_ovS reaches a minimum approximately 25 years before AMOC collapse. Our observed F_ovS trajectory is consistent with the theoretical prediction in sign and approximate magnitude of the decline rate. However, we cannot determine the system's distance from the tipping threshold because the absolute value of F_ovS is model-dependent and sensitive to the reference salinity, grid resolution, and the inclusion or exclusion of the barotropic component.

What the reanalysis record does establish is that the trajectory is declining. Combined with the South Atlantic salinity pile-up—which provides a higher signal-to-noise indicator of the same underlying process—this constitutes evidence that the salt-advection feedback is currently active. Whether this feedback will drive the system to collapse depends on the rate and duration of high-latitude freshwater forcing, which is determined by future emissions.

---

## 5. Conclusions

Using ORAS5 (1958–2025) and GLORYS12V1 (1993–2025), we assess multiple AMOC weakening fingerprints and validate against RAPID observations. Our main findings are:

1. F_ovS at 34.5°S has a mean of −0.033 Sv and a trend of −1.31 mSv yr⁻¹ (p < 0.001) over 68 years, placing the Atlantic in the theoretically bistable regime with an actively declining freshwater transport.

2. Both reanalyses reproduce observed MOC variability at 26.5°N (ORAS5: r = 0.75; GLORYS12: r = 0.71), but skill deteriorates at 34.5°S where the observing network is sparse.

3. The salinity pile-up index shows a significant positive trend (+0.006 PSU yr⁻¹, p < 0.001) over the satellite era, consistent with reduced northward salt advection.

4. The NAWH and Gulf Stream destabilization point do not show significant trends over 1993–2025, limiting the case for multi-fingerprint convergence.

5. Atlantic SSS trends are not explained by hydrological cycle intensification (R² = 0.06, p = 0.24). The residual pattern—South Atlantic salinification exceeding the amplification prediction by 64%, subpolar freshening—is diagnostic of reduced meridional overturning.

6. ORAS5 and GLORYS12 disagree on the sign of South Atlantic SSS trends, indicating that SSS-based conclusions are product-sensitive.

These results indicate that the salt-advection feedback mechanism underlying theoretical AMOC tipping is detectable in the reanalysis record, but that not all proposed fingerprints provide supporting evidence over the available time period. Continued monitoring through the RAPID, SAMBA, and OSNAP arrays, together with ongoing reanalysis development, will be essential for constraining the trajectory of this critical component of the climate system.

The ARDP analysis pipeline (https://github.com/bijanf/ardp) enables replication and continuous updating of these diagnostics.

---

## References

Boers, N. (2021). Observation-based early-warning signals for a collapse of the AMOC. Nature Climate Change, 11, 680–688.

Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., & Bladé, I. (1999). The effective number of spatial degrees of freedom of a time-varying field. Journal of Climate, 12, 1990–2009.

Buckley, M. W., & Marshall, J. (2016). Observations, inferences, and mechanisms of the Atlantic Meridional Overturning Circulation: A review. Reviews of Geophysics, 54, 5–63.

Cunningham, S. A., et al. (2007). Temporal variability of the Atlantic meridional overturning circulation at 26.5°N. Science, 317, 935–938.

Durack, P. J., & Wijffels, S. E. (2010). Fifty-year trends in global ocean salinities and their relationship to broad-scale warming. Journal of Climate, 23, 4342–4362.

Durack, P. J., Wijffels, S. E., & Matear, R. J. (2012). Ocean salinities reveal strong global water cycle intensification during 1950 to 2000. Science, 336, 455–458.

Fu, Y., et al. (2025). AMOC has not declined over the last 60 years. Science.

Ganachaud, A., & Wunsch, C. (2000). Improved estimates of global ocean circulation, heat transport and mixing from hydrographic data. Nature, 408, 453–457.

Held, I. M., & Soden, B. J. (2006). Robust responses of the hydrological cycle to global warming. Journal of Climate, 19, 5686–5699.

Jean-Michel, L., et al. (2021). The Copernicus global 1/12° oceanic and sea ice GLORYS12 reanalysis. Frontiers in Earth Science, 9, 698876.

Keil, P., et al. (2020). Multiple drivers of the North Atlantic warming hole. Nature Climate Change, 10, 667–671.

Latif, M., et al. (2022). Likely accelerated weakening of Atlantic overturning circulation emerges in optimal salinity fingerprint. Nature Climate Change, 12, 1106–1113.

Meinen, C. S., et al. (2018). Meridional overturning circulation transport variability at 34.5°S during 2009–2017. Journal of Geophysical Research: Oceans, 123, 4803–4821.

Rahmstorf, S. (1996). On the freshwater forcing and transport of the Atlantic thermohaline circulation. Climate Dynamics, 12, 799–811.

Rahmstorf, S. (2002). Ocean circulation and climate during the past 120,000 years. Nature, 419, 207–214.

Thompson, M. A., et al. (2025). Atlantic Ocean current expected to undergo limited weakening. Nature.

van Westen, R. M., et al. (2024). Physics-based early warning signal shows that AMOC is on tipping course. Science Advances, 10, eadk1189.

Wunsch, C. (2018). Towards determining uncertainties in global oceanic mean values of heat, salt, and surface elevation. Tellus A, 70, 1–14.

Zhu, C., & Liu, Z. (2020). Weakening Atlantic overturning circulation causes South Atlantic salinity pile-up. Nature Climate Change, 10, 998–1003.

Zuo, H., et al. (2019). The ECMWF operational ensemble reanalysis-analysis system for ocean and sea ice: OCEAN5. Geoscience Model Development, 12, 3287–3312.

---

## Figure Captions

**Figure 1.** Overturning freshwater transport at 34.5°S (F_ovS) from ORAS5 (1958–2025, 816 months). Monthly values (light blue) with 12-month running mean (dark blue). Two linear trends are overlaid: the full-record trend (dashed red; −1.31 mSv yr⁻¹, p < 0.001) and the satellite-era trend (dashed purple; −1.43 mSv yr⁻¹, p < 0.001). The steeper satellite-era trend is consistent with an accelerating decline, though the shorter period is more sensitive to start/end date selection. The horizontal dotted line marks F_ovS = 0; negative values indicate the bistable regime where the salt-advection feedback is self-amplifying.

**Figure 2.** Validation of reanalysis MOC against RAPID observations at 26.5°N. (a) Monthly time series of upper-cell MOC transport: RAPID observations (red), ORAS5 (blue), and GLORYS12V1 (green). Thin lines show monthly values; thick lines show 12-month running means. (b) ORAS5 versus RAPID scatter (r = 0.75, n = 240 months). (c) GLORYS12 versus RAPID scatter (r = 0.71, n = 240 months). Dashed lines show the regression fit; dotted lines show the 1:1 line.

**Figure 3.** Salinity pile-up index and its relationship to F_ovS. (a) Monthly salinity pile-up (SSS differential between the subtropical South Atlantic and subtropical South Indo-Pacific), 1993–2025. Linear trend: +0.006 PSU yr⁻¹ (p < 0.001). (b) Annual-mean F_ovS versus salinity pile-up, showing the expected anti-correlation: more negative F_ovS corresponds to increasing salt accumulation.

**Figure 4.** ORAS5 MOC upper-cell transport at 26.5°N (blue) and 34.5°S (green) over 1958–2025. Monthly values (faint) with 12-month running means (solid) and linear trends (dashed). The horizontal dotted line marks the RAPID observational mean (17.0 Sv). Both latitudes show long-term declining trends, with the decline steeper at 34.5°S.

**Figure 5.** Decomposition of Atlantic SSS trends into hydrological cycle and ocean dynamics components (GLORYS12V1, 1993–2025). (a) Zonal-mean climatological SSS versus zonal-mean SSS trend across 25 latitude bands, coloured by latitude. The dashed line shows the amplification regression (R² = 0.06, p = 0.24), which predicts that SSS trends should mirror the mean pattern under greenhouse-driven water cycle intensification. The weak relationship indicates that ocean dynamics, not the hydrological cycle, dominate Atlantic SSS trends. (b) Residual SSS trend after removing the amplification-predicted component. Positive residuals (red) indicate salinification exceeding atmospheric expectations; negative residuals (blue) indicate anomalous freshening. The prominent positive residual in the subtropical South Atlantic corresponds to the AMOC-driven salinity pile-up. Contour lines highlight regions where residuals exceed ±0.04 PSU decade⁻¹; the dashed line marks zero residual.

**Figure 5.** Attribution of the F_ovS decline. (a) Annual-mean F_ovS (blue, left axis) and AMO index (red, right axis) with 10-year running means. Light red shading highlights the positive AMV phase (1998–present). Despite the AMO recovering to positive values, F_ovS continues to decline, indicating decoupling from internal Atlantic variability. (b) F_ovS after statistically removing the AMO-correlated component (purple). The residual trend (−0.74 mSv yr⁻¹, p < 10⁻⁷) represents 57% of the raw trend, indicating a substantial forced signal independent of the AMV. Raw annual values (grey dots) and raw trend (dashed blue) shown for comparison. (c) Block bootstrap null distribution (5-year blocks, 10,000 iterations) of the F_ovS trend. The observed trend (red line, −1.31 mSv yr⁻¹) falls outside the 99.99th percentile of the null distribution (dashed lines mark 95% CI), indicating that the record's own autocorrelation structure cannot produce a trend this steep by chance.

**Figure 6.** GLORYS12V1 SSS trend map (1993–2025). (a) Per-pixel deseasonalized linear trend (PSU decade⁻¹) with significance stippling (hatching where p ≥ 0.05). Contour lines trace smoothed ±0.08 PSU decade⁻¹ thresholds. (b) Atlantic zonal-mean SSS trend profile with 95% confidence intervals adjusted for spatial autocorrelation (N_eff; Bretherton et al., 1999). Solid segments indicate statistically significant trends; dashed segments are not significant.
