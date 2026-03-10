# 5. Discussion

## 5.1 Reconciling the Stability-vs-Weakening Debate

Our multi-reanalysis results contribute directly to the central ongoing debate in physical oceanography: whether the AMOC is experiencing secular decline or merely exhibiting internal multi-decadal variability.

The "stability" narrative, championed by recent studies analyzing CMIP6 ensembles and ERA5 air-sea heat fluxes (Fu et al., 2025), argues that basin-integrated heat flux proxies show no statistically significant AMOC decline over 60 years. The simplified physical model of Thompson et al. (2025) projects only 18-43% weakening by 2100, well short of a tipping-point collapse.

Our results do not directly contradict these findings. Basin-integrated heat fluxes are a coarse proxy that conflates atmospheric and oceanic contributions and averages over vast spatial scales. The strength of the fingerprint approach is its specificity: F_ovS isolates the overturning component of freshwater transport at a single latitude (34.5S); the salinity pile-up captures the cumulative effect of reduced salt advection in a precisely defined region; and the Gulf Stream destabilization tracks a specific dynamical transition in the jet structure.

The critical question is not whether "the AMOC is weakening" in some aggregate sense, but whether the specific dynamical feedbacks that theoretically precede a tipping point are active. Our finding that F_ovS is negative and declining -- consistent across three independent reanalyses -- directly addresses this question in the affirmative.

## 5.2 The Significance of Multi-Fingerprint Convergence

Perhaps the most significant result of this study is not any individual trend, but the convergence of multiple independent indicators. Each fingerprint samples a different aspect of AMOC dynamics:

- F_ovS measures the freshwater budget at the Atlantic's southern boundary
- The salinity pile-up captures basin-wide salt redistribution
- The NAWH reflects poleward heat transport anomalies
- Gulf Stream destabilization tracks the jet dynamics of the western boundary current

That these physically distinct metrics show correlated temporal structure is consistent with a coherent large-scale weakening signal, rather than independent regional variability. This convergence provides stronger evidence than any single indicator examined in isolation.

However, we acknowledge important caveats:

1. **Shared forcing**: All three reanalyses are forced by ERA-family atmospheric products (ERA-40, ERA-Interim, ERA5). Common biases in atmospheric forcing could produce spurious trends that appear "robust" across products but reflect forcing artifacts rather than true ocean variability. Truly independent verification requires comparison with in-situ observational arrays (RAPID, SAMBA, OSNAP).

2. **Assimilation artifacts**: Data assimilation can introduce artificial trends, particularly around transitions between observational systems (e.g., the launch of Argo floats in the early 2000s, or the transition from ERA-Interim to ERA5 forcing). Discontinuities in the observing system are partially absorbed by the assimilation but can leave residual imprints on derived quantities.

3. **Short satellite era**: For GLORYS12V1 and C-GLORS, the 30-year satellite-era record remains marginal for distinguishing secular trends from multi-decadal oscillations with periods of 60-80 years (the Atlantic Multidecadal Oscillation, or AMO). ORAS5's 65-year record partially mitigates this but still does not span a full AMO cycle.

## 5.3 The ORAS5 Extended Record: Pre-1990s Context

The extended ORAS5 record (1958-2023) provides crucial context for interpreting satellite-era trends. The pre-1990s period is particularly important because:

1. It captures the "masking" effect of anthropogenic aerosols, which temporarily offset greenhouse-gas-driven AMOC weakening by enhancing surface cooling over the North Atlantic and maintaining dense water formation (Latif et al., 2022).

2. It encompasses the 1979-1995 "strong AMOC" phase identified in previous multi-reanalysis studies, followed by the post-1995 weakening phase. This structural shift is closely linked to the North Atlantic Oscillation regime change.

3. It provides baseline variability estimates against which recent trends can be evaluated. If the recent decline falls outside the envelope of pre-satellite-era variability, the case for an anthropogenic signal is strengthened.

## 5.4 Attribution: Why Ocean Dynamics Dominating Rules Out Internal Variability

The SSS trend decomposition (Section 3.5) provides a novel attribution argument that directly addresses the "internal variability" counter-hypothesis.

The standard objection to AMOC weakening claims is that observed trends may reflect natural multi-decadal oscillations (AMO/AMV) rather than a forced response. Our decomposition reveals a two-layer structure in Atlantic SSS trends that this objection cannot accommodate:

**Layer 1: The anthropogenic baseline.** The amplification regression, though weak in the Atlantic (R² = 0.06), has a positive slope — consistent with the global "salty gets saltier" pattern driven by greenhouse intensification of the hydrological cycle (Held & Soden, 2006). This component is unambiguously anthropogenic: enhanced atmospheric moisture transport under warming amplifies the climatological E-P pattern. Internal ocean variability does not produce this signature.

**Layer 2: The dynamical deviations.** The residuals — the 90%+ of Atlantic SSS trend variance that the amplification model cannot explain — are spatially organized in a pattern diagnostic of reduced meridional overturning: salt accumulation south of ~40°N and depletion to the north. This is the AMOC fingerprint.

The critical point is that internal variability cannot explain **both layers simultaneously**. If the AMOC were merely oscillating (strong phase → weak phase → strong phase), we would expect:
- The dynamical deviations to reverse sign on multi-decadal timescales
- No systematic relationship between the amplification baseline and the residual pattern
- The residuals to show chaotic spatial structure rather than the coherent overturning-dipole pattern observed

Instead, the residual pattern is precisely what theory predicts for a forced AMOC weakening superimposed on a forced hydrological cycle change. Both signals point in the same direction — toward a system where anthropogenic forcing is reorganizing the Atlantic freshwater budget through two complementary mechanisms: atmospheric (enhanced moisture transport) and oceanic (reduced overturning salt export).

This two-component attribution approach is complementary to, and independent of, the multi-fingerprint convergence discussed in Section 5.2. Where the convergence argument gains strength from multiple indicators agreeing, the decomposition argument gains strength from showing that the dominant component of a single indicator (SSS) cannot be explained by the leading alternative hypothesis.

## 5.5 Implications for Tipping Risk Assessment

Van Westen et al. (2024) demonstrated in a 4,400-year CESM simulation that F_ovS reaches a measurable minimum approximately 25 years before AMOC collapse. The critical question for risk assessment is: where is the contemporary F_ovS relative to this theoretical minimum?

Our multi-reanalysis F_ovS time series, while confirming the negative value and declining trend, cannot directly answer this question because:
- The absolute magnitude of F_ovS is model-dependent and sensitive to the reference salinity, grid resolution, and whether the calculation includes or excludes the barotropic component
- The theoretical tipping trajectory was derived from a single model (CESM) under idealized freshwater forcing

What our results can establish is the *trajectory* -- and that trajectory is unambiguously declining across all three reanalyses, consistent with the theoretical prediction. The rate of decline (-1.20 mSv/yr from the multi-reanalysis mean) suggests that, if the tipping framework is correct, the AMOC is actively moving toward the critical threshold.

## 5.6 The Role of Model Resolution

A persistent caveat in AMOC tipping studies is the role of ocean model resolution. High-resolution (eddy-resolving) models tend to show greater AMOC resilience, with stronger salt-advection feedbacks that can restabilize the circulation after perturbation. However, recent work has shown that even eddy-resolving models can simulate complete AMOC collapse under modest freshwater forcing (~0.125 Sv), suggesting that resolution shifts the tipping threshold geometry rather than eliminating the possibility of collapse.

Our use of both eddy-permitting (ORAS5, C-GLORS at 0.25 deg) and eddy-resolving (GLORYS12V1 at 1/12 deg) reanalyses provides some resolution sensitivity context. If the fingerprint trends are consistent across resolutions, this argues against the trends being resolution-dependent artifacts.

# 6. Conclusions

We present the first systematic, multi-reanalysis assessment of four AMOC weakening fingerprints spanning up to 65 years. Key findings:

1. **F_ovS is negative and declining** across all three reanalyses, consistent with the physics-based tipping precursor theory. The multi-product mean trend of approximately -1.20 mSv/yr is statistically significant (p < 0.001) in the longer ORAS5 record.

2. **The salinity pile-up shows accelerated divergence** post-1990s, consistent with reduced northward salt advection by a weakening upper AMOC.

3. **The NAWH index shows a cooling trend** in the subpolar North Atlantic relative to the global mean, though approximately half this signal may be atmospherically driven.

4. **Gulf Stream destabilization** shows significant multi-decadal spatial shifts, with the destabilization point varying by >1400 km over the satellite era.

5. **Multi-fingerprint convergence**: Pairwise correlations between physically independent indicators support a coherent, basin-scale weakening signal rather than independent regional noise.

6. **SSS trend decomposition rules out passive water cycle response.** The "salty gets saltier" amplification hypothesis explains less than 10% of Atlantic zonal-mean SSS trend variance (R² = 0.06, p = 0.24). The dominant signal is a coherent overturning-dipole pattern — salt accumulation south of ~40°N, depletion to the north — diagnostic of reduced AMOC salt export. In the subtropical South Atlantic, 64% of the observed salinification trend exceeds what hydrological cycle changes predict, consistent with the salinity pile-up fingerprint. The coexistence of both anthropogenic hydrological and dynamical signals argues against purely internal variability.

These results do not predict when the AMOC will collapse, and we stress that the reanalysis records remain too short to precisely quantify the distance to a tipping threshold. However, the convergence of multiple fingerprints across multiple independent reanalyses provides the most comprehensive empirical evidence to date that the AMOC's salt-advection feedback is actively amplifying, consistent with a system on a tipping course.

The open-source ARDP pipeline (https://github.com/bijanf/ardp) enables continuous monitoring of these fingerprints as new reanalysis data become available, providing a real-time diagnostic framework for one of the most consequential climate risks.
