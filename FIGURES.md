# Figure ↔ Code mapping

This file maps every figure in the two manuscripts that use this code
to the script that produces it. PNG previews of the cited figures live
under `figures/grl/` (Paper 1) and `figures/paper2/` (Paper 2). The
PDF versions used for typesetting are preserved at git tag
`paper2-v4.4.1` and recoverable with:

```bash
git checkout paper2-v4.4.1 -- figures/
```

The full manuscripts (`paper/`, `paper2/`, `paper2_overleaf/`) live in
a separate private archive; this code repo only carries the figures
and the scripts that build them.

## Paper 1 — "The Salt-Advection Feedback Is Now Active in the Atlantic Ocean" (GRL submission)

| Fig | Subject | Preview | Script |
| --- | --- | --- | --- |
| 1 | F_ovS at 34.5°S from ORAS5 (1958–2025) and GLORYS12V1 (1993–2025), plus mean Atlantic streamfunction Ψ(φ,z) | `figures/grl/fig1_fovs_multiproduct.png` | `scripts/plot_grl_figures.py` (Fig 1 panel) |
| 2 | RAPID-array validation of reanalysis MOC at 26.5°N (ORAS5 r=0.75, GLORYS12 r=0.71) | `figures/grl/fig2_rapid_validation.png` | `scripts/plot_grl_figures.py` (Fig 2 panel) |
| 3 | AMOC anomalies at 26.5°N for 4 reanalyses + RAPID, with Santer N_eff–corrected trends | `figures/grl/fig_amoc_reanalysis_anomalies_santer.png` | `scripts/plot_amoc_reanalysis_anomalies.py` |
| 4 | GLORYS12V1 SSS trend map (1993–2025), zonal-mean profile, and salinity pile-up index | `figures/grl/fig_sss_trend_map.png` | `scripts/plot_sss_trend_map.py --product glorys12` |
| 5 | Decomposition of Atlantic SSS trends — hydrological cycle vs ocean dynamics (R²=0.06) | `figures/grl/fig_sss_decomposition.png` | `scripts/plot_sss_decomposition.py --product glorys12` |
| 6 | CMIP6 vs ORAS5/GLORYS12 F_ovS — 30-yr boxplots and historical-mean ranking (22 models) | `figures/grl/fig4_cmip6_comparison.png` | `scripts/plot_cmip6_fovs_trajectory.py` |
| 7 | Attribution of F_ovS decline — AMO partial regression (residual −0.74 mSv/yr) and bootstrap null | `figures/grl/fig_attribution.png` | `scripts/plot_attribution.py` |

## Paper 2 — "Atlantic bistability and opposing mechanisms for freshening" (v4.4.1, Nature Comms target)

Each main figure assembles sub-panels (a/b/c/…) into a single
combined PDF. Running `plot_paper2_FigureN.py` produces both the
panels and the combined version under `figures/paper2/`.

| Fig | Subject | Preview | Script |
| --- | --- | --- | --- |
| 1 | F_ovS time series at 34.5°S across 4 reanalyses + decomposition into ΔF_v, ΔF_s, ΔF_cross + vertical integrand profiles | `figures/paper2/Figure1.png` | `scripts/plot_paper2_Figure1.py` |
| 2 | CMIP6 mechanism classification (25 models) — velocity-share vs salinity-share scatter, mechanism-conditional AMOC26N trajectories, projected weakening (salinity 54% / velocity 42% / 12 pp gap) | `figures/paper2/Figure2.png` | `scripts/plot_paper2_Figure2.py` |
| 3 | AMOC projections for the bistable-only subset (6 forced-weakening bistable models) | `figures/paper2/Figure3.png` | `scripts/plot_paper2_Figure3.py` |
| 4 | Robustness — post-Argo split, 25 early/late window pairs, signal-to-noise vs piControl | `figures/paper2/Figure4.png` | `scripts/plot_paper2_Figure4.py` |
| 5 | Zonal structure of Δv and ΔS at 34.5°S (ORAS5 vs GLORYS12) | `figures/paper2/Figure5.png` | `scripts/plot_paper2_Figure5.py` |
| 6 | F_ovS↔AMOC cross-correlation, emergent regression (n=21, ΔAMOC=−1.82 Sv [−3.68, +0.04]), and MPI-ESM SMILE robustness | `figures/paper2/Figure6.png` | `scripts/plot_paper2_Figure6.py` |

### Paper 2 SI — diagnostics A1–A5

| Diag | Subject | Preview | Script |
| --- | --- | --- | --- |
| A1 | Timescale-consistency check of the mechanism partition | `figures/paper2/diagA1_timescale.png` | `scripts/diagnostic_a1_timescale_consistency.py` |
| A2 | Within-class regression of f_v vs ΔAMOC | `figures/paper2/diagA2_within_class.png` | `scripts/diagnostic_a2_within_class_regression.py` |
| A3 | Continuous correlation f_v vs ΔAMOC% (ρ=−0.56, p=0.018, n=17) | `figures/paper2/diagA3_continuous.png` | `scripts/diagnostic_a3_continuous_correlation.py` |
| A4 | Joint sensitivity of the mechanism gap to threshold + window choices | `figures/paper2/diagA4_joint_sensitivity.png` | `scripts/diagnostic_a4_joint_sensitivity.py` |
| A5 | Bootstrap test of the salinity-vs-velocity ΔAMOC gap (11.6 pp, p=0.026, 95% CI [−0.7, +24.7] pp) | `figures/paper2/diagA5_gap_bootstrap.png` | `scripts/diagnostic_a5_gap_bootstrap.py` |

## Other plot scripts

These scripts produce supplementary or diagnostic figures that were
explored during analysis but are not cited in the v4.4.1 manuscripts.
They remain in the repo because they consume the same upstream
products and may resurface in revisions.

| Script | Output |
| --- | --- |
| `scripts/plot_paper2_fig1_multiprod_fovs.py` | `figures/paper2/fig1_multiprod_fovs.{pdf,png}` (older v2 four-product F_ovS) |
| `scripts/plot_paper2_fig2_decomposition.py` | `figures/paper2/fig2_decomposition.{pdf,png}` (older v2 v/s decomposition) |
| `scripts/plot_paper2_fig3_tiebreaker.py` | `figures/paper2/fig3_tiebreaker.{pdf,png}` (older v2 CMIP6 tie-breaker) |
| `scripts/plot_paper2_fig4_mechanism_conditional.py` | `figures/paper2/fig4_mechanism_conditional.{pdf,png}` (older v2 mechanism-conditional) |
| `scripts/plot_paper2_fig5_bistable_subset.py` | `figures/paper2/fig5_bistable_subset.{pdf,png}` (older v2 bistable-subset) |
| `scripts/plot_paper2_figS1_leadlag.py` | `figures/paper2/fig3_leadlag.{pdf,png}` (CCF supplementary, superseded by Figure 6a) |
| `scripts/plot_paper2_figS3_zonal_structure.py` | `figures/paper2/figS3_zonal_structure.{pdf,png}` (Δv/ΔS maps, superseded by Figure 5) |
| `scripts/plot_paper2_figS4_signal_noise.py` | `figures/paper2/figS4_signal_noise.{pdf,png}` (forced vs piControl, superseded by Figure 4c) |
| `scripts/plot_paper2_figS6_postargo.py` | `figures/paper2/figS6_postargo.{pdf,png}` (post-Argo robustness, superseded by Figure 4a) |
| `scripts/plot_paper2_FigureSMILE.py` | `figures/paper2/diagSMILE.{pdf,png}` (50-member MPI-ESM SMILE, integrated into Figure 6c) |
| `scripts/plot_paper2_FigureSMILE_AMOC.py` | `figures/paper2/diagSMILE_amoc.{pdf,png}` (SMILE AMOC trajectories, integrated into Figure 6d) |
| `scripts/plot_amoc_strength_comparison.py` | `figures/grl/fig_amoc_strength_26N.{pdf,png}` (mean AMOC strength comparison) |
| `scripts/plot_amoc_rate_comparison.py` | `figures/grl/fig_amoc_rate_comparison.{pdf,png}` (decadal weakening-rate bar chart) |
| `scripts/plot_345s_cross_section.py` | `figures/grl/fig_345s_cross_section.{pdf,png}` (34.5°S section schematic) |
| `scripts/plot_moc_streamfunction_comparison.py` | `figures/grl/fig_moc_streamfunction_comparison.{pdf,png}` (Ψ(φ,z) panels) |
| `scripts/plot_freshwater_flux_maps.py` | `figures/grl/fig_freshwater_flux_maps.{pdf,png}` (surface freshwater flux climatology) |
| `scripts/plot_fovs_decomposition.py` | `figures/grl/fig_fovs_decomposition.{pdf,png}` (older single-product decomposition) |
| `scripts/plot_fovs_multi_study.py` | `figures/grl/fig_fovs_multi_study.{pdf,png}` (multi-study comparison) |
| `scripts/plot_publication_figures.py` | `figures/publication/...` (alternative publication-style fig set) |
| `scripts/plot_fingerprints.py`, `scripts/plot_f_ovs.py`, `scripts/plot_gulf_stream.py` | Diagnostic plots, no fixed output path |
