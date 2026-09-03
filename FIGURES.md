# Figure to script mapping

Every figure in the manuscript is produced by one script in `scripts/`, run from
the repository root against the derived results in `data/results/`.

```bash
pip install -e ".[dev]"
python scripts/<script>.py          # writes into figures/
```

## Main text

| Figure | Subject | Script |
| --- | --- | --- |
| 1 | Mechanism-conditional AMOC projections, basin-mean salinity, `F_ovS` time series, Cold Blob | `plot_paper2_Figure1.py` |
| 2 | Observational grounding: EN4 zonal section and upper-300 m salinity trends | `plot_obs_grounding.py` |
| 3 | CMIP6 mechanism shares, mechanism-conditional AMOC, weakening by 2100 | `plot_paper2_Figure2.py` |
| 4 | AMOC projections restricted to the bistable subset | `plot_paper2_Figure3.py` |
| 5 | Robustness: Argo-era windows, period sensitivity, piControl null | `plot_paper2_Figure4.py` |
| 6 | Zonal structure of the velocity and salinity changes at 34.5S | `plot_paper2_Figure5.py` |
| 7 | Lead-lag, emergent regression, and large-ensemble robustness | `plot_paper2_Figure6.py` |

## Supplementary

| Figure | Subject | Script |
| --- | --- | --- |
| S1 | Within-class emergent regression test | `diagnostic_a2_within_class_regression.py` |
| S2 | Continuous velocity share against projected weakening | `diagnostic_a3_continuous_correlation.py` |
| S3 | Timescale consistency of the mechanism class | `diagnostic_a1_timescale_consistency.py` |
| S4 | Joint sensitivity to threshold and floor | `diagnostic_a4_joint_sensitivity.py` |
| S5 | Bootstrap distribution of the class-median gap | `diagnostic_a5_gap_bootstrap.py` |
| S6 | Permutation null for the mechanism-class label | `robustness_permutation_and_bootstrap.py` |
| S7 | Lead-lag cross-correlation split by mechanism class | `plot_paper2_figS1_leadlag.py` |
| S8 | Salinity component against projected AMOC weakening | `analysis_r1_salinity_amoc_coupling.py` |
| S9 | Observation-constrained salinity-driven component | `plot_obs_hybrid_dfs.py` |

## Tables

| Table | Script |
| --- | --- |
| 2 (net volume transport) | `compute_paper2_tableS2_vnet.py` |
| S1 (model selection) | `compute_paper2_tableS1.py` |

The `F_ovS` kernel shared by every one of these is `ardp/physics/fovs.py`, and its
identity and barotropic-subtraction properties are covered by
`tests/test_fovs_decomposition.py`.
