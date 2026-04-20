#!/usr/bin/env python3
"""One-shot orchestrator for all Paper 2 analysis and figures.

Runs the full pipeline in dependency order:
  1. F_ovS time series for any reanalyses whose .nc is missing
     (ORAS5 / GLORYS12 / SODA / ECCO).  If a product's prerequisite raw
     data is missing, that step is skipped with a warning.
  2. Reanalysis mechanism decomposition (for all products whose time
     series exists).
  3. CMIP6 forced mechanism decomposition + piControl null.
  4. CMIP6 F_ovS-AMOC lead-lag.
  5. Period-sensitivity robustness.
  6. Trend table (CSV + LaTeX).
  7. All four main figures and four supplementary figures.

Designed to be idempotent: each step checks whether its output already
exists and skips if so, unless --force is passed.

Usage:
    python scripts/make_paper2.py            # run only missing steps
    python scripts/make_paper2.py --force    # regenerate everything
    python scripts/make_paper2.py --skip soda  # skip SODA time series
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("figures/paper2")


def _run(cmd: str, check: bool = True) -> int:
    """Run a shell command, echoing it first."""
    log.info(f"  $ {cmd}")
    r = subprocess.run(shlex.split(cmd))
    if check and r.returncode != 0:
        log.error(f"  FAILED with exit code {r.returncode}: {cmd}")
    return r.returncode


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _need_update(output: Path, force: bool) -> bool:
    if force:
        return True
    return not _exists(output)


def step1_fovs_timeseries(force: bool, skip: set) -> None:
    log.info("=== Step 1: F_ovS time series ===")
    steps = [
        ("oras5",    "data/results/oras5_f_ovs.nc",    "python scripts/compute_oras5_fovs.py"),
        ("glorys12", "data/results/glorys12_f_ovs.nc", "python scripts/compute_glorys12_fovs.py"),
        ("soda",     "data/results/soda_f_ovs.nc",     "python scripts/compute_soda_fovs.py"),
        ("ecco",     "data/results/ecco_f_ovs.nc",     "python scripts/compute_ecco_fovs.py"),
    ]
    for name, out, cmd in steps:
        if name in skip:
            log.info(f"  {name}: SKIP (user)")
            continue
        if _need_update(Path(out), force):
            _run(cmd, check=False)
        else:
            log.info(f"  {name}: up-to-date ({out})")


def step2_reanalysis_decomp(force: bool) -> None:
    log.info("=== Step 2: reanalysis mechanism decomposition ===")
    # The underlying script skips products whose raw data is missing.
    if _need_update(RESULTS_DIR / "fovs_decomposition_oras5.nc", force) or \
       _need_update(RESULTS_DIR / "fovs_decomposition_glorys12.nc", force) or \
       _need_update(RESULTS_DIR / "fovs_decomposition_ecco.nc", force):
        _run("python scripts/compute_fovs_decomposition.py --product all", check=False)
    else:
        log.info("  up-to-date")


def step3_cmip6_decomp(force: bool) -> None:
    log.info("=== Step 3: CMIP6 forced mechanism decomposition + piControl null ===")
    if _need_update(RESULTS_DIR / "fovs_decomposition_cmip6_summary.csv", force):
        _run("python scripts/compute_cmip6_fovs_decomposition.py", check=False)
    if _need_update(RESULTS_DIR / "fovs_decomposition_cmip6_null.csv", force):
        _run("python scripts/compute_cmip6_picontrol_null.py --n-bootstrap 200", check=False)


def step4_leadlag(force: bool) -> None:
    log.info("=== Step 4: CMIP6 F_ovS-AMOC lead-lag ===")
    if _need_update(RESULTS_DIR / "cmip6_fovs_amoc_leadlag.nc", force):
        _run("python scripts/compute_cmip6_fovs_amoc_leadlag.py", check=False)


def step5_sensitivity(force: bool) -> None:
    log.info("=== Step 5: period-sensitivity robustness ===")
    if _need_update(RESULTS_DIR / "fovs_decomposition_sensitivity.csv", force):
        _run(
            "python scripts/compute_fovs_decomposition_sensitivity.py "
            "--products oras5 glorys12",
            check=False,
        )


def step6_table(force: bool) -> None:
    log.info("=== Step 6: trend table (CSV + LaTeX) ===")
    if _need_update(RESULTS_DIR / "paper2_tableS1_trends.tex", force):
        _run("python scripts/compute_paper2_tableS1.py", check=False)


def step7_figures(force: bool) -> None:
    log.info("=== Step 7: figures ===")
    figures = [
        (FIGURES_DIR / "fig1_multiprod_fovs.png",
         "python scripts/plot_paper2_fig1_multiprod_fovs.py"),
        (FIGURES_DIR / "fig2_decomposition.png",
         "python scripts/plot_paper2_fig2_decomposition.py"),
        (FIGURES_DIR / "fig3_tiebreaker.png",
         "python scripts/plot_paper2_fig3_tiebreaker.py"),
        (FIGURES_DIR / "fig4_mechanism_conditional.png",
         "python scripts/plot_paper2_fig4_mechanism_conditional.py"),
        (FIGURES_DIR / "figS1_leadlag.png",
         "python scripts/plot_paper2_figS1_leadlag.py"),
        (FIGURES_DIR / "figS3_zonal_structure.png",
         "python scripts/plot_paper2_figS3_zonal_structure.py"),
        (FIGURES_DIR / "figS4_signal_noise.png",
         "python scripts/plot_paper2_figS4_signal_noise.py"),
    ]
    # figS2 is produced by the sensitivity script in step 5, not a separate plot
    for out, cmd in figures:
        if _need_update(out, force):
            _run(cmd, check=False)
        else:
            log.info(f"  up-to-date: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate everything even if outputs exist")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Reanalyses whose time-series step to skip "
                        "(e.g. --skip soda ecco)")
    args = parser.parse_args()

    skip = set(args.skip)
    log.info(f"Force regeneration: {args.force}; skipping: {skip or '(nothing)'}")

    step1_fovs_timeseries(args.force, skip)
    step2_reanalysis_decomp(args.force)
    step3_cmip6_decomp(args.force)
    step4_leadlag(args.force)
    step5_sensitivity(args.force)
    step6_table(args.force)
    step7_figures(args.force)

    log.info("Done.")


if __name__ == "__main__":
    main()
