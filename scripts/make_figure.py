#!/usr/bin/env python3
"""Master pipeline: produce any figure from scratch, auto-downloading missing data.

Usage:
    python scripts/make_figure.py amoc_strength
    python scripts/make_figure.py fovs_trajectory
    python scripts/make_figure.py sss_trend_map
    python scripts/make_figure.py amoc_rate
    python scripts/make_figure.py grl_figures
    python scripts/make_figure.py all
    python scripts/make_figure.py --list          # show available figures
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "results"


def run(cmd: str, check: bool = True) -> int:
    """Run a shell command, printing it first."""
    print(f"\n  >>> {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if check and result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        sys.exit(1)
    return result.returncode


def exists(*paths: str) -> bool:
    """Check if all paths exist (relative to ROOT)."""
    return all((ROOT / p).exists() for p in paths)


def has_files(directory: str, pattern: str, min_count: int = 1) -> bool:
    """Check if directory has at least min_count files matching pattern."""
    d = ROOT / directory
    if not d.exists():
        return False
    return len(list(d.glob(pattern))) >= min_count


# ═══════════════════════════════════════════════════════════════════════
# Data availability checks
# ═══════════════════════════════════════════════════════════════════════

def ensure_oras5_velocity():
    """Ensure ORAS5 3D velocity files are downloaded."""
    if has_files("data/oras5", "vomecrty_*_3D_*.nc", min_count=100):
        print("  [OK] ORAS5 velocity data found")
        return
    print("  [DOWNLOAD] ORAS5 velocity data missing — downloading...")
    run("python scripts/download_oras5_parallel.py")


def ensure_oras5_salinity():
    """Ensure ORAS5 3D salinity files are downloaded."""
    if has_files("data/oras5", "vosaline_*_3D_*.nc", min_count=100):
        print("  [OK] ORAS5 salinity data found")
        return
    print("  [DOWNLOAD] ORAS5 salinity data missing — downloading...")
    run("python scripts/download_oras5_parallel.py")


def ensure_oras5_sss():
    """Ensure ORAS5 2D SSS files are downloaded."""
    if has_files("data/oras5", "sosaline_*_2D_*.nc", min_count=100):
        print("  [OK] ORAS5 SSS data found")
        return
    print("  [DOWNLOAD] ORAS5 SSS data missing — downloading...")
    run("python scripts/download_oras5_parallel.py")


def ensure_glorys12():
    """Ensure GLORYS12 data is downloaded."""
    if has_files("data/glorys12", "glorys12_*.nc", min_count=10):
        print("  [OK] GLORYS12 data found")
        return
    print("  [ERROR] GLORYS12 data missing. Download manually via copernicusmarine.")
    print("  See README.md for instructions.")
    sys.exit(1)


def ensure_rapid():
    """Ensure RAPID observations are downloaded."""
    if exists("data/external/rapid_moc_transports.nc"):
        print("  [OK] RAPID data found")
        return
    print("  [DOWNLOAD] RAPID data missing — downloading...")
    run("python scripts/download_rapid.py")


def ensure_cmip6_fullfield(experiments: list[str] | None = None):
    """Ensure CMIP6 full-field vo_zonal files exist."""
    if experiments is None:
        experiments = ["historical", "ssp585"]
    for exp in experiments:
        if has_files("data/cmip6_fullfield", f"*_{exp}_vo_zonal.nc", min_count=5):
            print(f"  [OK] CMIP6 full-field {exp} data found")
        else:
            print(f"  [DOWNLOAD] CMIP6 {exp} full-field data missing — downloading...")
            run(f"python scripts/download_cmip6_vo_fullfield.py --experiments {exp}")


def ensure_cmip6_sections():
    """Ensure CMIP6 34.5S section files exist."""
    if has_files("data/cmip6_sections", "*_historical_vo.nc", min_count=3):
        print("  [OK] CMIP6 sections found")
        return
    print("  [DOWNLOAD] CMIP6 sections missing — downloading...")
    run("python scripts/download_cmip6_fovs_sections.py")


# ═══════════════════════════════════════════════════════════════════════
# Computed results checks
# ═══════════════════════════════════════════════════════════════════════

def ensure_yearly_amoc26n_oras5():
    """Ensure yearly AMOC(26.5N) from ORAS5 is computed."""
    if exists("data/results/yearly_amoc26n_oras5.npz"):
        print("  [OK] ORAS5 AMOC(26.5N) computed")
        return
    print("  [COMPUTE] Computing ORAS5 AMOC(26.5N)...")
    ensure_oras5_velocity()
    run("python scripts/compute_yearly_amoc26n.py --product oras5")


def ensure_yearly_amoc26n_glorys12():
    """Ensure yearly AMOC(26.5N) from GLORYS12 is computed."""
    if exists("data/results/yearly_amoc26n_glorys12.npz"):
        print("  [OK] GLORYS12 AMOC(26.5N) computed")
        return
    print("  [COMPUTE] Computing GLORYS12 AMOC(26.5N)...")
    ensure_glorys12()
    run("python scripts/compute_yearly_amoc26n.py --product glorys12")


def ensure_yearly_amoc26n_cmip6():
    """Ensure yearly AMOC(26.5N) from CMIP6 is computed."""
    if exists("data/results/yearly_amoc26n_cmip6.npz"):
        print("  [OK] CMIP6 AMOC(26.5N) computed")
        return
    print("  [COMPUTE] Computing CMIP6 AMOC(26.5N)...")
    ensure_cmip6_fullfield()
    run("python scripts/compute_yearly_amoc26n.py --product cmip6")


def ensure_rapid_amoc26n():
    """Ensure yearly RAPID AMOC is computed."""
    if exists("data/results/rapid_amoc26n.npz"):
        print("  [OK] RAPID AMOC(26.5N) computed")
        return
    print("  [COMPUTE] Processing RAPID data...")
    ensure_rapid()
    run("python scripts/download_rapid.py")


def ensure_oras5_fovs():
    """Ensure ORAS5 F_ovS is computed."""
    if exists("data/results/oras5_f_ovs.nc"):
        print("  [OK] ORAS5 F_ovS computed")
        return
    print("  [COMPUTE] Computing ORAS5 F_ovS...")
    ensure_oras5_velocity()
    ensure_oras5_salinity()
    run("python scripts/compute_oras5_fovs.py")


def ensure_cmip6_fovs():
    """Ensure CMIP6 F_ovS time series are computed."""
    if has_files("data/results/cmip6", "fovs_*_hist_ssp585.nc", min_count=5):
        print("  [OK] CMIP6 F_ovS computed")
        return
    print("  [COMPUTE] Computing CMIP6 F_ovS...")
    ensure_cmip6_sections()
    run("python scripts/compute_cmip6_fovs_timeseries.py")


def ensure_sss_pileup():
    """Ensure salinity pile-up index is computed."""
    if exists("data/results/salinity_pileup_glorys12.nc"):
        print("  [OK] GLORYS12 salinity pile-up computed")
        return
    print("  [ERROR] Salinity pile-up not computed. Run the pile-up computation manually.")
    print("  (Requires global GLORYS12 SSS data)")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Figure definitions
# ═══════════════════════════════════════════════════════════════════════

FIGURES = {
    "amoc_strength": {
        "description": "AMOC strength at 26.5N: ORAS5/GLORYS12/RAPID vs CMIP6 envelope",
        "output": "figures/grl/fig_amoc_strength_26N.png",
        "ensure": [
            ensure_yearly_amoc26n_oras5,
            ensure_yearly_amoc26n_glorys12,
            ensure_yearly_amoc26n_cmip6,
            ensure_rapid_amoc26n,
        ],
        "plot_cmd": "python scripts/plot_amoc_strength_comparison.py",
    },
    "fovs_trajectory": {
        "description": "F_ovS trajectory with CMIP6 boxplots (1850-2100)",
        "output": "figures/grl/fig4_cmip6_comparison.png",
        "ensure": [
            ensure_oras5_fovs,
            ensure_cmip6_fovs,
        ],
        "plot_cmd": "python scripts/plot_cmip6_fovs_trajectory.py",
    },
    "amoc_rate": {
        "description": "AMOC weakening rate bar chart: observations vs CMIP6",
        "output": "figures/grl/fig_amoc_rate_comparison.png",
        "ensure": [
            ensure_yearly_amoc26n_oras5,
            ensure_yearly_amoc26n_glorys12,
            ensure_yearly_amoc26n_cmip6,
            ensure_rapid_amoc26n,
        ],
        "plot_cmd": "python scripts/plot_amoc_rate_comparison.py",
    },
    "sss_trend_map": {
        "description": "SSS trend map + zonal mean + salinity pile-up (GLORYS12)",
        "output": "figures/grl/fig_sss_trend_map.png",
        "ensure": [
            ensure_glorys12,
            ensure_sss_pileup,
        ],
        "plot_cmd": "python scripts/plot_sss_trend_map.py --product glorys12 "
                    "--pileup data/results/salinity_pileup_glorys12.nc",
    },
    "sss_trend_map_oras5": {
        "description": "SSS trend map (ORAS5, full 1958-2025 period)",
        "output": "figures/grl/fig_sss_trend_map_oras5.png",
        "ensure": [
            ensure_oras5_sss,
        ],
        "plot_cmd": "python scripts/plot_sss_trend_map.py --product oras5",
    },
    "grl_figures": {
        "description": "All 4 GRL figures (F_ovS, RAPID, pile-up, CMIP6)",
        "output": "figures/grl/fig1_fovs_multiproduct.png",
        "ensure": [
            ensure_oras5_fovs,
            ensure_cmip6_fovs,
            ensure_rapid_amoc26n,
        ],
        "plot_cmd": "python scripts/plot_grl_figures.py",
    },
}


def make_figure(name: str, force: bool = False) -> None:
    """Produce a single figure, ensuring all dependencies exist."""
    if name not in FIGURES:
        print(f"Unknown figure: {name}")
        print(f"Available: {', '.join(sorted(FIGURES.keys()))}")
        sys.exit(1)

    fig = FIGURES[name]
    print(f"\n{'='*60}")
    print(f"Making: {name}")
    print(f"  {fig['description']}")
    print(f"{'='*60}")

    # Check if output already exists
    if not force and exists(fig["output"]):
        print(f"  [SKIP] Output already exists: {fig['output']}")
        print("  Use --force to regenerate")
        return

    # Ensure all data dependencies
    print("\nChecking dependencies...")
    for ensure_fn in fig["ensure"]:
        ensure_fn()

    # Plot
    print("\nGenerating figure...")
    run(fig["plot_cmd"])
    print(f"\n  [DONE] {fig['output']}")


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline: produce any figure from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:25s} {fig['description']}"
            for name, fig in sorted(FIGURES.items())
        ),
    )
    parser.add_argument(
        "figures", nargs="*", default=[],
        help="Figure name(s) to produce, or 'all' for everything",
    )
    parser.add_argument("--list", action="store_true", help="List available figures")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()

    if args.list or not args.figures:
        print("Available figures:\n")
        for name, fig in sorted(FIGURES.items()):
            status = "[exists]" if exists(fig["output"]) else "[missing]"
            print(f"  {name:25s} {status:10s} {fig['description']}")
        print("\nUsage: python scripts/make_figure.py <name> [--force]")
        return

    targets = args.figures
    if "all" in targets:
        targets = sorted(FIGURES.keys())

    for name in targets:
        make_figure(name, force=args.force)

    print(f"\n{'='*60}")
    print(f"All done. {len(targets)} figure(s) processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
