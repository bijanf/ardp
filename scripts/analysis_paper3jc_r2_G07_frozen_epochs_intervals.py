"""Frozen-field contrast trends with 95% intervals and minimum detectable trends (group G07, action A07).

Recomputes the full-record frozen-field split of the limb salinity contrast
dS (ORAS5 and GLORYS12V1) that is stored in PAPER_3_v2/analysis/final_checks.json
(keys oras5.frozen_field and glorys12.frozen_field), using the same section
files and the same construction as scripts/analysis_paper3v2_final_checks.py
(frozen-field block): the full contrast, the salinity-only contrast with the
velocity profile frozen at its record mean, and the velocity-only contrast
with the salinity profile frozen at its record mean.

For each of the three series the script reports the bias-corrected trend with
its 95% interval (inverse of the N_eff test of scripts/paper3r1_stats.py), the
minimum detectable trend (half-width of that interval), the effective sample
size and the lag-one autocorrelation. The cross term is the residual
full - salinity-only - velocity-only; its interval and those of the three
parts are also given from a joint AR(1) sieve bootstrap (10,000 resamples,
seed 20261104, innovations resampled at common time indices, so the
cross-correlation of the three series is kept). Shares of the full trend are
reported with sieve intervals.

Inputs: data/results/paper3r1_section_oras5.nc, data/results/paper3r1_section_glorys12.nc,
PAPER_3_v2/analysis/final_checks.json (reproduction check only).
Outputs: PAPER_3_v2/analysis/jc_r2_G07_frozen_epochs_intervals.json

Run: python3 scripts/analysis_paper3jc_r2_G07_frozen_epochs_intervals.py
     (from the worktree root, with OMP_NUM_THREADS=4 and related variables set to 4)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3v2_final_checks import limbs  # noqa: E402
from paper3r1_stats import sieve_coefficients, trend_design, trend_test  # noqa: E402

RESULTS = REPO / "data" / "results"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G07_frozen_epochs_intervals.json"
SEED = 20261104
N_BOOT = 10_000
SECTIONS = {
    "oras5": ("ORAS5", "paper3r1_section_oras5.nc"),
    "glorys12": ("GLORYS12V1", "paper3r1_section_glorys12.nc"),
}
NAMES = ["full", "salinity_only", "velocity_only"]


def annual_series(fname: str) -> dict:
    """Annual means of the full, salinity-only and velocity-only contrasts."""
    ds = xr.open_dataset(RESULTS / fname)
    depth = ds["depth"].values
    e3t = ds["e3t"].values if "e3t" in ds else np.diff(depth, prepend=0.0)
    vbc = ds["V_bc"].values
    s = ds["S_bar"].values
    yrs = ds["time"].dt.year.values
    years = np.unique(yrs)
    v_bar = vbc.mean(axis=0)
    s_bar = s.mean(axis=0)
    variants = {
        "full": [limbs(vbc[i], s[i], e3t) for i in range(len(yrs))],
        "salinity_only": [limbs(v_bar, s[i], e3t) for i in range(len(yrs))],
        "velocity_only": [limbs(vbc[i], s_bar, e3t) for i in range(len(yrs))],
    }
    out = {"years": years.astype(int).tolist(), "record": [int(years[0]), int(years[-1])]}
    for name, arr in variants.items():
        ds_m = np.array([r[1] - r[2] for r in arr])
        out[name] = np.array([np.nanmean(ds_m[yrs == y]) for y in years])
    return out


def trend_entry(years: np.ndarray, y: np.ndarray) -> dict:
    r = trend_test(years.astype(float), y)
    return {
        "trend_per_decade": r["value"] * 10,
        "ci_95_per_decade": [r["ci"][0] * 10, r["ci"][1] * 10],
        "min_detectable_trend_per_decade": r["half_width"] * 10,
        "p_value": r["p"],
        "n_eff": r["n_eff"],
        "n_eff_at_floor": r["n_eff_at_floor"],
        "lag1_raw": r["lag1"],
        "phi_corrected": r["phi_corrected"],
        "n_years": r["n"],
    }


def pct(a: np.ndarray) -> list[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main() -> None:
    final = json.loads((ANALYSIS / "final_checks.json").read_text())
    out: dict = {
        "meta": {
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G07_frozen_epochs_intervals.py",
            "group": "G07_frozen_epochs",
            "action": "A07",
            "inputs": [
                "data/results/paper3r1_section_oras5.nc",
                "data/results/paper3r1_section_glorys12.nc",
                "PAPER_3_v2/analysis/final_checks.json (reproduction check)",
            ],
            "seed": SEED,
            "n_boot_sieve": N_BOOT,
            "units": "PSU per decade for dS and its parts; shares are dimensionless; p values are the N_eff test of paper3r1_stats.py",
            "interval_method": "t interval with bias-corrected N_eff (trend_test); cross term and shares: joint AR(1) sieve bootstrap (sieve_coefficients)",
            "cross_term_definition": "full - salinity_only - velocity_only (per decade)",
            "min_detectable_definition": "half-width of the 95% interval",
        }
    }
    for key, (label, fname) in SECTIONS.items():
        ser = annual_series(fname)
        years = np.array(ser["years"], dtype=float)
        x = trend_design(years)
        ys = np.column_stack([ser[n] for n in NAMES])

        block: dict = {"label": label, "record": ser["record"], "n_years": len(years)}
        for n in NAMES:
            block[n] = trend_entry(years, ser[n])

        stored = final[key]["frozen_field"]
        repro = {}
        for n in NAMES:
            repro[n] = {
                "stored_dS_trend_per_decade": stored[n]["dS_trend_per_decade"],
                "recomputed_dS_trend_per_decade": block[n]["trend_per_decade"],
                "stored_dS_min_detectable_per_decade": stored[n]["dS_min_detectable"],
                "recomputed_min_detectable_per_decade": block[n]["min_detectable_trend_per_decade"],
                "max_abs_diff_trend": abs(stored[n]["dS_trend_per_decade"] - block[n]["trend_per_decade"]),
            }
        block["reproduction_check_vs_final_checks"] = repro

        full = block["full"]["trend_per_decade"]
        sal = block["salinity_only"]["trend_per_decade"]
        vel = block["velocity_only"]["trend_per_decade"]
        cross = full - sal - vel
        rng = np.random.default_rng(SEED)
        draws = sieve_coefficients(x, ys, 1, rng, n_boot=N_BOOT) * 10  # (n_boot, 3) per decade
        d_full, d_sal, d_vel = draws[:, 0], draws[:, 1], draws[:, 2]
        d_cross = d_full - d_sal - d_vel

        block["three_term_split_per_decade"] = {
            "full": full,
            "salinity_only": sal,
            "velocity_only": vel,
            "cross_term": cross,
            "sum_of_parts": sal + vel + cross,
            "full_ci_95_t": block["full"]["ci_95_per_decade"],
            "salinity_only_ci_95_t": block["salinity_only"]["ci_95_per_decade"],
            "velocity_only_ci_95_t": block["velocity_only"]["ci_95_per_decade"],
            "full_ci_95_sieve": pct(d_full),
            "salinity_only_ci_95_sieve": pct(d_sal),
            "velocity_only_ci_95_sieve": pct(d_vel),
            "cross_term_ci_95_sieve": pct(d_cross),
            "cross_term_sieve_two_sided_p": float(2 * min(np.mean(d_cross > 0), np.mean(d_cross < 0))),
        }
        shares = {}
        for nm, arr, pt in (("salinity_only", d_sal, sal), ("velocity_only", d_vel, vel)):
            ratio = arr / d_full
            shares[nm] = {
                "point": pt / full,
                "ci_95_sieve": pct(ratio),
            }
        shares["cross_term"] = {"point": cross / full, "ci_95_sieve": pct(d_cross / d_full)}
        block["shares_of_full_trend"] = shares
        block["resolved_at_5pct_by_t_test"] = {n: bool(block[n]["p_value"] < 0.05) for n in NAMES}
        out[key] = block

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT}")
    for key in SECTIONS:
        b = out[key]
        print(key, b["record"])
        for n in NAMES:
            e = b[n]
            print(f"  {n:14s} {e['trend_per_decade']:+.4f} [{e['ci_95_per_decade'][0]:+.4f}, {e['ci_95_per_decade'][1]:+.4f}] mdt {e['min_detectable_trend_per_decade']:.4f} n_eff {e['n_eff']:.2f} p {e['p_value']:.3g}")
        t = b["three_term_split_per_decade"]
        print(f"  cross {t['cross_term']:+.4f} sieve CI {t['cross_term_ci_95_sieve']}")


if __name__ == "__main__":
    main()
