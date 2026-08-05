#!/usr/bin/env python3
"""PAPER_3 round-2 WP9: recompute every canonical number the manuscript quotes.

Reads ONLY the plain-named canonical files in data/results/ (never *.preBT or
*OLD_BUGGY*) plus the round-2 JSON outputs, and dumps one machine-readable
record of canonical values for the number audit. It does not read or edit the
manuscript; the comparison table is written by hand into WP9_number_audit.md.

Output: revision/rev_papaer3_02/results/WP9_canonical_values.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from plot_amoc_rate_comparison import compute_rate  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT = REPO / "revision" / "rev_papaer3_02" / "results"


def annual(path: Path, var: str = "F_ovS") -> tuple[np.ndarray, np.ndarray]:
    """Annual means. ORAS5/GLORYS12 are monthly; ECCO/SODA are already yearly."""
    ds = xr.open_dataset(path)
    da = ds[var]
    g = da.groupby("time.year").mean() if "time" in da.dims else da
    ds.close()
    return g["year"].values.astype(float), g.values.astype(float)


def fovs_block(name: str, fname: str) -> dict:
    ds = xr.open_dataset(RESULTS / fname)
    da = ds["F_ovS"]
    is_monthly = "time" in da.dims
    t = da["time"].values if is_monthly else da["year"].values
    monthly_mean = float(da.mean())
    ds.close()
    yr, val = annual(RESULTS / fname)
    full = ols_santer(yr, val)
    out = {
        "file": f"data/results/{fname}",
        "cadence": "monthly" if is_monthly else "annual",
        "n_records": int(len(t)),
        "first": str(t[0])[:10],
        "last": str(t[-1])[:10],
        "mean_Sv_native": monthly_mean,
        "mean_Sv_annual": float(np.mean(val)),
        "trend_mSv_per_yr_annual": full["slope"] * 1e3,
        "p_ols_annual": full["p_ols"],
        "p_santer_annual": full["p_santer"],
        "n_years": full["n_years"],
    }
    # Sub-period means the manuscript might intend
    for lo, hi, key in ((1995, 2024, "1995_2024"), (1993, 2025, "1993_2025")):
        m = (yr >= lo) & (yr <= hi)
        if m.sum() >= 5:
            out[f"mean_Sv_{key}"] = float(np.mean(val[m]))
    print(
        f"{name}: mean {out['mean_Sv_native']:+.4f} Sv (native), "
        f"{out['mean_Sv_annual']:+.4f} Sv (annual); trend "
        f"{out['trend_mSv_per_yr_annual']:+.3f} mSv/yr, n={out['n_years']} yr"
    )
    return out


def pileup_vs_fovs() -> dict:
    """Annual-mean F_ovS versus the pile-up index, both ORAS5 and GLORYS12."""
    fy, fv = annual(RESULTS / "oras5_f_ovs.nc")
    out = {}
    for tag, fname in (
        ("oras5", "salinity_pileup.nc"),
        ("glorys12", "salinity_pileup_glorys12.nc"),
    ):
        da = xr.open_dataarray(RESULTS / fname)
        g = da.groupby("time.year").mean()
        py, pv = g["year"].values.astype(float), g.values.astype(float)
        da.close()
        common = np.intersect1d(fy, py)
        a = fv[np.isin(fy, common)]
        b = pv[np.isin(py, common)]
        ok = np.isfinite(a) & np.isfinite(b)
        r, p = stats.pearsonr(a[ok], b[ok])
        rho, prho = stats.spearmanr(a[ok], b[ok])
        out[tag] = {
            "pileup_file": f"data/results/{fname}",
            "n_years": int(ok.sum()),
            "years": [int(common[ok].min()), int(common[ok].max())],
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "spearman_p": float(prho),
        }
        # The figure caption says n = 33 yr over 1993-2025, so also report the
        # correlation restricted to that window for each pile-up series.
        w = ok & (common >= 1993) & (common <= 2025)
        if w.sum() >= 10:
            rw, pw = stats.pearsonr(a[w], b[w])
            out[tag]["window_1993_2025"] = {
                "n_years": int(w.sum()),
                "pearson_r": float(rw),
                "pearson_p": float(pw),
            }
            print(f"   restricted 1993-2025: r={rw:+.4f} p={pw:.4f} n={int(w.sum())}")
        print(
            f"pileup vs F_ovS ({tag}): r={r:+.4f} p={p:.4f} n={int(ok.sum())} "
            f"({int(common[ok].min())}-{int(common[ok].max())})"
        )
    return out


def pileup_trends() -> dict:
    out = {}
    for tag, fname in (
        ("oras5", "salinity_pileup.nc"),
        ("glorys12", "salinity_pileup_glorys12.nc"),
    ):
        da = xr.open_dataarray(RESULTS / fname)
        g = da.groupby("time.year").mean()
        y, v = g["year"].values.astype(float), g.values.astype(float)
        da.close()
        for lo, hi in ((1993, 2025), (1993, 2024)):
            m = (y >= lo) & (y <= hi) & np.isfinite(v)
            if m.sum() < 10:
                continue
            r = ols_santer(y[m], v[m])
            out[f"{tag}_{lo}_{hi}"] = {
                "trend_PSU_per_yr": r["slope"],
                "trend_PSU_per_decade": r["slope"] * 10,
                "p_ols": r["p_ols"],
                "p_santer": r["p_santer"],
                "n_years": r["n_years"],
            }
    return out


def amoc_rates() -> dict:
    rapid = np.load(RESULTS / "rapid_amoc26n.npz")
    out: dict = {
        "rapid_record": {
            "first_year": int(rapid["years"][0]),
            "last_year": int(rapid["years"][-1]),
            "n_years": int(len(rapid["years"])),
        }
    }
    y0, y1 = int(rapid["years"][0]), int(rapid["years"][-1])
    for name, f in (
        ("RAPID", "rapid_amoc26n.npz"),
        ("ORAS5", "yearly_amoc26n_oras5.npz"),
        ("GLORYS12", "yearly_amoc26n_glorys12.npz"),
    ):
        d = np.load(RESULTS / f)
        # window as plotted (RAPID overlap)
        rate, p = compute_rate(d["years"], d["amoc"], y0, y1)
        # and the window the manuscript states, 2005-2022
        rate22, p22 = compute_rate(d["years"], d["amoc"], 2005, 2022)
        win = (d["years"] >= y0) & (d["years"] <= y1)
        out[name] = {
            "file": f"data/results/{f}",
            "record": [int(d["years"][0]), int(d["years"][-1])],
            "mean_Sv_full_record": float(np.nanmean(d["amoc"])),
            f"mean_Sv_{y0}_{y1}": float(np.nanmean(d["amoc"][win])),
            f"rate_Sv_dec_{y0}_{y1}": float(rate),
            f"p_{y0}_{y1}": float(p),
            "rate_Sv_dec_2005_2022": float(rate22),
            "p_2005_2022": float(p22),
        }
        print(
            f"{name}: {y0}-{y1} {rate:+.3f} Sv/dec | 2005-2022 "
            f"{rate22:+.3f} Sv/dec | mean(full) "
            f"{out[name]['mean_Sv_full_record']:.2f} Sv | "
            f"mean({y0}-{y1}) {out[name][f'mean_Sv_{y0}_{y1}']:.2f} Sv"
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "note": (
            "Canonical values only: plain-named files in data/results/. "
            "No *.preBT or *OLD_BUGGY* file was read."
        ),
        "fovs": {
            name: fovs_block(name, fname)
            for name, fname in (
                ("ORAS5", "oras5_f_ovs.nc"),
                ("GLORYS12", "glorys12_f_ovs.nc"),
                ("ECCO", "ecco_f_ovs.nc"),
                ("SODA", "soda_f_ovs.nc"),
            )
        },
        "pileup_vs_fovs": pileup_vs_fovs(),
        "pileup_trends": pileup_trends(),
        "amoc_rates": amoc_rates(),
    }

    # Decomposition epoch means, for the "mean F_ovS" definition question
    ds = xr.open_dataset(RESULTS / "fovs_decomposition_oras5_paper3_epochs.nc")
    payload["decomposition_paper3_epochs"] = {
        k: (float(v) if isinstance(v, int | float | np.floating) else str(v))
        for k, v in ds.attrs.items()
    }
    ds.close()

    # Round-2 JSON products already computed
    for tag, fname in (
        ("A3_split_period", "A3_split_period.json"),
        ("A2_fig10_count", "A2_fig10_count.json"),
        ("A6_multiproduct", "A6_multiproduct_fovs.json"),
        ("A8_direct_transports", "A8_direct_transports.json"),
        ("A10_amo", "A10_amo_lowfreq.json"),
        ("WP8_scatter", "WP8_scatter_stats.json"),
    ):
        p = OUT / fname
        if p.exists():
            payload[tag] = json.loads(p.read_text())

    (OUT / "WP9_canonical_values.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT / 'WP9_canonical_values.json'}")


if __name__ == "__main__":
    main()
