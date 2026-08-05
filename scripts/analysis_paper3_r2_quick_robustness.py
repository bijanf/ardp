#!/usr/bin/env python3
"""PAPER_3 round-2 quick robustness checks A6, A8, A10 (cached data only).

A6:  Four-product F_ovS trends over each product's full record (anchor
     against data/results/fovs_multiprod_trends.csv) and over the common
     window 1993-2017, plus ORAS5 recomputed inside each shorter product's
     window. Feeds the R2-principal and R3.1 (non-independence) responses.

A8:  RAPID AMOC(26.5N) trend with Santer p and block-bootstrap CI over the
     full RAPID record; SAMBA local record summarized (too short for
     trends; the response letter cites Frajka-Williams et al. 2019).

A10: AMO vs F_ovS at low frequency (R3: the 10-yr smoothed AMO still
     tracks F_ovS late in the record). Annual and 10-yr running-mean
     correlations, full record and post-1995, with Bretherton
     N_eff-adjusted significance.

Outputs JSON + a markdown summary into revision/rev_papaer3_02/results/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import bootstrap_slopes, ols_santer  # noqa: E402

RESULTS_DIR = REPO / "data" / "results"
OUT_DIR = REPO / "revision" / "rev_papaer3_02" / "results"

COMMON_WINDOW = (1993, 2017)


def load_annual_fovs(product: str) -> tuple[np.ndarray, np.ndarray]:
    """Annual-mean F_ovS series for a product from its canonical file."""
    ds = xr.open_dataset(RESULTS_DIR / f"{product}_f_ovs.nc")
    if "time" in ds.dims:
        annual = ds["F_ovS"].groupby("time.year").mean()
        years = annual["year"].values.astype(float)
        vals = annual.values.astype(float)
    else:
        years = ds["year"].values.astype(float)
        vals = ds["F_ovS"].values.astype(float)
    ds.close()
    good = np.isfinite(vals)
    return years[good], vals[good]


def trend_over(years: np.ndarray, vals: np.ndarray, y0: int, y1: int) -> dict | None:
    m = (years >= y0) & (years <= y1)
    if m.sum() < 8:
        return None
    st = ols_santer(years[m], vals[m])
    st["window"] = f"{int(years[m][0])}-{int(years[m][-1])}"
    st["slope_mSv_yr"] = st["slope"] * 1e3
    return st


# ══════════════════════════════════════════════════════════════════════
# A6: four-product comparison
# ══════════════════════════════════════════════════════════════════════


def a6_multiproduct() -> dict:
    products = ["oras5", "glorys12", "soda", "ecco"]
    series = {p: load_annual_fovs(p) for p in products}

    out: dict = {
        "common_window": f"{COMMON_WINDOW[0]}-{COMMON_WINDOW[1]}",
        "full_record": {},
        "common": {},
        "oras5_in_windows": {},
    }

    for p in products:
        yrs, vals = series[p]
        out["full_record"][p] = trend_over(yrs, vals, int(yrs[0]), int(yrs[-1]))
        out["common"][p] = trend_over(yrs, vals, *COMMON_WINDOW)

    o_yrs, o_vals = series["oras5"]
    for p in ["glorys12", "soda", "ecco"]:
        yrs, _ = series[p]
        w = (int(yrs[0]), int(yrs[-1]))
        out["oras5_in_windows"][p] = trend_over(o_yrs, o_vals, *w)

    return out


# ══════════════════════════════════════════════════════════════════════
# A8: RAPID trend, SAMBA record summary
# ══════════════════════════════════════════════════════════════════════


def a8_direct_transports() -> dict:
    rapid = np.load(RESULTS_DIR / "rapid_amoc26n.npz")
    yrs = rapid["years"].astype(float)
    am = rapid["amoc"].astype(float)
    st = ols_santer(yrs, am)
    boots = bootstrap_slopes(yrs, am, 10_000)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    ds = xr.open_dataset(REPO / "data" / "external" / "samba_moc_monthly.nc")
    samba_n = int(ds.sizes["time"])
    samba_span = f"{str(ds['time'].values[0])[:7]} to {str(ds['time'].values[-1])[:7]}"
    samba_mean = float(ds["upper_cell"].mean())
    samba_std = float(ds["upper_cell"].std())
    ds.close()

    return {
        "rapid": {
            "window": f"{int(yrs[0])}-{int(yrs[-1])}",
            "n_years": len(yrs),
            "slope_Sv_dec": st["slope"] * 10,
            "p_santer": st["p_santer"],
            "bootstrap_ci_Sv_dec": [c * 10 for c in ci],
            "ci_contains_zero": bool(ci[0] < 0 < ci[1]),
        },
        "samba_local_record": {
            "span": samba_span,
            "n_months": samba_n,
            "upper_cell_mean_Sv": samba_mean,
            "upper_cell_std_Sv": samba_std,
            "note": (
                "47-month record; too short for trend inference. "
                "Cite Frajka-Williams et al. 2019 for the absence of a "
                "significant long-term decline at 34.5S."
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# A10: AMO vs F_ovS at low frequency
# ══════════════════════════════════════════════════════════════════════


def _amo_annual() -> tuple[np.ndarray, np.ndarray]:
    """Parse the AMO monthly block embedded in scripts/plot_attribution.py."""
    src = (REPO / "scripts" / "plot_attribution.py").read_text()
    m = re.search(r'amo_lines = """(.*?)"""', src, re.S)
    if m is None:
        raise RuntimeError("AMO block not found in plot_attribution.py")
    years, vals = [], []
    for line in m.group(1).strip().splitlines():
        parts = line.split()
        yr = int(parts[0])
        monthly = [float(v) for v in parts[1:] if abs(float(v)) < 90]
        if monthly:
            years.append(yr)
            vals.append(np.mean(monthly))
    return np.array(years, dtype=float), np.array(vals)


def _corr_neff(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson r with Bretherton et al. (1999) N_eff-adjusted p."""
    r, p_naive = stats.pearsonr(a, b)
    r1a = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    r1b = float(np.corrcoef(b[:-1], b[1:])[0, 1])
    n = len(a)
    neff = n * (1 - r1a * r1b) / (1 + r1a * r1b)
    neff = max(neff, 3.0)
    t = r * np.sqrt((neff - 2) / max(1e-12, 1 - r * r))
    p = 2 * stats.t.sf(abs(t), df=neff - 2)
    return {
        "r": float(r),
        "p_naive": float(p_naive),
        "n": n,
        "n_eff": float(neff),
        "p_neff": float(p),
    }


def _runmean(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


def a10_amo_lowfreq() -> dict:
    amo_yrs, amo = _amo_annual()
    f_yrs, f = load_annual_fovs("oras5")

    y0, y1 = max(amo_yrs[0], f_yrs[0]), min(amo_yrs[-1], f_yrs[-1])
    am = amo[(amo_yrs >= y0) & (amo_yrs <= y1)]
    fv = f[(f_yrs >= y0) & (f_yrs <= y1)]
    yrs = np.arange(y0, y1 + 1)

    w = 10
    am_lp = _runmean(am, w)
    fv_lp = _runmean(fv, w)
    yrs_lp = yrs[w - 1 :]

    post = yrs >= 1995
    post_lp = yrs_lp >= 1995

    return {
        "overlap": f"{int(y0)}-{int(y1)}",
        "annual_full": _corr_neff(am, fv),
        "annual_post1995": _corr_neff(am[post], fv[post]),
        "lowpass10_full": _corr_neff(am_lp, fv_lp),
        "lowpass10_post1995": _corr_neff(am_lp[post_lp], fv_lp[post_lp]),
        "note": (
            "R3's concern: the 10-yr smoothed AMO visually tracks F_ovS "
            "late in the record. Low-pass N_eff is small; interpret "
            "p-values as honest, weak constraints."
        ),
    }


# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a6 = a6_multiproduct()
    a8 = a8_direct_transports()
    a10 = a10_amo_lowfreq()
    (OUT_DIR / "A6_multiproduct_fovs.json").write_text(json.dumps(a6, indent=2))
    (OUT_DIR / "A8_direct_transports.json").write_text(json.dumps(a8, indent=2))
    (OUT_DIR / "A10_amo_lowfreq.json").write_text(json.dumps(a10, indent=2))

    def _fmt(st: dict | None) -> str:
        if st is None:
            return "insufficient data"
        return (
            f"{st['slope_mSv_yr']:+.2f} mSv/yr "
            f"(p_Santer={st['p_santer']:.3f}, N_eff={st['n_eff']:.0f}, "
            f"{st['window']})"
        )

    lines = [
        "# Quick robustness checks A6 + A8 + A10 (canonical cached data)",
        "",
        "## A6: four-product F_ovS trends",
        "",
        "Full records (anchor vs fovs_multiprod_trends.csv):",
    ]
    for p in ["oras5", "glorys12", "soda", "ecco"]:
        lines.append(f"- {p}: {_fmt(a6['full_record'][p])}")
    lines += ["", f"Common window {a6['common_window']}:"]
    for p in ["oras5", "glorys12", "soda", "ecco"]:
        lines.append(f"- {p}: {_fmt(a6['common'][p])}")
    lines += ["", "ORAS5 recomputed inside each product's own window:"]
    for p, st in a6["oras5_in_windows"].items():
        lines.append(f"- {p} window: {_fmt(st)}")

    r = a8["rapid"]
    s = a8["samba_local_record"]
    lines += [
        "",
        "## A8: direct transport trends",
        "",
        f"- RAPID {r['window']}: {r['slope_Sv_dec']:+.2f} Sv/dec, "
        f"Santer p={r['p_santer']:.3f}, bootstrap CI "
        f"[{r['bootstrap_ci_Sv_dec'][0]:+.2f}, {r['bootstrap_ci_Sv_dec'][1]:+.2f}] "
        f"Sv/dec, contains zero: {r['ci_contains_zero']}",
        f"- SAMBA local record: {s['span']} ({s['n_months']} months), "
        f"upper cell {s['upper_cell_mean_Sv']:+.1f} "
        f"+/- {s['upper_cell_std_Sv']:.1f} Sv. {s['note']}",
        "",
        "## A10: AMO vs ORAS5 F_ovS",
        "",
    ]
    for k in ["annual_full", "annual_post1995", "lowpass10_full", "lowpass10_post1995"]:
        c = a10[k]
        lines.append(
            f"- {k}: r={c['r']:+.2f} (n={c['n']}, N_eff={c['n_eff']:.0f}, "
            f"p_neff={c['p_neff']:.3f})"
        )
    lines.append("")
    (OUT_DIR / "QUICK_ROBUSTNESS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
