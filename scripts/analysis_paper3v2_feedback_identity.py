#!/usr/bin/env python3
"""Separate the two terms of the F_ovS variation identity.

Reviewer 1 of the previous submission objected, correctly, that a falling
F_ovS cannot be read as a weakening overturning. Writing

    F_ovS = -(1/S0) * Psi * DvS

with Psi the upper-cell overturning strength at the section and DvS the
overturning-weighted vertical salinity contrast, the variation splits as

    dF_ovS = -(1/S0) * (DvS * dPsi + Psi * dDvS + dPsi * dDvS)

Whenever F_ovS < 0 and Psi > 0 the definition forces DvS > 0, so a weakening
overturning (dPsi < 0) pushes F_ovS *up*, not down.

This script evaluates the three terms for ORAS5 and GLORYS12V1, whose Psi and
F_ovS are computed on the identical section row, and reconciles them with the
velocity/salinity decomposition stored in ``data/results/``. The reconciliation
uses the exact identity

    dF_v = (Psi_late/Psi_early - 1) * F_ov_early + structural residual,

which holds because the barotropic correction forces the depth integral of the
corrected velocity to vanish, so a pure rescaling of the velocity profile
rescales F_ov by the same factor.

Writes ``PAPER_3_v2/analysis/feedback_identity.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
OUT_DIR = REPO / "PAPER_3_v2" / "analysis"

S0 = 35.0

# Epoch pairs that match the decomposition files, so the identity terms and the
# profile decomposition describe exactly the same pair of epochs.
SETUPS = {
    "oras5": {
        "label": "ORAS5",
        "fovs": "oras5_f_ovs.nc",
        "moc": "oras5_moc_34S.nc",
        "epochs": {
            "pre_registered": {
                "early": (1993, 2005),
                "late": (2013, 2025),
                "decomp": "fovs_decomposition_oras5.nc",
            },
            "long_epochs": {
                "early": (1960, 1989),
                "late": (1995, 2024),
                "decomp": "fovs_decomposition_oras5_paper3_epochs.nc",
            },
        },
        "trend_windows": [
            ("own record", 1958, 2025),
            ("common window", 1993, 2017),
            ("1993 to 2025", 1993, 2025),
        ],
    },
    "glorys12": {
        "label": "GLORYS12V1",
        "fovs": "glorys12_f_ovs.nc",
        "moc": "glorys12_moc_34S.nc",
        "epochs": {
            "pre_registered": {
                "early": (1993, 2005),
                "late": (2013, 2025),
                "decomp": "fovs_decomposition_glorys12.nc",
            },
            "halves": {
                "early": (1993, 2008),
                "late": (2009, 2024),
                "decomp": "fovs_decomposition_glorys12_paper3_halves.nc",
            },
        },
        "trend_windows": [
            ("own record", 1993, 2025),
            ("common window", 1993, 2017),
        ],
    },
}


def annual(path: Path, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Calendar-year means of a monthly series."""
    with xr.open_dataset(path) as ds:
        grouped = ds[var].groupby("time.year").mean()
        years = grouped["year"].values.astype(int)
        values = grouped.values.astype(float)
    good = np.isfinite(values)
    return years[good], values[good]


def epoch_mean(years: np.ndarray, values: np.ndarray, span: tuple[int, int]) -> float:
    mask = (years >= span[0]) & (years <= span[1])
    return float(values[mask].mean())


def identity_terms(
    psi_1: float, psi_2: float, fov_1: float, fov_2: float
) -> dict[str, float]:
    """Exact three-term expansion of -(1/S0) * Psi * DvS between two epochs.

    DvS is taken from the epoch means themselves, DvS = -S0 * F_ov / Psi, so
    the three terms sum exactly to the epoch difference in F_ovS.
    """
    dvs_1 = -S0 * fov_1 / psi_1
    dvs_2 = -S0 * fov_2 / psi_2
    d_psi = psi_2 - psi_1
    d_dvs = dvs_2 - dvs_1
    term_psi = -(1.0 / S0) * dvs_1 * d_psi
    term_dvs = -(1.0 / S0) * psi_1 * d_dvs
    term_cross = -(1.0 / S0) * d_psi * d_dvs
    total = fov_2 - fov_1
    return {
        "psi_early_Sv": psi_1,
        "psi_late_Sv": psi_2,
        "delta_psi_Sv": d_psi,
        "delta_psi_pct": 100.0 * d_psi / psi_1,
        "dvs_early_PSU": dvs_1,
        "dvs_late_PSU": dvs_2,
        "delta_dvs_PSU": d_dvs,
        "fovs_early_Sv": fov_1,
        "fovs_late_Sv": fov_2,
        "term_psi_mSv": term_psi * 1e3,
        "term_dvs_mSv": term_dvs * 1e3,
        "term_cross_mSv": term_cross * 1e3,
        "total_mSv": total * 1e3,
        "closure_mSv": (term_psi + term_dvs + term_cross - total) * 1e3,
        "share_psi_pct": 100.0 * term_psi / total,
        "share_dvs_pct": 100.0 * term_dvs / total,
        "share_cross_pct": 100.0 * term_cross / total,
    }


def compute_product(key: str, cfg: dict) -> dict:
    yrs_f, fovs = annual(RESULTS / cfg["fovs"], "F_ovS")
    yrs_p, psi = annual(RESULTS / cfg["moc"], "moc_upper")
    assert np.array_equal(yrs_f, yrs_p), f"{key}: F_ovS and Psi years differ"
    years = yrs_f
    assert psi.min() > 1.0, f"{key}: Psi approaches zero, factorisation unsafe"

    with xr.open_dataset(RESULTS / cfg["moc"]) as ds:
        section_lat = float(ds["moc_upper"].attrs["section_latitude"])

    dvs = -S0 * fovs / psi
    out: dict = {
        "label": cfg["label"],
        "section_latitude": section_lat,
        "record": [int(years[0]), int(years[-1])],
        "n_years": int(len(years)),
        "years": years.tolist(),
        "psi_Sv": psi.tolist(),
        "fovs_Sv": fovs.tolist(),
        "dvs_PSU": dvs.tolist(),
        "psi_mean_Sv": float(psi.mean()),
        "dvs_mean_PSU": float(dvs.mean()),
        "n_years_fovs_negative": int((fovs < 0).sum()),
        "sign_rule_holds": bool(np.all((fovs < 0) == (dvs > 0))),
    }

    out["psi_trends"] = {}
    out["fovs_trends"] = {}
    for label, y0, y1 in cfg["trend_windows"]:
        mask = (years >= y0) & (years <= y1)
        for series, store, scale in (
            (psi, out["psi_trends"], 10.0),  # Sv per decade
            (fovs, out["fovs_trends"], 1e3),  # mSv per year
        ):
            fit = ols_santer(years[mask].astype(float), series[mask])
            store[label] = {
                "window": [int(y0), int(y1)],
                "n": int(mask.sum()),
                "trend": fit["slope"] * scale,
                "p_santer": fit["p_santer"],
                "significant": bool(fit["p_santer"] < 0.05),
            }

    detrend = lambda a: a - np.polyval(np.polyfit(years, a, 1), years)  # noqa: E731
    out["corr_psi_fovs"] = {
        "pearson_r": float(np.corrcoef(psi, fovs)[0, 1]),
        "pearson_r_detrended": float(np.corrcoef(detrend(psi), detrend(fovs))[0, 1]),
        "n": int(len(years)),
    }

    out["epochs"] = {}
    for tag, spec in cfg["epochs"].items():
        fov_1 = epoch_mean(years, fovs, spec["early"])
        fov_2 = epoch_mean(years, fovs, spec["late"])
        terms = identity_terms(
            epoch_mean(years, psi, spec["early"]),
            epoch_mean(years, psi, spec["late"]),
            fov_1,
            fov_2,
        )
        terms["early_period"] = f"{spec['early'][0]}-{spec['early'][1]}"
        terms["late_period"] = f"{spec['late'][0]}-{spec['late'][1]}"

        with xr.open_dataset(RESULTS / spec["decomp"]) as ds:
            delta_v = float(ds.attrs["delta_v_Sv"]) * 1e3
            terms["decomposition"] = {
                "delta_total_mSv": float(ds.attrs["delta_total_Sv"]) * 1e3,
                "delta_v_mSv": delta_v,
                "delta_s_mSv": float(ds.attrs["delta_s_Sv"]) * 1e3,
                "delta_cross_mSv": float(ds.attrs["delta_cross_Sv"]) * 1e3,
            }
        # Split the profile velocity term into a pure amplitude rescaling and
        # a structural residual. The amplitude piece equals the identity's Psi
        # term exactly.
        amplitude = terms["term_psi_mSv"]
        terms["velocity_split"] = {
            "amplitude_mSv": amplitude,
            "structure_mSv": delta_v - amplitude,
            "structure_share_of_velocity_pct": 100.0 * (delta_v - amplitude) / delta_v,
        }
        out["epochs"][tag] = terms

    return out


def report(res: dict) -> None:
    for r in res.values():
        print(
            f"\n=== {r['label']} at {r['section_latitude']:.2f} deg N "
            f"({r['record'][0]}-{r['record'][1]}, n={r['n_years']}) ==="
        )
        print(
            f"  mean Psi {r['psi_mean_Sv']:.2f} Sv, mean DvS "
            f"{r['dvs_mean_PSU']:+.4f} PSU, F_ovS < 0 in "
            f"{r['n_years_fovs_negative']}/{r['n_years']} years, "
            f"sign rule holds: {r['sign_rule_holds']}"
        )
        print(
            f"  corr(Psi, F_ovS) r = {r['corr_psi_fovs']['pearson_r']:+.3f}, "
            f"detrended {r['corr_psi_fovs']['pearson_r_detrended']:+.3f}"
        )
        print(f"  {'window':16s} {'Psi (Sv/dec)':>22s} {'F_ovS (mSv/yr)':>22s}")
        for label in r["psi_trends"]:
            p = r["psi_trends"][label]
            f = r["fovs_trends"][label]
            print(
                f"  {label:16s} {p['trend']:+10.3f} p={p['p_santer']:6.4f}"
                f" {f['trend']:+10.3f} p={f['p_santer']:6.4f}"
            )
        for tag, t in r["epochs"].items():
            print(f"\n  -- {tag}: {t['early_period']} vs {t['late_period']}")
            print(
                f"     Psi   {t['psi_early_Sv']:6.2f} -> {t['psi_late_Sv']:6.2f} Sv "
                f"({t['delta_psi_Sv']:+.2f}, {t['delta_psi_pct']:+.1f}%)"
            )
            print(
                f"     DvS   {t['dvs_early_PSU']:+.4f} -> "
                f"{t['dvs_late_PSU']:+.4f} PSU ({t['delta_dvs_PSU']:+.4f})"
            )
            print(
                f"     F_ovS {t['fovs_early_Sv']:+.4f} -> {t['fovs_late_Sv']:+.4f} Sv"
            )
            print(
                f"     terms (mSv): Psi {t['term_psi_mSv']:+7.2f} "
                f"({t['share_psi_pct']:+6.1f}%)  DvS {t['term_dvs_mSv']:+7.2f} "
                f"({t['share_dvs_pct']:+6.1f}%)  cross {t['term_cross_mSv']:+6.2f} "
                f"({t['share_cross_pct']:+6.1f}%)  total {t['total_mSv']:+7.2f}"
                f"  closure {t['closure_mSv']:+.1e}"
            )
            d = t["decomposition"]
            v = t["velocity_split"]
            print(
                f"     profile     : total {d['delta_total_mSv']:+7.2f}  "
                f"v {d['delta_v_mSv']:+7.2f}  s {d['delta_s_mSv']:+7.2f}  "
                f"cross {d['delta_cross_mSv']:+6.2f}"
            )
            print(
                f"     velocity split: amplitude {v['amplitude_mSv']:+6.2f}  "
                f"structure {v['structure_mSv']:+7.2f} "
                f"({v['structure_share_of_velocity_pct']:.1f}% of v)"
            )


def main() -> None:
    res = {k: compute_product(k, cfg) for k, cfg in SETUPS.items()}
    report(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "feedback_identity.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
