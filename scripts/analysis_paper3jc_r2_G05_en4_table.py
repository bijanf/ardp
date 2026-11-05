#!/usr/bin/env python3
"""EN4 own-circulation reconciliation, EN4 F_ovS trends and printed-number checks.

Group G05_en4 of the paper 3 (J Climate) second review round, actions A48, A49
and A75.

What it computes
----------------
1. One table of the limb-contrast trends over 2005-2024, built from the same
   monthly inputs for every row:
     EN4.2.2 salinity with the monthly circulation of each reanalysis
       ("own"; the text at L431-433 and the EN4-own rows of Table S7);
     EN4.2.2 salinity with the record-mean circulation of each reanalysis
       ("fixed"; Fig. 5d shaded rows, Table S6);
     each reanalysis's own-salinity and own-circulation contrast (Fig. 5d
       lower rows, Table S7(c));
     each reanalysis's salinity under its fixed circulation (Fig. 5d).
   For each row: dS, S_north, S_south and F_ovS (annual series, trends with
   the bias-corrected lag-one test of paper3r1_stats.trend_test, intervals,
   p and n_eff).
2. The F_ovS series of EN4 with each reanalysis's monthly circulation, saved
   as annual values over 2005-2024, and its trend (A49).
3. The Table S6 start-year and end-year blocks recomputed from the annual EN4
   contrast under the ORAS5 record-mean circulation, and the 2005-2017 and
   2018-2024 northward-limb and contrast trends (S-L144 to S-L145).
4. Checks of every printed number of Fig. 1c, Fig. 4e, Fig. 5d, Table S6,
   Table S7 and the text sentences at L428-433 and S-L144-S-L147 against the
   values regenerated here or stored in earlier outputs. Printed values are
   typed in from the manuscript text extracted for the review round, with the
   decimals as printed. A value matches when it rounds to the printed value.

Inputs
------
- data/en4/EN.4.2.2.f.analysis.g10.*.nc (EN4.2.2 monthly, 2005-2024)
- data/results/paper3r1_section_{oras5,glorys12}.nc (V_bc, S_bar, e3t, dS_limb,
  S_north, S_south, F_ov, time 1993 to 2025 for GLORYS12V1; 1958 to 2025 for ORAS5)
- data/results/paper3r1_section3d_{oras5,glorys12}.nc (through the fixed-circulation
  reanalysis salinity, via analysis_paper3jc_n02_ratio.reanalysis_fixed)
- PAPER_3_v2/analysis/jc_n02_ratio.json, jc_n01_en4_window.json,
  en4.json and r1_salinification_structure.json (stored values for comparison
  and for the rows that are not regenerated here)

Outputs
-------
- PAPER_3_v2/analysis/jc_r2_G05_en4_table.json   (A48, A49, Table S6 recomputed)
- PAPER_3_v2/analysis/jc_r2_G05_en4_fig_checks.json (A75 checks, Fig. 1c, 4e,
  5d, Table S6, Table S7, S-L144 to S-L147)

No random draws are used: trend_test is analytic, so no seed applies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3jc_n02_ratio import (  # noqa: E402
    ANALYSIS,
    RESULTS,
    WINDOW,
    YEARS,
    annual_mean,
    load_en4_window,
    reanalysis_fixed,
    reanalysis_own,
)
from analysis_paper3v2_en4 import S0, frozen_velocity, limb_quantities  # noqa: E402
from paper3r1_stats import trend_test  # noqa: E402

PRODUCTS = ("oras5", "glorys12")
FIELDS = ("dS", "S_north", "S_south", "F_ov_Sv", "T_Sv")
DATE = "2026-11-05"
OUT_TABLE = ANALYSIS / "jc_r2_G05_en4_table.json"
OUT_CHECKS = ANALYSIS / "jc_r2_G05_en4_fig_checks.json"


# ── construction ─────────────────────────────────────────────────────


def monthly_limb(en4: dict, velocity_for_month) -> list:
    """Limb quantities for each EN4 month; None where the velocity is missing."""
    rows = []
    for i in range(len(en4["sbar"])):
        v = velocity_for_month(i)
        rows.append(
            None if v is None else limb_quantities(v, en4["sbar"][i], en4["e3t"])
        )
    return rows


def en4_own_rows(en4: dict, own: dict) -> list:
    """EN4 salinity with the monthly V_bc of the reanalysis on the EN4 levels.

    Same construction as analysis_paper3jc_n02_ratio.en4_limb_series(own=...).
    """
    index = {k: i for i, k in enumerate(own["months"])}

    def velocity(i: int):
        j = index.get((int(en4["year"][i]), int(en4["month"][i])))
        if j is None:
            return None
        vb = own["vbc_monthly"][j]
        if not np.all(np.isfinite(vb)):
            return None
        return np.interp(en4["depth"], own["depth"], vb, left=vb[0], right=0.0)

    return monthly_limb(en4, velocity)


def en4_fixed_rows(en4: dict, product: str) -> list:
    """EN4 salinity with the record-mean V_bc of the reanalysis."""
    v = frozen_velocity(f"paper3r1_section_{product}.nc", en4["depth"], en4["e3t"])
    return monthly_limb(en4, lambda i: v)


def annual_fields(en4: dict, rows: list) -> dict:
    out = {}
    for f in FIELDS:
        vals = np.array(
            [np.nan if m is None else m[f] for m in rows], dtype=float
        )
        out[f] = annual_mean(en4["year"], vals)
    out["n_valid_months"] = int(sum(m is not None for m in rows))
    return out


def reanalysis_F_ov_annual(product: str) -> np.ndarray:
    """Annual mean of the F_ov column of the reanalysis section file."""
    with xr.open_dataset(RESULTS / f"paper3r1_section_{product}.nc") as ds:
        t = pd.DatetimeIndex(ds["time"].values)
        f = np.asarray(ds["F_ov"].values, dtype=float)
    m = (t.year >= WINDOW[0]) & (t.year <= WINDOW[1])
    return annual_mean(t.year.values[m], f[m])


# ── trends ───────────────────────────────────────────────────────────


def trend(y, years: np.ndarray = YEARS) -> dict:
    """Trend of an annual series with paper3r1_stats.trend_test.

    Returns the slope per year and per decade, the 95% interval per decade,
    p, n_eff (after the effective-sample-size floor) and the diagnostics.
    """
    x = np.asarray(years, dtype=float)
    r = trend_test(x, np.asarray(y, dtype=float))
    return {
        "slope_per_year": float(r["value"]),
        "value_per_decade": float(10.0 * r["value"]),
        "ci_per_decade": [float(10.0 * r["ci"][0]), float(10.0 * r["ci"][1])],
        "p": float(r["p"]),
        "n": int(r["n"]),
        "n_eff": float(r["n_eff"]),
        "n_eff_at_floor": bool(r["n_eff_at_floor"]),
        "lag1_residual": float(r["lag1"]),
        "phi_corrected": float(r["phi_corrected"]),
    }


def series_block(values: np.ndarray, years: np.ndarray = YEARS) -> dict:
    """Series, its trend and the window of the trend."""
    values = np.asarray(values, dtype=float)
    block = {
        "years": [int(y) for y in years],
        "annual": [float(v) for v in values],
    }
    block.update(trend(values, years))
    return block


def max_abs_diff(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)))


# ── record-mean limbs (Fig. 1c) ──────────────────────────────────────


def record_mean_limbs(product: str, t0: int, t1: int) -> dict:
    """Transport-weighted limb salinities from the record-mean profiles.

    The velocity and the zonal-mean salinity are averaged over the years t0 to
    t1 first, then the limbs are split and weighted as in limb_quantities. This
    is the construction behind the vertical lines of Fig. 1c.
    """
    with xr.open_dataset(RESULTS / f"paper3r1_section_{product}.nc") as ds:
        t = pd.DatetimeIndex(ds["time"].values)
        m = (t.year >= t0) & (t.year <= t1)
        vbc = np.nanmean(np.asarray(ds["V_bc"].values, dtype=float)[m], axis=0)
        sbar = np.nanmean(np.asarray(ds["S_bar"].values, dtype=float)[m], axis=0)
        e3 = np.asarray(ds["e3t"].values, dtype=float)
        sn = np.asarray(ds["S_north"].values, dtype=float)[m]
        ss = np.asarray(ds["S_south"].values, dtype=float)[m]
        n_months = int(m.sum())
    tr = vbc * e3
    ok = np.isfinite(sbar) & np.isfinite(vbc)
    north = (tr > 0) & ok
    south = (tr < 0) & ok
    s_up = float((tr[north] * sbar[north]).sum() / tr[north].sum())
    s_down = float((-tr[south] * sbar[south]).sum() / (-tr[south]).sum())
    return {
        "window": [t0, t1],
        "n_months": n_months,
        "S_up_record_mean_profile": s_up,
        "S_down_record_mean_profile": s_down,
        "S_north_mean_of_monthly": float(np.nanmean(sn)),
        "S_south_mean_of_monthly": float(np.nanmean(ss)),
    }


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    en4 = load_en4_window()
    if en4["n_months"] != 240:
        raise RuntimeError(f"expected 240 EN4 months, found {en4['n_months']}")

    stored_n02 = json.loads((ANALYSIS / "jc_n02_ratio.json").read_text())
    stored_en4 = json.loads((ANALYSIS / "en4.json").read_text())
    stored_n01 = json.loads((ANALYSIS / "jc_n01_en4_window.json").read_text())
    stored_sal = json.loads((ANALYSIS / "r1_salinification_structure.json").read_text())

    table: dict = {}
    reproduction: dict = {}
    f_ov_means: dict = {}

    for p in PRODUCTS:
        own = reanalysis_own(p)
        fixed_re = reanalysis_fixed(p)
        A_own = annual_fields(en4, en4_own_rows(en4, own))
        A_fix = annual_fields(en4, en4_fixed_rows(en4, p))
        re_F = reanalysis_F_ov_annual(p)

        # Reproduction of the stored series, the same construction.
        s_n02 = stored_n02["products"][p]["series"]
        reproduction[p] = {
            "en4_own_dS_max_abs_diff_vs_jc_n02": max_abs_diff(
                A_own["dS"], s_n02["en4_own_dS"]
            ),
            "en4_fixed_dS_max_abs_diff_vs_jc_n02": max_abs_diff(
                A_fix["dS"], s_n02["en4_fixed_dS"]
            ),
            "en4_fixed_S_north_max_abs_diff_vs_en4_json": max_abs_diff(
                A_fix["S_north"], stored_en4[p]["S_north"]["series"]
            ),
            "en4_fixed_dS_max_abs_diff_vs_en4_json": max_abs_diff(
                A_fix["dS"], stored_en4[p]["dS"]["series"]
            ),
            "reanalysis_own_dS_max_abs_diff_vs_jc_n02": max_abs_diff(
                own["annual_dS"], s_n02["reanalysis_own_dS"]
            ),
            "reanalysis_fixed_dS_max_abs_diff_vs_jc_n02": max_abs_diff(
                fixed_re["annual_dS"], s_n02["reanalysis_fixed_dS"]
            ),
            "en4_own_valid_months": A_own["n_valid_months"],
            "en4_fixed_valid_months": A_fix["n_valid_months"],
        }

        table[p] = {
            "EN4 salinity, own circulation": {
                "dS": series_block(A_own["dS"]),
                "S_north": series_block(A_own["S_north"]),
                "S_south": series_block(A_own["S_south"]),
                "F_ov_Sv": series_block(A_own["F_ov_Sv"]),
                "T_Sv": series_block(A_own["T_Sv"]),
            },
            "EN4 salinity, fixed circulation (record-mean V_bc)": {
                "dS": series_block(A_fix["dS"]),
                "S_north": series_block(A_fix["S_north"]),
                "S_south": series_block(A_fix["S_south"]),
                "F_ov_Sv": series_block(A_fix["F_ov_Sv"]),
                "T_Sv": series_block(A_fix["T_Sv"]),
            },
            "reanalysis salinity, own circulation": {
                "dS": series_block(own["annual_dS"]),
                "F_ov_Sv_section_file": series_block(re_F),
            },
            "reanalysis salinity, fixed circulation": {
                "dS": series_block(fixed_re["annual_dS"]),
                "S_north": series_block(fixed_re["annual_S_north"]),
                "S_south": series_block(fixed_re["annual_S_south"]),
            },
        }

        f_ov_means[p] = {
            "EN4_fixed_implied_mean_Sv": float(np.mean(A_fix["F_ov_Sv"])),
            "EN4_own_mean_Sv": float(np.mean(A_own["F_ov_Sv"])),
            "reanalysis_section_file_mean_Sv_2005_2024": float(np.mean(re_F)),
        }

    # ── Table S6 recomputed from the ORAS5 record-mean circulation ─────
    fix_o = table["oras5"]["EN4 salinity, fixed circulation (record-mean V_bc)"]
    table_s6 = {}
    for start in range(2005, 2011):
        m = YEARS >= start
        table_s6[f"dS_start_{start}_end_2024"] = trend(
            np.asarray(fix_o["dS"]["annual"])[m], YEARS[m]
        )
    for end in range(2015, 2025):
        m = YEARS <= end
        table_s6[f"dS_start_2005_end_{end}"] = trend(
            np.asarray(fix_o["dS"]["annual"])[m], YEARS[m]
        )
    for name, (a, b) in {"2005_2017": (2005, 2017), "2018_2024": (2018, 2024)}.items():
        m = (YEARS >= a) & (YEARS <= b)
        for field in ("dS", "S_north", "S_south"):
            table_s6[f"{field}_window_{name}"] = trend(
                np.asarray(fix_o[field]["annual"])[m], YEARS[m]
            )
            table_s6[f"{field}_window_{name}"]["window"] = [a, b]
    table_s6["dS_full_2005_2024"] = fix_o["dS"]

    # Northward-limb and contrast trends for both circulations, 2005-2017.
    northward_2005_2017 = {}
    for p in PRODUCTS:
        blk = table[p]["EN4 salinity, fixed circulation (record-mean V_bc)"]
        m = YEARS <= 2017
        northward_2005_2017[p] = {
            "S_north": trend(np.asarray(blk["S_north"]["annual"])[m], YEARS[m]),
            "dS": trend(np.asarray(blk["dS"]["annual"])[m], YEARS[m]),
        }

    # Stored analogue of the 2005-2017 northward-limb trend, for comparison.
    stored_s_north_2005_2017 = {
        "oras5_full_column_jc_n01": stored_n01["series"]["oras5_full_column"][
            "S_north"
        ]["window_2005_2017"],
    }

    # ── record-mean limb salinities (Fig. 1c) ─────────────────────────
    fig1c = {
        "glorys12_1993_2025": record_mean_limbs("glorys12", 1993, 2025),
        "oras5_1993_2025": record_mean_limbs("oras5", 1993, 2025),
        "oras5_2005_2024": record_mean_limbs("oras5", 2005, 2024),
        "glorys12_2005_2024": record_mean_limbs("glorys12", 2005, 2024),
    }

    meta = {
        "date": DATE,
        "script": "scripts/analysis_paper3jc_r2_G05_en4_table.py",
        "group": "G05_en4",
        "actions": ["A48", "A49", "A75"],
        "inputs": [
            "data/en4/EN.4.2.2.f.analysis.g10.*.nc",
            "data/results/paper3r1_section_oras5.nc",
            "data/results/paper3r1_section_glorys12.nc",
            "data/results/paper3r1_section3d_oras5.nc",
            "data/results/paper3r1_section3d_glorys12.nc",
            "PAPER_3_v2/analysis/jc_n02_ratio.json (reproduction only)",
            "PAPER_3_v2/analysis/en4.json (reproduction only)",
            "PAPER_3_v2/analysis/jc_n01_en4_window.json (reproduction only)",
            "PAPER_3_v2/analysis/r1_salinification_structure.json (reproduction only)",
        ],
        "seeds": "none; trend_test is analytic",
        "replicate_counts": "none",
        "window": [2005, 2024],
        "n_years": int(len(YEARS)),
        "n_en4_months": int(en4["n_months"]),
        "units": {
            "salinity trends": "PSU per decade",
            "F_ov trends": "Sv per decade (slope_per_year in Sv per year)",
            "F_ov means": "Sv",
            "intervals": "95% of paper3r1_stats.trend_test",
            "p": "two-sided p of trend_test",
        },
        "sign_convention": "F_ov = -(1/S0) T dS / 1e6 with S0 = 35 (analysis_paper3v2_en4.limb_quantities)",
        "S0": S0,
    }

    table_out = {
        "meta": meta,
        "id": "G05_en4_table",
        "description": (
            "One table of EN4 and reanalysis limb-contrast trends over 2005-2024 "
            "with own and fixed circulations; EN4 F_ov series with own monthly "
            "circulation (A49); Table S6 start and end blocks recomputed (A75)."
        ),
        "table": table,
        "F_ov_means_2005_2024": f_ov_means,
        "table_S6_recomputed_oras5_fixed_circulation": table_s6,
        "northward_limb_2005_2017_S_north_and_dS": northward_2005_2017,
        "stored_S_north_2005_2017_jc_n01": stored_s_north_2005_2017,
        "fig1c_record_mean_limbs": fig1c,
        "reproduction_of_stored_values": reproduction,
    }
    OUT_TABLE.write_text(json.dumps(table_out, indent=2))

    # ── checks against printed numbers ───────────────────────────────
    checks, fig4e_shares = build_checks(
        table,
        table_s6,
        northward_2005_2017,
        fig1c,
        stored_n01,
        stored_sal,
        f_ov_means,
    )
    checks_out = {
        "meta": {**meta, "description": "printed-number checks, Fig. 1c, 4e, 5d, Table S6, S7 and S-L144 to S-L147"},
        "id": "G05_en4_fig_checks",
        "checks": checks,
        "fig4e_shares": fig4e_shares,
        "n_checks": len(checks),
        "n_mismatch": int(sum(not c["match"] for c in checks)),
    }
    OUT_CHECKS.write_text(json.dumps(checks_out, indent=2))

    print(f"wrote {OUT_TABLE}")
    print(f"wrote {OUT_CHECKS}: {checks_out['n_checks']} checks, {checks_out['n_mismatch']} mismatches")


# ── printed numbers and checks ──────────────────────────────────────


def _ok(got: float, printed: float, decimals: int) -> bool:
    return abs(got - printed) <= 0.5 * 10 ** (-decimals) + 1e-9


def build_checks(
    table, table_s6, nw, fig1c, stored_n01, stored_sal, table_fov
) -> tuple:
    out: list = []

    def add(where, quantity, printed, decimals, got, source):
        ok = _ok(got, printed, decimals) if printed is not None else None
        out.append(
            {
                "where": where,
                "quantity": quantity,
                "printed": printed,
                "printed_decimals": decimals,
                "regenerated": float(got) if got is not None else None,
                "source": source,
                "match": ok,
            }
        )

    own_key = "EN4 salinity, own circulation"
    fix_key = "EN4 salinity, fixed circulation (record-mean V_bc)"
    re_own = "reanalysis salinity, own circulation"
    re_fix = "reanalysis salinity, fixed circulation"

    # A48: text L432-433, EN4 with own circulation.
    for p, est, lo, hi, pp in (
        ("oras5", 0.055, -0.048, 0.158, 0.195),
        ("glorys12", 0.018, -0.025, 0.061, 0.345),
    ):
        t = table[p][own_key]["dS"]
        src = f"jc_r2_G05_en4_table.json table/{p}/{own_key}/dS"
        add("L432-433", f"EN4 own-circulation dS trend {p}", est, 3, t["value_per_decade"], src)
        add("L432-433", f"EN4 own-circulation dS interval lower {p}", lo, 3, t["ci_per_decade"][0], src)
        add("L432-433", f"EN4 own-circulation dS interval upper {p}", hi, 3, t["ci_per_decade"][1], src)
        add("L432-433", f"EN4 own-circulation dS p {p}", pp, 3, t["p"], src)

    # Fig. 5d lower rows and Table S7(c): reanalysis salinity with own circulation.
    for p, est, lo, hi, pp in (
        ("oras5", 0.058, -0.011, 0.127, 0.081),
        ("glorys12", 0.035, -0.003, 0.074, 0.065),
    ):
        t = table[p][re_own]["dS"]
        src = f"jc_r2_G05_en4_table.json table/{p}/{re_own}/dS"
        add("Fig. 5d; Table S7(c)", f"reanalysis own dS trend {p}", est, 3, t["value_per_decade"], src)
        add("Fig. 5d; Table S7(c)", f"reanalysis own dS lower {p}", lo, 3, t["ci_per_decade"][0], src)
        add("Fig. 5d; Table S7(c)", f"reanalysis own dS upper {p}", hi, 3, t["ci_per_decade"][1], src)
        add("Table S7(c)", f"reanalysis own dS p {p}", pp, 3, t["p"], src)

    # Fig. 5d lower rows: reanalysis salinity, fixed circulation.
    for p, est, lo, hi, pp in (
        ("oras5", 0.029, -0.298, 0.356, 0.456),
        ("glorys12", 0.042, -0.327, 0.411, 0.386),
    ):
        t = table[p][re_fix]["dS"]
        src = f"jc_r2_G05_en4_table.json table/{p}/{re_fix}/dS"
        add("Fig. 5d", f"reanalysis fixed dS trend {p}", est, 3, t["value_per_decade"], src)
        add("Fig. 5d", f"reanalysis fixed dS lower {p}", lo, 3, t["ci_per_decade"][0], src)
        add("Fig. 5d", f"reanalysis fixed dS upper {p}", hi, 3, t["ci_per_decade"][1], src)
        add("Table S7(c)", f"reanalysis fixed dS p {p}", pp, 3, t["p"], src)

    # Fig. 5d shaded EN4 rows (full column, ORAS5 and GLORYS12V1 circulations).
    for p, est, lo, hi in (
        ("oras5", 0.024, 0.011, 0.037),
        ("glorys12", 0.026, 0.013, 0.038),
    ):
        t = table[p][fix_key]["dS"]
        src = f"jc_r2_G05_en4_table.json table/{p}/{fix_key}/dS"
        add("Fig. 5d; L428-429", f"EN4 fixed full-column dS {p}", est, 3, t["value_per_decade"], src)
        add("Fig. 5d; L428-429", f"EN4 fixed full-column lower {p}", lo, 3, t["ci_per_decade"][0], src)
        add("Fig. 5d; L428-429", f"EN4 fixed full-column upper {p}", hi, 3, t["ci_per_decade"][1], src)

    # Fig. 5d shaded EN4 rows, upper 2000 m (stored in jc_n01).
    for p, key, est, lo, hi in (
        ("oras5", "oras5_upper_2000m", 0.035, 0.015, 0.055),
        ("glorys12", "glorys12_upper_2000m", 0.038, 0.017, 0.059),
    ):
        t = stored_n01["series"][key]["dS"]["full_2005_2024"]
        src = f"jc_n01_en4_window.json series/{key}/dS/full_2005_2024"
        add("Fig. 5d; Table S7(a)", f"EN4 fixed upper-2000 m dS {p}", est, 3, t["slope_per_decade"], src)
        add("Fig. 5d; Table S7(a)", f"EN4 fixed upper-2000 m lower {p}", lo, 3, t["ci_per_decade"][0], src)
        add("Fig. 5d; Table S7(a)", f"EN4 fixed upper-2000 m upper {p}", hi, 3, t["ci_per_decade"][1], src)

    # Table S6 (a) and (b): start and end years, ORAS5 record-mean circulation.
    s6_start = [
        (2005, 0.024, 0.011, 0.037, 0.002, 13.2),
        (2006, 0.026, 0.016, 0.037, 0.001, 18.8),
        (2007, 0.031, 0.021, 0.041, 0.001, 18.0),
        (2008, 0.031, 0.020, 0.042, 0.001, 17.0),
        (2009, 0.036, 0.026, 0.047, 0.001, 16.0),
        (2010, 0.037, 0.025, 0.049, 0.001, 15.0),
    ]
    for y0, est, lo, hi, pp, ne in s6_start:
        t = table_s6["dS_full_2005_2024"] if y0 == 2005 else table_s6[f"dS_start_{y0}_end_2024"]
        src = f"table_S6_recomputed dS_start_{y0}_end_2024"
        add(f"Table S6(a) {y0}-2024", "dS estimate", est, 3, t["value_per_decade"], src)
        add(f"Table S6(a) {y0}-2024", "dS lower", lo, 3, t["ci_per_decade"][0], src)
        add(f"Table S6(a) {y0}-2024", "dS upper", hi, 3, t["ci_per_decade"][1], src)
        add(f"Table S6(a) {y0}-2024", "dS n_eff", ne, 1, t["n_eff"], src)
        if pp == 0.001:
            ok = t["p"] < 0.001 if y0 > 2005 else None
            out.append({"where": f"Table S6(a) {y0}-2024", "quantity": "dS p < 0.001",
                        "printed": "<0.001", "printed_decimals": None,
                        "regenerated": t["p"], "source": src, "match": ok})
        else:
            add(f"Table S6(a) {y0}-2024", "dS p", pp, 3, t["p"], src)

    s6_end = [
        (2015, 0.003, -0.019, 0.024, 0.782, 11.0),
        (2016, 0.002, -0.016, 0.019, 0.811, 12.0),
        (2017, 0.005, -0.010, 0.020, 0.510, 13.0),
        (2018, 0.013, -0.003, 0.029, 0.112, 14.0),
        (2019, 0.018, -0.011, 0.047, 0.168, 7.2),
        (2020, 0.019, -0.007, 0.045, 0.118, 7.3),
        (2021, 0.023, -0.000, 0.047, 0.051, 8.0),
        (2022, 0.022, 0.005, 0.039, 0.016, 10.8),
        (2023, 0.022, 0.007, 0.036, 0.008, 11.6),
    ]
    for e0, est, lo, hi, pp, ne in s6_end:
        t = table_s6[f"dS_start_2005_end_{e0}"]
        src = f"table_S6_recomputed dS_start_2005_end_{e0}"
        add(f"Table S6(b) 2005-{e0}", "dS estimate", est, 3, t["value_per_decade"], src)
        add(f"Table S6(b) 2005-{e0}", "dS lower", lo, 3, t["ci_per_decade"][0], src)
        add(f"Table S6(b) 2005-{e0}", "dS upper", hi, 3, t["ci_per_decade"][1], src)
        add(f"Table S6(b) 2005-{e0}", "dS p", pp, 3, t["p"], src)
        add(f"Table S6(b) 2005-{e0}", "dS n_eff", ne, 1, t["n_eff"], src)

    # Table S6 (c): 2005-2017 and 2018-2024 contrast rows.
    t = table_s6["dS_window_2005_2017"]
    add("Table S6(c)", "dS 2005-2017 estimate", 0.005, 3, t["value_per_decade"], "table_S6_recomputed dS_window_2005_2017")
    add("Table S6(c)", "dS 2005-2017 lower", -0.010, 3, t["ci_per_decade"][0], "table_S6_recomputed dS_window_2005_2017")
    add("Table S6(c)", "dS 2005-2017 upper", 0.020, 3, t["ci_per_decade"][1], "table_S6_recomputed dS_window_2005_2017")
    add("Table S6(c)", "dS 2005-2017 p", 0.510, 3, t["p"], "table_S6_recomputed dS_window_2005_2017")
    t = table_s6["dS_window_2018_2024"]
    add("Table S6(c)", "dS 2018-2024 estimate", 0.014, 3, t["value_per_decade"], "table_S6_recomputed dS_window_2018_2024")
    add("Table S6(c)", "dS 2018-2024 lower", -0.038, 3, t["ci_per_decade"][0], "table_S6_recomputed dS_window_2018_2024")
    add("Table S6(c)", "dS 2018-2024 upper", 0.066, 3, t["ci_per_decade"][1], "table_S6_recomputed dS_window_2018_2024")
    add("Table S6(c)", "dS 2018-2024 p", 0.511, 3, t["p"], "table_S6_recomputed dS_window_2018_2024")
    add("Table S6(c)", "dS 2018-2024 n_eff", 7.0, 1, t["n_eff"], "table_S6_recomputed dS_window_2018_2024")

    # Table S6 (c) rows taken from jc_n01 (hinge, split, covariate).
    # The Table S6 row "2005-2017" is the contrast; the northward-limb analogue
    # is checked below from the table itself.
    hinge = stored_n01["series"]["oras5_full_column"]["dS"]["hinge_2017"]
    ch = hinge["change_of_slope"]
    add("Table S6(c)", "change of slope at 2017 estimate", 0.037, 3, ch["value_per_decade"], "jc_n01 hinge_2017/change_of_slope")
    add("Table S6(c)", "change of slope at 2017 lower", -0.012, 3, ch["ci_per_decade"][0], "jc_n01 hinge_2017/change_of_slope")
    add("Table S6(c)", "change of slope at 2017 upper", 0.085, 3, ch["ci_per_decade"][1], "jc_n01 hinge_2017/change_of_slope")
    add("Table S6(c)", "change of slope at 2017 p", 0.125, 3, ch["p"], "jc_n01 hinge_2017/change_of_slope")
    add("Table S6(c)", "change of slope at 2017 n_eff", 15.3, 1, ch["n_eff"], "jc_n01 hinge_2017/change_of_slope")
    post = hinge["slope_post_2017_within_model"]
    add("Table S6(c)", "slope after 2017 estimate", 0.049, 3, post["value_per_decade"], "jc_n01 hinge_2017/slope_post_2017_within_model")
    add("Table S6(c)", "slope after 2017 lower", 0.015, 3, post["ci_per_decade"][0], "jc_n01 hinge_2017/slope_post_2017_within_model")
    add("Table S6(c)", "slope after 2017 upper", 0.084, 3, post["ci_per_decade"][1], "jc_n01 hinge_2017/slope_post_2017_within_model")
    add("Table S6(c)", "slope after 2017 p", 0.009, 3, post["p"], "jc_n01 hinge_2017/slope_post_2017_within_model")
    diff = stored_n01["series"]["oras5_full_column"]["dS"]["subperiod_split_2017"]["difference_post_minus_pre"]
    add("Table S6(c)", "post minus pre split estimate", 0.010, 3, diff["value_per_decade"], "jc_n01 subperiod_split_2017/difference_post_minus_pre")
    add("Table S6(c)", "post minus pre split p", 0.629, 3, diff["p"], "jc_n01 subperiod_split_2017/difference_post_minus_pre")
    cov = stored_n01["series"]["oras5_full_column"]["dS"]["covariate"]["time_plus_both_weights"]
    add("Table S6(c)", "trend with both limb weights estimate", 0.031, 3, cov["slope_per_decade"], "jc_n01 covariate/time_plus_both_weights (interval not stored)")
    add("Table S6(c)", "trend with both limb weights p", 0.013, 3, cov["p"], "jc_n01 covariate/time_plus_both_weights")
    add("Table S6(c)", "trend with both limb weights n_eff", 16.0, 1, cov["n_eff"], "jc_n01 covariate/time_plus_both_weights")
    add("Table S6 footnote", "both-weight coefficient north", -0.27, 2, cov["coef_north_weight"], "jc_n01 covariate/time_plus_both_weights")
    add("Table S6 footnote", "both-weight p north", 0.26, 2, cov["p_north_weight"], "jc_n01 covariate/time_plus_both_weights")
    add("Table S6 footnote (not printed in text)", "both-weight coefficient south", 0.035, 3, cov["coef_south_weight"], "jc_n01 covariate/time_plus_both_weights")

    # S-L144 to S-L145: EN4 northward-limb, southward-limb and contrast p values.
    for p, pn, ps, pd in (("oras5", 0.006, 0.20, 0.002), ("glorys12", 0.003, 0.19, 0.0009)):
        blk = table[p][fix_key]
        add("S-L143-144", f"EN4 northward-limb p {p}", pn, 3, blk["S_north"]["p"], f"table/{p}/{fix_key}/S_north")
        add("S-L143-144", f"EN4 southward-limb p {p}", ps, 2, blk["S_south"]["p"], f"table/{p}/{fix_key}/S_south")
        add("S-L143-144", f"EN4 contrast p {p}", pd, 4 if p == "glorys12" else 3, blk["dS"]["p"], f"table/{p}/{fix_key}/dS")

    # S-L144 to S-L145: 2005-2017 northward-limb trend (not printed in the table).
    nw_o = nw["oras5"]["S_north"]
    add("S-L144-145", "EN4 (ORAS5 circulation) northward-limb 2005-2017 trend", 0.005, 3, nw_o["value_per_decade"], "jc_r2_G05_en4_table northward_limb_2005_2017_S_north_and_dS/oras5/S_north")
    add("S-L144-145", "EN4 (ORAS5 circulation) northward-limb 2005-2017 p", 0.51, 2, nw_o["p"], "jc_r2_G05_en4_table northward_limb_2005_2017_S_north_and_dS/oras5/S_north")

    # S-L145 to S-L147: reanalysis contrast p values over 2005-2024.
    add("S-L145-147", "ORAS5 own-salinity contrast p", 0.081, 3, table["oras5"][re_own]["dS"]["p"], "table/oras5/reanalysis own/dS")
    add("S-L145-147", "GLORYS12V1 own-salinity contrast p", 0.065, 3, table["glorys12"][re_own]["dS"]["p"], "table/glorys12/reanalysis own/dS")
    add("S-L147", "ORAS5 fixed-circulation contrast p", 0.46, 2, table["oras5"][re_fix]["dS"]["p"], "table/oras5/reanalysis fixed/dS")
    add("S-L147", "GLORYS12V1 fixed-circulation contrast p", 0.39, 2, table["glorys12"][re_fix]["dS"]["p"], "table/glorys12/reanalysis fixed/dS")

    # Text L427: northward-limb and southward-limb EN4 trends (fixed circulation).
    for p, est_n, est_s in (("oras5", 0.026, 0.002), ("glorys12", 0.028, 0.002)):
        blk = table[p][fix_key]
        add("L427", f"EN4 fixed northward-limb trend {p}", est_n, 3, blk["S_north"]["value_per_decade"], f"table/{p}/{fix_key}/S_north")
        add("L427", f"EN4 fixed southward-limb trend {p}", est_s, 3, blk["S_south"]["value_per_decade"], f"table/{p}/{fix_key}/S_south")

    # Fig. 1c: transport-weighted limb salinities, GLORYS12V1 1993-2025.
    g = fig1c["glorys12_1993_2025"]
    add("Fig. 1c", "S-up (northward limb) GLORYS12V1 1993-2025", 34.97, 2, g["S_up_record_mean_profile"], "jc_r2_G05_en4_table fig1c_record_mean_limbs/glorys12_1993_2025")
    add("Fig. 1c", "S-down (southward limb) GLORYS12V1 1993-2025", 34.81, 2, g["S_down_record_mean_profile"], "jc_r2_G05_en4_table fig1c_record_mean_limbs/glorys12_1993_2025")
    add("Table S4(a) GLORYS12V1 (monthly mean construction)", "northward limb GLORYS12V1 1993-2025", 34.951, 3, g["S_north_mean_of_monthly"], "jc_r2_G05_en4_table fig1c_record_mean_limbs/glorys12_1993_2025")
    add("Table S4(a) GLORYS12V1 (monthly mean construction)", "southward limb GLORYS12V1 1993-2025", 34.805, 3, g["S_south_mean_of_monthly"], "jc_r2_G05_en4_table fig1c_record_mean_limbs/glorys12_1993_2025")

    # Fig. 4e and text L413-417: upper-layer and band shares of the northward trend.
    sh = {}
    cases = {
        "EN4 frozen, ORAS5 circulation": stored_sal["en4"]["frozen"]["oras5"],
        "EN4 frozen, GLORYS12V1 circulation": stored_sal["en4"]["frozen"]["glorys12"],
        "ORAS5 own, 2005-2024": stored_sal["oras5"]["windows"]["2005-2024"],
        "GLORYS12V1 own, 2005-2024": stored_sal["glorys12"]["windows"]["2005-2024"],
    }
    for name, blk in cases.items():
        lay = blk["northward_by_layer"]
        tot = float(sum(lay.values()))
        band = blk["northward_by_band"]
        btot = float(sum(band.values()))
        sh[name] = {
            "upper_500m_share_pct": 100.0 * (lay["0-100 m"] + lay["100-500 m"]) / tot,
            "upper_100m_share_pct": 100.0 * lay["0-100 m"] / tot,
            "total_northward_trend_PSU_per_decade": tot,
            "band_share_pct": {k: 100.0 * v / btot for k, v in band.items()},
        }
    # Printed ranges in the text are compared with the min and max over the
    # four cases named in the text (EN4 under both circulations, own ORAS5 and
    # own GLORYS12V1). A range matches when both ends round to the printed ends.
    def band_vals(key):
        return [sh[c]["band_share_pct"][key] for c in cases]

    def top_vals(key):
        return [sh[c][key] for c in cases]

    def add_range(where, quantity, lo_p, hi_p, values, source):
        lo_g, hi_g = float(min(values)), float(max(values))
        ok = round(lo_g) == lo_p and round(hi_g) == hi_p
        out.append(
            {
                "where": where,
                "quantity": quantity,
                "printed": [lo_p, hi_p],
                "printed_decimals": 0,
                "regenerated": [lo_g, hi_g],
                "source": source,
                "match": bool(ok),
            }
        )

    src_sh = "jc_r2_G05_en4_table fig4e_shares (from r1_salinification_structure.json)"
    add_range("Fig. 4e; L413", "upper 500 m share range, 4 cases (printed 83% to 92%)", 83, 92, top_vals("upper_500m_share_pct"), src_sh)
    add_range("L414", "upper 100 m share range, 4 cases (printed 42% to 58%)", 42, 58, top_vals("upper_100m_share_pct"), src_sh)
    add_range("L414-415", "eastern interior share range, 4 cases (printed 38% to 48%)", 38, 48, band_vals("eastern interior (20 W to 0)"), src_sh)
    add_range("L416", "Cape Basin share range, 4 cases (printed 15% to 22%)", 15, 22, band_vals("Cape Basin (east of 0)"), src_sh)
    add_range("L417", "western boundary share range, 4 cases (printed 1% to 10%)", 1, 10, band_vals("western boundary (west of 45 W)"), src_sh)

    # Fig. 4e single shares, one per case (printed in the figure as 90, 83, 88, 92).
    add("Fig. 4e", "upper 500 m share, EN4 frozen, ORAS5 circulation (printed 90)", 90, 0, sh["EN4 frozen, ORAS5 circulation"]["upper_500m_share_pct"], src_sh)
    add("Fig. 4e", "upper 500 m share, ORAS5 own (printed 83)", 83, 0, sh["ORAS5 own, 2005-2024"]["upper_500m_share_pct"], src_sh)
    add("Fig. 4e", "upper 500 m share, GLORYS12V1 own (printed 88)", 88, 0, sh["GLORYS12V1 own, 2005-2024"]["upper_500m_share_pct"], src_sh)
    add("Fig. 4e", "upper 500 m share, EN4 frozen, GLORYS12V1 circulation (printed 92)", 92, 0, sh["EN4 frozen, GLORYS12V1 circulation"]["upper_500m_share_pct"], src_sh)

    # L430: EN4 implied FovS (fixed circulation) against the reanalysis means over 2005-2024.
    fo = table_fov
    for p, est, pct in (("oras5", -0.080, 5.0), ("glorys12", -0.128, 1.0)):
        ref = fo[p]["reanalysis_section_file_mean_Sv_2005_2024"]
        rel = 100.0 * abs(fo[p]["EN4_fixed_implied_mean_Sv"] - ref) / abs(ref)
        out.append(
            {
                "where": "L429-431",
                "quantity": f"EN4 implied F_ovS mean within {pct:g}% of {p} reanalysis 2005-2024 mean",
                "printed": est,
                "printed_decimals": 3,
                "regenerated": float(fo[p]["EN4_fixed_implied_mean_Sv"]),
                "source": f"jc_r2_G05_en4_table F_ov_means_2005_2024/{p}; reanalysis mean {ref:.4f} Sv, relative difference {rel:.2f} %",
                "match": bool(rel <= pct and _ok(fo[p]["EN4_fixed_implied_mean_Sv"], est, 3)),
            }
        )
    return out, sh


if __name__ == "__main__":
    main()
