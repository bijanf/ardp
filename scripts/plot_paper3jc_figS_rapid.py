#!/usr/bin/env python3
"""Fig. S5 (FigS_rapid): reanalysis overturning at 26.5 N against the RAPID array.

(a) Monthly upper-cell overturning transport in RAPID, ORAS5 and GLORYS12V1 over
    the 240 months from 2004-04 to 2024-03 (thin: monthly values; thick: 12-month
    running means).
(b, c) ORAS5 and GLORYS12V1 against RAPID over the same months, with the
    least-squares fit (dashed) and the one-to-one line (dotted).

The raw correlations are recomputed here with paper3r1_stats.corr_test and checked
against PAPER_3_v2/analysis/jc_r2_G10_obs26n_overlap_corr.json, which also supplies
the effective sample sizes, the intervals and the annual overlap offsets quoted in
the caption (offset = ORAS5 minus product over the overlap).

Writes PAPER_3_v2/figures_r1/FigS_rapid.{pdf,png} and the numbers to
PAPER_3_v2/analysis/jc_r2_figS_rapid_numbers.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402

import paper3jc_style as jc  # noqa: E402
from paper3r1_stats import corr_test  # noqa: E402

RES = REPO / "data" / "results"
EXT = REPO / "data" / "external"
AN = REPO / "PAPER_3_v2" / "analysis"
FIG = REPO / "PAPER_3_v2" / "figures_r1"
G10 = AN / "jc_r2_G10_obs26n_overlap_corr.json"
OUT_JSON = AN / "jc_r2_figS_rapid_numbers.json"
C_RAPID = "black"


def to_month(s: pd.Series) -> pd.Series:
    """Index on the first day of each calendar month (RAPID uses the 1st,
    the reanalyses mid-month)."""
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).to_period("M").to_timestamp()
    return out


def load_monthly():
    rp = xr.open_dataset(EXT / "rapid_moc_monthly.nc")["moc_mar_hc10"].to_series()
    rp.index = pd.DatetimeIndex(rp.index)
    orm = xr.open_dataset(RES / "oras5_moc_26N.nc")["moc_upper"].to_series()
    orm.index = pd.DatetimeIndex(orm.index)
    glm = xr.open_dataset(RES / "glorys12_moc_26N.nc")["moc_upper"].to_series()
    glm.index = pd.DatetimeIndex(glm.index)
    rap, orr, gl = to_month(rp), to_month(orm), to_month(glm)
    months = rap.index[(rap.index >= rp.index[0]) & (rap.index <= rp.index[-1])]
    rap_w = rap.reindex(months)
    or_w = orr.reindex(months)
    gl_w = gl.reindex(months)
    assert not (rap_w.isna().any() or or_w.isna().any() or gl_w.isna().any())
    return months, rap_w, or_w, gl_w


def main() -> None:
    g10 = json.loads(G10.read_text())
    prim = g10["monthly_correlations_with_RAPID"]["primary_detrend_false"]
    rap_ov = g10["annual_overlaps"]["products"]["RAPID"]

    months, rap_w, or_w, gl_w = load_monthly()
    raw_or = corr_test(or_w.values, rap_w.values, detrend=False)
    raw_gl = corr_test(gl_w.values, rap_w.values, detrend=False)
    # The recomputed correlations must reproduce the stored analysis.
    assert abs(raw_or["r"] - prim["ORAS5_vs_RAPID_raw"]["r"]) < 1e-9
    assert abs(raw_gl["r"] - prim["GLORYS12V1_vs_RAPID_raw"]["r"]) < 1e-9

    jc.set_style()
    fig = plt.figure(figsize=(jc.TWO_COL, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    series = (
        ("RAPID", rap_w, C_RAPID),
        ("ORAS5", or_w, jc.C_ORAS5),
        ("GLORYS12V1", gl_w, jc.C_GLORYS),
    )
    for label, s, col in series:
        ax_a.plot(months, s.values, color=col, lw=0.4, alpha=0.5)
        ax_a.plot(
            months,
            s.rolling(12, center=True).mean().values,
            color=col,
            lw=1.4,
            label=label,
        )
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Upper-cell overturning (Sv)")
    ax_a.legend(loc="upper left", ncol=3)
    jc.panel_label(ax_a, "a", x=-0.07, y=1.02)

    for ax, letter, name, s, col, res in (
        (ax_b, "b", "ORAS5", or_w, jc.C_ORAS5, raw_or),
        (ax_c, "c", "GLORYS12V1", gl_w, jc.C_GLORYS, raw_gl),
    ):
        x = rap_w.values
        y = s.values
        ax.plot(x, y, "o", ms=2.2, color=col, alpha=0.7, mec="none")
        lo = float(min(x.min(), y.min())) - 0.5
        hi = float(max(x.max(), y.max())) + 0.5
        xx = np.array([lo, hi])
        ax.plot(xx, xx, ":", color="0.35", lw=0.8)
        slope, icpt = np.polyfit(x, y, 1)
        ax.plot(xx, slope * xx + icpt, "--", color=col, lw=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("RAPID (Sv)")
        ax.set_ylabel(f"{name} (Sv)")
        ax.text(
            0.04,
            0.93,
            f"$r$ = {res['r']:.2f}",
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="top",
        )
        jc.panel_label(ax, letter, x=-0.2, y=1.02)

    fig.savefig(FIG / "FigS_rapid.pdf", bbox_inches="tight")
    fig.savefig(FIG / "FigS_rapid.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print("wrote FigS_rapid")

    def fit(x, y):
        s, i = np.polyfit(x, y, 1)
        return {"slope": float(s), "intercept": float(i)}

    numbers = {
        "figure": "FigS_rapid (Fig. S5)",
        "window": [str(months[0].date()), str(months[-1].date()), int(len(months))],
        "raw_correlation_with_RAPID": {
            "ORAS5": {
                "r": raw_or["r"],
                "r_ci95": prim["ORAS5_vs_RAPID_raw"]["r_ci95"],
                "n_eff": raw_or["n_eff"],
            },
            "GLORYS12V1": {
                "r": raw_gl["r"],
                "r_ci95": prim["GLORYS12V1_vs_RAPID_raw"]["r_ci95"],
                "n_eff": raw_gl["n_eff"],
            },
        },
        "deseasonalised_correlation_with_RAPID_from_G10": {
            "ORAS5": prim["ORAS5_vs_RAPID_deseasonalised"]["r"],
            "GLORYS12V1": prim["GLORYS12V1_vs_RAPID_deseasonalised"]["r"],
        },
        "monthly_mean_Sv": {
            "RAPID": float(rap_w.mean()),
            "ORAS5": float(or_w.mean()),
            "GLORYS12V1": float(gl_w.mean()),
        },
        "least_squares_fit_reanalysis_on_RAPID": {
            "ORAS5": fit(rap_w.values, or_w.values),
            "GLORYS12V1": fit(rap_w.values, gl_w.values),
        },
        "annual_overlap_offset_ORAS5_minus_RAPID": {
            "overlap_years": rap_ov["overlap_years"],
            "mean_ORAS5_Sv": rap_ov["mean_ORAS5_over_overlap_Sv"],
            "mean_RAPID_Sv": rap_ov["mean_product_over_overlap_Sv"],
            "offset_Sv": rap_ov["offset_ORAS5_minus_product_Sv"],
            "offset_ci95_Sv": rap_ov["offset_ci95_Sv"],
            "source": "jc_r2_G10_obs26n_overlap_corr.json (annual_overlaps.products.RAPID)",
        },
        "source": "data/external/rapid_moc_monthly.nc; data/results/oras5_moc_26N.nc; "
        "data/results/glorys12_moc_26N.nc; jc_r2_G10_obs26n_overlap_corr.json",
    }
    OUT_JSON.write_text(json.dumps(numbers, indent=1))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
