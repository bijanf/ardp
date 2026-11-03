#!/usr/bin/env python3
"""Figure 5: the limb asymmetry in observed salinity and the reanalysis trends.

The velocity structure is frozen at the ORAS5 record-mean profile and only the
EN4.2.2 objective analysis varies (2005-2024).

a  Transport-weighted salinity of the northward (solid) and southward (dashed)
   limb over the full column, as anomalies, with least-squares trends.
b  The same with both limbs defined inside the upper 2000 m.
c  Zonal-mean salinity trend against depth, 2005-2024, in EN4 and in the two
   reanalyses on the EN4 levels; the shading gives the EN4 salinity observation
   weight averaged along the section (upper axis).
d  Trend of the limb salinity contrast over 2005-2024 with 95% intervals from
   the Santer-adjusted standard error.

Writes PAPER_3_v2/figures_r1/Fig5_en4.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_GLORYS,
    C_ORAS5,
    TWO_COL,
    panel_letter,
    set_style,
)

AN = REPO / "PAPER_3_v2" / "analysis"
RES = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "figures_r1" / "Fig5_en4"
C_EN4 = "#111111"


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def trend_ci(years, series):
    x = np.asarray(years, float)
    y = np.asarray(series, float)
    f = ols_santer(x, y)
    dof = max(f["n_eff"] - 2, 1)
    h = stats.t.ppf(0.975, dof) * f["stderr_santer"]
    return 10 * f["slope"], 10 * (f["slope"] - h), 10 * (f["slope"] + h), f["p_santer"]


def limb_panel(ax, block: dict, colour: str):
    yrs = np.array(block["years"])
    for var, ls in (("S_north", "-"), ("S_south", (0, (2.5, 1.2)))):
        s = np.array(block[var]["series"])
        a = s - s.mean()
        ax.plot(yrs, a, color=colour, lw=0.8, ls=ls)
        coef = np.polyfit(yrs, a, 1)
        sig = block[var]["full"]["p_santer"] < 0.05
        ax.plot(
            yrs,
            np.polyval(coef, yrs),
            color=colour,
            lw=1.6 if sig else 0.8,
            ls="-" if sig else (0, (1.5, 1.5)),
        )
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_xlim(2004.5, 2024.5)


def main() -> None:
    set_style()
    dt = json.loads((AN / "en4_depth_test.json").read_text())
    attr = json.loads((AN / "attribution.json").read_text())
    prof = xr.open_dataset(RES / "paper3r1_salinity_trend_profiles.nc")

    fig = plt.figure(figsize=(TWO_COL, 4.6))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        height_ratios=[1, 1],
        wspace=0.42,
        hspace=0.5,
        left=0.07,
        right=0.99,
        top=0.95,
        bottom=0.09,
    )

    axa = fig.add_subplot(gs[0, 0])
    limb_panel(axa, dt["oras5"]["full_column"], "black")
    axa.set_ylabel("Limb salinity anomaly (PSU)")
    axa.legend(
        handles=[
            plt.Line2D([], [], color="k", lw=0.8, label="northward limb"),
            plt.Line2D(
                [], [], color="k", lw=0.8, ls=(0, (2.5, 1.2)), label="southward limb"
            ),
        ],
        loc="upper left",
    )
    panel_letter(axa, "a", x=-0.3)

    axb = fig.add_subplot(gs[0, 1], sharey=axa)
    limb_panel(axb, dt["oras5"]["upper_2000m"], "black")
    panel_letter(axb, "b", x=-0.18)

    # c: trend profiles
    axc = fig.add_subplot(gs[:, 2])
    z = prof["depth"].values
    w = prof["obs_weight_en4"].values
    axw = axc.twiny()
    axw.fill_betweenx(z, 0, w, color="0.88", lw=0, zorder=0)
    axw.set_xlim(0, 1)
    axw.set_xlabel("EN4 observation weight", fontsize=6)
    axw.tick_params(labelsize=5.5)
    axc.set_zorder(axw.get_zorder() + 1)
    axc.patch.set_visible(False)
    for key, label, col in (
        ("en4", "EN4.2.2", C_EN4),
        ("oras5", "ORAS5", C_ORAS5),
        ("glorys12", "GLORYS12V1", C_GLORYS),
    ):
        tr = prof[f"trend_{key}"].values
        pv = prof[f"p_{key}"].values
        axc.plot(tr, z, color=col, lw=0.7, label=label)
        axc.plot(np.where(pv < 0.05, tr, np.nan), z, color=col, lw=1.7)
    axc.axvline(0, color="black", lw=0.4)
    for ax in (axc, axw):
        ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    axc.set_ylim(5400, 0)
    axc.set_yticks([0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000])
    axc.set_ylabel("Depth (m)")
    axc.set_xlabel("Salinity trend, 2005–2024 (PSU decade$^{-1}$)")
    axc.legend(loc="lower right")
    panel_letter(axc, "c", x=-0.25, y=1.09)

    # d: contrast trends
    axd = fig.add_subplot(gs[1, 0:2])
    rows = []
    for key, lab in (("oras5", "ORAS5"), ("glorys12", "GLORYS12V1")):
        for blk, bl in (
            ("full_column", "full column"),
            ("upper_2000m", "upper 2000 m"),
        ):
            b = dt[key][blk]
            rows.append(
                (
                    f"EN4, {bl}, {lab} circulation",
                    *trend_ci(b["years"], b["dS"]["series"]),
                    C_EN4,
                )
            )
    for key, lab, col in (
        ("oras5", "ORAS5", C_ORAS5),
        ("glorys12", "GLORYS12V1", C_GLORYS),
    ):
        yrs = np.array(attr[key]["years"])
        ds_ = np.array(attr[key]["dS_limb_PSU"])
        m = (yrs >= 2005) & (yrs <= 2024)
        rows.append(
            (f"{lab}, own salinity and circulation", *trend_ci(yrs[m], ds_[m]), col)
        )
    y = np.arange(len(rows))[::-1]
    for yy, (_lab, t, lo, hi, p, col) in zip(y, rows, strict=False):
        axd.plot([lo, hi], [yy, yy], color=col, lw=1.0)
        axd.plot(t, yy, "o", color=col, ms=3.0, mfc=col if p < 0.05 else "white")
    axd.axvline(0, color="black", lw=0.4)
    axd.set_yticks(y)
    axd.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    axd.set_xlabel("Trend in $\\Delta S$, 2005–2024 (PSU decade$^{-1}$)")
    panel_letter(axd, "d", x=-0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=600)
    print("wrote", OUT)
    for r in rows:
        print(f"  {r[0]:45s} {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}] p={r[4]:.2g}")


if __name__ == "__main__":
    main()
