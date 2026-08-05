#!/usr/bin/env python3
"""Figure 4: CMIP6 sits on the other side of zero, and cannot be sorted by it.

(a) Historical (1950-1980) mean F_ovS at 34.5 S in 25 CMIP6 models, with the
    reanalysis estimates marked. No offset or bias correction is applied to
    either side.
(b) Model mean F_ovS against the projected overturning change at 26.5 N,
    1850-1900 to 2081-2100. This replaces the 19-year single-member trend used
    previously, which could not resolve any relationship: the grey band shows
    the spread of 19-year trends across the MPI-ESM1-2-LR large ensemble, that
    is, the noise a 19-year test has to compete against.

Reads PAPER_3_v2/analysis/{cmip6,attribution}.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    C_SODA,
    TWO_COL,
    panel_letter,
    save,
    set_style,
)

CMIP6 = REPO / "PAPER_3_v2" / "analysis" / "cmip6.json"
ATTR = REPO / "PAPER_3_v2" / "analysis" / "attribution.json"
SODA_F = REPO / "data" / "results" / "soda_f_ovs.nc"
OUTBASE = REPO / "PAPER_3_v2" / "figures" / "Fig4_models"

import csv  # noqa: E402

SUMMARY = REPO / "data" / "results" / "fovs_decomposition_cmip6_summary.csv"


def main() -> None:
    set_style()
    c6 = json.loads(CMIP6.read_text())
    attr = json.loads(ATTR.read_text())

    with open(SUMMARY) as fh:
        models = {r["model"]: float(r["F_ov_baseline"]) for r in csv.DictReader(fh)}

    obs = [
        ("ORAS5", attr["oras5"]["mean_F_ov_Sv"], C_ORAS5),
        ("GLORYS12V1", attr["glorys12"]["mean_F_ov_Sv"], C_GLORYS),
        ("ECCO-V4r4", attr["ecco"]["mean_F_ov_Sv"], C_ECCO),
    ]
    with xr.open_dataset(SODA_F) as ds:
        obs.append(("SODA 3.15.2", float(ds["F_ovS"].values.mean()), C_SODA))

    fig = plt.figure(figsize=(TWO_COL, 2.7))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.30,
        left=0.16,
        right=0.995,
        top=0.92,
        bottom=0.16,
    )

    # ── a: the distribution ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    order = sorted(models, key=models.get)
    vals = [models[m] for m in order]
    y = np.arange(len(order))
    ax.barh(
        y,
        vals,
        color=["#7f8fa6" if v < 0 else "#c46a4d" for v in vals],
        height=0.7,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=4.6)
    ax.axvline(0, color="0.3", lw=0.6)
    for _label, v, colour in obs:
        ax.axvline(v, color=colour, lw=0.9, ls=(0, (2.5, 1.2)))
    ax.set_xlabel("Mean $F_{\\mathrm{ov}}^{S}$ (Sv)")
    ax.set_ylim(-0.8, len(order) - 0.2)
    handles = [
        plt.Line2D([], [], color=c, lw=0.9, ls=(0, (2.5, 1.2)), label=lab)
        for lab, _, c in obs
    ]
    ax.legend(handles=handles, loc="lower right")
    panel_letter(ax, "a", x=-0.42)

    # ── b: the forced-response test ───────────────────────────────────
    ax = fig.add_subplot(gs[1])
    forced = c6["forced_response"]
    names = sorted(forced)
    x = np.array([forced[m]["fovs_Sv"] for m in names])
    yv = np.array([forced[m]["delta_Sv"] for m in names])
    ax.scatter(
        x,
        yv,
        s=9,
        c=["#7f8fa6" if v < 0 else "#c46a4d" for v in x],
        edgecolors="none",
    )
    fl = c6["internal_variability_floor"]
    lo, hi = fl["range_Sv_per_dec"]
    ax.axvline(0, color="0.3", lw=0.6)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xlabel("Model mean $F_{\\mathrm{ov}}^{S}$ (Sv)")
    ax.set_ylabel("Overturning change at 26.5$^\\circ$N (Sv)")
    panel_letter(ax, "b", x=-0.22)

    save(fig, OUTBASE)
    t = c6["rate_test_forced_absolute"]
    print(
        f"  forced-response test: n={t['n']}, r={t['pearson_r']:+.3f} "
        f"(p={t['pearson_p']:.3f}), 95% CI [{t['pearson_ci'][0]:+.2f}, "
        f"{t['pearson_ci'][1]:+.2f}], smallest detectable |r| = "
        f"{t['min_detectable_r']:.2f}"
    )
    print(
        f"  19-year internal-variability spread: {lo:+.2f} to {hi:+.2f} Sv per decade "
        f"across {fl['n_members']} members"
    )


if __name__ == "__main__":
    main()
