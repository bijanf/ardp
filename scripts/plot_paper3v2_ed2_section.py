#!/usr/bin/env python3
"""Extended Data Figure 2: the section profiles and the structure of the limbs.

This replaces the basin-wide streamfunction figure previously carried here. That
figure was drawn from a precomputed basin mask which produces an unphysical
strongly negative column between about 12 and 25 N in GLORYS12V1, it compared
the two products over different periods, and it did not show the quantity the
paper actually uses. Everything here is computed from the archived section
files on the contiguity-based section used throughout, over matched periods.

(a) Time-mean barotropic-corrected zonally integrated velocity against depth.
    The sign changes of this profile define the limbs, so this is the figure a
    reader needs in order to check the factorisation.
(b) Time-mean zonally averaged salinity against depth.
(c) The northward-limb transport split into its upper cell and its abyssal
    (Antarctic Bottom Water) part, as time series, since a change in the
    partition between them would alter the transport-weighted limb salinity
    without any water mass changing salinity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from analysis_paper3v2_limb_composition import sublimbs  # noqa: E402
from paper3v2_style import (  # noqa: E402
    C_ECCO,
    C_GLORYS,
    C_ORAS5,
    TWO_COL,
    panel_letter,
    save,
    set_style,
)

RESULTS = REPO / "data" / "results"
OUTBASE = REPO / "PAPER_3_v2" / "extended_data" / "ED2_section"

PRODUCTS = [
    ("ORAS5", "paper3v2_section_oras5.nc", C_ORAS5),
    ("GLORYS12V1", "paper3v2_section_glorys12.nc", C_GLORYS),
    ("ECCO-V4r4", "paper3v2_section_ecco.nc", C_ECCO),
]
MATCH = (1993, 2017)  # the window all three products share


def main() -> None:
    set_style()
    fig = plt.figure(figsize=(TWO_COL, 2.8))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.35],
        wspace=0.42,
        left=0.07,
        right=0.995,
        top=0.92,
        bottom=0.16,
    )
    ax_v = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])
    ax_t = fig.add_subplot(gs[2])

    for label, fname, colour in PRODUCTS:
        with xr.open_dataset(RESULTS / fname) as ds:
            depth = ds["depth"].values
            e3t = ds["e3t"].values if "e3t" in ds else np.diff(depth, prepend=0.0)
            a = ds.groupby("time.year").mean()
        years = a["year"].values.astype(int)
        m = (years >= MATCH[0]) & (years <= MATCH[1])

        vbc = a["V_bc"].values[m].mean(axis=0) / 1e6  # Sv per metre of depth
        sbar = a["S_bar"].values[m].mean(axis=0)
        ok = depth <= 5000
        ax_v.plot(vbc[ok] * 1e3, depth[ok], color=colour, lw=0.9, label=label)
        ax_s.plot(sbar[ok], depth[ok], color=colour, lw=0.9, label=label)

        # Sub-limb transports on MONTHLY fields, then averaged to years, so the
        # two curves sum to T. Identifying the limbs on annual-mean profiles
        # instead smooths out the deep sign reversals and understates the
        # abyssal cell by 30 to 40 per cent.
        with xr.open_dataset(RESULTS / fname) as ds2:
            vm = ds2["V_bc"].values
            sm = ds2["S_bar"].values
            ym = ds2["time"].dt.year.values
        mu, md = [], []
        for i in range(len(ym)):
            subs = sorted(sublimbs(vm[i], sm[i], e3t, depth), key=lambda r: r["top_m"])
            if not subs:
                mu.append(np.nan)
                md.append(np.nan)
                continue
            mu.append(subs[0]["T_Sv"])
            md.append(sum(r["T_Sv"] for r in subs[1:]))
        mu, md = np.array(mu), np.array(md)
        t_up = [np.nanmean(mu[ym == y]) for y in years]
        t_deep = [np.nanmean(md[ym == y]) for y in years]
        ax_t.plot(years, t_up, color=colour, lw=0.9)
        ax_t.plot(years, t_deep, color=colour, lw=0.9, ls=(0, (2.5, 1.2)))

    for ax in (ax_v, ax_s):
        ax.invert_yaxis()
        ax.set_ylim(5000, 0)
        ax.set_ylabel("Depth (m)")
    ax_v.axvline(0, color="0.4", lw=0.5)
    ax_v.set_xlabel("$V_{bc}$ (mSv m$^{-1}$)")
    ax_v.legend(loc="lower right")
    panel_letter(ax_v, "a")
    ax_s.set_xlabel("Zonal-mean salinity (PSU)")
    panel_letter(ax_s, "b")

    ax_t.set_xlabel("Year")
    ax_t.set_ylabel("Northward transport (Sv)")
    ax_t.set_xlim(1958, 2026)
    handles = [
        plt.Line2D([], [], color="0.35", lw=0.9, label="upper cell"),
        plt.Line2D(
            [], [], color="0.35", lw=0.9, ls=(0, (2.5, 1.2)), label="abyssal cell"
        ),
    ]
    ax_t.legend(handles=handles, loc="center left")
    panel_letter(ax_t, "c", x=-0.16)

    save(fig, OUTBASE)


if __name__ == "__main__":
    main()
