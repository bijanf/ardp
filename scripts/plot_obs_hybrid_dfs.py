#!/usr/bin/env python3
"""SI figure: observation-constrained salinity component of dF_ovS.

Left: observed dS(z) at 34.5S from EN4.2.2 and RG09 (2006-2012 vs
2018-2024). Right: for each reanalysis, its own delta_s versus the
hybrid delta_s obtained by pairing the OBSERVED dS(z) with that
product's baroclinic overturning profile, over identical reference
windows, showing that the salinity-driven component is negative for
every velocity field.

Reads:  revision/results/obs_hybrid_dFs.json
Writes: revision/latex/SI/figures/FigureS9_obs_hybrid_dfs.pdf (+ main_clean copy)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "revision/results/obs_hybrid_dFs.json"
OUT_DIRS = [
    ROOT / "revision/latex/SI/figures",
    ROOT / "revision/latex/main_clean/figures",
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)

PROD_COLORS = {"ORAS5": "#1f4e79", "GLORYS12V1": "#2e7d32", "ECCO-V4r4": "#7f7f7f"}


def main() -> None:
    d = json.loads(SRC.read_text())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.09, 3.3))

    # ----- left: observed dS(z) -----
    for prod, ls, col in [("EN4", "-", "#00838f"), ("RG09", "--", "#616161")]:
        if prod not in d["obs"]:
            continue
        z = np.array(d["obs"][prod]["depth_m"])
        ds = np.array(d["obs"][prod]["dS_PSU"])
        ok = np.isfinite(ds) & (z <= 1000)
        ax_l.plot(ds[ok], z[ok], ls, color=col, lw=1.4, label=prod)
    ax_l.axvline(0, color="0.7", lw=0.6)
    ax_l.set_ylim(1000, 0)
    ax_l.set_xlabel(r"observed $\Delta\bar{S}$ (PSU), 2018-2024 minus 2006-2012")
    ax_l.set_ylabel("depth (m)")
    ax_l.set_title("a   Observed salinity change at 34.5$^\\circ$S", loc="left")
    ax_l.legend(frameon=False, loc="lower right")

    # ----- right: own vs hybrid delta_s (reference windows) -----
    prods = list(d["products"].keys())
    x = np.arange(len(prods))
    own = [d["products"][p]["own_delta_s_Sv"] * 1e3 for p in prods]
    hyb_en4 = [
        d["products"][p]["hybrid_dFs_ref_windows_Sv"]["EN4_le2000m"] * 1e3
        for p in prods
    ]
    hyb_rg = [
        d["products"][p]["hybrid_dFs_ref_windows_Sv"].get("RG09", np.nan) * 1e3
        for p in prods
    ]
    w = 0.26
    ax_r.bar(
        x - w,
        own,
        w,
        color=[PROD_COLORS[p] for p in prods],
        edgecolor="0.2",
        linewidth=0.4,
        label="own $\\Delta F_s$",
    )
    ax_r.bar(
        x,
        hyb_en4,
        w,
        color="#00838f",
        edgecolor="0.2",
        linewidth=0.4,
        alpha=0.85,
        label="hybrid (obs EN4 $\\times$ product $V_1$)",
    )
    ax_r.bar(
        x + w,
        hyb_rg,
        w,
        color="#9e9e9e",
        edgecolor="0.2",
        linewidth=0.4,
        alpha=0.85,
        label="hybrid (obs RG09 $\\times$ product $V_1$)",
    )
    ax_r.axhline(0, color="0.5", lw=0.6)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(prods, rotation=12)
    ax_r.set_ylabel(r"salinity-driven component $\Delta F_s$ (mSv)")
    ax_r.set_title(
        "b   Own vs observation-constrained $\\Delta F_s$\n"
        "(identical 2006-2012 vs 2018-2024 windows)",
        loc="left",
    )
    ax_r.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "FigureS9_obs_hybrid_dfs.pdf", bbox_inches="tight")
    print("saved FigureS9_obs_hybrid_dfs.pdf")


if __name__ == "__main__":
    main()
