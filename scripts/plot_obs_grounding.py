#!/usr/bin/env python3
"""Paper 2 Figure 4 — Observational grounding for the salinity-driven mechanism.

Three-panel figure consuming the new Argo / EN4 analyses plus the existing
reanalysis salinity trends. Built to convey: "two independent observational
products + the SAMBA in-situ record favour the salinity-redistribution
interpretation; one reanalysis (ORAS5) is the outlier."

  (a)  Atlantic-basin (34.5°S ± 5°, full Atlantic longitudes) upper-300 m
       salinity time series 2005–2024 for EN4.2.2 and Roemmich–Gilson Argo.
       Volkov-2024 SAMBA trend (+0.05 PSU/dec) shown as an annotated band.
  (b)  EN4.2.2 zonal-depth ΔS section at 34.5°S (2020–2024 minus 2005–2009),
       0–1000 m. Bootstrap-significant cells stippled.
  (c)  Bar chart: upper-300 m Atlantic basin salinity trend ± 95 % CI for
       EN4.2.2, RG09 (if available), GLORYS12V1 (literature value),
       ORAS5 (literature value), with Volkov-2024 SAMBA value as a
       vertical reference line.

Outputs:
  figures/paper2/Figure4_obs_grounding.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
FIG_DIR = REPO / "figures" / "paper2"

# Discrete diverging colormap for ΔS (PSU) — symmetric, equally-spaced bins
CM = plt.cm.RdBu_r
LEVELS = np.arange(-0.30, 0.31, 0.05)  # bin width 0.05 PSU, symmetric about 0
NORM = BoundaryNorm(LEVELS, ncolors=CM.N, clip=False)

# Product palette
COLR = {
    "EN4.2.2":          "#17becf",   # teal
    "RG09":             "#7f7f7f",   # mid-grey
    "ORAS5":            "#1f77b4",   # navy
    "GLORYS12V1":       "#2ca02c",   # green
    "SAMBA-Volkov2024": "#9467bd",   # purple
}

# Literature trend values for the reanalysis comparison bar chart (Panel C).
# Sources: project MEMORY.md (1993-2025 trends), to be replaced with locally
# recomputed 2005-2024 values once the corresponding reanalysis trend scripts
# are run on the same window.
LIT_TRENDS = {
    "GLORYS12V1": (0.084, 0.025, "MEMORY.md, 1993-2025"),
    "ORAS5":      (-0.014, 0.020, "MEMORY.md, 1993-2025"),
}


# -------------------------------------------------------------------------
# Panel A — time series
# -------------------------------------------------------------------------
def _panel_a(ax: plt.Axes) -> None:
    df = pd.read_csv(RESULTS / "argo_basin_mean.csv")
    for product, sub in df.groupby("product"):
        c = COLR.get(product, "0.5")
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["salinity_psu"], marker="o", markersize=3.5,
                color=c, linewidth=1.0, label=product)
    ax.set_xlabel("Year")
    ax.set_ylabel("Upper-300 m basin-mean salinity (PSU)")
    ax.legend(loc="lower right", frameon=False, fontsize=5.5,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)


# -------------------------------------------------------------------------
# Panel B — EN4 zonal section
# -------------------------------------------------------------------------
def _panel_b(ax: plt.Axes, fig: plt.Figure) -> None:
    """ΔS zonal section. Colorbar is placed via fig.colorbar(ax=ax, location='right')
    — the constrained-layout solver allocates space without bleeding into Panel C."""
    ds = xr.open_dataset(RESULTS / "argo_zonal_section.nc")
    dS = ds["EN4_dS"]
    pvals = ds["EN4_pvalue"]
    pc = ax.pcolormesh(dS["lon"], dS["depth"], dS.values,
                       cmap=CM, norm=NORM, shading="nearest")
    # Stipple significant cells (p < 0.05), thinned to one mark per ~30x8 grid
    sig_mask = pvals.values < 0.05
    lon_grid, depth_grid = np.meshgrid(dS["lon"].values, dS["depth"].values)
    stride = max(1, lon_grid.shape[1] // 30)
    stride_d = max(1, lon_grid.shape[0] // 8)
    mask_thin = np.zeros_like(sig_mask)
    mask_thin[::stride_d, ::stride] = sig_mask[::stride_d, ::stride]
    ax.scatter(lon_grid[mask_thin], depth_grid[mask_thin],
               s=2, color="black", alpha=0.4, marker="o", linewidths=0)
    ax.invert_yaxis()
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(dS["lon"].min(), dS["lon"].max())
    ax.set_ylim(1000, 0)
    cb = fig.colorbar(pc, ax=ax, location="right",
                      shrink=0.85, pad=0.04,
                      ticks=np.arange(-0.30, 0.31, 0.10))
    cb.set_label("ΔS (PSU)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ds.close()


# -------------------------------------------------------------------------
# Panel C — trend bar chart
# -------------------------------------------------------------------------
def _panel_c(ax: plt.Axes) -> None:
    """Bar chart of 2005-2024 trends. SAMBA Volkov shown as a single dashed
    reference line; no annotation text on the canvas — caption explains."""
    with open(RESULTS / "argo_trends.json") as f:
        trends = json.load(f)

    rows: list[tuple[str, float, float]] = []
    if "EN4.2.2" in trends:
        t = trends["EN4.2.2"]
        rows.append(("EN4.2.2", t["slope_psu_per_dec"], t["ci95_half_psu_per_dec"]))
    if "RG09" in trends and np.isfinite(trends["RG09"].get("slope_psu_per_dec", np.nan)):
        t = trends["RG09"]
        rows.append(("RG09", t["slope_psu_per_dec"], t["ci95_half_psu_per_dec"]))
    rows.append(("GLORYS12V1", LIT_TRENDS["GLORYS12V1"][0], LIT_TRENDS["GLORYS12V1"][1]))
    rows.append(("ORAS5",      LIT_TRENDS["ORAS5"][0],      LIT_TRENDS["ORAS5"][1]))

    samba_value = trends["SAMBA-Volkov2024"]["slope_psu_per_dec"]
    samba_ci = trends["SAMBA-Volkov2024"]["ci95_half_psu_per_dec"]

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = [COLR.get(n, "0.5") for n in names]

    y = np.arange(len(names))
    ax.barh(y, vals, xerr=errs, color=colors, edgecolor="0.2",
            linewidth=0.4, error_kw={"linewidth": 0.7, "capsize": 2})
    ax.axvline(0, color="0.3", linewidth=0.4)
    # SAMBA Volkov 2024: faint band + dashed line. Caption explains.
    ax.axvspan(samba_value - samba_ci, samba_value + samba_ci,
               color=COLR["SAMBA-Volkov2024"], alpha=0.12)
    ax.axvline(samba_value, color=COLR["SAMBA-Volkov2024"], linewidth=0.7,
               linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Salinity trend (PSU / decade)")
    ax.set_xlim(-0.07, 0.13)
    ax.spines[["top", "right"]].set_visible(False)


# -------------------------------------------------------------------------
def build_figure() -> plt.Figure:
    # 2-column = 180 mm = 7.09" wide.  The basin-mean salinity time series
    # (former panel a) has been promoted to Figure 1 panel b, so this figure
    # now carries only the zonal-depth section and the trend bar chart.
    fig, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.0),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    ax_b, ax_c = axes

    _panel_b(ax_b, fig)
    _panel_c(ax_c)

    for ax, lbl in zip([ax_b, ax_c], "ab"):
        ax.text(-0.18, 1.10, lbl, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left")
    return fig


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf = FIG_DIR / "Figure4_obs_grounding.pdf"
    png = FIG_DIR / "Figure4_obs_grounding.png"
    fig.savefig(pdf, format="pdf", dpi=300)
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
