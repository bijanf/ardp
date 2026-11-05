#!/usr/bin/env python3
"""Figure 4, Journal of Climate version: where the northward limb salinifies.

Same data and computations as scripts/plot_paper3r1_fig4_salinification.py.
Panels keep the letters cited in Main.tex.

(a) Transport-weighted salinity of the northward limb (solid) and southward
    limb (dashed) as anomalies from their record means; grey zero line.
(b) Zonal-mean salinity trend against depth over the full water column, bold
    where the Santer-adjusted p is below 0.05; gray bands mark 0-1000 m and
    4000 m to the bottom.
(c) Longitude-depth section of the salinity trend in GLORYS12V1, 2005-2024.
(d) The same in the EN4.2.2 objective analysis on its 34 S row.
(e) Contributions of four longitude bands and three layers to the trend of the
    northward-limb salinity with the circulation held at its record mean,
    2005-2024, one panel per product; a black horizontal bar gives each band's
    contribution for a trend that is uniform along the section.

Depth axes use a square-root scale (stated on the axes and in the caption).
Stipples in (c) and (d) sit on every second longitude (2 degrees) and every
second depth level in both sections.

Writes PAPER_3_v2/figures_jclim/Fig4_salinification.{pdf,png} at 6.5 in wide.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

import paper3jc_style as st  # noqa: E402

RES = REPO / "data" / "results"
AN = REPO / "PAPER_3_v2" / "analysis"
OUTDIR = REPO / "PAPER_3_v2" / "figures_jclim"
OUT = OUTDIR / "Fig4_salinification"

W_IN, H_IN = st.TWO_COL, 7.9

PROD = [
    ("oras5", "ORAS5", st.C_ORAS5),
    ("glorys12", "GLORYS12V1", st.C_GLORYS),
    ("ecco", "ECCO-V4r4", st.C_ECCO),
]
BANDS = [(-54.0, -45.0), (-45.0, -20.0), (-20.0, 0.0), (0.0, 19.4)]
BAND_LABELS = ["WB", "WI", "EI", "CB"]
BAND_KEYS = [
    "western boundary (west of 45 W)",
    "western interior (45 W to 20 W)",
    "eastern interior (20 W to 0)",
    "Cape Basin (east of 0)",
]
LAYERS = ["0-100 m", "100-500 m", "500-1300 m"]
LAYER_TEXT = ["0–100 m", "100–500 m", "500–1300 m"]
LAYER_SHADE = ["0.14", "0.50", "0.84"]
LEV = np.array([-0.15, -0.10, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10, 0.15])
ANN = 7  # annotation size, pt
SEAFLOOR = "#4d4d4d"

# Line styles: solid northward limb and products; dashed southward limb (G3).
DASH_SOUTH = (0, (3.0, 1.4))
ZERO_LINE = dict(color="0.55", lw=0.5, zorder=1)  # thin grey zero line under the lines
STIPPLE_DEG = 2.0       # stipple spacing in longitude, degrees
STIPPLE_DEPTH_STEP = 2  # every second depth level
E_BAR_MARKER = dict(marker="_", ms=8, color="black", mew=1.3, ls="none", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])


def place(x0, y0, w, h):
    """Axes from top-left coordinates in inches (the figure is authored at size)."""
    return plt.gcf().add_axes([x0 / W_IN, 1 - (y0 + h) / H_IN, w / W_IN, h / H_IN])


def ftext(x, y, s, **kw):
    """Figure text at top-left coordinates in inches."""
    plt.gcf().text(x / W_IN, 1 - y / H_IN, s, **kw)


def product_label(ax, x, y, name, colour, dx_pt=5.0):
    """Direct label for a product line: its marker shape at the data point, then the name."""
    ax.plot([x], [y], ls="none", marker=st.PRODUCT_MARKERS[name], ms=3.2, mfc=colour,
            mec=colour, mew=0.6, zorder=6, clip_on=False)
    ax.annotate(name, xy=(x, y), xytext=(dx_pt, 0), textcoords="offset points",
                color=colour, fontsize=ANN, ha="left", va="center", zorder=6)


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def trend_profile(ds: xr.Dataset, y0: int, y1: int):
    a = ds["S_bar"].groupby("time.year").mean()
    yrs = a["year"].values
    m = (yrs >= y0) & (yrs <= y1)
    tr, pv = [], []
    for k in range(a.sizes["depth"]):
        y = a.values[m, k]
        if np.ptp(y) == 0 or not np.isfinite(y).all():
            tr.append(np.nan)
            pv.append(np.nan)
            continue
        f = ols_santer(yrs[m].astype(float), y)
        tr.append(f["slope"] * 10)
        pv.append(f["p_santer"])
    return np.array(tr), np.array(pv)


def upper500_share(rec: dict) -> float:
    """Share of the northward-limb trend from the layers above 500 m (as in panel e)."""
    total = rec["S_north_frozen_trend_sum_of_cells"]
    return (rec["northward_by_layer"]["0-100 m"] + rec["northward_by_layer"]["100-500 m"]) / total


def section_panel(ax, fname: str):
    """Longitude-depth trend section, as in the reference script. Returns the mappable."""
    ds = xr.open_dataset(RES / fname)
    lon, depth = ds["lon"].values, ds["depth"].values.copy()
    depth[0] = 0.0  # draw the uppermost level up to the surface
    tr = ds["trend"].values
    fin = np.isfinite(tr)
    # seafloor: the depth midway between each column's deepest finite level and the next level
    zb = np.empty(len(lon))
    for i in range(len(lon)):
        idx = np.where(fin[:, i])[0]
        if idx.size == 0:
            zb[i] = depth[-1]
            continue
        k = idx.max()
        zb[i] = 0.5 * (depth[k] + depth[k + 1]) if k + 1 < len(depth) else depth[k]
    ax.fill_between(lon, zb, 2000.0, color=SEAFLOOR, lw=0, zorder=0.5, step="mid")
    cmap = plt.get_cmap("BrBG_r", len(LEV) + 1)
    cf = ax.contourf(
        lon,
        depth,
        tr,
        levels=LEV,
        cmap=cmap,
        extend="both",
        norm=colors.BoundaryNorm(LEV, cmap.N, extend="both"),
        zorder=1,
    )
    ax.contour(lon, depth, tr, levels=[-0.03], colors="k", linewidths=0.35,
               linestyles="--", zorder=2)
    ax.contour(lon, depth, tr, levels=[0.03], colors="k", linewidths=0.35,
               linestyles="-", zorder=2)
    # stipples: every second depth level, and one longitude per STIPPLE_DEG in both products
    p = ds["p"].values
    lon_step = max(1, int(round(STIPPLE_DEG / float(np.median(np.diff(lon))))))
    i_lon = np.arange(0, len(lon), lon_step)
    i_dep = np.arange(0, len(depth), STIPPLE_DEPTH_STEP)
    lon2, dep2 = np.meshgrid(lon[i_lon], depth[i_dep])
    sub = np.ix_(i_dep, i_lon)
    sig = (p[sub] < 0.05) & fin[sub]
    ax.plot(lon2[sig], dep2[sig], ls="none", marker="o", ms=2.5, mfc="black",
            mec="white", mew=0.3, zorder=4)
    for lo in [b[0] for b in BANDS[1:]]:
        ax.axvline(lo, color="0.3", lw=0.4, ls=":", zorder=3)
    ax.axhline(500, color="0.1", lw=0.6, zorder=3)
    ax.text(-27.0, 500, "500 m", fontsize=ANN, ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(1500, 0)
    ax.set_yticks([0, 100, 300, 500, 1000, 1500])
    ax.set_xlim(-54, 19.4)
    lt = [-50, -40, -30, -20, -10, 0, 10]
    ax.set_xticks(lt)
    ax.set_xticklabels([f"{-v}°W" if v < 0 else ("0°" if v == 0 else f"{v}°E") for v in lt])
    ax.set_facecolor(SEAFLOOR)
    ax.set_xlabel("Longitude")
    return cf


def main() -> None:
    st.set_style()
    plt.close("all")
    fig = plt.figure(figsize=(W_IN, H_IN))
    struct = json.loads((AN / "r1_salinification_structure.json").read_text())

    # records for panel (e) and the section titles
    sets = [
        ("EN4.2.2", "ORAS5 circulation", struct["en4"]["frozen"]["oras5"],
         struct["en4"]["band_width_fraction"]),
        ("ORAS5", "own circulation", struct["oras5"]["windows"]["2005-2024"],
         struct["oras5"]["windows"]["2005-2024"]["band_width_fraction"]),
        ("GLORYS12V1", "own circulation", struct["glorys12"]["windows"]["2005-2024"],
         struct["glorys12"]["windows"]["2005-2024"]["band_width_fraction"]),
    ]
    up_en4 = upper500_share(sets[0][2])
    up_glo = upper500_share(sets[2][2])

    # ------------------------------------------------------------ (a)
    ax = place(0.62, 0.30, 2.75, 1.25)
    ax.axhline(0, **ZERO_LINE)
    for key, _label, col in PROD:
        ds = xr.open_dataset(RES / f"paper3r1_section_{key}.nc").groupby("time.year").mean()
        yrs = ds["year"].values
        lw = 0.9 if key != "ecco" else 0.8
        for var, ls in (("S_north", "-"), ("S_south", DASH_SOUTH)):
            v = ds[var].values
            ax.plot(yrs, v - v.mean(), color=col, lw=lw, ls=ls, zorder=2)
    ax.set_xlim(1958, 2026)
    ax.set_ylim(-0.2, 0.16)
    ax.set_xlabel("Year")
    ax.set_ylabel("Salinity anomaly\n(PSU)")
    product_label(ax, 1960, -0.125, "ORAS5", st.C_ORAS5)
    product_label(ax, 2003, -0.150, "GLORYS12V1", st.C_GLORYS)
    product_label(ax, 2012.5, -0.064, "ECCO-V4r4", st.C_ECCO)
    # key with line samples, black, same styles as the lines
    ax.plot([1959, 1965], [0.145, 0.145], color="black", lw=1.0, ls="-")
    ax.plot([1959, 1965], [0.112, 0.112], color="black", lw=1.0, ls=DASH_SOUTH)
    ax.text(1966.5, 0.145, "northward limb", fontsize=ANN, ha="left", va="center")
    ax.text(1966.5, 0.112, "southward limb", fontsize=ANN, ha="left", va="center")
    st.panel_label_pt(ax, "a")

    # ------------------------------------------------------------ (b)
    ax = place(4.05, 0.30, 2.2, 1.25)
    for key, _label, col in PROD:
        ds = xr.open_dataset(RES / f"paper3r1_section_{key}.nc")
        y0, y1 = (1992, 2017) if key == "ecco" else (1993, 2025)
        tr, pv = trend_profile(ds, y0, y1)
        z = ds["depth"].values
        ax.plot(tr, z, color=col, lw=0.6)
        sig = np.where(pv < 0.05, tr, np.nan)
        ax.plot(sig, z, color=col, lw=1.8)
    ax.axhspan(0, 1000, color="0.85", lw=0, zorder=0)
    ax.axhspan(4000, 5800, color="0.85", lw=0, zorder=0)
    ax.axvline(0, color="0.45", lw=0.4, zorder=1)
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(5800, 0)
    ax.set_yticks([0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel("Salinity trend (PSU decade$^{-1}$)")
    ax.set_ylabel("Depth (m),\nsquare-root scale")
    product_label(ax, 0.052, 70, "GLORYS12V1", st.C_GLORYS, dx_pt=4.0)
    product_label(ax, 0.030, 330, "ORAS5", st.C_ORAS5, dx_pt=4.0)
    product_label(ax, 0.005, 2900, "ECCO-V4r4", st.C_ECCO, dx_pt=4.0)
    st.panel_label_pt(ax, "b")

    # ------------------------------------------------------------ (c), (d)
    SEC_Y, SEC_H = 2.40, 2.45
    axc = place(0.62, SEC_Y, 2.80, SEC_H)
    cf = section_panel(axc, "paper3r1_salinity_trend_section_glorys12_2005-2024.nc")
    axc.set_ylabel("Depth (m),\nsquare-root scale")
    axd = place(3.62, SEC_Y, 2.80, SEC_H)
    section_panel(axd, "paper3r1_salinity_trend_section_en4_2005-2024.nc")
    axd.set_yticklabels([])
    for a_ in (axc, axd):
        for lo, hi, lab in zip(
            [b[0] for b in BANDS], [b[1] for b in BANDS], BAND_LABELS, strict=False
        ):
            a_.text(0.5 * (lo + hi), 1.01, lab, transform=a_.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=ANN)
    st.panel_label_pt(axc, "c")
    st.panel_label_pt(axd, "d")
    ftext(0.42, 1.98, "GLORYS12V1, 2005–2024", fontsize=ANN + 1, ha="left", va="top")
    ftext(0.42, 2.13, f"upper 500 m {100 * up_glo:.0f}% of limb trend", fontsize=ANN,
          ha="left", va="top")
    ftext(4.02, 1.98, "EN4.2.2, 2005–2024", fontsize=ANN + 1, ha="left", va="top")
    ftext(4.02, 2.13, f"upper 500 m {100 * up_en4:.0f}% of limb trend", fontsize=ANN,
          ha="left", va="top")

    # shared colour bar under both sections, the width of the section pair;
    # every colour boundary carries a tick label
    cax = place(0.62, 5.40, 5.80, 0.12)
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEV)
    cb.ax.set_xticklabels([f"{v:+.2f}".replace("-", "−") for v in LEV])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.4)
    ftext(0.62, 5.74, "freshening", fontsize=ANN, ha="left", va="top")
    ftext(6.42, 5.74, "salinification", fontsize=ANN, ha="right", va="top")
    ftext(3.52, 5.74, "Salinity trend (PSU decade$^{-1}$)", fontsize=8, ha="center", va="top")

    # ------------------------------------------------------------ (e)
    E_Y, E_H = 6.58, 1.12
    colour_title = [st.C_EN4, st.C_ORAS5, st.C_GLORYS]
    prod_name = ["EN4.2.2", "ORAS5", "GLORYS12V1"]
    xs = [0.66, 2.36, 4.06]
    panel_w = 1.5
    axes_e = []
    printed = []
    for j, ((name, circ, rec, wfrac), x0) in enumerate(zip(sets, xs, strict=False)):
        ax = place(x0, E_Y, panel_w, E_H)
        axes_e.append(ax)
        total = rec.get("S_north_frozen_trend_sum_of_cells", rec.get("S_north_frozen_trend_direct"))
        tab = rec.get("northward_by_band_and_layer")
        upper500 = upper500_share(rec)
        for i, b in enumerate(BAND_KEYS):
            x = i
            pos_bottom, neg_bottom = 0.0, 0.0
            for ly, sh in zip(LAYERS, LAYER_SHADE, strict=False):
                if tab is not None:
                    val = tab[f"{b} | {ly}"]
                else:
                    val = (rec["northward_by_band"][b] * rec["northward_by_layer"][ly]
                           / sum(rec["northward_by_layer"][q] for q in LAYERS))
                if val >= 0:
                    ax.bar(x, val, 0.55, bottom=pos_bottom, color=sh, ec="black", lw=0.4)
                    pos_bottom += val
                else:
                    ax.bar(x, val, 0.55, bottom=neg_bottom, color=sh, ec="black", lw=0.4)
                    neg_bottom += val
            uniform = total * wfrac[b]
            ax.plot(x, uniform, **E_BAR_MARKER)
            share = 100 * rec["northward_by_band"][b] / total
            printed.append((name, BAND_LABELS[i], round(share, 1), round(uniform, 5)))
            top = max(pos_bottom, uniform) + 0.00045
            ax.text(x, top, f"{share:.0f}%", ha="center", va="bottom", fontsize=ANN)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.0008, 0.0185)
        ax.set_yticks([0, 0.005, 0.010, 0.015])
        ax.set_yticklabels(["0", "0.005", "0.010", "0.015"] if j == 0 else [])
        ax.set_xticks(range(4))
        ax.set_xticklabels(BAND_LABELS)
        ax.tick_params(axis="x", length=0)
        # product name in black, its marker in the product colour beside it
        tr_mark = offset_copy(ax.transAxes, fig=fig, x=0, y=26, units="points")
        ax.plot([0.0], [1.0], transform=tr_mark, ls="none",
                marker=st.PRODUCT_MARKERS[prod_name[j]], ms=4.2, mfc=colour_title[j],
                mec=colour_title[j], mew=0.6, clip_on=False)
        ax.annotate(prod_name[j], xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(8, 23), textcoords="offset points",
                    color="black", fontsize=8, ha="left", va="bottom")
        ax.annotate(circ, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=ANN, ha="left", va="bottom")
        ax.annotate(f"upper 500 m {100 * upper500:.0f}%", xy=(0.0, 1.0),
                    xycoords="axes fraction", xytext=(8, 2), textcoords="offset points",
                    fontsize=ANN, ha="left", va="bottom")
        if j > 0:
            ax.set_yticklabels([])
    axes_e[0].set_ylabel("$\\mathrm{S_N}$ contribution\n(PSU decade$^{-1}$)")
    st.panel_label_pt(axes_e[0], "e")

    # key for the bars and the percentages, top right of the row
    fig.add_artist(Line2D([5.66 / W_IN], [1 - 6.20 / H_IN], transform=fig.transFigure,
                          **E_BAR_MARKER))
    ftext(5.80, 6.20, "uniform trend", fontsize=ANN, ha="left", va="center")
    ftext(5.66, 6.34, "% share of total", fontsize=ANN, ha="left", va="center")

    # direct labels for the layers, beside the Cape Basin bars of the last panel
    last = axes_e[-1]
    rec = sets[-1][2]
    cb_key = BAND_KEYS[3]
    levels = [rec["northward_by_band_and_layer"][f"{cb_key} | {ly}"] for ly in LAYERS]
    base = 0.0
    for ly_text, val in zip(LAYER_TEXT, levels, strict=False):
        mid = base + 0.5 * val
        base += val
        last.text(3.36, mid, ly_text, fontsize=ANN, ha="left", va="center")

    # values plotted, for the numbers check
    print(f"upper 500 m share GLORYS12V1 {100 * up_glo:.1f}%, EN4.2.2 {100 * up_en4:.1f}%")
    for row in printed:
        print("panel e", row)

    small = st.check_min_fontsize(fig)
    if small:
        print(f"{len(small)} text items below 7 pt")
    st.save(fig, str(OUT))


if __name__ == "__main__":
    main()
