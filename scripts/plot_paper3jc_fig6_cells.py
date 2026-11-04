#!/usr/bin/env python3
"""Figure 6 (Journal of Climate): the overturning cells of the streamfunction.

a  Time-mean streamfunction of each product after removal of the net section
   transport, on a depth axis. The ORAS5 upper cell (mid grey) and abyssal cell
   (pale grey) are shaded. Each product marks its streamfunction maximum with its
   product marker (ORAS5 circle, GLORYS12V1 square, ECCO-V4r4 triangle), and the
   depth at which its time-mean curve returns to zero (the boundary between the two
   cells) with a horizontal bar in the product colour. Direct labels carry the
   maximum in Sv and its depth in m.
b  Annual freshwater transport. Upper panel: the upper cell (solid). Lower strip,
   with its own scale: the abyssal cell (dashed). Each product in its colour, with
   its product marker at the last year of its record.
c  Trends over each product's own record (95 per cent intervals). Rows are the
   full column (total), the upper cell, its transport term, its salinity term, the
   residual (no interval) and the abyssal cell, in the greys and marker shapes of
   TERM_STYLE. Filled markers with solid (abyssal: dashed) lines mark trends
   significant at 5 per cent (Santer-corrected p); open markers with dotted grey
   lines are intervals that include zero. The right-hand column gives each trend in
   mSv per yr.

Data handling follows scripts/plot_paper3r1_fig6_cells.py: the time-mean
streamfunction from data/results/paper3r1_section_<key>.nc, annual means of
data/results/paper3r1_cells_<key>.nc, and the trends and intervals of
PAPER_3_v2/analysis/r1_cells.json (windows 'own'). Record periods are read from
the time coordinate of the cells NetCDF files, never typed.

Writes PAPER_3_v2/figures_jclim/Fig6_cells.{pdf,png} at the final size. Pass a
preview path as the first argument to also write a 200 dpi preview there.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper3jc_style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402  (tick labels use the true minus)

RES = REPO / "data" / "results"
AN = REPO / "PAPER_3_v2" / "analysis"
OUTDIR = REPO / "PAPER_3_v2" / "figures_jclim"
OUT = OUTDIR / "Fig6_cells"

# (key, label, product colour, product marker)
PROD = [
    ("oras5", "ORAS5", st.PRODUCT_COLOURS["ORAS5"], st.PRODUCT_MARKERS["ORAS5"]),
    ("glorys12", "GLORYS12V1", st.PRODUCT_COLOURS["GLORYS12V1"], st.PRODUCT_MARKERS["GLORYS12V1"]),
    ("ecco", "ECCO-V4r4", st.PRODUCT_COLOURS["ECCO-V4r4"], st.PRODUCT_MARKERS["ECCO-V4r4"]),
]
MINUS = "−"
ALPHA = 0.05
FIG_W, FIG_H = st.TWO_COL, 6.60
C_NS = "0.45"      # interval that includes zero (dotted)
C_NS_EDGE = "0.25"
C_SIG = "black"    # interval that excludes zero

# Panel c rows: (label, indent level, source, key, term in TERM_STYLE)
# source 'win': trend of the window; 'att': term of the upper-cell attribution;
# 'res': residual of the attribution (upper cell minus the two terms), no interval.
ROWS = [
    ("full column", 0, "win", "trend_F_ov", "total"),
    ("upper cell", 1, "win", "trend_F_up", "upper_cell"),
    ("transport term", 2, "att", "T", "transport"),
    ("salinity term", 2, "att", "S", "salinity"),
    ("residual", 2, "res", "res", "residual"),
    ("abyssal cell", 1, "win", "trend_F_ab", "abyssal_cell"),
]
ABYSSAL_ROW = "abyssal_cell"
XLIM_C = (-8.0, 2.5)
GROUP_STEP = 8.0
N_ROWS = len(ROWS)
N_TESTS = sum(1 for row in ROWS if row[2] != "res")  # rows with a trend and a p value


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def mean_psi(key: str):
    """Identical to the reference script: time-mean streamfunction and its extrema."""
    ds = xr.open_dataset(RES / f"paper3r1_section_{key}.nc")
    e3t = ds["e3t"].values
    q = ds["V_bc"].mean("time").values * e3t
    psi = np.concatenate([[0.0], np.cumsum(q)]) / 1e6
    zw = np.concatenate([[0.0], np.cumsum(e3t)])
    kmax = int(np.argmax(psi))
    i = next(i for i in range(kmax + 1, len(psi) - 1) if psi[i] < 0)
    z0 = zw[i - 1] + psi[i - 1] / (psi[i - 1] - psi[i]) * (zw[i] - zw[i - 1])
    return psi, zw, zw[kmax], psi[kmax], z0


def annual_cells(key: str):
    """Annual means of the upper and abyssal cell transports (reference script)."""
    ds = xr.open_dataset(RES / f"paper3r1_cells_{key}.nc").groupby("time.year").mean()
    return ds["year"].values, ds["F_up"].values, ds["F_ab"].values


def record_period(key: str) -> str:
    """First and last year of the product record, from the time coordinate."""
    ds = xr.open_dataset(RES / f"paper3r1_cells_{key}.nc")
    years = ds["time"].dt.year.values
    return f"{int(years.min())}–{int(years.max())}"


def minus(text: str) -> str:
    return text.replace("-", MINUS)


def add_ax(x0: float, y0: float, w: float, h: float):
    """Axes placed in inches from the top-left corner of the figure."""
    return FIG.add_axes([x0 / FIG_W, 1 - (y0 + h) / FIG_H, w / FIG_W, h / FIG_H])


def draw_interval(ax, y, lo, hi, xlo, xhi, dashed_when_sig: bool = False):
    """95 per cent interval: solid (dashed for the abyssal row) when it excludes zero,
    dotted grey when it includes zero. Bounds beyond the axis end in an arrow."""
    inc0 = lo <= 0 <= hi
    if inc0:
        ls, col = ":", C_NS
    else:
        ls, col = ("--" if dashed_when_sig else "-"), C_SIG
    a, b = min(lo, hi), max(lo, hi)
    ca, cb = max(a, xlo), min(b, xhi)
    if cb > ca:
        ax.plot([ca, cb], [y, y], color=col, lw=0.8, ls=ls,
                solid_capstyle="butt", zorder=3)
    for bound, sign in ((a, -1), (b, +1)):
        if (sign < 0 and bound < xlo) or (sign > 0 and bound > xhi):
            edge = xlo if sign < 0 else xhi
            ax.annotate(
                "", xy=(edge, y), xytext=(edge - sign * 0.45, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.6, mutation_scale=4.5),
                zorder=4,
            )
    return inc0


def main(preview: str | None = None) -> None:
    st.set_style()
    global FIG
    cells = json.loads((AN / "r1_cells.json").read_text())

    # ------------------------------------------------------------ data
    psi_data = {key: mean_psi(key) for key, *_ in PROD}
    cell_data = {key: annual_cells(key) for key, *_ in PROD}
    periods = {key: record_period(key) for key, *_ in PROD}

    FIG = plt.figure(figsize=(FIG_W, FIG_H))

    # ------------------------------------------------------------ a
    ax_a = add_ax(0.62, 0.30, 1.95, 5.60)
    psi, zw, zmax, pmax, z0 = psi_data["oras5"]
    i_cross = int(np.searchsorted(zw, z0))  # zw[i_cross-1] < z0 <= zw[i_cross]
    zs_up = np.append(zw[:i_cross], z0)
    ps_up = np.interp(zs_up, zw, psi)
    zs_ab = np.concatenate([[z0], zw[i_cross:]])
    ps_ab = np.interp(zs_ab, zw, psi)
    ax_a.fill_betweenx(zs_up, 0.0, ps_up, color=st.C_CELL_UP, alpha=0.55, lw=0, zorder=1)
    ax_a.fill_betweenx(zs_ab, 0.0, ps_ab, color=st.C_CELL_AB, alpha=0.9, lw=0, zorder=1)
    ax_a.axvline(0, color="black", lw=0.5, zorder=2)
    for key, label, col, mark in PROD:
        psi, zw, zmax, pmax, z0 = psi_data[key]
        ax_a.plot(psi, zw, color=col, lw=1.0, zorder=3)
        # maximum: product marker; zero crossing: horizontal bar in the product colour
        ax_a.plot(pmax, zmax, mark, color=col, ms=4.4, mec="black", mew=0.5, zorder=4)
        ax_a.plot(0, z0, marker="_", color=col, ms=6.0, mew=1.5, ls="none", zorder=4)
    ax_a.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax_a.set_ylim(5800, 0)
    ax_a.set_xlim(-4.5, 21.0)
    ax_a.set_xticks([0, 10, 20])
    ax_a.set_yticks([0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000])
    ax_a.set_xlabel("Streamfunction (Sv)")
    ax_a.set_ylabel("Depth (m)")
    # horizontal label inside the upper-cell shading, clear of every curve at this depth
    ax_a.text(4.6, 760, "upper cell", ha="center", va="center", fontsize=7,
              color="black", zorder=6)
    ax_a.annotate("abyssal cell", xy=(-0.5, 4650), xytext=(3.0, 4650), fontsize=7,
                  ha="left", va="center", color="black", zorder=6,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color="black",
                                  shrinkA=2, shrinkB=0))
    # direct labels: the product marker, then the maximum of the plotted (time-mean)
    # streamfunction (the depth is given in the text as the median monthly value), the values the markers in the panel mark
    label_depth = {"oras5": 3300, "glorys12": 3720, "ecco": 4140}
    for key, label, col, mark in PROD:
        _psi, _zw, zmax_p, pmax_p, _z0 = psi_data[key]
        ylab = label_depth[key]
        ax_a.plot([4.9], [ylab], mark, color=col, ms=4.4, mec="black", mew=0.5,
                  ls="none", zorder=6, clip_on=False)
        ax_a.text(5.9, ylab, f"{label}: {pmax_p:.1f} Sv", color="black",
                  fontsize=7, ha="left", va="center", zorder=6)
    # key for the two marker kinds, in black (the product colour is on the curves)
    ax_a.plot([8.2], [5150], "o", color="black", ms=4.0, mec="black", mew=0.5,
              ls="none", zorder=6)
    ax_a.text(9.2, 5150, "maximum", ha="left", va="center", fontsize=7, zorder=6)
    ax_a.plot([8.2], [5450], marker="_", color="black", ms=6.0, mew=1.5, ls="none",
              zorder=6)
    ax_a.text(9.2, 5450, "zero crossing", ha="left", va="center", fontsize=7, zorder=6)
    st.panel_label_pt(ax_a, "a")

    # ------------------------------------------------------------ b
    ax_bu = add_ax(3.05, 0.30, 3.25, 1.50)
    ax_bs = add_ax(3.05, 1.95, 3.25, 0.78)
    for key, label, col, mark in PROD:
        yr, fup, fab = cell_data[key]
        ax_bu.plot(yr, fup, color=col, lw=1.0, zorder=3)
        ax_bu.plot([yr[-1]], [fup[-1]], mark, color=col, ms=3.6, mec="black", mew=0.4,
                   zorder=4)
        ax_bu.text(yr[-1] + 2.5, fup[-1], label, color="black", fontsize=7, ha="left",
                   va="center")
        ax_bs.plot(yr, fab, color=col, lw=0.9, ls="--", zorder=3)
        ax_bs.plot([yr[-1]], [fab[-1]], mark, color=col, ms=3.6, mec="black", mew=0.4,
                   zorder=4)
    ax_bu.axhline(0, color="0.5", lw=0.5, zorder=1)
    ax_bu.set_xlim(1957, 2044)
    ax_bu.set_ylim(-0.20, 0.03)
    ax_bu.set_xticks(range(1960, 2021, 10))
    ax_bu.tick_params(labelbottom=False)
    ax_bu.set_yticks([-0.20, -0.15, -0.10, -0.05, 0.0])
    ax_bu.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: minus(f"{v:.2f}")))
    ax_bu.set_ylabel("Freshwater transport (Sv)")
    ax_bu.plot([1958.5, 1964.5], [-0.186, -0.186], color="black", lw=1.0, zorder=5)
    ax_bu.text(1966.0, -0.186, "upper cell", fontsize=7, ha="left", va="center")
    # strip with its own scale for the abyssal cell
    ax_bs.axhline(0, color="0.5", lw=0.5, zorder=1, ls=":")
    ax_bs.set_xlim(1957, 2044)
    ax_bs.set_ylim(0.0, 0.024)
    ax_bs.set_xticks(range(1960, 2021, 10))
    ax_bs.set_yticks([0.0, 0.01, 0.02])
    ax_bs.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.2f}"))
    ax_bs.set_xlabel("Year")
    ax_bs.set_ylabel("Abyssal cell (Sv)")
    ax_bs.plot([1958.5, 1964.5], [0.0215, 0.0215], color="black", lw=0.9, ls="--",
               zorder=5)
    ax_bs.text(1966.0, 0.0215, "abyssal cell", fontsize=7, ha="left", va="center")
    st.panel_label_pt(ax_bu, "b")

    # ------------------------------------------------------------ c
    ax_c = add_ax(3.92, 3.22, 2.00, 2.95)
    yticks, ylabels = [], []
    for g, (key, label, col, mark) in enumerate(PROD):
        base = g * GROUP_STEP
        att = cells[key]["windows"]["own"]["upper_cell_attribution"]
        win = cells[key]["windows"]["own"]
        n_sig = 0
        for r, (name, indent, src, k, term) in enumerate(ROWS):
            y = base + 1.0 + r
            if src == "win":
                t = win[k]
                est, lo, hi, p = t["trend"], t["ci"][0], t["ci"][1], t["p_santer"]
            elif src == "att":
                if k == "T":
                    est = att["trend_T_mSv_per_yr"]
                    lo, hi = att["ci_T_mSv_per_yr"][0], att["ci_T_mSv_per_yr"][1]
                    p = att["p_T"]
                else:
                    est = att["trend_S_mSv_per_yr"]
                    lo, hi = att["ci_S_mSv_per_yr"][0], att["ci_S_mSv_per_yr"][1]
                    p = att["p_S"]
            else:
                est, lo, hi, p = att["residual_mSv_per_yr"], None, None, None
            if src != "res":
                sig = p < ALPHA
                n_sig += int(sig)
                inc0 = draw_interval(ax_c, y, lo, hi, XLIM_C[0], XLIM_C[1],
                                     dashed_when_sig=(term == ABYSSAL_ROW))
                if inc0 == sig:  # open marker (interval includes zero) must agree with p
                    print(f"CHECK: p and interval disagree for {key} {name}: "
                          f"p={p:.3g} ci=({lo:.3g}, {hi:.3g})")
                ax_c.plot([est], [y], **st.term_marker(term, significant=not inc0, ms=4.2),
                          zorder=5)
            else:
                ax_c.plot([est], [y], **st.term_marker(term, significant=True, ms=4.2),
                          zorder=5)
            yticks.append(y)
            ylabels.append(" " * (2 * indent) + name)
            ax_c.text(1.02, y, minus(f"{est:+.2f}"), transform=ax_c.get_yaxis_transform(),
                      ha="left", va="center", fontsize=7, clip_on=False)
        # group header: product marker and name, record period and count
        yticks.append(base)
        ylabels.append(label)
        ax_c.plot([XLIM_C[0] + 0.30], [base], mark, color=col, ms=4.0, mec="black",
                  mew=0.5, ls="none", zorder=6, clip_on=False)
        hdr_box = dict(fc="white", ec="none", pad=0.3)
        ax_c.text(XLIM_C[0] + 0.75, base, periods[key], color="black", fontsize=7,
                  ha="left", va="center", zorder=6, bbox=hdr_box)
        ax_c.text(XLIM_C[1] - 0.25, base, f"{n_sig} of {N_TESTS} significant",
                  color="black", fontsize=7, ha="right", va="center", zorder=6, bbox=hdr_box)
        if key == "oras5":
            # the transport term share of the upper-cell trend, from the attribution block
            share = att["share_T_point"] * 100
            lo_s, hi_s = att["share_T_ci"][0] * 100, att["share_T_ci"][1] * 100
            ax_c.text(-0.35, base + 3.0, f"transport share {share:.0f}% [{lo_s:.0f}%, {hi_s:.0f}%]",
                      fontsize=7, ha="right", va="center", zorder=6)
    ax_c.text(1.02, 1.0, "trend\nmSv yr$^{-1}$", transform=ax_c.transAxes, ha="left",
              va="bottom", fontsize=7, linespacing=1.1, clip_on=False)
    ax_c.axvline(0, color="black", lw=0.5, zorder=1)
    ax_c.set_xlim(*XLIM_C)
    ax_c.set_ylim(GROUP_STEP * 2 + N_ROWS + 1.0, -0.7)
    ax_c.set_yticks(yticks)
    ax_c.set_yticklabels(ylabels)
    ax_c.tick_params(axis="y", length=0)
    for lbl in ax_c.get_yticklabels():
        if lbl.get_text() in [p[1] for p in PROD]:
            lbl.set_fontweight("bold")
            lbl.set_color("black")
    ax_c.set_xticks([-8, -6, -4, -2, 0, 2])
    ax_c.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: minus(f"{v:g}")))
    ax_c.set_xlabel("Trend (mSv yr$^{-1}$)")
    st.panel_label_pt(ax_c, "c", dx_pt=-30.0, dy_pt=5.0)

    # ------------------------------------------------------------ checks and save
    st.check_min_fontsize(FIG)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    st.save(FIG, str(OUT))
    if preview:
        FIG.savefig(preview, format="png", dpi=200)
        print("wrote preview", preview)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
