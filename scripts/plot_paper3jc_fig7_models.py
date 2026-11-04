#!/usr/bin/env python3
"""Figure 7 (Journal of Climate version): CMIP6 at 34.5 S on the reanalysis basis.

Same data handling as scripts/plot_paper3r1_fig7_models.py, which reads only
PAPER_3_v2/analysis/r1_cmip6.json (per_model, reanalysis_windows and summary).
Panel letters keep their meaning from the manuscript:

  a  Mean F_ovS over 1993-2014 in 25 CMIP6 models (grey bars) with the 1950-1980
     values (ticks) and the three reanalyses over 1993-2014 (solid thin lines in
     the product colours, with product markers in the key at the top).
  b  Hero. Models and reanalyses in the plane of the northward-limb transport T
     and the limb salinity contrast dS, shaded by F_ovS = -T dS / S0.
  c  Change at 34.5 S against change at 26.5 N, 1850-1900 to 2081-2100 (17 models).
  d  Mean F_ovS over 1950-1980 against the change at 26.5 N (21 models), with the
     least-squares line and its 95 per cent band of the mean, coloured by the
     1850-1900 overturning strength at 26.5 N. The panel prints the Spearman rank
     correlation (computed here from the plotted values, with a permutation p),
     and the partial correlation of the JSON that holds that strength fixed.

Every number written in the figure is read from the JSON at run time, except the
Spearman correlation, the permutation p and the least-squares band of (d), which are
computed here from the same plotted values. Writes
PAPER_3_v2/figures_jclim/Fig7_models.{pdf,png} at final print size.
Optional: --preview writes a 200 dpi check render to the figwork folder, and
the numbers written into the figure go to a JSON log there for verification.
Label positions in panel b are chosen by a clearance test against every
model and reanalysis marker, and the chosen offsets are written to the log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import paper3jc_style as st  # noqa: E402

AN = REPO / "PAPER_3_v2" / "analysis"
OUTDIR = REPO / "PAPER_3_v2" / "figures_jclim"
OUTBASE = OUTDIR / "Fig7_models"
WORK = OUTDIR / "checks"  # preview and the log of plotted numbers
PREVIEW = WORK / "Fig7_models_preview200.png"
LOG = WORK / "fig7_numbers.json"

S0 = 35.0          # PSU, as in the manuscript Methods and the old script
W = st.TWO_COL     # 6.5 in
H = 6.26           # in (content is shifted by DY so the margins are even)
DY = -0.21         # in, shift of all content towards the bottom edge
TOP_Y0, TOP_H = 2.66, 3.09    # top row (a, b), in inches before DY
BOT_Y0, BOT_H = 0.62, 1.23    # bottom row (c, d)

REAN = [
    ("oras5", "ORAS5", st.C_ORAS5, "o"),
    ("glorys12", "GLORYS12V1", st.C_GLORYS, "s"),
    ("ecco", "ECCO-V4r4", st.C_ECCO, "^"),
]

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
MINUS = "−"
LOG_NUMBERS: dict = {}
MS_MODEL = 4.5     # CMIP6 model circles, pt (filled grey: open is reserved for "includes zero")
MS_CROSS = 4.6     # MPI-ESM1-2-LR cross, pt
MEC_MODEL = "white"
MEW_MODEL = 0.5
C_MODEL = st.C_CMIP6   # neutral grey for CMIP6 models


def sgn(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}".replace("-", MINUS)


def num(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}".replace("-", MINUS)


def ax_in(fig, x0, y0, w, h):
    """Axes placed in inches from the bottom-left corner of the figure."""
    return fig.add_axes([x0 / W, (y0 + DY) / H, w / W, h / H])


def text_in(fig, x, y, s, **kw):
    return fig.text(x / W, (y + DY) / H, s, **kw)


def tilted(r) -> bool:
    """Model row spanning more than 1 degree of latitude (as in the old script)."""
    return abs((r.get("lat_north") or 0) - (r.get("lat_south") or 0)) > 1.0


def _hits(px, py, pr, bb: Bbox) -> bool:
    """True if a disc of radius pr (display px) centred at (px, py) touches bb."""
    nx = min(max(px, bb.x0), bb.x1)
    ny = min(max(py, bb.y0), bb.y1)
    return (nx - px) ** 2 + (ny - py) ** 2 < pr ** 2


def _text_bbox(text, renderer) -> Bbox:
    """Window extent of the words only (an annotation's extent includes its arrow)."""
    text.update_positions(renderer)
    return Text.get_window_extent(text, renderer)


def place_label(ax, renderer, text, anchor, obstacles, placed, cands, pad=1.5):
    """Pick the first candidate offset whose text box clears every marker.

    obstacles: list of (display x, display y, radius) for markers.
    placed: list of Bbox already used by other labels.
    Returns the chosen (dx, dy, ha, va) and sets the annotation accordingly.
    """
    axbb = ax.bbox
    chosen = None
    for dx, dy, ha, va in cands:
        text.xyann = (dx, dy)
        text.set_ha(ha)
        text.set_va(va)
        bb = _text_bbox(text, renderer)
        if bb.x0 < axbb.x0 + 2 or bb.x1 > axbb.x1 - 2 or bb.y0 < axbb.y0 + 2 or bb.y1 > axbb.y1 - 2:
            continue
        if any(_hits(px, py, pr + pad, bb) for px, py, pr in obstacles):
            continue
        if any(bb.overlaps(p) for p in placed):
            continue
        chosen = (dx, dy, ha, va, bb)
        break
    if chosen is None:
        dx, dy, ha, va = cands[-1]
        text.xyann = (dx, dy)
        text.set_ha(ha)
        text.set_va(va)
        bb = _text_bbox(text, renderer)
        chosen = (dx, dy, ha, va, bb)
        print("WARNING no clear position for", text.get_text())
    placed.append(chosen[4])
    return chosen[:4]


def product_key_marker(fig, x_in, y_in, colour, shape):
    """Product marker in the key at the top of panel (a): colour and shape together."""
    fig.add_artist(Line2D([x_in / W], [(y_in + DY) / H], marker=shape, ms=3.6,
                          mfc=colour, mec=colour, mew=0.6, ls="none"))


def main() -> None:
    st.set_style()
    WORK.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = json.loads((AN / "r1_cmip6.json").read_text())
    rows = d["per_model"]
    rw = d["reanalysis_windows"]

    fig = plt.figure(figsize=(W, H))

    # ------------------------------------------------------------------ a
    ax = ax_in(fig, 1.00, TOP_Y0, 1.25, TOP_H)
    rr = sorted(rows, key=lambda r: r["F_1993-2014"])
    y = np.arange(len(rr))
    f = np.array([r["F_1993-2014"] for r in rr])
    ax.barh(y, f, color=np.where(f > 0, st.C_CMIP6, st.C_CMIP6_LIGHT),
            height=0.74, lw=0, zorder=2)
    ax.plot([r["F_1950-1980"] for r in rr], y, "|", color="black",
            ms=3.6, mew=0.8, zorder=3,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    for key, _label, col, _mk in REAN:
        # Solid thin line in the product colour; the white halo keeps it apart from the bars.
        ax.axvline(rw[key]["1993-2014"]["F_ov"], color=col, lw=0.8,
                   ls="-", zorder=4,
                   path_effects=[pe.withStroke(linewidth=2.1, foreground="white")])
    ax.axvline(0, color="black", lw=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in rr], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.6, len(rr) - 0.4)
    ax.set_xlim(-0.25, 0.5)
    ax.set_xticks([-0.2, 0, 0.2, 0.4])
    ax.set_xticklabels([sgn(v, 1) if v else "0" for v in [-0.2, 0, 0.2, 0.4]])
    ax.set_xlabel("Mean $F_{\\mathrm{ovS}}$, 1993–2014 (Sv)")
    st.panel_label_pt(ax, "a")

    # Reanalysis means as a key above the column: product colour, product marker, label.
    x_head = 1.00
    for i, (key, label, col, mk) in enumerate(REAN):
        val = rw[key]["1993-2014"]["F_ov"]
        yy = 5.98 + 0.15 * (2 - i)
        fig.add_artist(Line2D([x_head / W, (x_head + 0.22) / W],
                              [(yy + DY) / H, (yy + DY) / H],
                              color=col, lw=0.8, ls="-"))
        product_key_marker(fig, x_head + 0.11, yy, col, mk)
        text_in(fig, x_head + 0.28, yy, f"{label}  {num(val)} Sv",
                fontsize=7, va="center", ha="left", color="black")
        LOG_NUMBERS[f"a_reanalysis_{key}_F_ov"] = val

    # Count of positive models and the model median, from per_model F_1993-2014.
    f_all = np.array([r["F_1993-2014"] for r in rows])
    n_pos = int(np.sum(f_all > 0))
    med = float(np.median(f_all))
    LOG_NUMBERS["a_n_positive"] = n_pos
    LOG_NUMBERS["a_n_models"] = len(rows)
    LOG_NUMBERS["a_median_F_1993_2014"] = med
    x0 = 0.02
    ax.text(x0, 1.6, f"{n_pos} of {len(rows)}", fontsize=7, va="center", ha="left")
    ax.text(x0, 2.9, "models positive", fontsize=7, va="center", ha="left")
    ax.text(x0, 4.5, f"median {sgn(med)} Sv", fontsize=7, va="center", ha="left")
    ax.text(x0, 6.0, "tick: 1950–1980", fontsize=7, va="center", ha="left")

    # ------------------------------------------------------------------ b
    ax = ax_in(fig, 3.35, TOP_Y0, 3.05, TOP_H)
    ax.set_xlim(9.5, 27.0)      # set before the clearance test, which uses transData
    ax.set_ylim(-1.0, 0.8)
    t_grid = np.linspace(9.5, 27.0, 300)
    d_grid = np.linspace(-1.0, 0.8, 300)
    tt, dd = np.meshgrid(t_grid, d_grid)
    ff = -tt * dd / S0
    edges = np.round(np.arange(-0.4, 0.401, 0.1), 2)     # 8 bands, extended
    cmap = plt.get_cmap("PuOr_r", 8)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(7))
    ax.contourf(tt, dd, ff, levels=edges, cmap=cmap, extend="both", zorder=1)
    lev_lines = edges[np.abs(edges) > 1e-9]
    cs = ax.contour(tt, dd, ff, levels=lev_lines, colors="0.2", linewidths=0.35,
                    zorder=2)
    lab = {lv: sgn(lv, 1) for lv in lev_lines}
    # Label positions chosen on each contour inside the axes, clear of markers
    # and annotations: F = -T dS / S0 gives dS = -S0 F / T on each contour.
    manual = [(20.0, 0.70), (20.0, 0.525), (12.0, 0.583), (10.8, 0.324),
              (14.0, -0.25), (14.0, -0.50), (18.0, -0.583), (15.5, -0.903)]
    assert len(manual) == len(lev_lines)
    txts = ax.clabel(cs, levels=list(lev_lines), fmt=lab, inline=True, fontsize=7,
                     inline_spacing=4, manual=manual)
    for t in txts:
        t.set_color("black")
        t.set_path_effects(HALO)
    ax.axhline(0, color="black", lw=0.6, zorder=2)

    level = [r for r in rows if not tilted(r)]
    slanted = [r for r in rows if tilted(r)]
    # CMIP6 models are filled grey (open markers are reserved for "interval includes zero").
    ax.plot([r["T_1993-2014"] for r in level], [r["dS_1993-2014"] for r in level],
            "o", ms=MS_MODEL, mfc=C_MODEL, mec=MEC_MODEL, mew=MEW_MODEL, ls="none",
            zorder=5)
    if slanted:
        ax.plot([r["T_1993-2014"] for r in slanted],
                [r["dS_1993-2014"] for r in slanted],
                "x", ms=MS_CROSS, color="black", mew=1.0, ls="none", zorder=6)
    for key, _label, col, mk in REAN:
        w = rw[key]["1993-2014"]
        ax.plot(w["T"], w["dS"], mk, ms=5.6, mfc=col, mec="black", mew=0.5,
                ls="none", zorder=7)
        LOG_NUMBERS[f"b_reanalysis_{key}_T_dS"] = [w["T"], w["dS"]]

    # Clearance test: every marker is an obstacle for the direct labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    pts_model = [(r["T_1993-2014"], r["dS_1993-2014"]) for r in rows]
    pts_rean = [(rw[k]["1993-2014"]["T"], rw[k]["1993-2014"]["dS"]) for k, *_ in REAN]
    r_model = (MS_MODEL / 2 + MEW_MODEL / 2) * dpi / 72
    r_rean = (5.6 / 2 + 0.5) * dpi / 72
    r_cross = (MS_CROSS / 2 + 0.5) * dpi / 72
    marker_disp = [(ax.transData.transform(p), r_model) for p in pts_model
                   if np.all(np.isfinite(p))] + \
                  [(ax.transData.transform(p), r_rean) for p in pts_rean]
    marker_disp += [(ax.transData.transform((r["T_1993-2014"], r["dS_1993-2014"])), r_cross)
                    for r in slanted]

    def obstacles_except(anchor):
        out = []
        for (px, py), rad in marker_disp:
            if abs(px - anchor[0]) < 0.5 and abs(py - anchor[1]) < 0.5:
                continue
            out.append((px, py, rad))
        return out

    generic = [(5, -5, "left", "top"), (-5, -5, "right", "top"),
               (5, 5, "left", "bottom"), (-5, 5, "right", "bottom"),
               (0, -9, "center", "top"), (0, 9, "center", "bottom"),
               (9, 0, "left", "center"), (-9, 0, "right", "center"),
               (8, -12, "left", "top"), (-8, -12, "right", "top"),
               (8, 12, "left", "bottom"), (-8, 12, "right", "bottom"),
               (0, -16, "center", "top"), (0, 16, "center", "bottom"),
               (14, -6, "left", "top"), (-14, -6, "right", "top"),
               (14, 6, "left", "bottom"), (-14, 6, "right", "bottom"),
               (22, 0, "left", "center"), (-22, 0, "right", "center"),
               (0, -24, "center", "top"), (0, 24, "center", "bottom"),
               (-6, -26, "right", "top"), (6, 26, "left", "bottom")]

    placed: list = []
    label_choice = {}
    # Reanalysis labels, then the model cross, then the slanted model names.
    specs = [
        ("ORAS5", pts_rean[0], dict(fontsize=7)),
        ("GLORYS12V1", pts_rean[1], dict(fontsize=7)),
        ("ECCO-V4r4", pts_rean[2], dict(fontsize=7)),
    ]
    for name, anchor_data, kw in specs:
        anchor = ax.transData.transform(anchor_data)
        t = ax.annotate(name, anchor_data, xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", path_effects=HALO, zorder=8, **kw)
        ch = place_label(ax, renderer, t, anchor, obstacles_except(anchor), placed, generic)
        label_choice[name] = ch

    for r in slanted:
        anchor_data = (r["T_1993-2014"], r["dS_1993-2014"])
        anchor = ax.transData.transform(anchor_data)
        t = ax.annotate(r["model"], anchor_data, xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", fontsize=7, path_effects=HALO, zorder=8,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.2",
                                        shrinkA=1, shrinkB=3))
        cands = [(-6, -9, "right", "top"), (6, -9, "left", "top"),
                 (-6, 9, "right", "bottom"), (6, 9, "left", "bottom"),
                 (0, -16, "center", "top"), (0, 16, "center", "bottom"),
                 (-24, 0, "right", "center"), (24, 0, "left", "center"),
                 (-6, 30, "right", "bottom"), (-4, 40, "right", "bottom"),
                 (-4, 55, "right", "bottom"), (-12, 40, "right", "bottom"),
                 (0, 55, "center", "bottom"), (-10, 60, "right", "bottom"),
                 (-30, -20, "right", "top")]
        ch = place_label(ax, renderer, t, anchor, obstacles_except(anchor), placed, cands)
        label_choice[r["model"]] = ch
        # Leader line only when the name is set off from its cross.
        if max(abs(ch[0]), abs(ch[1])) <= 16:
            t.arrow_patch.set_visible(False)

    # Sign message: which limb is saltier, with the counts behind it.
    n_neg_reanalysis = int(sum(rw[k]["1993-2014"]["F_ov"] < 0 for k, *_ in REAN))
    LOG_NUMBERS["b_n_reanalyses_negative"] = n_neg_reanalysis
    LOG_NUMBERS["b_n_models_positive"] = n_pos
    ax.text(26.8, 0.72,
            f"northward limb saltier\n{n_neg_reanalysis} of {len(REAN)} reanalyses, F$_{{\\mathrm{{ovS}}}}$ < 0",
            fontsize=7, ha="right", va="top", color="white", zorder=9)
    ax.text(26.8, -0.62,
            f"northward limb fresher\n{n_pos} of {len(rows)} models, F$_{{\\mathrm{{ovS}}}}$ > 0",
            fontsize=7, ha="right", va="top", color="white", zorder=9)

    dsv = np.array([r["dS_1993-2014"] for r in rows]
                   + [rw[k]["1993-2014"]["dS"] for k, *_ in REAN])
    tv = np.array([r["T_1993-2014"] for r in rows]
                  + [rw[k]["1993-2014"]["T"] for k, *_ in REAN])
    finite_b = np.isfinite(np.array([r["T_1993-2014"] for r in rows])) & \
        np.isfinite(np.array([r["dS_1993-2014"] for r in rows]))
    LOG_NUMBERS["b_n_models_plotted"] = int(finite_b.sum())
    LOG_NUMBERS["b_n_models_slanted"] = len(slanted)
    LOG_NUMBERS["b_T_range_models"] = [float(tv[:len(rows)][finite_b].min()),
                                       float(tv[:len(rows)][finite_b].max())]
    LOG_NUMBERS["b_dS_range_all"] = [float(np.nanmin(dsv)), float(np.nanmax(dsv))]
    LOG_NUMBERS["b_label_offsets"] = {k: list(v) for k, v in label_choice.items()}
    ax.set_xticks([10, 14, 18, 22, 26])
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_yticklabels([sgn(v, 1) if v else "0" for v in [-0.8, -0.4, 0, 0.4, 0.8]])
    ax.set_xlabel("Northward limb transport $T$, 1993–2014 (Sv)")
    ax.set_ylabel("Limb salinity contrast $\\Delta S$ (PSU)")
    st.panel_label_pt(ax, "b")
    text_in(fig, 3.35, TOP_Y0 + TOP_H + 0.18, "Shading and contours: F$_{\\mathrm{ovS}}$ (Sv)",
            fontsize=7, va="center", ha="left")
    # Key for the grey model circles sits in the header line, off the data field.
    fig.add_artist(Line2D([5.50 / W], [(TOP_Y0 + TOP_H + 0.18 + DY) / H], marker="o", ms=MS_MODEL,
                          mfc=C_MODEL, mec=MEC_MODEL, mew=MEW_MODEL, ls="none"))
    text_in(fig, 5.62, TOP_Y0 + TOP_H + 0.18, "CMIP6 model", fontsize=7, va="center", ha="left")

    # ------------------------------------------------------------------ c
    ax = ax_in(fig, 0.95, BOT_Y0, 2.10, BOT_H)
    sub = [r for r in rows
           if r.get("dpsi34") is not None and r.get("damoc26") is not None
           and np.isfinite(r.get("dpsi34", np.nan))
           and np.isfinite(r.get("damoc26", np.nan))]
    x = np.array([r["damoc26"] for r in sub])
    yv = np.array([r["dpsi34"] for r in sub])
    lim = [-14, 2]
    ax.plot(lim, lim, color="0.5", lw=0.5, ls=":", zorder=2)
    coef = np.polyfit(x, yv, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(coef, xx), color="black", lw=0.8, zorder=3)
    # Filled grey models (open markers are reserved for "interval includes zero").
    ax.plot(x, yv, "o", ms=3.2, mfc=C_MODEL, mec="white", mew=0.4, ls="none", zorder=4)
    ax.axhline(0, color="black", lw=0.3, zorder=1)
    ax.axvline(0, color="black", lw=0.3, zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(-10, 5)
    ax.set_xticks([-12, -8, -4, 0])
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_xlabel("Change at 26.5°N (Sv)")
    ax.set_ylabel("Change at 34.5°S (Sv)")
    st.panel_label_pt(ax, "c")
    # Slope and correlation, from the same 17 models (JSON summary checks r).
    r_c = float(np.corrcoef(x, yv)[0, 1])
    summ_r = d["summary"]["dpsi34_vs_damoc26"]["r"]
    summ_n = d["summary"]["dpsi34_vs_damoc26"]["n"]
    LOG_NUMBERS["c_n"] = len(sub)
    LOG_NUMBERS["c_slope"] = float(coef[0])
    LOG_NUMBERS["c_r"] = r_c
    LOG_NUMBERS["c_summary_r"] = summ_r
    LOG_NUMBERS["c_summary_n"] = summ_n
    # Upper-left corner is empty: the fit runs from lower left to upper right.
    ax.text(-13.6, 4.6, f"slope {coef[0]:.2f}\nr = {sgn(r_c, 2)}",
            fontsize=7, ha="left", va="top", zorder=5)
    ax.text(1.9, 2.4, "1:1", fontsize=7, ha="right", va="bottom", color="0.35")

    # ------------------------------------------------------------------ d
    ax = ax_in(fig, 3.60, BOT_Y0, 2.15, BOT_H)
    sub_d = [r for r in rows
             if r.get("damoc26") is not None and np.isfinite(r.get("damoc26", np.nan))]
    xd = np.array([r["damoc26"] for r in sub_d])
    yd = np.array([r["F_1950-1980"] for r in sub_d])
    cd = [r["amoc26_1850-1900"] for r in sub_d]
    n_d = len(xd)
    lo_c, hi_c = 5.0, 24.0       # colour range covers the data (5.93 to 23.98 Sv)
    ax.axhline(0, color="black", lw=0.4, zorder=2)
    # Least-squares line with the 95 per cent band of the fitted mean (t, n - 2 df).
    b1, b0 = np.polyfit(xd, yd, 1)
    xl = np.linspace(xd.min(), xd.max(), 100)
    resid = yd - (b0 + b1 * xd)
    s_res = np.sqrt(np.sum(resid ** 2) / (n_d - 2))
    sxx = np.sum((xd - xd.mean()) ** 2)
    se_mean = s_res * np.sqrt(1.0 / n_d + (xl - xd.mean()) ** 2 / sxx)
    t_crit = stats.t.ppf(0.975, n_d - 2)
    yl = b0 + b1 * xl
    ax.fill_between(xl, yl - t_crit * se_mean, yl + t_crit * se_mean,
                    color="0.45", alpha=0.25, lw=0, zorder=1)
    ax.plot(xl, yl, color="black", lw=0.8, zorder=2.5)
    sc = ax.scatter(xd, yd, c=cd, cmap="cividis", s=11, edgecolors="black",
                    linewidths=0.4, zorder=3, vmin=lo_c, vmax=hi_c)
    # Rank statistics: Spearman rho from the plotted values, permutation p (seeded).
    rho, p_t = stats.spearmanr(xd, yd)
    rng = np.random.default_rng(20261104)
    rx, ry = stats.rankdata(xd), stats.rankdata(yd)
    perm_rho = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(20000)])
    perm_p = float(np.mean(np.abs(perm_rho) >= abs(rho) - 1e-12))
    part = d["summary"]["partial_F_vs_damoc26_given_base"]
    n11 = json.loads((AN / "jc_n11_cmip6_relation.json").read_text())
    pb = n11["sets"]["26.5N_21"]["bootstrap_family_cluster"]["partial_r"]
    pr_pt, (pr_lo, pr_hi) = float(pb["point"]), [float(v) for v in pb["ci95"]]
    ax.set_xlim(-14, 2)
    ax.set_ylim(-0.3, 0.9)
    ax.set_xticks([-12, -8, -4, 0])
    ax.set_yticks([-0.2, 0, 0.2, 0.4])
    ax.set_yticklabels([sgn(v, 1) if v else "0" for v in [-0.2, 0, 0.2, 0.4]])
    ax.set_xlabel("Change at 26.5°N (Sv)")
    ax.set_ylabel("$F_{\\mathrm{ovS}}$, 1950–1980 (Sv)")
    st.panel_label_pt(ax, "d")
    ax.text(1.9, 0.87, f"Spearman ρ = {sgn(rho, 2)}, n = {n_d}\n"
            f"partial r = {sgn(pr_pt, 2)}, family bootstrap [{sgn(pr_lo, 2)}, {sgn(pr_hi, 2)}]",
            fontsize=7, ha="right", va="top", linespacing=1.25, zorder=6)
    LOG_NUMBERS["d_n"] = n_d
    LOG_NUMBERS["d_spearman_rho"] = float(rho)
    LOG_NUMBERS["d_spearman_p_t"] = float(p_t)
    LOG_NUMBERS["d_permutation_p"] = perm_p
    LOG_NUMBERS["d_ls_slope"] = float(b1)
    LOG_NUMBERS["d_ls_intercept"] = float(b0)
    LOG_NUMBERS["d_partial_family_bootstrap"] = [pr_pt, pr_lo, pr_hi]
    LOG_NUMBERS["d_partial_r_json"] = float(part["r"])
    LOG_NUMBERS["d_partial_p_json"] = float(part["p"])
    LOG_NUMBERS["d_partial_loo_range_json"] = [float(v) for v in part["leave_one_out_range"]]
    LOG_NUMBERS["d_colour_range"] = [float(min(cd)), float(max(cd))]
    LOG_NUMBERS["d_colour_axis"] = [lo_c, hi_c]

    cax = ax_in(fig, 5.85, BOT_Y0, 0.08, BOT_H)
    cb = fig.colorbar(sc, cax=cax, ticks=[10, 15, 20])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    # Colour-bar label horizontal, above the bar (7 pt).
    text_in(fig, 5.89, BOT_Y0 + BOT_H + 0.07, "Overturning at 26.5°N,\n1850–1900 (Sv)",
            fontsize=7, ha="center", va="bottom", linespacing=1.15)

    # ------------------------------------------------------------------ save
    fig.canvas.draw()
    small = st.check_min_fontsize(fig)
    LOG_NUMBERS["small_text_count"] = len(small)
    st.save(fig, str(OUTBASE))
    if "--preview" in sys.argv:
        fig.savefig(str(PREVIEW), dpi=200)
        print("preview", PREVIEW)
    LOG.write_text(json.dumps(LOG_NUMBERS, indent=1, default=float))
    print("numbers log", LOG)
    print("label offsets", json.dumps(label_choice, default=float))


if __name__ == "__main__":
    main()
