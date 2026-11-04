#!/usr/bin/env python3
"""Figure 3 (Journal of Climate): attribution of the F_ovS trend.

Every number written in the figure is read from PAPER_3_v2/analysis/*.json at
run time (attribution.json, stepfit.json, r1_cells.json, r1_grid_sensitivity.json,
r1_step_adjusted.json).

(a) Two stacked axes that share the time axis and use the same range in per
    cent of the record mean: upper, the limb salinity contrast dS; lower, the
    northward-limb transport T. Products by colour (ORAS5, GLORYS12V1), solid
    lines, direct labels at the line ends. Upper and lower axes carry the one
    label (a). The table in the lower axes gives the own-record fractional rates
    per decade of F_ovS and of T.
(b) Trend of F_ovS (total, circle) with its transport term (diamond) and
    salinity term (square), 95 per cent intervals, by window and product. Terms
    use the TERM_STYLE greys; product group labels are black. Filled marker and
    solid interval: the interval excludes zero. Open marker and dotted interval:
    it includes zero. A pale marker with no interval: the effective sample size
    is below NEFF_FLOOR. Bounds beyond the axis are printed at the right.
(c) Transport share of the change over all non-overlapping epoch pairs, one
    horizontal strip per product on a 0 to 1.05 axis: median (dot), 90th and
    95th percentiles (ticks), maximum (cross). Pair count and share below 0.25
    are printed under each strip.
(d) Transport share of the trend with 95 per cent intervals under each
    robustness test (unchanged in content).

Writes PAPER_3_v2/figures_jclim/Fig3_attribution.{pdf,png} at 6.5 in wide.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402

import paper3jc_style as st  # noqa: E402

AN = REPO / "PAPER_3_v2" / "analysis"
OUT = REPO / "PAPER_3_v2" / "figures_jclim" / "Fig3_attribution"

W = st.TWO_COL   # 6.5 in
H = 8.00         # inches, final height (limit 8 in)

MINUS = "−"
INK = "#111111"
PALE = "#bdbdbd"          # face of a point without an interval (no interval, not open)
C_ORAS = st.C_ORAS5
C_GLO = st.C_GLORYS
PM_ORAS = st.PRODUCT_MARKERS["ORAS5"]
PM_GLO = st.PRODUCT_MARKERS["GLORYS12V1"]
TERM_OF = {"total": "total", "T": "transport", "S": "salinity"}

# An interval whose effective sample size is below this value sits at the floor
# of the fit (n_eff = 3, two parameters); such intervals are not drawn.
NEFF_FLOOR = 3.5

# Layout, inches from the top-left corner of the figure.
KEY_H = 0.52
A_LEFT, A_W = 0.85, 3.10
A_TOP, A_H, A_GAP = 0.80, 0.90, 0.12
C_LEFT, C_W, C_H = 5.10, 1.30, 2.00
B_TOP, B_LEFT, B_W = 3.30, 1.45, 4.85
B_PITCH, B_GROUP = 0.215, 0.10
D_TOP, D_LEFT, D_W = 6.07, 2.45, 3.85
D_PITCH = 0.135

YLO_A, YHI_A = -150.0, 200.0


def sgn(v: float, nd: int) -> str:
    """Signed number with a typographic minus sign."""
    return f"{v:+.{nd}f}".replace("-", MINUS)


def add_ax(fig, left: float, top: float, width: float, height: float):
    """Axes placed in inches from the top-left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, 1 - (top + height) / fh, width / fw, height / fh])


def fig_text(fig, x: float, y: float, s: str, **kw):
    """Text at inches from the top-left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.text(x / fw, 1 - y / fh, s, **kw)


def spread(ys: list[float], gap: float) -> list[float]:
    """Push label positions apart (top to bottom) so that none is closer than gap."""
    pos = list(ys)
    order = sorted(range(len(pos)), key=lambda i: -pos[i])
    prev = None
    for i in order:
        if prev is not None and pos[prev] - pos[i] < gap:
            pos[i] = pos[prev] - gap
        prev = i
    return pos


def load():
    d = json.loads((AN / "attribution.json").read_text())
    cells = json.loads((AN / "r1_cells.json").read_text())
    grid = json.loads((AN / "r1_grid_sensitivity.json").read_text())
    step = json.loads((AN / "r1_step_adjusted.json").read_text())
    rates = json.loads((AN / "stepfit.json").read_text())
    return d, cells, grid, step, rates


# ------------------------------------------------------------------ key
def draw_key(fig) -> None:
    """One key for the whole figure: products, terms, interval shapes, strips."""
    ax = add_ax(fig, 0.0, 0.0, W, KEY_H)
    ax.set_xlim(0, W)
    ax.set_ylim(KEY_H, 0)
    ax.axis("off")
    y1, y2, y3 = 0.10, 0.27, 0.44

    def txt(x, y, s):
        ax.text(x, y, s, fontsize=7, ha="left", va="center")

    def mk(x, y, **kw):
        ax.plot([x], [y], clip_on=False, **kw)

    txt(0.00, y1, "Products (a, c, d)")
    mk(1.12, y1, marker=PM_ORAS, ms=3.8, mfc=C_ORAS, mec=C_ORAS, mew=0.8, ls="none")
    txt(1.22, y1, "ORAS5")
    mk(1.78, y1, marker=PM_GLO, ms=3.5, mfc=C_GLO, mec=C_GLO, mew=0.8, ls="none")
    txt(1.88, y1, "GLORYS12V1")
    txt(2.80, y1, "Terms (b)")
    for term, x, label in (("total", 3.45, "total $F_{\\mathrm{ovS}}$"),
                           ("salinity", 4.62, "salinity $\\Delta S$"),
                           ("transport", 5.64, "transport $T$")):
        mk(x, y1, **st.term_marker(term, True, ms=3.8))
        txt(x + 0.11, y1, label)

    txt(0.00, y2, "Intervals")
    ax.plot([0.70, 0.94], [y2, y2], color=INK, lw=1.1, clip_on=False)
    mk(0.82, y2, **st.term_marker("total", True, ms=3.8))
    txt(1.02, y2, "filled: interval excludes zero")
    ax.plot([2.62, 2.86], [y2, y2], color=INK, lw=1.1, ls=(0, (1.0, 1.0)), clip_on=False)
    mk(2.74, y2, **st.term_marker("total", False, ms=3.8))
    txt(2.94, y2, "open, dotted: interval includes zero")
    mk(4.98, y2, marker="o", ms=3.8, mfc=PALE, mec=INK, mew=0.8, ls="none")
    txt(5.12, y2, "pale: no interval")

    txt(0.00, y3, "Strips (c)")
    mk(0.82, y3, marker="o", ms=4.2, mfc=INK, mec=INK, mew=0.8, ls="none")
    txt(0.92, y3, "median")
    mk(1.62, y3, marker="|", ms=5.0, mec=INK, mew=1.2, ls="none")
    txt(1.72, y3, "p90, p95")
    mk(2.50, y3, marker="x", ms=4.2, mec=INK, mew=1.2, ls="none")
    txt(2.60, y3, "maximum")
    ax.plot([3.30, 3.54], [y3, y3], color="0.35", lw=0.7, ls=st.LS_REFERENCE, clip_on=False)
    txt(3.64, y3, "one quarter")
    txt(4.55, y3, "colour: product, as in (a) and (d)")


# ------------------------------------------------------------------ panel (a)
def panel_a(fig, d, rates):
    ax_u = add_ax(fig, A_LEFT, A_TOP, A_W, A_H)
    ax_l = add_ax(fig, A_LEFT, A_TOP + A_H + A_GAP, A_W, A_H)
    ax_l.sharex(ax_u)
    ax_u.tick_params(labelbottom=False)
    ends_u, ends_l = [], []
    for key, label, colour in (("oras5", "ORAS5", C_ORAS), ("glorys12", "GLORYS12V1", C_GLO)):
        r = d[key]
        yrs = np.array(r["years"])
        t = np.array(r["T_limb_Sv"])
        s = np.array(r["dS_limb_PSU"])
        ta = 100 * (t - t.mean()) / t.mean()
        sa = 100 * (s - s.mean()) / s.mean()
        ax_u.plot(yrs, sa, color=colour, lw=1.0, zorder=3)
        ax_l.plot(yrs, ta, color=colour, lw=1.0, zorder=3)
        pm = st.PRODUCT_MARKERS[label]
        for ax, yv in ((ax_u, sa[-1]), (ax_l, ta[-1])):
            ax.plot([yrs[-1]], [yv], marker=pm, ms=3.2, mfc=colour, mec=colour, mew=0.6,
                    ls="none", clip_on=False, zorder=4)
        ends_u.append((sa[-1], label))
        ends_l.append((ta[-1], label))
        print(f"(a) {label}: dS anomaly min {sa.min():.1f} max {sa.max():.1f} "
              f"last {sa[-1]:.1f} %; T anomaly min {ta.min():.1f} max {ta.max():.1f} "
              f"last {ta[-1]:.1f} %")
    for ax, ends in ((ax_u, ends_u), (ax_l, ends_l)):
        gap = 0.12 * (YHI_A - YLO_A) / A_H
        pos = spread([e[0] for e in ends], gap)
        for (_, label), yl in zip(ends, pos):
            ax.text(2026.5, yl, label, color="black", fontsize=7, ha="left", va="center",
                    clip_on=False)
    for ax in (ax_u, ax_l):
        ax.axhline(0, color="0.55", lw=0.5, zorder=1)
        ax.set_xlim(1958, 2025)
        ax.set_ylim(YLO_A, YHI_A)
        ax.set_yticks([-150, -100, -50, 0, 50, 100, 150, 200])
    ax_u.set_ylabel("ΔS anomaly (%)")
    ax_l.set_ylabel("T anomaly (%)")
    ax_l.set_xticks(range(1960, 2021, 10))
    ax_l.set_xlabel("Year")

    # Own-record fractional rates per decade, read from stepfit.json.
    fr = {k: rates[k]["fractional_rates"]["own record"] for k in ("oras5", "glorys12")}
    cols = (0.02, 0.37, 0.71)
    rows_y = (0.97, 0.79, 0.61)
    tt = dict(transform=ax_l.transAxes, fontsize=7, va="center", ha="left")
    ax_l.text(cols[0], rows_y[0], "per decade", color="0.25", **tt)
    ax_l.text(cols[1], rows_y[0], "ORAS5", **tt)
    ax_l.text(cols[2], rows_y[0], "GLORYS12V1", **tt)
    ax_l.plot([0.02, 0.95], [0.88, 0.88], transform=ax_l.transAxes, color="0.5", lw=0.5,
              clip_on=False)
    ax_l.text(cols[0], rows_y[1], r"$F_{\mathrm{ovS}}$", **tt)
    ax_l.text(cols[0], rows_y[2], r"$T$", **tt)
    for j, key in enumerate(("oras5", "glorys12")):
        fF = fr[key]["F_ov"]["fractional_pct_per_decade"]
        fT = fr[key]["T_limb"]["fractional_pct_per_decade"]
        ax_l.text(cols[1 + j], rows_y[1], f"{sgn(fF, 0)}%", **tt)
        ax_l.text(cols[1 + j], rows_y[2], f"{sgn(fT, 1)}%", **tt)
        print(f"(a) table {key}: F_ovS {fF:.2f} %/decade, T {fT:.2f} %/decade")
    return ax_u, ax_l


# ------------------------------------------------------------------ panel (c)
def panel_c(fig, d):
    ax = add_ax(fig, C_LEFT, A_TOP, C_W, C_H)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, C_H)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.axvline(0.25, color="0.35", lw=0.7, ls=st.LS_REFERENCE, zorder=1)
    ax.text(0.25, C_H + 0.05, "one quarter", fontsize=7, ha="center", va="bottom",
            color="0.25")
    strips = (("oras5", "ORAS5", C_ORAS, 1.60), ("glorys12", "GLORYS12V1", C_GLO, 0.62))
    for key, name, colour, y in strips:
        c = d[key]["census"]
        med = c["share_T_sym_median"]
        p90 = c["share_T_sym_p90"]
        p95 = c["share_T_sym_p95"]
        mx = c["share_T_sym_max"]
        n = int(c["n_pairs_usable"])
        share_below = float(c["frac_share_below_0p25"])
        ax.plot([0, 1.05], [y, y], color="0.85", lw=0.6, zorder=1)
        ax.text(0.0, y + 0.28, name, fontsize=7, fontweight="bold", color="black",
                ha="left", va="bottom")
        ax.plot([med], [y], marker="o", ms=4.2, mfc=colour, mec=colour, mew=0.6, ls="none",
                zorder=4)
        for p in (p90, p95):
            ax.plot([p, p], [y - 0.09, y + 0.09], color=colour, lw=1.2, zorder=4)
        ax.plot([mx], [y], marker="x", ms=4.2, mew=1.2, color=colour, ls="none",
                clip_on=False, zorder=4)
        halo = dict(fc="white", ec="none", pad=0.6)
        ax.text(0.0, y - 0.14, f"{n} pairs, {100 * share_below:.0f}% below 0.25",
                fontsize=7, ha="left", va="top", color="black", bbox=halo, zorder=5)
        ax.text(0.0, y - 0.27, f"median {med:.2f}, max {mx:.2f}", fontsize=7,
                ha="left", va="top", color="black", bbox=halo, zorder=5)
        print(f"(c) {name}: n {n}, median {med:.3f}, p90 {p90:.3f}, p95 {p95:.3f}, "
              f"max {mx:.3f}, share below 0.25 {share_below:.3f}")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
    ax.set_xlabel("Transport share")
    return ax


# ------------------------------------------------------------------ panel (b)
def interval_resolved(c: dict, term: str) -> bool:
    """A term has a usable interval when its effective sample size is above the floor."""
    key = {"total": "n_eff_total", "T": "n_eff_T", "S": "n_eff_S"}[term]
    return float(c[key]) >= NEFF_FLOOR


def draw_interval(ax, yy, est, lo, hi, term_key, sig, xl, xr):
    """95 per cent interval with its estimate, drawn in the term grey."""
    colour = st.TERM_STYLE[term_key]["colour"]
    a, b = min(lo, hi), max(lo, hi)
    ls = "-" if sig else (0, (1.0, 1.0))
    cl, ch = max(a, xl), min(b, xr)
    if ch > cl:
        ax.plot([cl, ch], [yy, yy], color=colour, lw=1.1, ls=ls,
                solid_capstyle="butt", zorder=3)
    for bound in (a, b):
        if xl <= bound <= xr:
            ax.plot([bound, bound], [yy - 0.05, yy + 0.05], color=colour, lw=1.1, zorder=3)
    ax.plot([est], [yy], zorder=5, **st.term_marker(term_key, sig, ms=3.6))


def panel_b(fig, d):
    groups = [
        ("ORAS5", [
            d["oras5"]["continuous"]["own record"],
            d["oras5"]["consolidated_only"]["continuous"],
            d["oras5"]["continuous"]["argo"],
            d["oras5"]["continuous"]["common"],
        ]),
        ("GLORYS12V1", [
            d["glorys12"]["continuous"]["own record"],
            d["glorys12"]["continuous"]["argo"],
            d["glorys12"]["continuous"]["argo to 2014"],
            d["glorys12"]["continuous"]["common"],
        ]),
        ("ECCO-V4r4", [
            d["ecco"]["continuous"]["common"],
        ]),
    ]
    xl, xr = -8.0, 3.0
    # Row positions in inches from the top of the axes (y increases downwards).
    rows = []
    y = 0.12
    for name, cs in groups:
        first = True
        for c in cs:
            rows.append((name, c, y, first))
            first = False
            y += B_PITCH
        y += B_GROUP
    y_last = rows[-1][2]
    b_h = y_last + 0.14
    ax = add_ax(fig, B_LEFT, B_TOP, B_W, b_h)
    ax.set_xlim(xl, xr)
    ax.set_ylim(b_h, 0)

    off = {"total": -0.075, "T": 0.0, "S": 0.075}
    comp = {
        "total": ("trend_total_mSv_per_yr", "ci_total_mSv_per_yr", "total"),
        "T": ("trend_T_mSv_per_yr", "ci_T_mSv_per_yr", "transport"),
        "S": ("trend_S_mSv_per_yr", "ci_S_mSv_per_yr", "salinity"),
    }
    # Highlight band: the full ORAS5 record, the headline row.
    ax.axhspan(rows[0][2] - 0.115, rows[0][2] + 0.115, color="0.93", lw=0, zorder=0)
    incons, unresolved, outside = [], [], []
    for idx, (name, c, yr, first) in enumerate(rows):
        for term in ("total", "T", "S"):
            vk, ck, term_key = comp[term]
            est = float(c[vk])
            yy = yr + off[term]
            if not interval_resolved(c, term):
                unresolved.append((name, c["window"], term, round(est, 2)))
                ax.plot([est], [yy], zorder=5, **st.term_marker(term_key, True, ms=3.6)
                        | {"mfc": PALE, "mec": st.TERM_STYLE[term_key]["colour"]
                           if term_key != "transport" else INK})
                continue
            lo, hi = (float(v) for v in c[ck])
            if term == "total":
                sig = bool(c["p_total"] < 0.05)
                if sig != (not (lo <= 0.0 <= hi)):
                    incons.append((name, c["window"], float(c["p_total"]), lo, hi))
            else:
                sig = not (lo <= 0.0 <= hi)
            if lo < xl or hi > xr:
                outside.append((name, c["window"], term, lo, hi))
                ax.text(xr + 0.12, yy, f"{sgn(lo, 0)} to {sgn(hi, 0)}", fontsize=7,
                        ha="left", va="center", clip_on=False)
            draw_interval(ax, yy, est, lo, hi, term_key, sig, xl, xr)

    # Key numbers of the full ORAS5 record, printed beside the zero line.
    c0 = rows[0][1]
    # 7 pt glyphs are about 0.09 in tall, so the three numbers sit 0.105 in apart.
    label_off = {"total": -0.105, "T": 0.0, "S": 0.105}
    for term, val in (("total", c0["trend_total_mSv_per_yr"]),
                      ("T", c0["trend_T_mSv_per_yr"]),
                      ("S", c0["trend_S_mSv_per_yr"])):
        ax.text(0.40, rows[0][2] + label_off[term], sgn(val, 2), color=INK,
                ha="left", va="center", fontsize=7, zorder=6)
    print(f"(b) ORAS5 own record: total {c0['trend_total_mSv_per_yr']:.2f}, "
          f"T {c0['trend_T_mSv_per_yr']:.2f}, S {c0['trend_S_mSv_per_yr']:.2f} mSv/yr")

    ax.axvline(0, color="0.4", lw=0.5, zorder=1)
    ax.set_yticks([r[2] for r in rows])
    ax.set_yticklabels([f"{r[1]['window'][0]}–{r[1]['window'][1]}" for r in rows])
    ax.tick_params(axis="y", length=0)
    xt = [-8, -6, -4, -2, 0, 2]
    ax.set_xticks(xt)
    ax.set_xticklabels([sgn(v, 0) if v else "0" for v in xt])
    ax.set_xlabel("Trend of $F_{\\mathrm{ovS}}$ (mSv yr$^{-1}$)")
    # Black bold product group labels, on the first row of each group.
    for name, c, yr, first in rows:
        if first:
            fig_text(fig, 0.08, B_TOP + yr, name, fontsize=7, fontweight="bold",
                     color="black", ha="left", va="center")
    return ax, incons, unresolved, outside


# ------------------------------------------------------------------ panel (d)
def panel_d(fig, d, cells, grid, step):
    scan = d["oras5"]["start_year_scan"]
    resolved_starts = [s for s in scan if s["p_total"] < 0.05]
    smin = min(s["start"] for s in resolved_starts)
    smax = max(s["start"] for s in resolved_starts)
    tests = [
        ("baseline, own record", lambda k: d[k]["continuous"]["own record"]),
        ("Argo era", lambda k: d[k]["continuous"]["argo"]),
        ("upper cell only", lambda k: cells[k]["windows"]["own"]["upper_cell_attribution"]),
        ("$\\Psi_{\\max}$ from surface",
         lambda k: d[k]["psi_definition_sensitivity"]["streamfunction max, surface start"]),
        ("$\\Psi_{\\max}$ below 250 m",
         lambda k: d[k]["psi_definition_sensitivity"]["streamfunction max, 250 m start"]),
        ("removal in proportion to $|V|$", lambda k: grid[k]["_absV"]),
        ("thickness from level centres", lambda k: grid[k]["_centres"]),
        ("bottom cells at one fifth", lambda k: grid[k]["_bottom0p2"]),
        ("observing-system step fitted", lambda k: step[k]),
        ("consolidated stream only",
         lambda k: d[k]["consolidated_only"]["continuous"] if k == "oras5" else None),
        (f"start years {smin}–{smax}", lambda k: "scan" if k == "oras5" else None),
    ]
    xmax = 0.35
    yy = np.array([0.08 + k * D_PITCH for k in range(len(tests))])
    d_h = yy[-1] + 0.08
    ax = add_ax(fig, D_LEFT, D_TOP, D_W, d_h)
    ax.set_ylim(d_h, 0)
    ax.set_xlim(0, xmax)

    def resolved(key: str, r: dict) -> bool:
        """Shares are quoted only where the trend of the total is resolved."""
        if "total_resolved" in r:
            return bool(r["total_resolved"])
        if "F_trend_resolved" in r:
            return bool(r["F_trend_resolved"])
        if "p_F_ov" in r:
            return bool(r["p_F_ov"] < 0.05)
        return bool(d[key]["continuous"]["own record"]["total_resolved"])

    labels_end = {}
    for j, (key, colour, label) in enumerate((("oras5", C_ORAS, "ORAS5"),
                                              ("glorys12", C_GLO, "GLORYS12V1"))):
        off = -0.035 if j == 0 else 0.035
        pm = st.PRODUCT_MARKERS[label]
        for y0, (name, getter) in zip(yy, tests, strict=False):
            try:
                r = getter(key)
            except KeyError:
                r = None
            if r is None:
                continue
            y = y0 + off
            if r == "scan":
                sh = [s["share_T"] for s in resolved_starts]
                lo_, hi_ = min(sh), max(sh)
                ax.plot([lo_, hi_], [y, y], color=colour, lw=2.4,
                        solid_capstyle="butt", zorder=3)
                ax.text(lo_ - 0.006, y, f"{100 * lo_:.1f}%", color=colour, ha="right",
                        va="center", fontsize=7)
                ax.text(hi_ + 0.006, y, f"{100 * hi_:.1f}%", color=colour, ha="left",
                        va="center", fontsize=7)
                continue
            p = r.get("share_T_point", r.get("share_T"))
            if resolved(key, r):
                ci = r["share_T_ci"]
                ax.plot([ci[0], min(ci[1], xmax)], [y, y], color=colour, lw=1.0, zorder=3,
                        solid_capstyle="butt")
                ax.plot([p], [y], marker=pm, ms=3.4, mfc=colour, mec=colour, mew=0.6,
                        ls="none", zorder=4)
                labels_end[(key, name)] = (min(ci[1], xmax), p, y, colour)
            else:
                ax.plot([min(p, xmax)], [y], marker=pm, ms=3.4, mfc=PALE, mec=colour,
                        mew=0.8, ls="none", zorder=4)
    # Point values quoted in the text for the streamfunction definitions.
    for nm in ("$\\Psi_{\\max}$ from surface", "$\\Psi_{\\max}$ below 250 m"):
        if ("oras5", nm) in labels_end:
            hi_, p, yv, colour = labels_end[("oras5", nm)]
            ax.text(hi_ + 0.008, yv, f"{100 * p:.0f}%", color=colour, ha="left",
                    va="center", fontsize=7)
    ax.axvline(0.25, color="0.35", lw=0.7, ls=st.LS_REFERENCE, zorder=1)
    ax.text(0.255, yy[0], "one quarter", color="0.25", fontsize=7, ha="left", va="center")
    ax.set_yticks(yy)
    ax.set_yticklabels([t[0] for t in tests])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(0, 0.351, 0.05))
    ax.set_xticklabels([f"{v:.2f}" if v else "0" for v in np.arange(0, 0.351, 0.05)])
    ax.set_xlabel("Transport share of the $F_{\\mathrm{ovS}}$ trend")
    return ax, resolved_starts


def main() -> None:
    st.set_style()
    d, cells, grid, step, rates = load()

    fig = plt.figure(figsize=(W, H))
    draw_key(fig)

    ax_u, _ax_l = panel_a(fig, d, rates)
    st.panel_label_pt(ax_u, "a")

    ax_c = panel_c(fig, d)
    st.panel_label_pt(ax_c, "c")

    ax_b, incons, unresolved, outside = panel_b(fig, d)
    st.panel_label_pt(ax_b, "b")

    ax_d, resolved_starts = panel_d(fig, d, cells, grid, step)
    st.panel_label_pt(ax_d, "d")

    small = st.check_min_fontsize(fig)
    print("text items below 7 pt:", len(small))
    st.save(fig, str(OUT))

    print("panel (b) significance rule mismatches (total: p<0.05 vs CI excludes 0):")
    for row in incons:
        print("  ", row)
    print(f"panel (b) rows without interval (n_eff below {NEFF_FLOOR}):")
    for row in unresolved:
        print("  ", row)
    print("panel (b) bounds printed beyond the axis:")
    for row in outside:
        print("  ", row)
    print("panel (d) resolved start years:", min(s["start"] for s in resolved_starts),
          "to", max(s["start"] for s in resolved_starts), "n =", len(resolved_starts))


if __name__ == "__main__":
    main()
