#!/usr/bin/env python3
"""Figure 2 of the Journal of Climate version: sign and magnitude of F_ovS.

Panel letters and meanings are those of the Journal of Climate Figure 2:

  a  Annual F_ovS at 34.5 S in ORAS5, GLORYS12V1 and ECCO-V4r4 (solid lines,
     product markers at the line ends), with the count of years in which
     F_ovS is negative.
  b  Forest plot: for each product the record mean and the mean of the last
     ten years of its record, with 95% intervals. Open marker and dotted
     interval where the interval includes zero.
  c  Mean F_ovS over 1993-2025 against section latitude with 95% intervals;
     open marker where the interval includes zero.
  d  Per product, a horizontal bar pair from zero: the overturning (negative)
     and azonal (positive) record means; the total as a marker with its value.
  e  Forest plot of the trends of the three components, one row per product
     and component, with 95% intervals; filled and solid where significant at
     5%, open and dotted where the interval includes zero.

Writes PAPER_3_v2/figures_jclim/Fig2_sign.{pdf,png} at the final printed width
(6.5 in). Every number is read at run time from
PAPER_3_v2/analysis/{attribution,latitude_sensitivity,r1_common_window}.json.
Style constants and the term and product encodings come from
scripts/paper3jc_style.py (not edited here).
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
OUT = REPO / "PAPER_3_v2" / "figures_jclim" / "Fig2_sign"

KEYS = ["oras5", "glorys12", "ecco"]
PRODUCT_COLOUR = {"oras5": st.C_ORAS5, "glorys12": st.C_GLORYS, "ecco": st.C_ECCO}
MINUS = "−"
EN = "–"

# Panel geometry in inches from the top-left corner of the figure.
W, H = st.TWO_COL, 7.9
A_X, A_W, A_TOP, A_H = 0.66, 5.64, 0.36, 1.95
ROW2_TOP, ROW2_H = 2.85, 1.60
B_X, B_W = 1.45, 1.55
C_X, C_W = 3.85, 2.45
ROW3_TOP, ROW3_H = 5.05, 2.45
D_X, D_W = 1.05, 1.70
E_X, E_W = 4.10, 1.85

E_LIM = (-10.0, 10.0)   # trend axis of e, mSv per yr


def num(v: float, nd: int = 2, signed: bool = False) -> str:
    """Format a number with a typographic minus sign."""
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def prod_name(key: str) -> str:
    return {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}[key]


def ax_in(fig, x: float, y_top: float, w: float, h: float):
    """Axes placed in inches from the top-left corner of the figure."""
    return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])


def load():
    d = json.loads((AN / "attribution.json").read_text())
    lat = json.loads((AN / "latitude_sensitivity.json").read_text())
    cw = json.loads((AN / "r1_common_window.json").read_text())
    return d, lat, cw


def _bold_header_ticks(ax, header_texts) -> None:
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in header_texts:
            lbl.set_fontweight("bold")


# ----------------------------------------------------------------- panel a
def panel_a(fig, d):
    ax = ax_in(fig, A_X, A_TOP, A_W, A_H)
    for key in KEYS:
        r = d[key]
        label = r["label"]
        colour = PRODUCT_COLOUR[key]
        yrs = np.asarray(r["years"])
        v = np.asarray(r["F_ov_Sv"])
        ax.plot(yrs, v, color=colour, lw=1.0, ls="-", zorder=3)
        ax.plot([yrs[-1]], [v[-1]], ls="none", marker=st.PRODUCT_MARKERS[label],
                ms=4.2, mfc=colour, mec=colour, mew=0.6, zorder=4)
        ax.text(yrs[-1] + 1.8, v[-1], label, color=colour, fontsize=7,
                ha="left", va="center")
    # Count of years with F_ovS < 0, each with its product marker.
    x0, y0 = 1959.0, -0.128
    ax.text(x0, y0, "Years with $F_{\\mathrm{ovS}}<0$", fontsize=7, ha="left",
            va="center", color="black")
    for i, key in enumerate(KEYS):
        r = d[key]
        colour = PRODUCT_COLOUR[key]
        yy = y0 - 0.0140 * (i + 1)
        ax.plot([x0 + 0.8], [yy], ls="none", marker=st.PRODUCT_MARKERS[r["label"]],
                ms=3.4, mfc=colour, mec=colour, mew=0.6, zorder=4)
        ax.text(x0 + 2.4, yy, f"{r['label']}: {r['n_years_F_ov_negative']} of {r['n_years']}",
                fontsize=7, ha="left", va="center", color="black")
    ax.axhline(0, color="0.35", lw=0.5, zorder=1)
    ax.set_xlim(1958, 2036)
    ax.set_ylim(-0.18, 0.03)
    ax.set_xticks(range(1960, 2021, 10))
    ax.set_yticks([-0.15, -0.10, -0.05, 0.0])
    ax.set_yticklabels([num(t) for t in [-0.15, -0.10, -0.05, 0.0]])
    ax.set_xlabel("Year")
    ax.set_ylabel("$F_{\\mathrm{ovS}}$ (Sv)")
    st.panel_label_pt(ax, "a")
    return ax


# ----------------------------------------------------------------- panel b
def panel_b(fig, d):
    ax = ax_in(fig, B_X, ROW2_TOP, B_W, ROW2_H)
    ticks, labels, headers = [], [], set()
    y = 9.0
    for key in KEYS:
        r = d[key]
        name = r["label"]
        colour = PRODUCT_COLOUR[key]
        mk = st.PRODUCT_MARKERS[name]
        ticks.append(y)
        labels.append(name)
        headers.add(name)
        y -= 1.0
        for kind, text in (("rec", "record mean"), ("rd", "last 10 yr")):
            if kind == "rec":
                est = r["mean_stats"]["mean"]
                lo, hi = r["mean_stats"]["ci"]
            else:
                est = r["recent_decade"]["mean_F_ov_Sv"]
                lo, hi = r["recent_decade"]["ci"]
            incl0 = lo <= 0.0 <= hi
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0,
                    ls=st.LS_NOT_SIGNIFICANT if incl0 else "-",
                    solid_capstyle="butt", zorder=3)
            ax.plot([est], [y], ls="none", marker=mk, ms=4.0,
                    mfc="white" if incl0 else colour, mec=colour, mew=0.8, zorder=5)
            ax.text(hi + 0.006, y, num(est, 3), color=colour, fontsize=7,
                    ha="left", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.3), zorder=6)
            ticks.append(y)
            labels.append(text)
            y -= 1.0
    ax.axvline(0, color="0.35", lw=0.5, zorder=1)
    ax.set_xlim(-0.36, 0.17)
    ax.set_ylim(0.4, 9.6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    _bold_header_ticks(ax, headers)
    ax.tick_params(axis="y", length=0)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in headers:
            key = [k for k in KEYS if d[k]["label"] == lbl.get_text()][0]
            lbl.set_color(PRODUCT_COLOUR[key])
    ax.set_xticks([-0.3, -0.2, -0.1, 0.0])
    ax.set_xticklabels([num(t, 1) for t in [-0.3, -0.2, -0.1, 0.0]])
    ax.set_xlabel("$F_{\\mathrm{ovS}}$ (Sv)")
    st.panel_label_pt(ax, "b")
    return ax


# ----------------------------------------------------------------- panel c
def panel_c(fig, lat):
    ax = ax_in(fig, C_X, ROW2_TOP, C_W, ROW2_H)
    marker = {"oras5": "o", "glorys12": "s"}
    for prod in ("oras5", "glorys12"):
        colour = PRODUCT_COLOUR[prod]
        entries = sorted(lat[prod].values(), key=lambda s: s["section_latitude"])
        lats = np.array([s["section_latitude"] for s in entries])
        means = np.array([s["mean_F_ov_Sv"] for s in entries])
        lo = np.array([s["ci_F_ov_Sv"][0] for s in entries])
        hi = np.array([s["ci_F_ov_Sv"][1] for s in entries])
        crosses = lo * hi <= 0
        mk = marker[prod]
        ax.plot(means, lats, "-", color=colour, lw=0.9, zorder=2)
        ax.errorbar(means, lats, xerr=[means - lo, hi - means], fmt="none",
                    ecolor=colour, elinewidth=0.7, capsize=1.2, zorder=3)
        ax.plot(means[~crosses], lats[~crosses], mk, color=colour, ms=3.2, zorder=4)
        ax.plot(means[crosses], lats[crosses], mk, ms=3.2, mfc="white",
                mec=colour, mew=0.8, zorder=4)
        top_i = int(np.argmax(lats))
        ax.text(means[top_i], lats[top_i] + 0.55, prod_name(prod), color=colour,
                fontsize=7, ha="center", va="bottom")
    ax.axvline(0, color="0.35", lw=0.5, zorder=1)
    ax.set_xlim(-0.158, 0.012)
    ax.set_ylim(-35.1, -24.1)
    ax.set_yticks([-34.5, -32, -30, -28, -25])
    ax.set_yticklabels(["34.5", "32", "30", "28", "25"])
    ax.set_xticks([-0.15, -0.10, -0.05, 0.0])
    ax.set_xticklabels([num(t) for t in [-0.15, -0.10, -0.05, 0.0]])
    ax.set_xlabel("$F_{\\mathrm{ovS}}$ (Sv)")
    ax.set_ylabel("Section latitude (°S)")
    ax.text(-0.152, -33.9, f"1993{EN}2025", fontsize=7, ha="left", va="center")
    st.panel_label_pt(ax, "c")
    return ax


# ----------------------------------------------------------------- panel d
def panel_d(fig, cw):
    ax = ax_in(fig, D_X, ROW3_TOP, D_W, ROW3_H)
    row_y = {"oras5": 2.0, "glorys12": 1.0, "ecco": 0.0}
    bar_h = 0.34
    ov_c = st.TERM_STYLE["overturning"]["colour"]
    az_c = st.TERM_STYLE["azonal"]["colour"]
    for key in KEYS:
        w = cw[key]["own"]
        y = row_y[key]
        fov, faz, ftot = w["mean_F_ov_Sv"], w["mean_F_az_Sv"], w["mean_F_total_Sv"]
        ax.barh(y, fov, height=bar_h, color=ov_c, lw=0, zorder=2)
        ax.barh(y, faz, height=bar_h, color=az_c, lw=0, zorder=2)
        ax.plot([ftot], [y], **st.term_marker("total", True, ms=4.6), zorder=5)
        ax.text(fov - 0.012, y, num(fov, 3, signed=True), ha="right", va="center",
                fontsize=7)
        ax.text(faz + 0.012, y, num(faz, 3, signed=True), ha="left", va="center",
                fontsize=7)
        ax.text(ftot, y + 0.25, num(ftot, 3, signed=True), ha="center", va="bottom",
                fontsize=7)
    for x, text in ((-0.09, "overturning"), (0.165, "azonal"), (0.31, "total")):
        ax.text(x, 2.72, text, ha="center", va="bottom", fontsize=7, color="black")
    ax.axvline(0, color="0.35", lw=0.5, zorder=1)
    ax.set_xlim(-0.36, 0.50)
    ax.set_ylim(-0.6, 3.1)
    ax.set_yticks([row_y[k] for k in KEYS])
    ax.set_yticklabels([prod_name(k) for k in KEYS])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
    ax.set_xticklabels([num(t, 1) for t in [-0.2, 0.0, 0.2, 0.4]])
    ax.set_xlabel("Mean transport (Sv)")
    st.panel_label_pt(ax, "d")
    return ax


# ----------------------------------------------------------------- panel e
def _forest_interval(ax, y, lo, hi, term, sig, lim):
    """95% interval in TERM_STYLE grey: solid when significant, dotted when the
    interval includes zero. A bound beyond the axis is marked by an arrowhead at
    the edge; the bound value is printed inside the axis at the lower edge and
    outside the axis at the upper edge."""
    colour = st.TERM_STYLE[term]["colour"]
    ls = "-" if sig else st.LS_NOT_SIGNIFICANT
    a, b = min(lo, hi), max(lo, hi)
    clip_lo, clip_hi = max(a, lim[0]), min(b, lim[1])
    if clip_hi > clip_lo:
        ax.plot([clip_lo, clip_hi], [y, y], color=colour, lw=1.4, ls=ls,
                solid_capstyle="butt", zorder=3)
    if a < lim[0]:
        ax.annotate("", xy=(lim[0], y), xytext=(lim[0] + 0.9, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.6,
                                    mutation_scale=5), zorder=4)
        ax.text(lim[0] + 0.35, y, num(a, 1, signed=True), fontsize=7, ha="left",
                va="center", color="black", zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.2))
    if b > lim[1]:
        ax.annotate("", xy=(lim[1], y), xytext=(lim[1] - 0.9, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.6,
                                    mutation_scale=5), zorder=4)
        ax.text(1.02, y, num(b, 1, signed=True), transform=ax.get_yaxis_transform(),
                fontsize=7, ha="left", va="center", color="black", clip_on=False)


def panel_e(fig, cw):
    ax = ax_in(fig, E_X, ROW3_TOP, E_W, ROW3_H)
    group_top = {"oras5": 11.0, "glorys12": 7.0, "ecco": 3.0}
    comps = [("overturning", "F_ov"), ("azonal", "F_az"), ("total", "F_total")]
    ticks, labels, headers = [], [], set()
    mismatches = []
    for key in KEYS:
        w = cw[key]["own"]
        y0 = group_top[key]
        ticks.append(y0)
        labels.append(prod_name(key))
        headers.add(prod_name(key))
        for j, (term, comp) in enumerate(comps):
            y = y0 - 1.0 - j
            t = w[f"trend_{comp}"]
            lo, hi = t["ci"]
            sig = not (lo <= 0.0 <= hi)
            if sig != bool(t["significant"]):
                mismatches.append((key, comp))
            _forest_interval(ax, y, lo, hi, term, sig, E_LIM)
            ax.plot([t["trend"]], [y], **st.term_marker(term, sig, ms=4.4), zorder=5)
            ticks.append(y)
            labels.append(term)
    if mismatches:
        print("WARNING: significance flag differs from CI for", mismatches)
    ax.axvline(0, color="0.35", lw=0.5, zorder=1)
    ax.set_xlim(*E_LIM)
    ax.set_ylim(-0.6, 11.6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    _bold_header_ticks(ax, headers)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels([num(t, 0) for t in [-10, -5, 0, 5, 10]])
    ax.set_xlabel("Trend (mSv yr$^{-1}$)")
    st.panel_label_pt(ax, "e")
    return ax


def build():
    st.set_style()
    d, lat, cw = load()
    fig = plt.figure(figsize=(W, H))
    panel_a(fig, d)
    panel_b(fig, d)
    panel_c(fig, lat)
    panel_d(fig, cw)
    panel_e(fig, cw)
    return fig


def main() -> None:
    fig = build()
    st.check_min_fontsize(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", format="pdf", metadata={"Creator": "paper3jc_fig2"})
    fig.savefig(f"{OUT}.png", format="png", dpi=600)
    print("wrote", OUT, "size", fig.get_size_inches())


if __name__ == "__main__":
    main()
