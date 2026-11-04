#!/usr/bin/env python3
"""Figure 5 (Journal of Climate version): the limb salinity contrast in EN4.

The circulation is held at the record-mean profile of each reanalysis and only
the EN4.2.2 objective analysis varies (2005-2024). Data handling is that of
scripts/plot_paper3r1_fig5_en4.py (same JSON keys, same profile file, same
Santer-adjusted trend and interval function); only the layout and the
annotations differ.

(a)  Transport-weighted salinity of the northward (solid) and southward (dashed)
     limb over the full column, as anomalies, with least-squares trends. The
     trend is drawn bold where its Santer-adjusted p-value is below 0.05 and
     thin otherwise (thin trends are not dotted: dotting is reserved for
     intervals that include zero).
(b)  As (a) with both limbs defined inside the upper 2000 m.
(c)  Zonal-mean salinity trend against depth, truncated at 2500 m, on a
     square-root depth axis, 2005-2024, on the EN4 levels. EN4 (raspberry, circle
     marker "P"), ORAS5 (blue, "o") and GLORYS12V1 (orange, "s") are all solid
     lines with product markers at 50, 200, 500, 1000 and 2000 m and direct
     labels; bold where p < 0.05. The grey shading gives the EN4 observation
     weight averaged along the section (upper axis), cut at the same depth.
(d)  Hero: trend of the limb salinity contrast over 2005-2024 with 95 per cent
     intervals, grouped into EN4 under each reanalysis circulation and the
     reanalysis salinities with fixed and with own circulation. Markers follow
     PRODUCT_MARKERS. Open markers and dotted intervals mark intervals that
     include zero (equivalently p >= 0.05; checked at run time). Arrows mark
     intervals beyond the axis. The ratio of fixed-circulation to EN4 trends is
     not shown: its intervals are too wide to resolve it (see the text).

Writes PAPER_3_v2/figures_jclim/Fig5_en4.{pdf,png} at its printed size.
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
from matplotlib import colors as mcolors  # noqa: E402
from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

import paper3jc_style as st  # noqa: E402

AN = REPO / "PAPER_3_v2" / "analysis"
RES = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "figures_jclim" / "Fig5_en4"

W_IN = st.TWO_COL   # 6.5 in
H_IN = 5.6          # inches; within the 8 in limit
MINUS = "−"
DELTA = "Δ"
Z_MAX = 2500.0      # depth truncation of (c) and of its observation-weight shading (m)
MARK_DEPTHS = [50, 200, 500, 1000, 2000]   # product markers on the profiles (m)
VALUE_X = 1.035     # axes fraction of (d) where the trend column starts
DATA_ALPHA = 0.45   # the raw limb series are drawn lighter than their trends
REANALYSIS_ALPHA = 0.6  # lighter intervals for the reanalysis rows of (d)
MARKER_MS = {"P": 4.6, "o": 3.5, "s": 3.2, "^": 3.6, "D": 3.2}  # filled glyphs look alike at equal size
PROFILES = {        # key in the profile file: (colour, product name for its marker)
    "en4": (st.C_EN4, "EN4.2.2"),
    "oras5": (st.C_ORAS5, "ORAS5"),
    "glorys12": (st.C_GLORYS, "GLORYS12V1"),
}


# ------------------------------------------------------------------ data handling
def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def trend_ci(years, series):
    """Trend in PSU per decade, 95 per cent interval and Santer p-value.

    Identical to the old script: the interval uses the Santer-adjusted standard
    error with a t quantile at n_eff - 2 degrees of freedom.
    """
    x = np.asarray(years, float)
    y = np.asarray(series, float)
    f = ols_santer(x, y)
    dof = max(f["n_eff"] - 2, 1)
    h = stats.t.ppf(0.975, dof) * f["stderr_santer"]
    return 10 * f["slope"], 10 * (f["slope"] - h), 10 * (f["slope"] + h), f["p_santer"]


def signed(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}".replace("-", MINUS)


def tick(v: float) -> str:
    """Axis tick labels: two decimals everywhere (values in the text use three)."""
    return f"{v:.2f}".replace("-", MINUS)


def p_text(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    if p < 0.1:
        return f"{p:.3f}"
    return f"{p:.2f}"


# ------------------------------------------------------------------ geometry (inches)
def ax_in(fig, x0: float, y0: float, w: float, h: float, **kw):
    """Axes placed by its lower-left corner and size in inches from the figure corner."""
    return fig.add_axes([x0 / W_IN, y0 / H_IN, w / W_IN, h / H_IN], **kw)


def limb_series(ax, block: dict, colour: str):
    """Limb salinity anomalies and their least-squares trends, as in the old script.

    Raw series are light; the trend is solid (northward) or dashed (southward),
    bold where significant and thin otherwise.
    """
    yrs = np.array(block["years"])
    ax.axhline(0, color="0.55", lw=0.5, zorder=1)
    for var, ls in (("S_north", "-"), ("S_south", st.LS_SOUTH)):
        s = np.array(block[var]["series"])
        a = s - s.mean()
        ax.plot(yrs, a, color=colour, lw=0.7, ls=ls, alpha=DATA_ALPHA, zorder=2)
        coef = np.polyfit(yrs, a, 1)
        sig = block[var]["full"]["p_santer"] < 0.05
        ax.plot(
            yrs,
            np.polyval(coef, yrs),
            color=colour,
            lw=1.9 if sig else 0.9,
            ls=ls,
            zorder=3,
        )
    ax.set_xlim(2004.5, 2024.5)
    ax.set_xticks([2005, 2010, 2015, 2020])
    ax.set_xlabel("Year")


def limb_key(ax, block: dict, colour: str, x0: float = 0.04, y_top: float = 0.885):
    """Two key lines in the empty upper left corner: line style, trend and p-value.

    The key replaces the legend: each line sample has the style of its limb.
    A header gives the unit once for both lines.
    """
    ax.text(x0, 0.975, "trend (PSU decade$^{-1}$)", transform=ax.transAxes,
            fontsize=7, va="center", ha="left")
    rows = (
        ("northward", "S_north", "-"),
        ("southward", "S_south", st.LS_SOUTH),
    )
    for i, (name, var, ls) in enumerate(rows):
        yy = y_top - i * 0.105
        ax.plot([x0, x0 + 0.08], [yy, yy], transform=ax.transAxes, color=colour,
                lw=0.9, ls=ls, clip_on=False)
        tr = block[var]["full"]["trend_per_decade"]
        p = block[var]["full"]["p_santer"]
        ax.text(x0 + 0.11, yy, f"{name} {signed(tr)} ($p$ = {p_text(p)})",
                transform=ax.transAxes, fontsize=7, va="center", ha="left")


def main() -> None:
    st.set_style()
    dt = json.loads((AN / "en4_depth_test.json").read_text())
    attr = json.loads((AN / "attribution.json").read_text())
    fc = json.loads((AN / "final_checks.json").read_text())
    prof = xr.open_dataset(RES / "paper3r1_salinity_trend_profiles.nc")

    fig = plt.figure(figsize=(W_IN, H_IN))

    # ---------------------------------------------------------------- top row
    top_y, top_h = 3.3, 1.85
    axa = ax_in(fig, 0.62, top_y, 1.50, top_h)
    axb = ax_in(fig, 2.45, top_y, 1.50, top_h, sharey=axa)
    axc = ax_in(fig, 4.60, top_y, 1.45, top_h)

    limb_series(axa, dt["oras5"]["full_column"], st.C_EN4)
    axa.set_ylim(-0.045, 0.09)
    axa.set_yticks([-0.04, -0.02, 0, 0.02, 0.04, 0.06])
    axa.set_yticklabels([tick(v) for v in [-0.04, -0.02, 0, 0.02, 0.04, 0.06]])
    axa.set_ylabel("Limb salinity anomaly (PSU)")
    limb_key(axa, dt["oras5"]["full_column"], st.C_EN4)

    limb_series(axb, dt["oras5"]["upper_2000m"], st.C_EN4)
    axb.tick_params(axis="y", labelleft=False)
    limb_key(axb, dt["oras5"]["upper_2000m"], st.C_EN4)

    # (c) profiles of the trend, truncated at Z_MAX
    z = prof["depth"].values
    w = prof["obs_weight_en4"].values
    axw = axc.twiny()
    axw.fill_betweenx(z, 0, w, color="0.90", lw=0, zorder=0)   # clipped at Z_MAX below
    axw.set_xlim(0, 1)
    axw.set_xticks([0, 0.5, 1])
    axw.set_xticklabels([tick(v) for v in [0, 0.5, 1]])
    axw.tick_params(labelsize=7)
    axw.set_xlabel("EN4 observation weight", fontsize=8, labelpad=2)
    axc.set_zorder(axw.get_zorder() + 1)
    axc.patch.set_visible(False)
    idx = [int(np.argmin(np.abs(z - d))) for d in MARK_DEPTHS]
    for key, (col, name) in PROFILES.items():
        tr = prof[f"trend_{key}"].values
        pv = prof[f"p_{key}"].values
        # every product line is solid; colour, marker and direct label identify it
        axc.plot(tr, z, color=col, lw=0.8, ls="-", zorder=3)
        axc.plot(np.where(pv < 0.05, tr, np.nan), z, color=col, lw=2.0, ls="-", zorder=4)
        mk = st.PRODUCT_MARKERS[name]
        axc.plot(tr[idx], z[idx], ls="none", marker=mk, ms=MARKER_MS[mk],
                 mfc=col, mec="white", mew=0.4, zorder=5)
    axc.axvline(0, color="black", lw=0.4, zorder=2)
    for ax in (axc, axw):
        ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    axc.set_ylim(Z_MAX, 0)
    axc.set_yticks([0, 100, 300, 600, 1000, 2000])
    axc.set_xlim(-0.012, 0.145)
    axc.set_xticks([0, 0.04, 0.08])
    axc.set_xticklabels([tick(v) for v in [0, 0.04, 0.08]])
    axc.set_ylabel("Depth (m)")
    axc.set_xlabel("Trend (PSU decade$^{-1}$)")
    # direct labels in a stack to the right of the profiles: a solid line sample
    # with the product marker, then its name in the profile colour
    label_stack = [("GLORYS12V1", "glorys12", 6.0),
                   ("EN4.2.2", "en4", 45.0),
                   ("ORAS5", "oras5", 150.0)]
    x_sample = (0.085, 0.097)
    for text, key, zl in label_stack:
        col, name = PROFILES[key]
        axc.plot(x_sample, [zl, zl], color=col, lw=1.0, ls="-", zorder=5,
                 solid_capstyle="butt")
        mk = st.PRODUCT_MARKERS[name]
        axc.plot([np.mean(x_sample)], [zl], ls="none", marker=mk, ms=MARKER_MS[mk],
                 mfc=col, mec="white", mew=0.4, zorder=6)
        axc.text(0.102, zl, text, color=col, fontsize=7, va="center", ha="left",
                 zorder=5)

    # ---------------------------------------------------------------- hero (d)
    axd = ax_in(fig, 2.05, 0.60, 3.20, 2.30)
    xl = (-0.06, 0.15)

    fixed = {}
    for key in ("oras5", "glorys12"):
        fixed[key] = fc[key]["frozen_field"]["salinity_only"]["window_2005_2024"]

    def row(label, est, lo, hi, p, colour, marker):
        # significance is the interval excluding zero; the p-value must agree
        sig = (lo > 0) or (hi < 0)
        if sig != (p < 0.05):
            print(f"CHECK: interval and p disagree for {label}: [{lo:+.4f}, {hi:+.4f}] p={p:.3g}")
        return dict(kind="row", label=label, est=est, lo=lo, hi=hi, p=p,
                    colour=colour, marker=marker, sig=sig)

    def hdr(label):
        return dict(kind="hdr", label=label)

    rows = [hdr("EN4 salinity under reanalysis circulation")]
    for blk, bl in (("full_column", "full column"), ("upper_2000m", "upper 2000 m")):
        for key, lab in (("oras5", "ORAS5"), ("glorys12", "GLORYS12V1")):
            b = dt[key][blk]
            est, lo, hi, p = trend_ci(b["years"], b["dS"]["series"])
            rows.append(row(f"{bl.capitalize()}, {lab} circulation",
                            est, lo, hi, p, st.C_EN4, st.PRODUCT_MARKERS["EN4.2.2"]))
    rows.append(hdr("Reanalysis salinity, fixed circulation"))
    for key, lab, col in (("oras5", "ORAS5", st.C_ORAS5),
                          ("glorys12", "GLORYS12V1", st.C_GLORYS)):
        f = fixed[key]
        lo, hi = f["dS_ci_per_decade"]
        rows.append(row(lab, f["dS_trend_per_decade"], lo, hi, f["dS_p"], col,
                        st.PRODUCT_MARKERS[lab]))
    rows.append(hdr("Reanalysis salinity and circulation, own"))
    for key, lab, col in (("oras5", "ORAS5", st.C_ORAS5),
                          ("glorys12", "GLORYS12V1", st.C_GLORYS)):
        yrs = np.array(attr[key]["years"])
        ds_ = np.array(attr[key]["dS_limb_PSU"])
        m = (yrs >= 2005) & (yrs <= 2024)
        est, lo, hi, p = trend_ci(yrs[m], ds_[m])
        rows.append(row(lab, est, lo, hi, p, col, st.PRODUCT_MARKERS[lab]))

    n = len(rows)
    ypos = np.arange(n)[::-1].astype(float)
    axd.set_xlim(*xl)
    axd.set_ylim(-0.6, n - 0.4)
    axd.axvline(0, color="black", lw=0.5, zorder=1)
    # light band behind the EN4 block: the observation-based rows of the message
    en4_idx = [i for i, r in enumerate(rows) if r["kind"] == "row" and r["colour"] == st.C_EN4]
    axd.axhspan(ypos[max(en4_idx)] - 0.5, ypos[min(en4_idx)] + 0.5,
                color=st.C_EN4, alpha=0.07, lw=0, zorder=0)
    # computed message for the EN4 block, true only if every lower bound is above zero
    en4_all_exclude = all(rows[i]["lo"] > 0 for i in en4_idx)
    if en4_all_exclude:
        y_mid = float(np.mean(ypos[en4_idx]))
        axd.text(0.066, y_mid, "all four EN4 intervals\nexclude zero", color=st.C_EN4,
                 fontsize=7, ha="left", va="center", linespacing=1.3, zorder=6)
    ticks, labels = [], []
    trans_v = axd.get_yaxis_transform()  # x in axes fraction, y in data
    for yy, r in zip(ypos, rows, strict=True):
        ticks.append(yy)
        labels.append(r["label"])
        if r["kind"] == "hdr":
            continue
        col, est, lo, hi, sig = r["colour"], r["est"], r["lo"], r["hi"], r["sig"]
        # reanalysis rows are drawn lighter so that the EN4 rows carry the hero
        alpha = 1.0 if col == st.C_EN4 else REANALYSIS_ALPHA
        lcol = mcolors.to_rgba(col, alpha)
        a, b = max(lo, xl[0]), min(hi, xl[1])
        axd.plot([a, b], [yy, yy], color=lcol, lw=1.3 if sig else 0.9,
                 ls="-" if sig else (0, (1.0, 1.1)), solid_capstyle="butt", zorder=3)
        for edge, flag, sign in ((xl[0], lo < xl[0], -1), (xl[1], hi > xl[1], +1)):
            if flag:
                axd.annotate(
                    "", xy=(edge, yy), xytext=(edge - sign * 0.012, yy),
                    arrowprops=dict(arrowstyle="-|>", color=lcol, lw=0.6, mutation_scale=5),
                    zorder=4,
                )
        axd.plot([est], [yy], marker=r["marker"], ms=MARKER_MS[r["marker"]], mec=col, mew=0.8,
                 mfc=col if sig else "white", ls="none", zorder=5)
        value = f"{signed(est)} [{signed(lo)}, {signed(hi)}]"
        axd.text(VALUE_X, yy, value, transform=trans_v, fontsize=7, ha="left", va="center")
    axd.set_yticks(ticks)
    axd.set_yticklabels(labels, fontsize=7)
    axd.tick_params(axis="y", length=0, pad=4)
    for tl, r in zip(axd.get_yticklabels(), rows, strict=True):
        if r["kind"] == "hdr":
            tl.set_fontweight("bold")
    xt = [-0.05, 0, 0.05, 0.10]
    axd.set_xticks(xt)
    axd.set_xticklabels([tick(v) for v in xt])
    axd.set_xlabel(f"Trend in {DELTA}S, 2005–2024 (PSU decade$^{{-1}}$)")
    for s in ("left", "bottom"):
        axd.spines[s].set_visible(True)
    axd.spines["left"].set_visible(False)
    # column heads in the first header row, above the values
    hdr_y = ypos[0]
    axd.text(VALUE_X, hdr_y, "trend [95% interval]", transform=trans_v, fontsize=7,
             ha="left", va="center")

    # ---------------------------------------------------------------- panel labels
    # one offset for every panel (panel_label_pt default), so each row shares a height
    for ax, letter in ((axa, "a"), (axb, "b"), (axc, "c")):
        st.panel_label_pt(ax, letter)
    # (d) aligned with (a) at the left edge, clear of the x label of (a)
    wfig = fig.get_size_inches()[0]
    dx_d = (axa.get_position().x0 - axd.get_position().x0) * wfig * 72.0 - 30.0
    st.panel_label_pt(axd, "d", dx_pt=dx_d)

    small = st.check_min_fontsize(fig)
    if small:
        raise SystemExit(f"{len(small)} text items below 7 pt; fix before saving")

    save_base = str(OUT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st.save(fig, save_base)

    print("rows:")
    for r in rows:
        if r["kind"] == "row":
            print(f"  {r['label']:42s} {r['est']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                  f"p={r['p']:.3g} sig={r['sig']}")
    print(f"EN4 block: all lower bounds above zero = {en4_all_exclude}")
    print(f"(c) markers at depths {[round(float(z[i]), 1) for i in idx]} m; "
          f"axis truncated at {Z_MAX:.0f} m; deepest level shown {float(z[z <= Z_MAX].max()):.1f} m")


if __name__ == "__main__":
    main()
