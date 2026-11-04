"""Figure style for the Journal of Climate (AMS) revision of Paper 3.

Authored at final printed size: no figure is scaled after saving.
Widths follow the AMS standard figure widths (19, 27 and 39 picas).
Text: tick labels 7 pt, axis labels 8 pt, panel labels 9 pt, annotations at
least 7 pt. Vector PDF with TrueType fonts (fonttype 42) plus a 600 dpi PNG.

Semantic colour system (the same meaning in every figure):
  products   ORAS5, GLORYS12V1, ECCO-V4r4 and SODA 3.15.2 keep the colours of
             scripts/paper3v2_style.py, so supplementary figures stay unchanged
  EN4        observation-based salinity, raspberry
  limbs      northward limb red-rose, southward limb steel blue; always drawn
             with direct labels, and never by colour alone in a product panel
  terms      transport term T teal, salinity term dS amber, total F_ovS black
  cells      upper cell mid grey, abyssal cell pale grey
  CMIP6      model distributions in neutral grey
  maps       salinity cividis (sequential, monotonic lightness);
             salinity trend BrBG reversed (brown salinification, teal freshening);
             F_ovS PuOr (purple negative, orange positive)
"""

from __future__ import annotations

import logging
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

# AMS print widths in inches (picas / 6)
ONE_COL = 19 / 6      # 3.17 in
ONEHALF_COL = 27 / 6  # 4.50 in
TWO_COL = 39 / 6      # 6.50 in
MAX_HEIGHT = 8.0      # inches

# ---------------------------------------------------------------- products
# Kept from scripts/paper3v2_style.py (supplementary figures keep them).
C_ORAS5 = "#20558a"
C_GLORYS = "#b3541e"
C_ECCO = "#3f7d52"
C_SODA = "#8d6ea8"

PRODUCT_COLOURS = {
    "ORAS5": C_ORAS5,
    "GLORYS12V1": C_GLORYS,
    "ECCO-V4r4": C_ECCO,
    "SODA 3.15.2": C_SODA,
}

# ---------------------------------------------------------------- observations
C_EN4 = "#b0246b"

# ---------------------------------------------------------------- physical quantities
C_NORTH = "#c8474f"        # northward limb (red-rose)
C_SOUTH = "#4f7fb8"        # southward limb (steel blue)
C_TRANSPORT = "#1f9a8a"    # transport term T
C_SALINITY = "#e8a33d"     # salinity term dS
C_SALINITY_DARK = "#95620a"  # salinity term dS, darker ochre for print and colour-vision
                          # (added for Fig 3; clear of the GLORYS12V1 product colour)
C_TOTAL = "#111111"        # total F_ovS and its trend
C_CELL_UP = "#7a7a7a"      # upper overturning cell
C_CELL_AB = "#c4c4c4"      # abyssal overturning cell
C_CMIP6 = "#6f6f6f"        # CMIP6 model distributions
C_CMIP6_LIGHT = "#bdbdbd"
C_GREY = "#9a9a9a"         # reference lines and guides

# ---------------------------------------------------------------- colour maps
CMAP_SALINITY = plt.get_cmap("cividis")   # sequential, monotonic lightness
CMAP_SALT_TREND = plt.get_cmap("BrBG_r")  # diverging: brown salinification
CMAP_FOVS = plt.get_cmap("PuOr_r")        # diverging: purple negative F_ovS, orange positive

# Hatching for "not significant" (lines and bars), open markers for points.
HATCH_NS = "////"

FONT_FAMILY = ["Arial", "Liberation Sans", "Helvetica", "TeX Gyre Heros", "DejaVu Sans"]


def set_style() -> None:
    """Final-size rcParams: fonts embedded as TrueType, no scaling on save."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.default": "regular",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "hatch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.borderaxespad": 0.2,
            "legend.columnspacing": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "savefig.dpi": 600,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure(width: str = "two", height: float = 4.0):
    """Create a figure at its final size. width is 'one', 'onehalf' or 'two'."""
    w = {"one": ONE_COL, "onehalf": ONEHALF_COL, "two": TWO_COL}[width]
    if height > MAX_HEIGHT:
        raise ValueError(f"height {height} in exceeds {MAX_HEIGHT} in")
    fig = plt.figure(figsize=(w, height))
    return fig


def panel_label(ax, letter: str, x: float = -0.14, y: float = 1.02) -> None:
    """AMS panel label: lowercase letter in parentheses, 9 pt, plain weight."""
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="normal",
        ha="left",
        va="bottom",
    )


def direct_label(
    ax,
    x: float,
    y: float,
    text: str,
    colour: str = "black",
    ha: str = "left",
    va: str = "center",
    fontsize: float = 7,
    weight: str = "normal",
    **kwargs,
) -> None:
    """Label placed directly on the data, in place of a legend entry."""
    ax.text(
        x,
        y,
        text,
        color=colour,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=weight,
        **kwargs,
    )


def value_label(ax, x: float, y: float, value: float, fmt: str, unit: str = "", **kwargs) -> str:
    """Annotate a key number read from data. Returns the formatted string."""
    text = fmt.format(value) + (f" {unit}" if unit else "")
    direct_label(ax, x, y, text, **kwargs)
    return text


def ci_interval(
    ax,
    y: float,
    lo: float,
    hi: float,
    est: float | None = None,
    colour: str = "black",
    significant: bool = True,
    orientation: str = "horizontal",
    fmt: str = "{:+.2f}",
    marker_size: float = 3.5,
    lw: float = 1.0,
    cap: float = 2.5,
) -> None:
    """Draw a 95 per cent interval at position y (x for vertical) with an estimate.

    The interval is clipped to the axis limits. When a bound lies outside the
    axis, the line ends in an arrow at the edge and the bound value is printed
    there. Not significant intervals use open markers (white face) and a dotted
    line style so that significance does not depend on colour alone.
    """
    if orientation == "horizontal":
        xlo, xhi = ax.get_xlim()
        a, b = min(lo, hi), max(lo, hi)
        ls = "-" if significant else ":"
        clip_lo, clip_hi = max(a, xlo), min(b, xhi)
        out_lo, out_hi = a < xlo, b > xhi
        if clip_hi > clip_lo:
            ax.plot([clip_lo, clip_hi], [y, y], color=colour, lw=lw, ls=ls,
                    solid_capstyle="butt", zorder=3)
        for val, flag, sign in ((a, out_lo, -1), (b, out_hi, +1)):
            if flag:
                edge = xlo if sign < 0 else xhi
                ax.annotate(
                    "", xy=(edge, y), xytext=(edge - sign * 0.02 * (xhi - xlo), y),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.6, mutation_scale=5),
                    zorder=4,
                )
                direct_label(ax, edge, y - 0.22, fmt.format(val), colour=colour,
                             ha="left" if sign > 0 else "right", va="top", fontsize=7)
            elif val == a or val == b:
                ax.plot([val, val], [y - cap / 100, y + cap / 100], color=colour,
                        lw=lw, zorder=3)
    else:
        raise NotImplementedError("vertical intervals are added as needed")
    if est is not None:
        ax.plot([est], [y], marker="o", ms=marker_size,
                mfc=colour if significant else "white", mec=colour, mew=0.8, ls="none",
                zorder=5)


def significance_marker(significant: bool, colour: str):
    """Filled marker for significant values, open marker for not significant."""
    return dict(marker="o", mfc=colour if significant else "white", mec=colour,
                mew=0.8, ms=3.5, ls="none")


def contrast_text(colour: str) -> str:
    """Black or white text, whichever reads on the given fill colour."""
    r, g, b = mcolors.to_rgb(colour)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 0.45 else "white"


def save(fig, outbase: str) -> None:
    """Write PDF (embedded TrueType) and 600 dpi PNG at the fixed figure size.

    No bbox cropping: the final width equals the authored width."""
    fig.savefig(f"{outbase}.pdf", format="pdf", metadata={"Creator": "paper3jc_style"})
    fig.savefig(f"{outbase}.png", format="png", dpi=600)
    print(f"wrote {outbase}.pdf and .png at {fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in")


# Decomposition terms are drawn in greyscale with distinct marker shapes, so that
# no term shares a hue with a product. Filled marker and solid interval: the
# interval excludes zero. Open marker (white face) and dotted interval: it does not.
TERM_STYLE = {
    "total": dict(colour="#111111", marker="o"),
    "salinity": dict(colour="#4a4a4a", marker="s"),
    "transport": dict(colour="#a3a3a3", marker="D"),
    "residual": dict(colour="#111111", marker="x"),
    "upper_cell": dict(colour="#4a4a4a", marker="^"),
    "abyssal_cell": dict(colour="#a3a3a3", marker="v"),
    "overturning": dict(colour="#111111", marker="o"),
    "azonal": dict(colour="#a3a3a3", marker="D"),
}

# Products: colour, marker shape and direct label together; never line style alone.
PRODUCT_MARKERS = {
    "ORAS5": "o",
    "GLORYS12V1": "s",
    "ECCO-V4r4": "^",
    "SODA 3.15.2": "D",
    "EN4.2.2": "P",
}

# Line-style semantics shared by every figure:
#   solid  : products, the northward limb, the upper cell, totals
#   dashed : the southward limb and the abyssal cell only
#   dotted : intervals that include zero, and reference lines (1:1, one quarter)
LS_SOUTH = "--"
LS_ABYSSAL = "--"
LS_NOT_SIGNIFICANT = ":"
LS_REFERENCE = ":"


def term_marker(term: str, significant: bool = True, ms: float = 4.0) -> dict:
    """Marker keywords for a decomposition term (see TERM_STYLE)."""
    st = TERM_STYLE[term]
    face = st["colour"] if significant else "white"
    edge = "#111111" if term == "transport" else st["colour"]
    return dict(marker=st["marker"], mfc=face, mec=edge, mew=0.8, ms=ms, ls="none")


def panel_label_pt(ax, letter: str, dx_pt: float = -30.0, dy_pt: float = 5.0) -> None:
    """AMS panel label at a fixed offset in points from the top-left axes corner,
    so that labels sit at the same distance from their panels in every figure."""
    ax.annotate(
        f"({letter})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx_pt, dy_pt),
        textcoords="offset points",
        fontsize=9,
        fontweight="normal",
        ha="left",
        va="bottom",
    )


def check_min_fontsize(fig, minimum: float = 7.0) -> list:
    """List every visible text in the figure that is smaller than `minimum` pt.

    Call before save(); fix every item it prints."""
    small = []
    for t in fig.findobj(lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text")):
        try:
            txt = t.get_text()
        except Exception:
            continue
        if txt and t.get_visible() and t.get_fontsize() < minimum - 1e-6:
            small.append((round(t.get_fontsize(), 2), txt[:40]))
    for size, txt in small:
        print(f"SMALL TEXT {size} pt: {txt!r}")
    return small


def ams_mode() -> bool:
    """Panel labels are always AMS style in this module."""
    return os.environ.get("PANEL_STYLE", "ams").lower() == "ams"
