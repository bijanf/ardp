"""Nature Communications publication-quality plot styling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature Communications figure specifications:
# - Single column: 88 mm (3.46 in)
# - Double column: 180 mm (7.09 in)
# - Max height: 247 mm (9.72 in)
# - Font: sans-serif (Helvetica/Arial), min 5 pt, labels 7-8 pt
# - Resolution: 300 dpi minimum
# - File formats: PDF, EPS, or high-res PNG/TIFF

SINGLE_COL = 3.46  # inches
DOUBLE_COL = 7.09  # inches
MAX_HEIGHT = 9.72  # inches

# GRL (Geophysical Research Letters) figure specifications:
# - Single column: 8.4 cm (3.31 in)
# - Full width: 17.1 cm (6.73 in)
# - Max height: 23.7 cm (9.33 in)
GRL_SINGLE_COL = 3.31  # inches
GRL_FULL_WIDTH = 6.73  # inches
GRL_MAX_HEIGHT = 9.33  # inches

# Color palette — colorblind-safe, print-friendly
# Based on Paul Tol's qualitative palette
COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

FINGERPRINT_COLORS = {
    "f_ovs": COLORS["blue"],
    "salinity_pileup": COLORS["red"],
    "nawh": COLORS["cyan"],
    "gulf_stream": COLORS["green"],
}


def apply_nature_style() -> None:
    """Apply Nature Communications styling globally."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        # Axes
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.2,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Grid
        "axes.grid": False,
    })


def figure_single_col(
    nrows: int = 1,
    ncols: int = 1,
    height_ratio: float = 0.75,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Create a single-column Nature Comms figure."""
    apply_nature_style()
    h = SINGLE_COL * height_ratio * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(SINGLE_COL, min(h, MAX_HEIGHT)),
        **kwargs,
    )
    return fig, axes


def figure_double_col(
    nrows: int = 1,
    ncols: int = 1,
    height_ratio: float = 0.4,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Create a double-column Nature Comms figure."""
    apply_nature_style()
    h = DOUBLE_COL * height_ratio * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(DOUBLE_COL, min(h, MAX_HEIGHT)),
        **kwargs,
    )
    return fig, axes


def figure_grl_single(
    nrows: int = 1,
    ncols: int = 1,
    height_ratio: float = 0.75,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Create a single-column GRL figure."""
    apply_nature_style()
    h = GRL_SINGLE_COL * height_ratio * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(GRL_SINGLE_COL, min(h, GRL_MAX_HEIGHT)),
        **kwargs,
    )
    return fig, axes


def figure_grl_full(
    nrows: int = 1,
    ncols: int = 1,
    height_ratio: float = 0.4,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Create a full-width GRL figure."""
    apply_nature_style()
    h = GRL_FULL_WIDTH * height_ratio * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(GRL_FULL_WIDTH, min(h, GRL_MAX_HEIGHT)),
        **kwargs,
    )
    return fig, axes


def add_panel_label(
    ax: Any,
    label: str,
    x: float = -0.12,
    y: float = 1.08,
) -> None:
    """Add a panel label (a, b, c, ...) to an axes, Nature Comms style."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def format_year_axis(ax: Any) -> None:
    """Format x-axis to show years cleanly."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def add_trend_annotation(
    ax: Any,
    slope: float,
    unit: str = "",
    pvalue: float | None = None,
    position: str = "upper left",
) -> None:
    """Add a trend annotation box to an axes."""
    text = f"Trend: {slope:+.2f} {unit}/yr"
    if pvalue is not None:
        if pvalue < 0.001:
            text += " (p < 0.001)"
        elif pvalue < 0.01:
            text += f" (p = {pvalue:.3f})"
        else:
            text += f" (p = {pvalue:.2f})"

    loc_map = {
        "upper left": (0.03, 0.95),
        "upper right": (0.97, 0.95),
        "lower left": (0.03, 0.12),
        "lower right": (0.97, 0.12),
    }
    x, y = loc_map.get(position, (0.03, 0.95))
    ha = "right" if "right" in position else "left"

    ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        fontsize=6,
        ha=ha,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.9},
    )


def save_publication_figure(
    fig: Any,
    path: str | Path,
    formats: list[str] | None = None,
    bbox_inches: str | None = "tight",
) -> None:
    """Save figure in publication-quality formats.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str or Path
        Base path (without extension for multi-format).
    formats : list[str]
        File formats to save. Default: ["png", "pdf"].
    bbox_inches : str or None
        Passed through to ``fig.savefig``. Defaults to ``"tight"``, which
        expands the saved canvas to include any artists drawn outside the
        axes (titles, legends, leader lines from ``adjustText``, ...). Set
        to ``None`` to preserve the ``figsize`` aspect ratio exactly, at
        the cost of clipping anything outside the figure bounds. Use
        ``None`` for figures whose aspect is critical (otherwise leader
        lines can bloat the canvas into a near-square and LaTeX will
        render the figure tiny at ``width=0.9\\linewidth``).
    """
    if formats is None:
        formats = ["png", "pdf"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve special sentinels: None and the string "fixed" both mean
    # "use the figure's own canvas, don't recompute a tight bbox".
    if bbox_inches in (None, "fixed"):
        from matplotlib.transforms import Bbox
        w, h = fig.get_size_inches()
        bbox_inches = Bbox([[0, 0], [w, h]])

    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=300, bbox_inches=bbox_inches)
        print(f"Saved: {out}")

    plt.close(fig)
