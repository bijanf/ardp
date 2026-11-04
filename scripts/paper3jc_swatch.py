"""Render the Paper 3 Journal of Climate palette and colour maps.

Three columns per row: the colour as printed, its grayscale equivalent and a
simulated deuteranopia view (Machado, Oliveira and Fernandes 2009, severity 1.0,
applied in linear RGB). Output: PAPER_3_v2/figures_jclim/style_swatch.png.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

import paper3jc_style as st  # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "PAPER_3_v2", "figures_jclim")
OUT_PNG = os.path.join(OUT_DIR, "style_swatch.png")

# Machado et al. 2009, deuteranopia, severity 1.0 (linear RGB)
MACHADO_DEUT = np.array(
    [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]
)


def to_linear(c) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_deuteranopia(rgb) -> np.ndarray:
    lin = to_linear(np.asarray(rgb, dtype=float))
    return np.clip(to_srgb(lin @ MACHADO_DEUT.T), 0.0, 1.0)


def to_grey(rgb) -> np.ndarray:
    lin = to_linear(np.asarray(rgb, dtype=float))
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    g = to_srgb(y)
    return np.stack([g, g, g], axis=-1)


PALETTE_ROWS = [
    ("products", [
        ("ORAS5", st.C_ORAS5),
        ("GLORYS\n12V1", st.C_GLORYS),
        ("ECCO\nV4r4", st.C_ECCO),
        ("SODA\n3.15.2", st.C_SODA),
    ]),
    ("EN4 and terms", [
        ("EN4", st.C_EN4),
        ("T", st.C_TRANSPORT),
        ("dS", st.C_SALINITY),
        ("F_ovS\ntotal", st.C_TOTAL),
    ]),
    ("limbs and cells", [
        ("north\nlimb", st.C_NORTH),
        ("south\nlimb", st.C_SOUTH),
        ("upper\ncell", st.C_CELL_UP),
        ("abyssal\ncell", st.C_CELL_AB),
    ]),
    ("CMIP6 and reference", [
        ("CMIP6", st.C_CMIP6),
        ("CMIP6\nlight", st.C_CMIP6_LIGHT),
        ("reference", st.C_GREY),
    ]),
]

CMAPS = [
    ("salinity", st.CMAP_SALINITY, False),
    ("salinity trend", st.CMAP_SALT_TREND, True),
    ("F_ovS", st.CMAP_FOVS, True),
]

# column left edges and width, in figure fractions (three columns, 6.5 in wide)
COL_X = [0.16, 0.47, 0.78]
COL_W = 0.21
HEADERS = ["as printed", "grayscale", "deuteranopia"]


def min_pairwise_distance(hexes) -> float:
    """Smallest Euclidean distance in linear RGB among a set of colours."""
    rgb = np.array([to_linear(mcolors.to_rgb(h)) for h in hexes])
    d = np.sqrt(((rgb[:, None, :] - rgb[None, :, :]) ** 2).sum(-1))
    d[np.eye(len(hexes), dtype=bool)] = np.inf
    return float(d.min())


def variant(rgb, column: int) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if column == 0:
        return rgb
    if column == 1:
        return to_grey(rgb)
    return simulate_deuteranopia(rgb)


def monotonic(seq: np.ndarray) -> bool:
    d = np.diff(seq)
    return bool(np.all(d >= -1e-3) or np.all(d <= 1e-3))


def main() -> None:
    st.set_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    n_rows = len(PALETTE_ROWS) + len(CMAPS)
    fig = plt.figure(figsize=(st.TWO_COL, 6.6))
    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.set_axis_off()
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)

    for x, h in zip(COL_X, HEADERS):
        overlay.text(x, 0.975, h, ha="left", va="top", fontsize=8)

    row_h = 0.84 / n_rows
    top = 0.93

    row = 0
    for label, items in PALETTE_ROWS:
        y0 = top - (row + 1) * row_h
        overlay.text(0.01, y0 + row_h * 0.5, label, ha="left", va="center", fontsize=7)
        slot = COL_W / len(items)
        for ci, x0 in enumerate(COL_X):
            for k, (name, hexcolour) in enumerate(items):
                shown = variant(mcolors.to_rgb(hexcolour), ci)
                fx = x0 + k * slot
                fy = y0 + row_h * 0.10
                overlay.add_patch(
                    Rectangle((fx, fy), slot * 0.94, row_h * 0.80,
                              facecolor=shown, edgecolor="none")
                )
                overlay.text(fx + slot * 0.47, fy + row_h * 0.40, name,
                             ha="center", va="center", fontsize=5.5, linespacing=0.95,
                             color=st.contrast_text(mcolors.to_hex(shown)))
        row += 1

    grad = np.linspace(0, 1, 256)
    for label, cmap, diverging in CMAPS:
        y0 = top - (row + 1) * row_h
        overlay.text(0.01, y0 + row_h * 0.5, label, ha="left", va="center", fontsize=7)
        rgba = cmap(grad)[:, :3]
        for ci, x0 in enumerate(COL_X):
            img = variant(rgba, ci)[None, :, :]
            ax = fig.add_axes([x0, y0 + row_h * 0.10, COL_W, row_h * 0.80])
            ax.imshow(img, aspect="auto", interpolation="nearest")
            ax.set_axis_off()
        row += 1

    # grayscale lightness check: sequential maps monotonic; diverging maps
    # monotonic in each half, moving away from a central light value
    checks = {}
    for label, cmap, diverging in CMAPS:
        g = to_grey(cmap(grad)[:, :3])[:, 0]
        if diverging:
            ok = monotonic(g[:128]) and monotonic(g[128:])
        else:
            ok = monotonic(g)
        checks[label] = ok

    prod_hex = [h for _, h in PALETTE_ROWS[0][1]]
    prod_col = min_pairwise_distance(prod_hex)
    prod_sim = min_pairwise_distance(
        [mcolors.to_hex(simulate_deuteranopia(mcolors.to_rgb(h))) for h in prod_hex]
    )
    notes1 = (f"Product colours, minimum linear-RGB distance: as printed {prod_col:.3f}, "
              f"deuteranopia {prod_sim:.3f}.")
    notes2 = "Grayscale lightness monotonic per half: " + ", ".join(
        f"{k} {'yes' if v else 'no'}" for k, v in checks.items())
    overlay.text(0.01, 0.030, notes1, fontsize=6, ha="left", va="bottom")
    overlay.text(0.01, 0.008, notes2, fontsize=6, ha="left", va="bottom")
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print("wrote", OUT_PNG)
    print(notes1)
    print(notes2)


if __name__ == "__main__":
    main()
