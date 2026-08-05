#!/usr/bin/env python3
"""PAPER_3_v2 Figure 3: the salinity change that moves F_ovS.

Panel (a) interbasin surface salinity contrast, subtropical South Atlantic
minus subtropical South Indo-Pacific, for ORAS5 and GLORYS12V1 over the common
1993-2024 window. The Indo-Pacific box is wrapped across the date line, which
the earlier version of this index was not.

Panel (b) zonal-mean salinity trend profile at the 34.5 S F_ovS section over
1993-2025. Solid where the Santer N_eff p-value is below 0.05, dashed
elsewhere. The two products differ by roughly a factor of 29 in the upper
layer, so each gets its own colour-matched x-axis with a common zero.

Panel (c) the vertical salinity contrast at the section itself, the upper limb
(0 to 1000 m) minus the deep limb (1000 to 3000 m) thickness-weighted
zonal-mean salinity. This is the quantity that appears in the F_ovS variation
identity, and it is computed from the salinity field alone, independently of
F_ovS and of the overturning.

The 34.5 S profiles are read from
revision/rev_papaer3_02/results/WP5_section_salinity_trends.nc; the interbasin
index is read from the wrapped-box recomputation in
scripts/analysis_paper3v2_pileup.py.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS = REPO / "data" / "results"
WP5_NC = (
    REPO / "revision" / "rev_papaer3_02" / "results" / "WP5_section_salinity_trends.nc"
)
PILEUP_JSON = REPO / "PAPER_3_v2" / "analysis" / "pileup.json"
OUT_BASE = REPO / "PAPER_3_v2" / "figures" / "Fig3_saltredist"

C_ORAS5 = "#20558a"
C_GLORYS = "#b3541e"

TREND_WINDOW = (1993, 2024)
ALPHA_SIG = 0.05

UPPER_LIMB = (0.0, 1000.0)
DEEP_LIMB = (1000.0, 3000.0)

# Recomputed with the wrapped Indo-Pacific box, see analysis/pileup.json.
CANON_PILEUP = {
    "oras5": {"trend_per_decade": 0.0325, "p_santer": 0.0090},
    "glorys12": {"trend_per_decade": 0.0698, "p_santer": 0.0254},
}
CANON_LAYERS = {
    "oras5": {(0, 300): 0.0038, (300, 1200): 0.0086},
    "glorys12": {(0, 300): 0.1089, (300, 1200): 0.0426},
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "lines.solid_capstyle": "round",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def darken(hex_colour: str, factor: float = 0.68) -> tuple[float, float, float]:
    rgb = matplotlib.colors.to_rgb(hex_colour)
    return tuple(factor * c for c in rgb)


def cell_thickness(depth: np.ndarray) -> np.ndarray:
    """Thickness of each level, from the midpoints between level centres."""
    inner = 0.5 * (depth[:-1] + depth[1:])
    edges = np.concatenate(([0.0], inner, [depth[-1] + (depth[-1] - inner[-1])]))
    return np.diff(edges)


def layer_mean(depth: np.ndarray, field: np.ndarray, lo: float, hi: float) -> float:
    thick = cell_thickness(depth)
    sel = (depth >= lo) & (depth < hi)
    return float(np.sum(field[sel] * thick[sel]) / np.sum(thick[sel]))


def clip_profile(
    depth: np.ndarray, trend: np.ndarray, pval: np.ndarray, zmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(int(np.sum(depth <= zmax)) + 1, len(depth))
    return depth[:n], trend[:n], pval[:n]


def sig_segments(
    depth: np.ndarray, trend: np.ndarray, sig: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Solid overlay for each significant level, spanning half a level either
    side so the two line styles join without gaps."""
    n = len(depth)
    z_mid = 0.5 * (depth[:-1] + depth[1:])
    t_mid = 0.5 * (trend[:-1] + trend[1:])
    segments = []
    for i in np.flatnonzero(sig):
        zs, ts = [], []
        if i > 0:
            zs.append(z_mid[i - 1])
            ts.append(t_mid[i - 1])
        zs.append(depth[i])
        ts.append(trend[i])
        if i < n - 1:
            zs.append(z_mid[i])
            ts.append(t_mid[i])
        segments.append((np.array(ts), np.array(zs)))
    return segments


def panel_a(ax, pileup: dict) -> dict:
    out = {}
    y0, y1 = TREND_WINDOW
    for key, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        r = pileup[key]
        years = np.asarray(r["years"], dtype=float)
        values = np.asarray(r["index"], dtype=float)
        ax.plot(years, values, color=colour, lw=0.8, alpha=0.95, zorder=3)
        fit = ols_santer(years, values)
        xf = np.array([y0, y1], dtype=float)
        yf = fit["slope"] * xf + (values.mean() - fit["slope"] * years.mean())
        ax.plot(xf, yf, color=darken(colour), lw=1.4, zorder=4)

        canon = CANON_PILEUP[key]
        assert abs(fit["slope"] * 10.0 - canon["trend_per_decade"]) < 5e-4, key
        assert abs(fit["p_santer"] - canon["p_santer"]) < 5e-4, key
        out[key] = fit

    ax.set_xlim(1992, 2025)
    ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020])
    ax.set_xlabel("Year")
    ax.set_ylabel("Interbasin contrast (PSU)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color=C_ORAS5, lw=0.9, label="ORAS5"),
            Line2D([], [], color=C_GLORYS, lw=0.9, label="GLORYS12"),
        ],
        loc="upper left",
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    return out


def panel_b(ax, prof: dict, zmax: float = 1500.0) -> dict:
    """ORAS5 on a top x-axis, GLORYS12 on the bottom one, zeros aligned."""
    ax_o = ax.twiny()

    zero_frac = 1.0 / 6.0
    span_g, span_o = 0.156, 0.036
    lim_g = (-zero_frac * span_g, (1 - zero_frac) * span_g)
    lim_o = (-zero_frac * span_o, (1 - zero_frac) * span_o)

    out = {}
    for target, key, colour, lims, ticks in (
        (ax, "glorys12", C_GLORYS, lim_g, [0.0, 0.05, 0.10]),
        (ax_o, "oras5", C_ORAS5, lim_o, [0.0, 0.01, 0.02]),
    ):
        depth, trend, pval = clip_profile(*prof[key], zmax)
        target.plot(trend, depth, color=colour, lw=1.0, ls=(0, (2.4, 1.6)), zorder=3)
        for t_seg, z_seg in sig_segments(depth, trend, pval < ALPHA_SIG):
            target.plot(t_seg, z_seg, color=colour, lw=1.3, zorder=4)
        target.set_xlim(*lims)
        target.set_xticks(ticks)
        target.set_ylim(zmax, 0.0)
        target.tick_params(axis="x", colors=colour)
        for spine in target.spines.values():
            spine.set_visible(False)
        out[key] = {
            lay: layer_mean(prof[key][0], prof[key][1], *lay)
            for lay in CANON_LAYERS[key]
        }

    ax.axvline(0.0, color="0.35", lw=0.4, zorder=2)
    ax.set_ylabel("Depth (m)")
    ax.set_yticks([0, 300, 600, 900, 1200, 1500])
    ax.set_xlabel("GLORYS12 (PSU per decade)", color=C_GLORYS)
    ax_o.set_xlabel("ORAS5 (PSU per decade)", color=C_ORAS5)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(C_GLORYS)
    ax_o.spines["top"].set_visible(True)
    ax_o.spines["top"].set_color(C_ORAS5)

    for key, layers in out.items():
        for lay, val in layers.items():
            assert abs(val - CANON_LAYERS[key][lay]) < 1e-3, (key, lay, val)

    ax.legend(
        handles=[
            Line2D([], [], color="0.3", lw=1.3, label="p < 0.05"),
            Line2D([], [], color="0.3", lw=1.0, ls=(0, (2.4, 1.6)), label="p > 0.05"),
        ],
        loc="lower right",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.3,
        labelspacing=0.25,
    )
    return out


def vertical_contrast(ds: xr.Dataset, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Upper-limb minus deep-limb zonal-mean salinity at the section, per year."""
    depth = ds[f"depth_{key}"].values.astype(float)
    years = ds[f"year_{key}"].values.astype(float)
    s_bar = ds[f"s_bar_{key}"].values.astype(float)  # (year, depth)
    upper = np.array([layer_mean(depth, row, *UPPER_LIMB) for row in s_bar])
    deep = np.array([layer_mean(depth, row, *DEEP_LIMB) for row in s_bar])
    return years, upper - deep


def panel_c(ax, ds: xr.Dataset) -> dict:
    out = {}
    for key, colour in (("oras5", C_ORAS5), ("glorys12", C_GLORYS)):
        years, contrast = vertical_contrast(ds, key)
        ax.plot(years, contrast, color=colour, lw=0.8, zorder=3)
        fit = ols_santer(years, contrast)
        style = "-" if fit["p_santer"] < ALPHA_SIG else (0, (2.4, 1.6))
        ax.plot(
            years,
            fit["slope"] * years + (contrast.mean() - fit["slope"] * years.mean()),
            color=darken(colour),
            lw=1.4,
            ls=style,
            zorder=4,
        )
        out[key] = {
            "trend_per_decade": fit["slope"] * 10.0,
            "p_santer": fit["p_santer"],
            "mean": float(contrast.mean()),
            "n": int(len(years)),
        }
    ax.set_xlim(1992, 2026)
    ax.set_xticks([1995, 2005, 2015, 2025])
    ax.set_xlabel("Year")
    ax.set_ylabel("Vertical salinity contrast (PSU)")
    ax.spines[["top", "right"]].set_visible(False)
    return out


def main() -> None:
    set_style()

    pileup = json.loads(PILEUP_JSON.read_text())

    with xr.open_dataset(WP5_NC) as ds:
        prof = {
            key: (
                ds[f"depth_{key}"].values.astype(float),
                ds[f"trend_{key}"].values.astype(float),
                ds[f"p_santer_{key}"].values.astype(float),
            )
            for key in ("oras5", "glorys12")
        }
        ds_mem = ds.load()

    fig = plt.figure(figsize=(7.09, 2.65))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.35, 1.05, 1.20],
        wspace=0.44,
        left=0.062,
        right=0.988,
        bottom=0.155,
        top=0.845,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    stats_a = panel_a(ax_a, pileup)
    stats_b = panel_b(ax_b, prof)
    stats_c = panel_c(ax_c, ds_mem)

    for ax, letter, dx in ((ax_a, "a", -0.13), (ax_b, "b", -0.22), (ax_c, "c", -0.20)):
        ax.text(
            dx,
            1.10,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".pdf"))
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=300)
    plt.close(fig)

    print(f"wrote {OUT_BASE.with_suffix('.pdf')} and .png")
    for key, fit in stats_a.items():
        print(
            f"(a) {key}: {fit['slope'] * 10:+.4f} PSU/decade, "
            f"p={fit['p_santer']:.4f}, n={fit['n_years']}, N_eff={fit['n_eff']:.1f}"
            f"  | corr with F_ovS raw {pileup[key]['corr_with_fovs_raw']:+.3f}"
            f" detrended {pileup[key]['corr_with_fovs_detrended']:+.3f}"
            f" (p={pileup[key]['p_detrended']:.3f})"
        )
    for key, layers in stats_b.items():
        for lay, val in layers.items():
            print(f"(b) {key} {lay[0]}-{lay[1]} m: {val:+.4f} PSU/decade")
    for key, r in stats_c.items():
        print(
            f"(c) {key}: mean {r['mean']:+.4f} PSU, "
            f"trend {r['trend_per_decade']:+.4f} PSU/decade, "
            f"p={r['p_santer']:.4f}, n={r['n']}"
        )


if __name__ == "__main__":
    main()
