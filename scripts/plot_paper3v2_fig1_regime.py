#!/usr/bin/env python3
"""PAPER_3_v2 Figure 1: the bistable regime across four reanalyses.

(a) Annual-mean F_ovS at 34.5 S for ORAS5, GLORYS12, ECCO-V4r4 and SODA 3.15.2.
(b) Record-mean F_ovS with 95 % circular block-bootstrap confidence intervals
    (5-year blocks, 10,000 iterations). Solid interval = excludes zero.
(c) F_ovS trends over each product's own record and over the common window
    1993-2017. Filled marker = Santer N_eff p < 0.05, open marker = not
    significant. Nothing is significant in the common window.

All values are recomputed here from the canonical files in data/results/ and
asserted against PAPER_3_v2/CANONICAL_FACTS.md sections 1 and 2.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from analysis_paper3_r2_gate_checks import ols_santer  # noqa: E402

RESULTS_DIR = REPO / "data" / "results"
FIG_DIR = REPO / "PAPER_3_v2" / "figures"
OUT_BASE = FIG_DIR / "Fig1_regime"

SEED = 20260804
N_BOOT = 10_000
BLOCK_YEARS = 5
COMMON_WINDOW = (1993, 2017)

PRODUCTS = ["oras5", "glorys12", "ecco", "soda"]
LABELS = {
    "oras5": "ORAS5",
    "glorys12": "GLORYS12",
    "ecco": "ECCO",
    "soda": "SODA",
}
COLORS = {
    "oras5": "#20558a",
    "glorys12": "#b3541e",
    "ecco": "#3f7f5f",
    "soda": "#8a6bbf",
}

# CANONICAL_FACTS.md section 1: mean F_ovS (Sv) and 95 % block-bootstrap CI.
CANON_MEAN = {
    "oras5": (-0.0239, -0.0392, -0.0086),
    "glorys12": (-0.0709, -0.1022, -0.0365),
    "ecco": (-0.1161, -0.1193, -0.1123),
    "soda": (-0.0091, -0.0188, +0.0010),
}
# CANONICAL_FACTS.md section 2: trends (mSv/yr) over the own record and over
# the common window 1993-2017, with the significance verdict.
CANON_TREND_OWN = {
    "oras5": (-1.41, True),
    "glorys12": (-4.39, True),
    "ecco": (+0.09, False),
    "soda": (+0.59, False),
}
CANON_TREND_COMMON = {
    "oras5": (-0.91, False),
    "glorys12": (-4.78, False),
    "ecco": (+0.10, False),
    "soda": (+0.66, False),
}


def load_annual(product: str) -> tuple[np.ndarray, np.ndarray]:
    """Annual-mean F_ovS for a product.

    ORAS5 and GLORYS12 are stored monthly on a `time` dimension and are
    averaged by calendar year; ECCO and SODA are already annual on a `year`
    coordinate.
    """
    with xr.open_dataset(RESULTS_DIR / f"{product}_f_ovs.nc") as ds:
        if "time" in ds.dims:
            annual = ds["F_ovS"].groupby("time.year").mean()
            years = annual["year"].values.astype(float)
            vals = annual.values.astype(float)
        else:
            years = ds["year"].values.astype(float)
            vals = ds["F_ovS"].values.astype(float)
    good = np.isfinite(vals)
    return years[good], vals[good]


def block_bootstrap_mean_ci(
    vals: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """95 % CI of the record mean from a circular moving-block bootstrap.

    Blocks are BLOCK_YEARS long, wrapped around the end of the series, and the
    resampled series is truncated back to the original length. The generator is
    passed in and advanced in a fixed product order so the intervals are
    reproducible.
    """
    n = len(vals)
    n_blocks = int(np.ceil(n / BLOCK_YEARS))
    offsets = np.arange(BLOCK_YEARS)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]) % n
        means[i] = vals[idx].ravel()[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute() -> dict:
    """Recompute every number in the figure from the canonical files."""
    series = {p: load_annual(p) for p in PRODUCTS}
    rng = np.random.default_rng(SEED)

    out: dict = {}
    for p in PRODUCTS:
        years, vals = series[p]
        lo, hi = block_bootstrap_mean_ci(vals, rng)
        own = ols_santer(years, vals)
        mask = (years >= COMMON_WINDOW[0]) & (years <= COMMON_WINDOW[1])
        common = ols_santer(years[mask], vals[mask])
        out[p] = {
            "years": years,
            "values": vals,
            "record": (int(years[0]), int(years[-1])),
            "n_years": int(len(years)),
            "mean": float(vals.mean()),
            "ci": (lo, hi),
            "ci_excludes_zero": bool(hi < 0.0 or lo > 0.0),
            "trend_own_mSv": own["slope"] * 1e3,
            "p_own": own["p_santer"],
            "sig_own": bool(own["p_santer"] < 0.05),
            "trend_common_mSv": common["slope"] * 1e3,
            "p_common": common["p_santer"],
            "sig_common": bool(common["p_santer"] < 0.05),
            "n_common": int(mask.sum()),
        }
    return out


def check(res: dict) -> None:
    """Assert the recomputed values reproduce CANONICAL_FACTS.md."""
    for p in PRODUCTS:
        r = res[p]
        mean_c, lo_c, hi_c = CANON_MEAN[p]
        assert abs(r["mean"] - mean_c) < 5e-4, (p, r["mean"], mean_c)
        assert abs(r["ci"][0] - lo_c) < 5e-4, (p, r["ci"][0], lo_c)
        assert abs(r["ci"][1] - hi_c) < 5e-4, (p, r["ci"][1], hi_c)

        trend_c, sig_c = CANON_TREND_OWN[p]
        assert abs(r["trend_own_mSv"] - trend_c) < 1e-2, (p, r["trend_own_mSv"])
        assert r["sig_own"] is sig_c, (p, r["p_own"])

        trend_c, sig_c = CANON_TREND_COMMON[p]
        assert abs(r["trend_common_mSv"] - trend_c) < 1e-2, (p, r["trend_common_mSv"])
        assert r["sig_common"] is sig_c, (p, r["p_common"])

    assert sum(res[p]["ci_excludes_zero"] for p in PRODUCTS) == 3
    assert not res["soda"]["ci_excludes_zero"]
    assert not any(res[p]["sig_common"] for p in PRODUCTS)


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
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "lines.linewidth": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.16) -> None:
    ax.text(
        x,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def make_figure(res: dict) -> None:
    set_style()
    fig = plt.figure(figsize=(7.09, 2.35))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.55, 1.0, 1.0],
        left=0.075,
        right=0.995,
        bottom=0.215,
        top=0.90,
        wspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # ---------------------------------------------------------------- panel a
    ax_a.axhline(0.0, color="0.45", lw=0.5, zorder=1)
    for i, p in enumerate(PRODUCTS):
        r = res[p]
        # ORAS5, the longest record, is drawn on top of the shorter products.
        z = 3 + (len(PRODUCTS) - i)
        ax_a.plot(
            r["years"],
            r["values"],
            color=COLORS[p],
            lw=0.7,
            label=LABELS[p],
            zorder=z,
        )
        ax_a.plot(
            [r["years"][0], r["years"][-1]],
            [r["mean"], r["mean"]],
            color=COLORS[p],
            lw=0.7,
            alpha=0.50,
            zorder=2,
        )
    ax_a.set_xlim(1956, 2027)
    ax_a.set_ylim(-0.16, 0.075)
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel(r"$F_{ovS}$ (Sv)")
    ax_a.set_xticks(np.arange(1960, 2030, 10))
    ax_a.legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    for side in ("top", "right"):
        ax_a.spines[side].set_visible(False)
    panel_letter(ax_a, "a", x=-0.10)

    # ---------------------------------------------------------------- panel b
    ax_b.axvline(0.0, color="0.45", lw=0.5, zorder=1)
    ypos = {p: len(PRODUCTS) - 1 - i for i, p in enumerate(PRODUCTS)}
    for p in PRODUCTS:
        r = res[p]
        y = ypos[p]
        line_kw = {"ls": "-"} if r["ci_excludes_zero"] else {"ls": (0, (2.2, 1.3))}
        ax_b.plot(
            list(r["ci"]),
            [y, y],
            color=COLORS[p],
            lw=0.9,
            solid_capstyle="butt",
            zorder=3,
            **line_kw,
        )
        for edge in r["ci"]:
            ax_b.plot([edge, edge], [y - 0.14, y + 0.14], color=COLORS[p], lw=0.9)
        ax_b.plot(
            r["mean"],
            y,
            marker="o",
            ms=3.4,
            mfc=COLORS[p],
            mec=COLORS[p],
            ls="none",
            zorder=4,
        )
    ax_b.set_yticks([ypos[p] for p in PRODUCTS])
    ax_b.set_yticklabels([LABELS[p] for p in PRODUCTS])
    ax_b.set_ylim(-0.6, len(PRODUCTS) - 0.4)
    ax_b.set_xlim(-0.135, 0.022)
    ax_b.set_xticks([-0.12, -0.08, -0.04, 0.0])
    ax_b.set_xlabel(r"mean $F_{ovS}$ (Sv)")
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)
    panel_letter(ax_b, "b")

    # ---------------------------------------------------------------- panel c
    ax_c.axhline(0.0, color="0.45", lw=0.5, zorder=1)
    dx = 0.19
    for i, p in enumerate(PRODUCTS):
        r = res[p]
        ax_c.plot(
            [i - dx, i + dx],
            [r["trend_own_mSv"], r["trend_common_mSv"]],
            color=COLORS[p],
            lw=0.6,
            alpha=0.55,
            zorder=2,
        )
        for x, key, marker in (
            (i - dx, "own", "o"),
            (i + dx, "common", "D"),
        ):
            sig = r[f"sig_{key}"]
            ax_c.plot(
                x,
                r[f"trend_{key}_mSv"],
                marker=marker,
                ms=3.8 if sig else 3.2,
                mfc=COLORS[p] if sig else "white",
                mec=COLORS[p],
                mew=0.8,
                ls="none",
                zorder=3,
            )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ms=3.2,
            mfc="white",
            mec="0.35",
            mew=0.8,
            ls="none",
            label="own record",
        ),
        plt.Line2D(
            [],
            [],
            marker="D",
            ms=3.2,
            mfc="white",
            mec="0.35",
            mew=0.8,
            ls="none",
            label="1993 to 2017",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ms=3.6,
            mfc="0.35",
            mec="0.35",
            ls="none",
            label="p < 0.05",
        ),
    ]
    ax_c.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.35,
        borderaxespad=0.2,
    )
    ax_c.set_xticks(range(len(PRODUCTS)))
    ax_c.set_xticklabels(
        [LABELS[p] for p in PRODUCTS],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax_c.set_xlim(-0.55, len(PRODUCTS) - 0.45)
    ax_c.set_ylim(-5.6, 1.35)
    ax_c.set_yticks([-5, -4, -3, -2, -1, 0, 1])
    ax_c.set_ylabel(r"$F_{ovS}$ trend (mSv yr$^{-1}$)")
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    panel_letter(ax_c, "c")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_BASE}.pdf")
    fig.savefig(f"{OUT_BASE}.png", dpi=300)
    plt.close(fig)


def report(res: dict) -> None:
    print("Figure 1 plotted values")
    print("panel a/b: annual-mean F_ovS, record mean and 95 % bootstrap CI")
    for p in PRODUCTS:
        r = res[p]
        print(
            f"  {LABELS[p]:9s} {r['record'][0]}-{r['record'][1]} "
            f"n={r['n_years']:2d}  mean {r['mean']:+.4f} Sv  "
            f"CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  "
            f"excludes zero: {r['ci_excludes_zero']}"
        )
    print("panel c: trends (mSv/yr) with Santer N_eff p")
    for p in PRODUCTS:
        r = res[p]
        print(
            f"  {LABELS[p]:9s} own {r['record'][0]}-{r['record'][1]} "
            f"{r['trend_own_mSv']:+.3f} (p={r['p_own']:.4g}, "
            f"sig={r['sig_own']}) | 1993-2017 n={r['n_common']} "
            f"{r['trend_common_mSv']:+.3f} (p={r['p_common']:.4g}, "
            f"sig={r['sig_common']})"
        )
    print(f"wrote {OUT_BASE}.pdf and {OUT_BASE}.png")


def main() -> None:
    res = compute()
    check(res)
    make_figure(res)
    report(res)


if __name__ == "__main__":
    main()
