#!/usr/bin/env python3
"""Fig S6: main-window vs post-Argo-window mechanism decomposition.

Tests the Argo-observing-system-shock hypothesis directly. For each
reanalysis with a sufficiently long record we repeat the
velocity-vs-salinity decomposition on a fully-post-Argo window pair
(early = 2006-2012 or 2006-2011, late = 2018-2024 or 2012-2017) and
compare the mechanism shares against the main-window result
(early = 1993-2005, late = 2013-2025 or equivalent).

Products: ORAS5, GLORYS12V1, ECCO. SODA is added as a stub for
post-Argo that resolves once the decomposition is finalised.

Reads:  data/results/fovs_decomposition_{product}{,_postargo}.nc
Writes: figures/paper2/figS6_postargo.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ardp.viz.style import apply_nature_style, save_publication_figure


PRODUCTS = [
    ("ORAS5",      "oras5",    "#1f77b4"),
    ("GLORYS12V1", "glorys12", "#2ca02c"),
    ("SODA3.15.2", "soda",     "#e377c2"),
    ("ECCO-V4r4",  "ecco",     "#d62728"),
]


def _load(results_dir: Path, key: str, tag: str = "") -> dict | None:
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"fovs_decomposition_{key}{suffix}.nc"
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    out = {
        "delta_total": float(ds.attrs["delta_total_Sv"]),
        "delta_v": float(ds.attrs["delta_v_Sv"]),
        "delta_s": float(ds.attrs["delta_s_Sv"]),
        "delta_cross": float(ds.attrs["delta_cross_Sv"]),
        "early": ds.attrs["early_period"],
        "late": ds.attrs["late_period"],
        "V_net_early": float(ds.attrs.get("V_net_early_Sv", np.nan)),
        "V_net_late": float(ds.attrs.get("V_net_late_Sv", np.nan)),
    }
    ds.close()
    return out


def _shares(res: dict) -> tuple[float, float, float]:
    dt = res["delta_total"]
    if abs(dt) < 0.010:  # 10 mSv ill-defined threshold
        return np.nan, np.nan, np.nan
    return (100 * res["delta_v"] / dt,
            100 * res["delta_s"] / dt,
            100 * res["delta_cross"] / dt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/paper2/figS6_postargo"))
    args = parser.parse_args()

    apply_nature_style()

    rows = []
    for label, key, color in PRODUCTS:
        main = _load(args.results_dir, key)
        post = _load(args.results_dir, key, "postargo")
        rows.append((label, key, color, main, post))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax_main, ax_post = axes

    for ax, tag, title in (
        (ax_main, 0, "(a) Main window (1993–2005 vs 2013–2025*)"),
        (ax_post, 1, "(b) Post-Argo window (2006–2012 vs 2018–2024*)"),
    ):
        x = np.arange(len(rows))
        dv = np.array([r[3 + tag]["delta_v"] * 1000 if r[3 + tag] else np.nan
                       for r in rows])
        ds_ = np.array([r[3 + tag]["delta_s"] * 1000 if r[3 + tag] else np.nan
                        for r in rows])
        dc = np.array([r[3 + tag]["delta_cross"] * 1000 if r[3 + tag] else np.nan
                       for r in rows])
        dt = dv + ds_ + dc

        has_trend = np.abs(dt) >= 10.0
        has_data = np.array([r[3 + tag] is not None for r in rows])

        width = 0.55
        # Bars
        ax.bar(x[has_trend], dv[has_trend], width=width, color="#E69F00",
               label=r"$\Delta F_v$  (velocity)")
        ax.bar(x[has_trend], ds_[has_trend], width=width,
               bottom=dv[has_trend], color="#56B4E9",
               label=r"$\Delta F_s$  (salinity)")
        ax.bar(x[has_trend], dc[has_trend], width=width,
               bottom=dv[has_trend] + ds_[has_trend], color="0.6",
               label=r"$\Delta F_\mathrm{cross}$")
        no_trend_mask = has_data & ~has_trend
        if no_trend_mask.any():
            ax.bar(x[no_trend_mask], dt[no_trend_mask], width=width,
                   color="0.8", edgecolor="0.4", hatch="///",
                   label=r"$|\Delta F_\mathrm{total}| < 10$ mSv")
        no_data_mask = ~has_data
        if no_data_mask.any():
            ax.text(np.mean(x[no_data_mask]), 0.0, "pending",
                    ha="center", va="center", fontsize=7,
                    color="0.45", style="italic")

        ax.scatter(x[has_data], dt[has_data], color="black", s=25,
                   marker="D", zorder=5, label=r"$\Delta F_\mathrm{total}$")

        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right",
                           fontsize=6.5)
        ax.set_ylabel(r"$\Delta\mathrm{F}_{ovS}$ (mSv, late − early)" if tag == 0
                      else "")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-100, 25)
        ax.grid(axis="y", alpha=0.3, lw=0.3)

        # Share annotations inside plot area
        for i, r in enumerate(rows):
            result = r[3 + tag]
            if result is None:
                continue
            fv, fs, _ = _shares(result)
            if not np.isfinite(fv):
                ax.text(i, -2, "no\ntrend", ha="center", va="top",
                        fontsize=5.5, color="0.3", style="italic")
                continue
            y = 3 if dt[i] < 0 else -3
            va = "bottom" if dt[i] < 0 else "top"
            ax.text(i, y, f"v:{fv:+.0f}%\ns:{fs:+.0f}%",
                    ha="center", va=va, fontsize=5.5, color="0.3")

    # Single legend, outside
    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=6,
               frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_publication_figure(fig, args.output)

    # Print comparison table to stdout for the manuscript
    print("\nPost-Argo robustness table (shares in %):")
    print(f"{'Product':<12s} {'ΔF_main':>9s} {'fv_main':>9s} {'fs_main':>9s}"
          f" {'ΔF_post':>9s} {'fv_post':>9s} {'fs_post':>9s}")
    for r in rows:
        main, post = r[3], r[4]
        dfm = f"{main['delta_total'] * 1000:+.0f}" if main else "—"
        dfp = f"{post['delta_total'] * 1000:+.0f}" if post else "—"
        fvm, fsm, _ = _shares(main) if main else (np.nan, np.nan, np.nan)
        fvp, fsp, _ = _shares(post) if post else (np.nan, np.nan, np.nan)
        print(f"{r[0]:<12s} {dfm:>9s} {fvm:>+8.0f}% {fsm:>+8.0f}%"
              f" {dfp:>9s} {fvp:>+8.0f}% {fsp:>+8.0f}%")


if __name__ == "__main__":
    main()
