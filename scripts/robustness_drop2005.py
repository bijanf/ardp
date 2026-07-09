#!/usr/bin/env python3
"""Reviewer R3.8 robustness check: drop 2005 (sparse Argo year) and recompute.

Reuses the production analysis code unchanged by importing the two compute
modules and overriding their period constants:
  - compute_argo_sa_trend:      YEAR_START 2005 -> 2006  (trend 2006-2024)
  - compute_argo_zonal_section: EARLY (2005,2009) -> (2006,2009)

All outputs go to revision/ — the production files in data/results/ are
NOT touched:
  revision/results/argo_trends_drop2005.json
  revision/results/argo_basin_mean_drop2005.csv
  revision/results/argo_zonal_section_drop2005.nc
  revision/figures/Fig2_drop2005.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import compute_argo_sa_trend as sat  # noqa: E402
import compute_argo_zonal_section as zsec  # noqa: E402

REV_RESULTS = REPO / "revision" / "results"
REV_FIGURES = REPO / "revision" / "figures"

# Nature-style rc settings (mirrors plot_obs_grounding.py)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CM = plt.cm.RdBu_r
LEVELS = np.arange(-0.30, 0.31, 0.05)
NORM = BoundaryNorm(LEVELS, ncolors=CM.N, clip=False)

COLR = {
    "EN4.2.2": "#17becf",
    "RG09": "#7f7f7f",
    "ORAS5": "#1f77b4",
    "GLORYS12V1": "#2ca02c",
    "SAMBA-Volkov2024": "#9467bd",
}
LIT_TRENDS = {
    "GLORYS12V1": (0.084, 0.025),
    "ORAS5": (-0.014, 0.020),
}


# -------------------------------------------------------------------------
# Task A — basin-mean trends 2006-2024
# -------------------------------------------------------------------------
def compute_trends_drop2005() -> dict:
    sat.YEAR_START = 2006  # drop 2005 (sparse Argo coverage)

    print("Loading EN4.2.2 annual basin-mean salinity (2006-2024) ...")
    df_en4 = sat._load_en4_basin_annual()
    print(f"  EN4 years: {df_en4['year'].tolist() if not df_en4.empty else '(none)'}")

    print("\nLoading RG09 annual basin-mean salinity (2006-2024) ...")
    df_rg09 = sat._load_rg09_basin_annual()
    rg09_years = df_rg09["year"].tolist() if not df_rg09.empty else "(none)"
    print(f"  RG09 years: {rg09_years}")

    summary: dict[str, dict] = {}
    for label, df in [("EN4.2.2", df_en4), ("RG09", df_rg09)]:
        if df.empty:
            print(f"  {label}: no data — skipping")
            continue
        yrs = df["year"].values
        vals = df["salinity_psu"].values
        trend = sat._santer_neff_ci(yrs, vals)
        b_lo, b_hi = sat._bootstrap_trend(yrs, vals)
        summary[label] = {
            "n_years": int(len(yrs)),
            "year_range": [int(yrs.min()), int(yrs.max())],
            "slope_psu_per_dec": float(trend["slope_per_year"]) * 10.0,
            "ci95_half_psu_per_dec": float(trend["ci95_half"]) * 10.0,
            "bootstrap_ci95_psu_per_dec": [b_lo * 10.0, b_hi * 10.0],
            "n_eff": trend["n_eff"],
            "r1": trend["r1"],
            "p_raw": trend["p_raw"],
        }
        print(f"\n{label} (2006-2024): trend = "
              f"{summary[label]['slope_psu_per_dec']:+.4f} ± "
              f"{summary[label]['ci95_half_psu_per_dec']:.4f} PSU/decade (Santer)")
        print(f"  bootstrap 95% CI: [{b_lo*10:+.4f}, {b_hi*10:+.4f}] PSU/decade")

    summary["SAMBA-Volkov2024"] = {
        "slope_psu_per_dec": 0.050,
        "ci95_half_psu_per_dec": 0.020,
        "source": "Volkov et al. 2024, SAMBA repeat hydrography 2009-2023",
    }
    summary["_note"] = {
        "description": ("Drop-2005 robustness check (reviewer R3.8): "
                        "trends over 2006-2024"),
    }

    out_json = REV_RESULTS / "argo_trends_drop2005.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    combined = pd.DataFrame()
    if not df_en4.empty:
        df_en4["product"] = "EN4.2.2"
        combined = pd.concat([combined, df_en4])
    if not df_rg09.empty:
        df_rg09["product"] = "RG09"
        combined = pd.concat([combined, df_rg09])
    if not combined.empty:
        out_csv = REV_RESULTS / "argo_basin_mean_drop2005.csv"
        combined.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")
    return summary


# -------------------------------------------------------------------------
# Task B — zonal ΔS section with EARLY = 2006-2009
# -------------------------------------------------------------------------
def compute_section_drop2005() -> Path:
    zsec.EARLY = (2006, 2009)  # drop 2005 from the early epoch

    print("\nComputing EN4 34.5°S zonal ΔS section (early 2006-2009) ...")
    ds = zsec._en4_zonal_delta()
    ds.attrs["note"] = "Drop-2005 robustness check (reviewer R3.8)"
    out_path = REV_RESULTS / "argo_zonal_section_drop2005.nc"
    ds.to_netcdf(out_path)
    print(f"Wrote {out_path}")
    return out_path


# -------------------------------------------------------------------------
# Task C — drop-2005 version of the observational-grounding figure
# -------------------------------------------------------------------------
def plot_figure_drop2005() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.0),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    ax_b, ax_c = axes

    # Panel a — ΔS zonal section (drop-2005 early epoch)
    ds = xr.open_dataset(REV_RESULTS / "argo_zonal_section_drop2005.nc")
    d_s = ds["EN4_dS"]
    pvals = ds["EN4_pvalue"]
    pc = ax_b.pcolormesh(d_s["lon"], d_s["depth"], d_s.values,
                         cmap=CM, norm=NORM, shading="nearest")
    sig_mask = pvals.values < 0.05
    lon_grid, depth_grid = np.meshgrid(d_s["lon"].values, d_s["depth"].values)
    stride = max(1, lon_grid.shape[1] // 30)
    stride_d = max(1, lon_grid.shape[0] // 8)
    mask_thin = np.zeros_like(sig_mask)
    mask_thin[::stride_d, ::stride] = sig_mask[::stride_d, ::stride]
    ax_b.scatter(lon_grid[mask_thin], depth_grid[mask_thin],
                 s=2, color="black", alpha=0.4, marker="o", linewidths=0)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Longitude (°E)")
    ax_b.set_ylabel("Depth (m)")
    ax_b.set_xlim(d_s["lon"].min(), d_s["lon"].max())
    ax_b.set_ylim(1000, 0)
    cb = fig.colorbar(pc, ax=ax_b, location="right",
                      shrink=0.85, pad=0.04,
                      ticks=np.arange(-0.30, 0.31, 0.10))
    cb.set_label("ΔS (PSU)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ds.close()

    # Panel b — trend bar chart (2006-2024 Argo trends)
    with open(REV_RESULTS / "argo_trends_drop2005.json") as f:
        trends = json.load(f)

    rows: list[tuple[str, float, float]] = []
    if "EN4.2.2" in trends:
        t = trends["EN4.2.2"]
        rows.append(("EN4.2.2", t["slope_psu_per_dec"], t["ci95_half_psu_per_dec"]))
    rg09_slope = trends.get("RG09", {}).get("slope_psu_per_dec", np.nan)
    if np.isfinite(rg09_slope):
        t = trends["RG09"]
        rows.append(("RG09", t["slope_psu_per_dec"], t["ci95_half_psu_per_dec"]))
    rows.append(("GLORYS12V1", *LIT_TRENDS["GLORYS12V1"]))
    rows.append(("ORAS5", *LIT_TRENDS["ORAS5"]))

    samba_value = trends["SAMBA-Volkov2024"]["slope_psu_per_dec"]
    samba_ci = trends["SAMBA-Volkov2024"]["ci95_half_psu_per_dec"]

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = [COLR.get(n, "0.5") for n in names]

    y = np.arange(len(names))
    ax_c.barh(y, vals, xerr=errs, color=colors, edgecolor="0.2",
              linewidth=0.4, error_kw={"linewidth": 0.7, "capsize": 2})
    ax_c.axvline(0, color="0.3", linewidth=0.4)
    ax_c.axvspan(samba_value - samba_ci, samba_value + samba_ci,
                 color=COLR["SAMBA-Volkov2024"], alpha=0.12)
    ax_c.axvline(samba_value, color=COLR["SAMBA-Volkov2024"], linewidth=0.7,
                 linestyle="--")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(names, fontsize=6)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Salinity trend (PSU / decade)")
    ax_c.set_xlim(-0.07, 0.13)
    ax_c.spines[["top", "right"]].set_visible(False)

    for ax, lbl in zip([ax_b, ax_c], "ab", strict=False):
        ax.text(-0.18, 1.10, lbl, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left")

    pdf = REV_FIGURES / "Fig2_drop2005.pdf"
    png = REV_FIGURES / "Fig2_drop2005.png"
    fig.savefig(pdf, format="pdf", dpi=300)
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def main() -> int:
    REV_RESULTS.mkdir(parents=True, exist_ok=True)
    REV_FIGURES.mkdir(parents=True, exist_ok=True)
    compute_trends_drop2005()
    compute_section_drop2005()
    plot_figure_drop2005()
    return 0


if __name__ == "__main__":
    sys.exit(main())
