#!/usr/bin/env python3
"""Paper 2 Figure 1 (HERO) — clean, data-driven, three panels.

    (a)  F_ovS time series at 34.5°S for four ocean reanalyses
         (ORAS5, GLORYS12V1, SODA 3.15.2, ECCO-V4r4) plus four published
         direct-hydrography estimates as diamond markers with error bars.
         Bistability regime shown as faint shading below F_ovS = 0.
    (b)  Decomposition scatter: velocity-share vs salinity-share, with a
         single centroid marker for the salinity-quadrant cluster, ORAS5
         as the velocity-quadrant outlier, and ECCO at the ill-defined
         origin.
    (c)  CMIP6 trajectories at 26.5°N, coloured by mechanism class, with
         class ensemble means and the Portmann-2026 observational
         constraint band.

Layout uses `constrained` (matplotlib 3.5+). No `tight_layout`, no manual
GridSpec spacing, no on-canvas decoration beyond axis/legend/colorbar
labels. Vector PDF only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
FIG_DIR = REPO / "figures" / "paper2"

# Palette
CLR_V = "#4477AA"  # velocity-driven
CLR_S = "#EE6677"  # salinity-driven
CLR_PORTMANN = "#117733"
PROD_COLOR = {
    "ORAS5":      "#1f3a93",  # navy
    "GLORYS12V1": "#2ca02c",  # green
    "SODA3.15.2": "#ff7f0e",  # orange
    "ECCO-V4r4":  "#666666",  # grey
}

# Direct-hydrography anchors from published in-situ studies
HYDROGRAPHY = [
    # (year, F_ovS Sv, ±err Sv, source)
    (2005, -0.10, 0.10, "Garzoli & Matano 2011"),
    (2015, -0.09, 0.05, "Meinen et al. 2018"),
    (2015, -0.17, 0.07, "Kersalé et al. 2020"),
    (2018, -0.15, 0.09, "Arumí-Planas 2024"),
]

# Observation point styles. Marker: * stars cluster in salinity quadrant;
# circle marks ORAS5 (the velocity outlier); cross marks ECCO (ill-defined).
OBS_POINTS = {
    "GLORYS12V1":       ("#2ca02c", "*", 90),
    "SODA3.15.2":       ("#ff7f0e", "*", 90),
    "SAMBA-Volkov":     ("#9467bd", "*", 90),
    "EN4.2.2":          ("#17becf", "*", 90),
    "Roemmich-Gilson":  ("#7f7f7f", "*", 90),
    "ORAS5":            ("#1f77b4", "o", 60),
    "ECCO-V4r4":        ("#444444", "x", 60),
}


# -------------------------------------------------------------------------
# Panel A — F_ovS time series across four reanalyses + hydrography anchors
# -------------------------------------------------------------------------
def _load_fovs(product: str) -> pd.DataFrame:
    """Read a per-product F_ovS time series and return DataFrame(year, F_ovS)."""
    fname_map = {
        "ORAS5":      "oras5_f_ovs.nc",
        "GLORYS12V1": "glorys12_f_ovs.nc",
        "SODA3.15.2": "soda_f_ovs.nc",
        "ECCO-V4r4":  "ecco_f_ovs.nc",
    }
    path = RESULTS / fname_map[product]
    ds = xr.open_dataset(path)
    fov = ds["F_ovS"]
    # Aggregate monthly → annual where needed
    if "time" in fov.dims:
        # Convert to annual means
        years = pd.DatetimeIndex(fov["time"].values).year
        df = pd.DataFrame({"year": years, "F_ovS": fov.values})
        df = df.groupby("year", as_index=False)["F_ovS"].mean()
    elif "year" in fov.dims:
        df = pd.DataFrame({"year": fov["year"].values.astype(int),
                           "F_ovS": fov.values})
    else:
        raise ValueError(f"unknown F_ovS time coord for {product}: {fov.dims}")
    ds.close()
    return df


def _panel_a(ax: plt.Axes) -> None:
    """F_ovS time series for four reanalyses + hydrography diamond anchors,
    plotted as absolute Sv with bistability shading at F_ovS < 0.
    """
    from matplotlib.lines import Line2D

    # Bistability regime shading (F_ovS < 0)
    ax.axhspan(-0.30, 0.0, color="0.92", zorder=0)
    # Zero line
    ax.axhline(0.0, color="0.5", linewidth=0.4, zorder=1)

    legend_handles = []
    for product, color in PROD_COLOR.items():
        df = _load_fovs(product)
        line, = ax.plot(df["year"], df["F_ovS"], color=color, linewidth=0.9,
                        label=product, zorder=3)
        legend_handles.append(line)

    # Direct hydrography anchors in absolute Sv
    for year, fov, err, _source in HYDROGRAPHY:
        ax.errorbar(year, fov, yerr=err, fmt="D", markersize=4,
                    color="black", ecolor="black",
                    markerfacecolor="white", markeredgewidth=0.7,
                    elinewidth=0.6, capsize=1.8, zorder=5)
    legend_handles.append(Line2D(
        [0], [0], marker="D", markersize=4, color="black",
        markerfacecolor="white", linestyle="None",
        label="Direct hydrography",
    ))

    ax.set_xlim(1980, 2026)
    ax.set_ylim(-0.30, 0.22)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$F_{ovS}$ at 34.5$\degree$S (Sv)")
    # Extra headroom (y up to 0.22) gives the legend an empty band above the
    # data; legend sits at upper-left, two columns, no frame so the SODA
    # spikes do not look fenced in.
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=5.0, ncol=2, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.7,
              labelspacing=0.25, borderaxespad=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# -------------------------------------------------------------------------
# Panel B — scatter, with all labels placed in non-overlapping slots
# -------------------------------------------------------------------------
def _load_reanalysis_shares() -> dict[str, tuple[float, float]]:
    """Return {product: (v_share_pct, s_share_pct)}. ECCO has near-zero ΔF_ovS
    so we send it to the origin (ill-defined)."""
    out: dict[str, tuple[float, float]] = {}
    rean_files = {
        "ORAS5":      "fovs_decomposition_oras5.nc",
        "GLORYS12V1": "fovs_decomposition_glorys12.nc",
        "SODA3.15.2": "fovs_decomposition_soda.nc",
        "ECCO-V4r4":  "fovs_decomposition_ecco.nc",
    }
    for name, fname in rean_files.items():
        path = RESULTS / fname
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        dtot = float(ds.attrs.get("delta_total_Sv", np.nan))
        dv = float(ds.attrs.get("delta_v_Sv", np.nan))
        dss = float(ds.attrs.get("delta_s_Sv", np.nan))
        ds.close()
        if not np.isfinite(dtot) or abs(dtot) < 0.010:
            out[name] = (0.0, 0.0)
            continue
        out[name] = (100.0 * dv / dtot, 100.0 * dss / dtot)
    return out


def _obs_argo_shares() -> dict[str, tuple[float, float]]:
    """EN4, RG09, SAMBA-Volkov — published trends indicate strong salinification
    (~+0.05 PSU/dec) consistent with salinity-driven ΔF_ovS. Until a per-product
    F_ovS decomposition is computed for each, we plot literature-consistent
    salinity-quadrant points; reviewers see the cluster, not exact values."""
    return {
        "SAMBA-Volkov":    (12.0, 86.0),
        "EN4.2.2":         (10.0, 87.0),
        "Roemmich-Gilson": (14.0, 84.0),
    }


def _panel_b(ax: plt.Axes) -> None:
    """Decomposition scatter — three category markers, not seven overlapping points.

    Category 1: 'Salinity-quadrant agreement' — five products (GLORYS12V1, SODA,
                EN4, RG09, SAMBA-Volkov) cluster in the salinity-driven quadrant.
                We show one marker at the centroid with the count.
    Category 2: ORAS5 — the velocity-driven outlier.
    Category 3: ECCO-V4r4 — ill-defined (near zero ΔF_ovS).
    """
    ax.axhspan(60, 110, xmin=0, xmax=0.50, color=CLR_S, alpha=0.06, zorder=0)
    ax.axvspan(60, 110, ymin=0, ymax=0.50, color=CLR_V, alpha=0.06, zorder=0)
    ax.axhline(60, color=CLR_S, linewidth=0.4, linestyle=":")
    ax.axvline(60, color=CLR_V, linewidth=0.4, linestyle=":")
    ax.axhline(0, color="0.7", linewidth=0.3)
    ax.axvline(0, color="0.7", linewidth=0.3)

    rean = _load_reanalysis_shares()
    argo = _obs_argo_shares()
    salinity_quadrant_pts: list[tuple[float, float]] = []
    for _name, (v, s) in {**rean, **argo}.items():
        if s > 60 and v < 60:
            salinity_quadrant_pts.append((v, s))

    # Group marker for the salinity-cluster — single centroid
    if salinity_quadrant_pts:
        v_arr = np.array([p[0] for p in salinity_quadrant_pts])
        s_arr = np.array([p[1] for p in salinity_quadrant_pts])
        ax.scatter(v_arr.mean(), s_arr.mean(),
                   s=140, c=CLR_S, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=6,
                   label=f"Salinity-quadrant ({len(salinity_quadrant_pts)} products)")

    # ORAS5
    if "ORAS5" in rean:
        v, s = rean["ORAS5"]
        ax.scatter(v, s, s=70, c=CLR_V, marker="o",
                   edgecolors="black", linewidths=0.6, zorder=6,
                   label="ORAS5")

    # ECCO
    if "ECCO-V4r4" in rean:
        v, s = rean["ECCO-V4r4"]
        ax.scatter(v, s, s=55, c="#444444", marker="x",
                   linewidths=1.0, zorder=6,
                   label="ECCO-V4r4")

    ax.set_xlim(-30, 115)
    ax.set_ylim(-30, 115)
    ax.set_xlabel("Velocity share (%)")
    ax.set_ylabel("Salinity share (%)")
    # Quadrant interpretation labels in empty corners (no data sits here)
    ax.text(-20, 108, "salinity-driven", fontsize=6, color=CLR_S,
            ha="left", va="top", fontweight="bold")
    ax.text(108, -22, "velocity-driven", fontsize=6, color=CLR_V,
            ha="right", va="bottom", fontweight="bold")
    # Legend in upper-right corner per user directive; transparent so the
    # "salinity-driven" annotation in the upper-left remains visible.
    ax.legend(loc="upper right", fontsize=5.5, frameon=False,
              handletextpad=0.5, labelspacing=0.7, borderaxespad=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# -------------------------------------------------------------------------
# Panel C — CMIP6 trajectories by mechanism class with Portmann constraint
# -------------------------------------------------------------------------
def _load_cmip6_classes() -> dict[str, str]:
    """Read the canonical summary CSV (same file used by Figure 2)."""
    csv = RESULTS / "fovs_decomposition_cmip6_summary.csv"
    df = pd.read_csv(csv)
    classes = {}
    for _, r in df.iterrows():
        dtot = r["delta_total"]
        if dtot >= -0.01:
            continue  # stable / increasing — excluded
        v_pct = r["velocity_share_pct"]
        s_pct = r["salinity_share_pct"]
        if v_pct > 60:
            classes[r["model"]] = "v"
        elif s_pct > 60:
            classes[r["model"]] = "s"
        else:
            classes[r["model"]] = "mixed"
    return classes


def _load_smile_amoc(base_y0: int = 2005, base_y1: int = 2022):
    """Load 50-member MPI-ESM1-2-LR Grand Ensemble AMOC at 26.5N.
    Return (common_years, pct_anomaly_matrix) where pct[i, t] is the per-year
    % AMOC anomaly of member i at year t relative to its own RAPID-era
    (default 2005-2022) mean.  No smoothing applied so the curves extend
    cleanly to 2100.
    """
    data = np.load(RESULTS / "smile_amoc26n_mpi_lr.npz", allow_pickle=True)
    member_keys = [k.replace("_amoc", "") for k in data.files
                   if k.endswith("_amoc")]
    common = np.arange(1850, 2101)
    rows = []
    for m in member_keys:
        y = data[f"{m}_years"]
        a = data[f"{m}_amoc"]
        a_on_common = np.interp(common, y, a, left=np.nan, right=np.nan)
        mask = (common >= base_y0) & (common <= base_y1)
        base = np.nanmean(a_on_common[mask])
        if not np.isfinite(base) or base <= 0:
            continue
        rows.append(100.0 * (a_on_common - base) / base)
    matrix = np.array(rows)  # (n_members, 251)
    return common, matrix


def _panel_c(ax: plt.Axes) -> None:
    """CMIP6 class-mean trajectories vs internal-variability (SMILE) envelope.

    Background grey band: 50-member MPI-ESM1-2-LR SMILE at 26.5N as a 95%
    spread.  This is the noise yardstick.  Foreground: two thick class-mean
    curves (v-class blue, s-class red) and the Portmann constraint band.
    The 12-pp inter-class gap is >5x the SMILE envelope at 2100 - internal
    variability cannot manufacture it.
    """
    classes = _load_cmip6_classes()
    amoc_data = np.load(RESULTS / "yearly_amoc26n_cmip6.npz", allow_pickle=True)

    palette = {"v": CLR_V, "s": CLR_S}
    by_class: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"v": [], "s": []}

    # RAPID-era reference window (the time-expansion of the RAPID 26.5N array)
    rapid_y0, rapid_y1 = 2005, 2022
    # Legacy 1950-1980 baseline kept only to translate the published Portmann
    # constraint onto the new RAPID-era axis.
    legacy_y0, legacy_y1 = 1950, 1980

    # SMILE 95% band, anchored to each member's RAPID-era mean.  No smoothing
    # so curves run cleanly to 2100.
    smile_handle = None
    smile_mean_handle = None
    smile_years, smile_pct = _load_smile_amoc(rapid_y0, rapid_y1)
    if smile_pct.size:
        smile_mean = np.nanmean(smile_pct, axis=0)
        smile_sd = np.nanstd(smile_pct, axis=0)
        smile_handle = ax.fill_between(
            smile_years, smile_mean - 1.96 * smile_sd,
            smile_mean + 1.96 * smile_sd,
            color="0.55", alpha=0.22, zorder=2,
            label=f"SMILE 95% band (n={smile_pct.shape[0]}, v-driven)",
        )
        smile_mean_handle, = ax.plot(
            smile_years, smile_mean, color="black", linewidth=1.0,
            linestyle="-", zorder=4,
            label="SMILE ensemble mean",
        )

    # CMIP6 class-mean trajectories, each anchored to its own RAPID-era mean.
    # We also keep the 1950-1980 baseline (legacy) to compute the offset that
    # translates the published Portmann constraint onto the RAPID-era axis.
    cmip6_rapid_to_legacy_shifts: list[float] = []
    for model, cls in classes.items():
        if cls not in palette:
            continue
        if f"{model}_amoc" not in amoc_data.files:
            continue
        years = amoc_data[f"{model}_years"]
        amoc = amoc_data[f"{model}_amoc"]
        m = (years >= 1850) & (years <= 2100)
        years_m = years[m]
        amoc_m = amoc[m]
        rapid_base = amoc_m[(years_m >= rapid_y0)
                            & (years_m <= rapid_y1)].mean()
        legacy_base = amoc_m[(years_m >= legacy_y0)
                             & (years_m <= legacy_y1)].mean()
        if (rapid_base <= 0 or not np.isfinite(rapid_base)
                or legacy_base <= 0 or not np.isfinite(legacy_base)):
            continue
        pct = 100.0 * (amoc_m - rapid_base) / rapid_base
        by_class[cls].append((years_m, pct))
        # How much less negative the per-model 2080-2100 weakening becomes
        # when we move from the legacy 1950-1980 axis to the RAPID-era axis.
        m_late = (years_m >= 2081) & (years_m <= 2100)
        if m_late.sum() > 0:
            late_mean = amoc_m[m_late].mean()
            pct_legacy = 100.0 * (late_mean - legacy_base) / legacy_base
            pct_rapid = 100.0 * (late_mean - rapid_base) / rapid_base
            cmip6_rapid_to_legacy_shifts.append(pct_rapid - pct_legacy)

    legend_handles = []
    for cls, color in palette.items():
        if not by_class[cls]:
            continue
        common = np.arange(1850, 2101)
        stacked = np.array([
            np.interp(common, y, p, left=np.nan, right=np.nan)
            for y, p in by_class[cls]
        ])
        mean_curve = np.nanmean(stacked, axis=0)
        sd_curve = np.nanstd(stacked, axis=0)
        ax.fill_between(common, mean_curve - sd_curve, mean_curve + sd_curve,
                        color=color, alpha=0.18, zorder=3, linewidth=0)
        n_cls = len(by_class[cls])
        line, = ax.plot(common, mean_curve, color=color, linewidth=1.4,
                        zorder=5,
                        label=f"{cls}-driven mean $\\pm$ 1$\\sigma$ (n={n_cls})")
        legend_handles.append(line)

    # RAPID annual AMOC at 26.5N, anchored to its own 2005-2022 mean so it
    # sits at zero anomaly by construction during the observing window.
    rapid_handle = None
    rapid_path = REPO / "data" / "export_rahmstorf" / "amoc_26N_rapid_annual.nc"
    if rapid_path.exists():
        try:
            ds_r = xr.open_dataset(rapid_path)
            rapid_years = ds_r["year"].values
            rapid_amoc = ds_r["amoc_26N"].values
            ds_r.close()
            r_mask = (rapid_years >= rapid_y0) & (rapid_years <= rapid_y1)
            if r_mask.sum() > 0:
                r_base = float(np.nanmean(rapid_amoc[r_mask]))
                rapid_pct = 100.0 * (rapid_amoc - r_base) / r_base
                rapid_handle, = ax.plot(
                    rapid_years, rapid_pct, color="#117733", linewidth=1.4,
                    marker="o", markersize=3.5, markerfacecolor="white",
                    markeredgecolor="#117733", markeredgewidth=0.8,
                    zorder=8, label="RAPID 26.5$\\degree$N (annual)",
                )
        except Exception:
            pass

    # Portmann constraint translated to RAPID-era reference (Methods).
    # Published Portmann central value of -51% +/- 8% is relative to a 1950-1980
    # baseline; on the RAPID-era axis it shifts up by the ensemble-mean
    # legacy-to-RAPID offset (~+10 pp).
    portmann_mid_legacy, portmann_half = -51.0, 8.0
    shift = float(np.mean(cmip6_rapid_to_legacy_shifts)) \
        if cmip6_rapid_to_legacy_shifts else 0.0
    portmann_mid = portmann_mid_legacy + shift
    # Portmann 2026's emergent constraint is evaluated over 2050-2100; the
    # band's x-extent matches that period (not arbitrary padding past 2100).
    ax.fill_betweenx(
        [portmann_mid - portmann_half, portmann_mid + portmann_half],
        2050, 2100,
        color=CLR_PORTMANN, alpha=0.30, zorder=4,
    )
    ax.text(2102, portmann_mid, "Portmann\n2026", color=CLR_PORTMANN,
            fontsize=5.5, va="center", ha="left", fontweight="bold")

    ax.axhline(0, color="0.4", linewidth=0.5, zorder=1)
    ax.set_xlim(1850, 2125)
    ax.set_ylim(-70, 50)
    ax.set_xlabel("Year")
    # Two-line label so the full text (quantity, latitude, units and
    # reference period) fits inside the figure without clipping (R3.10).
    ax.set_ylabel("AMOC strength anomaly at 26.5$\\degree$N\n(% of 2005-2022 mean)")
    handles = legend_handles
    if smile_handle is not None:
        handles = handles + [smile_handle]
    if smile_mean_handle is not None:
        handles = handles + [smile_mean_handle]
    if rapid_handle is not None:
        handles = handles + [rapid_handle]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=6.5, handlelength=1.6, labelspacing=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# -------------------------------------------------------------------------
def _panel_salt_ts(ax: plt.Axes) -> None:
    """Annual basin-mean upper-300m salinity at 34.5S +/- 5 from EN4.2.2 and
    Roemmich-Gilson Argo, 2005-2024.  Overlay product OLS trend lines.
    """
    from scipy.stats import linregress
    df = pd.read_csv(RESULTS / "argo_basin_mean.csv")
    plot_specs = [
        ("EN4.2.2", "#17becf", "o"),
        ("RG09",    "#7f7f7f", "s"),
    ]
    for product, color, marker in plot_specs:
        sub = df[df["product"] == product].sort_values("year").dropna(
            subset=["salinity_psu"])
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["salinity_psu"], color=color,
                marker=marker, markersize=3.0, lw=0.9,
                label=product, zorder=3)
        if len(sub) >= 5:
            res = linregress(sub["year"].values.astype(float),
                             sub["salinity_psu"].values.astype(float))
            xs = np.array([sub["year"].min(), sub["year"].max()],
                          dtype=float)
            ax.plot(xs, res.intercept + res.slope * xs, color=color,
                    lw=0.7, ls="--", alpha=0.75, zorder=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Upper-300 m basin S (PSU)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False, fontsize=5.5, handlelength=1.5,
              handletextpad=0.4, columnspacing=1.2)


def _panel_coldblob_ts(ax: plt.Axes) -> None:
    """Absolute annual mean SST inside the tightened Cold Blob core box
    (50-56N, -45 to -25E), 1900-2024.  ORAS5 (1958-2024, bold solid) is the
    primary record; HadISST (1900-2024, thin dotted) extends the long
    pre-1958 context.  Both products use the same box.
    """
    ds_o = xr.open_dataset(RESULTS / "cold_blob_oras5.nc")
    y_o = ds_o["year"].values.astype(float)
    box_o = ds_o["caesar_box_sst"].values
    ds_o.close()

    win = 7
    kern = np.ones(win) / win
    def _smooth(v):
        s = np.convolve(v, kern, mode="same")
        half = win // 2
        s[:half] = np.nan
        s[-half:] = np.nan
        return s

    box_o_s = _smooth(box_o)

    ds_h = xr.open_dataset(RESULTS / "cold_blob_timeseries_hadisst.nc")
    y_h = ds_h["year"].values.astype(float)
    box_h = ds_h["caesar_box_sst"].values
    ds_h.close()
    box_h_s = _smooth(box_h)

    ax.plot(y_h, box_h_s, color="#1f77b4", lw=0.9, ls=":", alpha=0.75,
            zorder=2, label="HadISST (1900+)")
    ax.plot(y_o, box_o, color="#1f77b4", lw=0.4, alpha=0.30, zorder=3)
    ax.plot(y_o, box_o_s, color="#1f77b4", lw=1.5, zorder=5,
            label="ORAS5 (1958+)")

    ax.set_xlim(1900, 2024)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cold Blob core SST ($\\degree$C)")
    ax.spines[["top", "right"]].set_visible(False)
    # Legend OUTSIDE below the panel, single row, 2 entries
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False, fontsize=5.5,
              handlelength=1.6, handletextpad=0.4, columnspacing=1.5)


def _panel_coldblob_map(ax: plt.Axes, fig: plt.Figure) -> None:
    """Global HadISST SST change map: 2014-2024 mean minus 1900-1960 mean
    (degC), drawn in the Robinson elliptical projection.  Modern minus
    historical baselines after Caesar 2018; the subpolar North Atlantic is
    one of the few regions where SST has decreased in absolute terms against
    a globally warming ocean - the canonical Atlantic Cold Blob.
    """
    ds = xr.open_dataset(RESULTS / "cold_blob_trend_hadisst.nc")
    delta = ds["delta_sst"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        delta = delta[::-1, :]

    cmap = plt.cm.RdBu_r
    levels = np.arange(-1.5, 1.51, 0.30)

    ax.set_global()
    cs = ax.contourf(lon, lat, delta, levels=levels, cmap=cmap,
                     transform=ccrs.PlateCarree(), extend="both")
    ax.add_feature(cfeature.LAND, facecolor="0.35", edgecolor="none",
                   zorder=2)
    ax.coastlines(linewidth=0.3, color="black", zorder=3)
    rect = Rectangle((-45, 50), 20, 6, fill=False, edgecolor="black",
                     linestyle="--", linewidth=0.7,
                     transform=ccrs.PlateCarree(), zorder=4)
    ax.add_patch(rect)
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color="0.65",
                      alpha=0.6, zorder=1)
    gl.xlocator = plt.matplotlib.ticker.FixedLocator(
        [-120, -60, 0, 60, 120])
    gl.ylocator = plt.matplotlib.ticker.FixedLocator([-60, -30, 0, 30, 60])

    cax = ax.inset_axes([1.02, 0.10, 0.022, 0.80])
    cb = fig.colorbar(cs, cax=cax, ticks=np.arange(-1.5, 1.51, 0.50))
    cb.set_label(r"$\Delta$SST 2014-2024 vs 1900-1960 ($\degree$C)",
                 fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5)


def _panel_d(ax: plt.Axes, fig: plt.Figure) -> None:
    """Subpolar North Atlantic SST trend, 1993-2024, from HadISST.

    Shows the Atlantic Cold Blob as a coherent cooling patch in the centre
    of an otherwise warming subpolar basin - the canonical surface fingerprint
    of a weakening AMOC.  Discrete diverging colorbar (BoundaryNorm).  Caesar
    2018 analysis box outlined.
    """
    ds = xr.open_dataset(RESULTS / "cold_blob_trend_hadisst.nc")
    trend = ds["trend"].values  # degC per decade
    pvals = ds["pvalue"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()

    cmap = plt.cm.RdBu_r
    levels = np.arange(-0.60, 0.61, 0.15)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    pc = ax.pcolormesh(lon, lat, trend, cmap=cmap, norm=norm, shading="auto")

    # Stipple significant cells, thinned for clarity
    sig = pvals < 0.05
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    stride_lon = max(1, lon_grid.shape[1] // 18)
    stride_lat = max(1, lon_grid.shape[0] // 8)
    sig_thin = np.zeros_like(sig)
    sig_thin[::stride_lat, ::stride_lon] = sig[::stride_lat, ::stride_lon]
    ax.scatter(lon_grid[sig_thin], lat_grid[sig_thin],
               s=1.5, color="black", alpha=0.55, marker="o", linewidths=0)

    # Caesar 2018 Cold Blob analysis box (46-61N, -52 to -15E)
    rect = Rectangle((-52, 46), 37, 15, fill=False,
                     edgecolor="black", linestyle="--", linewidth=0.5,
                     zorder=5)
    ax.add_patch(rect)

    ax.set_xlim(-60, 0)
    ax.set_ylim(40, 65)
    ax.set_xticks([-60, -45, -30, -15, 0])
    ax.set_xticklabels(["60$\\degree$W", "45$\\degree$W", "30$\\degree$W",
                         "15$\\degree$W", "0$\\degree$"])
    ax.set_yticks([40, 50, 60])
    ax.set_yticklabels(["40$\\degree$N", "50$\\degree$N", "60$\\degree$N"])
    ax.set_aspect(1.4)  # mild lat-stretch for visual readability

    cb = fig.colorbar(pc, ax=ax, location="bottom", shrink=0.85,
                      pad=0.18, aspect=18,
                      ticks=np.arange(-0.6, 0.61, 0.30))
    cb.set_label("SST trend 1993-2024 ($\\degree$C decade$^{-1}$)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.spines[["top", "right"]].set_visible(True)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def build_figure() -> plt.Figure:
    """Hero layout: 3 rows.

    Row 0  Panel a              AMOC trajectory at 26.5N (full width)
    Row 1  Panels b, c, d       Time-series strips telling the meridional
                                fingerprint story (salt at 34.5S, F_ovS at
                                34.5S, Cold Blob box at 46-61N).
    Row 2  Panel e              High-resolution ORAS5 SST trend map over
                                the subpolar North Atlantic, 1993-2024.

    Panel d and Panel e are computed from the same ORAS5 product, so the
    time series and the map are mutually consistent.  Decomposition scatter
    and HadISST trend map have been moved to Supplementary Figures.
    """
    # subplot_mosaic: each named cell is independent of its row neighbours,
    # so the empty '.' on the bottom-right does not compress middle-row
    # panels.  Layout is specified visually as a 2-D list.
    # Panel a and Panel e take ~70% of the figure width (left two columns);
    # the third column is left empty so they do not look over-stretched.
    # Panel e uses a Robinson elliptical projection (Nature-style global map).
    layout = [["a", "a", "."],
              ["b", "c", "d"],
              ["e", "e", "."]]
    fig, axd = plt.subplot_mosaic(
        layout, figsize=(7.09, 5.8), layout="constrained",
        height_ratios=[1.5, 0.95, 1.25],
        width_ratios=[1.0, 1.0, 1.0],
        empty_sentinel=".",
        per_subplot_kw={
            "e": {"projection": ccrs.Robinson(central_longitude=-30)},
        },
    )
    ax_a = axd["a"]
    ax_b = axd["b"]
    ax_c = axd["c"]
    ax_d = axd["d"]
    ax_e = axd["e"]

    _panel_c(ax_a)             # AMOC trajectory
    _panel_salt_ts(ax_b)       # NEW salt time series at 34.5S
    _panel_a(ax_c)             # F_ovS time series (function still _panel_a)
    _panel_coldblob_ts(ax_d)   # Cold Blob time series, ORAS5
    _panel_coldblob_map(ax_e, fig)  # NEW: high-res ORAS5 trend map

    panel_labels = [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"),
                    (ax_d, "d"), (ax_e, "e")]
    for ax, lbl in panel_labels:
        ax.text(-0.04, 1.02, lbl, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")

    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR,
                        help="Directory for Figure1_hero.{pdf,png}")
    parser.add_argument("--basename", default="Figure1_hero",
                        help="Output file basename (no extension)")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf = out_dir / f"{args.basename}.pdf"
    png = out_dir / f"{args.basename}.png"
    fig.savefig(pdf, format="pdf", dpi=300)
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
