#!/usr/bin/env python3
"""PAPER_3 round-2 WP6: diagnostics for the Fig 1b overturning streamfunction.

R1 reports "pronounced sinking near 10 N" in Fig 1b. This script produces the
three diagnostic panels asked for and a factual listing of the streamfunction
extrema. It does NOT recompute the physics: it loads exactly the cached field
that scripts/paper3/make_paper3_figures.py:fig1_b() plots, namely
data/results/moc_streamfunction_2005_2024.npz (ORAS5, 2005-2024).

Panel (b) excludes the upper 250 m from the vertical integration start. Because
scripts/compute_moc_streamfunction.py builds Psi as a surface-downward cumulative
sum (`psi = np.cumsum(transport, axis=0)` with depth index 0 at 0.51 m), that
re-basing is exact rather than approximate:

    Psi_from250(z) = Psi(z) - Psi(250 m)

which is the same integral started at 250 m instead of at the surface.

Outputs:
    revision/rev_papaer3_02/figures/WP6_fig1b_diagnostics.pdf (+ .png)
    revision/rev_papaer3_02/results/WP6_extrema.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import maximum_filter, minimum_filter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "data" / "results"
OUT_FIG = REPO / "revision" / "rev_papaer3_02" / "figures"
OUT_RES = REPO / "revision" / "rev_papaer3_02" / "results"

MASK_DEPTH = 250.0  # m, panel (b) integration start
LAT_LO, LAT_HI = -35.0, 70.0  # plotted range, same as fig1_b
SCAN_LO, SCAN_HI = -5.0, 30.0  # extrema scan range asked for by WP6
PROFILE_LATS = (10.0, 26.5)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_fig1b_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load exactly the field fig1_b() plots (same file, same fallback order)."""
    path = RESULTS_DIR / "moc_streamfunction_2005_2024.npz"
    if not path.exists():
        path = RESULTS_DIR / "moc_streamfunction_oras5_1993_2024.npz"
    d = np.load(path)
    return d["psi"], d["lat"], d["depth"], path.name


def rebase_from(psi: np.ndarray, depth: np.ndarray, z0: float) -> np.ndarray:
    """Restart the surface-downward cumulative integration at depth z0."""
    k = int(np.argmin(np.abs(depth - z0)))
    out = psi - psi[k, :][None, :]
    out[:k, :] = np.nan  # above the new integration start: undefined
    return out


def local_extrema(
    field: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    nz: int = 5,
    nlat: int = 15,
    prominence: float = 0.25,
    exclude_depth_idx: int | None = None,
) -> list[dict]:
    """Local maxima and minima of field(depth, lat).

    A point qualifies only if it beats every other point in an
    ``nz`` x ``nlat`` window (the window centre is excluded, so flat plateaus
    produce no detections) by at least ``prominence`` Sv. That threshold is
    what separates genuine overturning cells from the numerical ripple of a
    smooth cumulative integral sampled on a 0.25 deg latitude grid.
    """
    work = np.where(np.isfinite(field), field, np.nan).astype(float)
    if exclude_depth_idx is not None:
        work[exclude_depth_idx, :] = np.nan
    finite = np.isfinite(work)

    # Window footprint with the centre removed.
    foot = np.ones((nz, nlat), dtype=bool)
    foot[nz // 2, nlat // 2] = False

    nbr_max = maximum_filter(np.nan_to_num(work, nan=-np.inf), footprint=foot)
    nbr_min = minimum_filter(np.nan_to_num(work, nan=+np.inf), footprint=foot)

    is_max = finite & (work > nbr_max + prominence)
    is_min = finite & (work < nbr_min - prominence)

    out = []
    for kind, mask in (("max", is_max), ("min", is_min)):
        for iz, ix in zip(*np.where(mask), strict=True):
            out.append(
                {
                    "kind": kind,
                    "latitude_N": float(lat[ix]),
                    "depth_m": float(depth[iz]),
                    "psi_Sv": float(field[iz, ix]),
                }
            )
    out.sort(key=lambda r: (r["kind"], r["latitude_N"], r["depth_m"]))
    return out


def dedupe_extrema(
    rows: list[dict], dlat: float = 2.0, dz_frac: float = 0.25
) -> list[dict]:
    """Collapse extrema that sit within dlat degrees and dz_frac in log-depth.

    Keeps the strongest member of each cluster, so a broad flat maximum is
    reported once rather than once per grid point.
    """
    kept: list[dict] = []
    for r in sorted(rows, key=lambda x: -abs(x["psi_Sv"])):
        dup = False
        for k in kept:
            if k["kind"] != r["kind"]:
                continue
            same_lat = abs(k["latitude_N"] - r["latitude_N"]) <= dlat
            same_z = abs(np.log(k["depth_m"]) - np.log(r["depth_m"])) <= dz_frac
            if same_lat and same_z:
                dup = True
                break
        if not dup:
            kept.append(r)
    kept.sort(key=lambda r: (r["kind"], r["latitude_N"], r["depth_m"]))
    return kept


def upper_cell_by_latitude(
    field: np.ndarray, lat: np.ndarray, depth: np.ndarray, zmax: float = 3000.0
) -> dict:
    """Psi_max(lat) and its depth, plus the local extrema of that 1D curve.

    This is the standard "overturning strength versus latitude" diagnostic. A
    distinguished feature at one latitude shows up here as a local extremum of
    the curve, which the 2D interior-extremum scan can miss when the feature is
    a shoulder rather than a closed cell.
    """
    from scipy.signal import find_peaks

    zsel = depth <= zmax
    sub = field[zsel, :]
    dz = depth[zsel]

    with np.errstate(invalid="ignore"):
        allnan = np.all(~np.isfinite(sub), axis=0)
    psi_max = np.full(len(lat), np.nan)
    z_at_max = np.full(len(lat), np.nan)
    good = ~allnan
    psi_max[good] = np.nanmax(sub[:, good], axis=0)
    z_at_max[good] = dz[np.nanargmax(sub[:, good], axis=0)]

    curve = psi_max.copy()
    finite = np.isfinite(curve)
    peaks, pprops = find_peaks(np.where(finite, curve, -np.inf), prominence=0.25)
    troughs, tprops = find_peaks(np.where(finite, -curve, -np.inf), prominence=0.25)

    return {
        "z_search_max_m": zmax,
        "latitude_N": [float(v) for v in lat],
        "psi_max_Sv": [float(v) for v in psi_max],
        "depth_of_psi_max_m": [float(v) for v in z_at_max],
        "local_maxima": [
            {
                "latitude_N": float(lat[i]),
                "psi_max_Sv": float(curve[i]),
                "depth_m": float(z_at_max[i]),
                "prominence_Sv": float(p),
            }
            for i, p in zip(peaks, pprops["prominences"], strict=True)
        ],
        "local_minima": [
            {
                "latitude_N": float(lat[i]),
                "psi_max_Sv": float(curve[i]),
                "depth_m": float(z_at_max[i]),
                "prominence_Sv": float(p),
            }
            for i, p in zip(troughs, tprops["prominences"], strict=True)
        ],
    }


def panel_map(ax, lat, depth, field, levels, title, label):
    cmap = plt.cm.RdBu_r.copy()
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend="both")
    cf = ax.contourf(
        lat, depth, field, levels=levels, cmap=cmap, norm=norm, extend="both"
    )
    ax.contour(lat, depth, field, levels=levels, colors="0.35", linewidths=0.2)
    ax.contour(lat, depth, field, levels=[0], colors="k", linewidths=0.6)
    ax.set_ylim(5500, 0)
    ax.set_xlim(LAT_LO, LAT_HI)
    ax.set_ylabel("Depth (m)")
    ax.set_title(title, fontsize=7, pad=3)
    ax.axvline(26.5, color="0.3", lw=0.4, ls=":")
    ax.axvline(-34.5, color="0.3", lw=0.4, ls=":")
    ax.axvline(10.0, color="0.3", lw=0.4, ls=":")
    ax.text(
        0.004,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )
    return cf


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    psi, lat, depth, src = load_fig1b_field()
    m = (lat >= LAT_LO) & (lat <= LAT_HI)
    lat_p = lat[m].astype(float)
    psi_p = psi[:, m]
    psi_b = rebase_from(psi, depth, MASK_DEPTH)[:, m]

    # Zero-centered discrete scale spanning both panels.
    vmax = float(
        np.ceil(np.nanmax(np.abs(np.concatenate([psi_p.ravel(), psi_b.ravel()]))) / 2.0)
        * 2.0
    )
    vmax = min(vmax, 24.0)
    levels = np.arange(-vmax, vmax + 2, 2)

    fig = plt.figure(figsize=(7.09, 7.2))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1, 1, 0.95],
        hspace=0.42,
        left=0.08,
        right=0.90,
        top=0.96,
        bottom=0.06,
    )

    ax_a = fig.add_subplot(gs[0])
    cf = panel_map(
        ax_a,
        lat_p,
        depth,
        psi_p,
        levels,
        "Time-mean Atlantic overturning streamfunction, ORAS5 2005-2024",
        "a",
    )

    ax_b = fig.add_subplot(gs[1])
    panel_map(
        ax_b,
        lat_p,
        depth,
        psi_b,
        levels,
        f"Vertical integration restarted at {MASK_DEPTH:.0f} m "
        "(wind-driven surface cells excluded)",
        "b",
    )
    ax_b.set_xlabel("Latitude (°N)")

    cax = fig.add_axes([0.915, 0.44, 0.014, 0.50])
    cbar = fig.colorbar(
        cf, cax=cax, orientation="vertical", extend="both", ticks=levels[::2]
    )
    cbar.set_label(r"$\Psi$ (Sv)", fontsize=7)
    cbar.ax.tick_params(labelsize=6, width=0.4, length=2)
    cbar.outline.set_linewidth(0.4)

    # --- (c) vertical profiles at 10 N and 26.5 N ---
    ax_c = fig.add_subplot(gs[2])
    styles = {10.0: ("#20558a", "-"), 26.5: ("#b3541e", "-")}
    profiles = {}
    for want in PROFILE_LATS:
        j = int(np.argmin(np.abs(lat_p - want)))
        col, ls = styles[want]
        ax_c.plot(
            psi_p[:, j],
            depth,
            color=col,
            linestyle=ls,
            linewidth=1.2,
            label=f"{lat_p[j]:.1f}°N, surface start",
        )
        ax_c.plot(
            psi_b[:, j],
            depth,
            color=col,
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
            label=f"{lat_p[j]:.1f}°N, {MASK_DEPTH:.0f} m start",
        )
        profiles[f"{want:g}N"] = {
            "nearest_latitude_N": float(lat_p[j]),
            "psi_surface_start_Sv": [float(v) for v in psi_p[:, j]],
            "psi_250m_start_Sv": [float(v) for v in psi_b[:, j]],
            "psi_max_Sv": float(np.nanmax(psi_p[:, j])),
            "psi_max_depth_m": float(depth[int(np.nanargmax(psi_p[:, j]))]),
            "psi_min_Sv": float(np.nanmin(psi_p[:, j])),
            "psi_min_depth_m": float(depth[int(np.nanargmin(psi_p[:, j]))]),
        }
    ax_c.axvline(0, color="0.6", lw=0.4)
    ax_c.set_ylim(5500, 0)
    ax_c.set_xlabel(r"$\Psi$ (Sv)")
    ax_c.set_ylabel("Depth (m)")
    ax_c.legend(fontsize=5, loc="lower right", frameon=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.text(
        0.004,
        1.03,
        "c",
        transform=ax_c.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    out = OUT_FIG / "WP6_fig1b_diagnostics"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {out}.pdf / .png")

    # --- extrema scan, 5 S to 30 N ---
    s = (lat >= SCAN_LO) & (lat <= SCAN_HI)
    lat_s = lat[s].astype(float)
    k_ref = int(np.argmin(np.abs(depth - MASK_DEPTH)))
    raw_a = local_extrema(psi[:, s], lat_s, depth)
    # In panel (b) the reference level is identically zero by construction, so
    # it is excluded rather than reported as a row of spurious extrema.
    raw_b = local_extrema(
        rebase_from(psi, depth, MASK_DEPTH)[:, s],
        lat_s,
        depth,
        exclude_depth_idx=k_ref,
    )
    ext_a = dedupe_extrema(raw_a)
    ext_b = dedupe_extrema(raw_b)

    cell_a = upper_cell_by_latitude(psi[:, s], lat_s, depth)
    cell_b = upper_cell_by_latitude(
        rebase_from(psi, depth, MASK_DEPTH)[:, s], lat_s, depth
    )

    def near10(rows: list[dict], win: float = 4.0) -> list[dict]:
        return [r for r in rows if abs(r["latitude_N"] - 10.0) <= win]

    payload = {
        "source_file": f"data/results/{src}",
        "field": "time-mean Atlantic MOC streamfunction, ORAS5 2005-2024",
        "integration": (
            "psi = np.cumsum(transport, axis=0), depth index 0 = 0.51 m, "
            "i.e. surface-downward"
        ),
        "scan_range_latitude_N": [SCAN_LO, SCAN_HI],
        "grid_points_in_scan": {"n_lat": int(s.sum()), "n_depth": int(len(depth))},
        "panel_a_extrema_raw_count": len(raw_a),
        "panel_b_extrema_raw_count": len(raw_b),
        "panel_a_extrema": ext_a,
        "panel_b_extrema": ext_b,
        "panel_a_extrema_near_10N": near10(ext_a),
        "panel_b_extrema_near_10N": near10(ext_b),
        "panel_a_upper_cell_by_latitude": cell_a,
        "panel_b_upper_cell_by_latitude": cell_b,
        "panel_a_upper_cell_extrema_near_10N": near10(
            cell_a["local_maxima"] + cell_a["local_minima"]
        ),
        "panel_b_upper_cell_extrema_near_10N": near10(
            cell_b["local_maxima"] + cell_b["local_minima"]
        ),
        "bottom_psi_Sv": {
            "note": (
                "Psi at the deepest level; a surface-downward cumulative "
                "integral closes at the net section transport"
            ),
            "at_10N": float(psi[-1, int(np.argmin(np.abs(lat - 10.0)))]),
            "at_26.5N": float(psi[-1, int(np.argmin(np.abs(lat - 26.5)))]),
            "min_over_scan": float(np.nanmin(psi[-1, s])),
            "max_over_scan": float(np.nanmax(psi[-1, s])),
        },
        "profiles": profiles,
        "colorbar_levels_Sv": [float(v) for v in levels],
    }
    (OUT_RES / "WP6_extrema.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUT_RES / 'WP6_extrema.json'}")
    print(
        f"panel a: {len(ext_a)} clustered extrema "
        f"({len(near10(ext_a))} within 4 deg of 10 N)"
    )
    print(
        f"panel b: {len(ext_b)} clustered extrema "
        f"({len(near10(ext_b))} within 4 deg of 10 N)"
    )
    for r in ext_a:
        print(
            f"  a {r['kind']}: {r['latitude_N']:+7.2f} N  "
            f"{r['depth_m']:8.1f} m  {r['psi_Sv']:+8.2f} Sv"
        )
    for r in ext_b:
        print(
            f"  b {r['kind']}: {r['latitude_N']:+7.2f} N  "
            f"{r['depth_m']:8.1f} m  {r['psi_Sv']:+8.2f} Sv"
        )
    for tag, cell in (("a", cell_a), ("b", cell_b)):
        print(f"upper-cell curve, panel {tag}:")
        for kind in ("local_maxima", "local_minima"):
            for r in cell[kind]:
                print(
                    f"  {kind[6:9]} {r['latitude_N']:+7.2f} N  "
                    f"{r['depth_m']:7.1f} m  {r['psi_max_Sv']:+7.2f} Sv  "
                    f"(prom {r['prominence_Sv']:.2f} Sv)"
                )
    bp = payload["bottom_psi_Sv"]
    print(
        f"bottom Psi: 10N {bp['at_10N']:+.2f} Sv, "
        f"26.5N {bp['at_26.5N']:+.2f} Sv, "
        f"scan range [{bp['min_over_scan']:+.2f}, {bp['max_over_scan']:+.2f}] Sv"
    )


if __name__ == "__main__":
    main()
