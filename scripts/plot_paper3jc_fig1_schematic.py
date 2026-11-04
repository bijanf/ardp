#!/usr/bin/env python3
"""Figure 1 of the Journal of Climate version: the section, the limbs and cells,
and the limb factorization of F_ovS.

a  Mean surface salinity of the Atlantic sector (EN4.2.2, 2005-2024) on an
   orthographic view, with the contiguous 34.5 S section and the four longitude
   bands WB, WI, EI and CB.
b  Longitude-depth section of the time-mean salinity in GLORYS12V1 (1993-2025),
   with the water masses, the streamfunction maximum and the zero crossing of
   the streamfunction that bound the cells.
c  The profiles that define the limbs and cells: the zonally integrated velocity
   after removal of the net transport, its streamfunction and the zonal-mean
   salinity, with the transport-weighted limb salinities S-up and S-down labelled
   at the tops of their vertical lines.
d  The plane of the two factors of F_ovS = -T dS / S0, shaded by F_ovS, with the
   5-year-mean path of each reanalysis through it.

The data handling of scripts/plot_paper3r1_fig1_schematic.py is kept: the same
NetCDF variables, the same cell and limb computations, the same 5-year blocks
and the same reference salinity S0 = 35 PSU. Every number written in the figure
is read from the JSON or NetCDF files at run time.

Writes PAPER_3_v2/figures_jclim/Fig1_schematic.{pdf,png} at the printed size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import paper3jc_style as st  # noqa: E402

RESULTS = REPO / "data" / "results"
EN4_DIR = REPO / "data" / "en4"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT_DIR = REPO / "PAPER_3_v2" / "figures_jclim"

FIG_W = st.TWO_COL          # 6.5 in, AMS two-column width
FIG_H = 7.25                # in, below the 8 in limit
S0 = 35.0                   # reference salinity of Eq. (factorization), PSU
BANDS = [(-54.0, -45.0), (-45.0, -20.0), (-20.0, 0.0), (0.0, 19.4)]
BAND_LABELS = ["WB", "WI", "EI", "CB"]
YTICKS = [0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000]
ENDPOINT_LABEL_FS = 7
BAND_LABEL_DLAT = 4.8       # latitude offset of the band labels above the section, degrees
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
DARK_HALO = [pe.withStroke(linewidth=2.0, foreground="black")]

# One marker shape per reanalysis, so that the paths separate without colour
# (the shapes are the marker keys of the shared style module).
PRODUCT_MARKER = {"oras5": st.PRODUCT_MARKERS["ORAS5"],
                  "glorys12": st.PRODUCT_MARKERS["GLORYS12V1"],
                  "ecco": st.PRODUCT_MARKERS["ECCO-V4r4"]}
# Offset (T in Sv, dS in PSU) and alignment of the "step" label from the block point
# that contains the observing-system step.
STEP_OFFSET = {"oras5": (0.14, -0.010, "left"), "glorys12": (-0.18, -0.030, "right")}


# ------------------------------------------------------------------ layout
def add_ax(fig, x_in: float, y_in: float, w_in: float, h_in: float, **kwargs):
    """Axes whose left-top corner sits x_in from the left and y_in from the top."""
    return fig.add_axes(
        [x_in / FIG_W, 1.0 - (y_in + h_in) / FIG_H, w_in / FIG_W, h_in / FIG_H],
        **kwargs,
    )


def minus(value: float, fmt: str = "{:.3g}") -> str:
    """Number with a typographic minus sign."""
    return fmt.format(value).replace("-", "−")


# ------------------------------------------------------------------ depth axis
def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def depth_axis(ax, label: bool = True) -> None:
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(5600, 0)
    ax.set_yticks(YTICKS)
    if label:
        ax.set_ylabel("Depth (m)")


# ------------------------------------------------------------------ data
def en4_sss_mean():
    """Mean surface salinity of EN4.2.2 over the monthly files (as in the old script)."""
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    acc = None
    for f in files:
        with xr.open_dataset(f, decode_timedelta=False) as ds:
            s = ds["salinity"].isel(time=0, depth=0).load()
        acc = s if acc is None else acc + s
    sss = acc / len(files)
    lon = sss["lon"].values
    return sss.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon"), len(files)


def cells_from_profile(vbc: np.ndarray, e3t: np.ndarray):
    """Streamfunction (Sv), its level boundaries, the depth of its maximum and the
    depth of its zero crossing below the maximum (as in the old script)."""
    q = vbc * e3t
    psi = np.concatenate([[0.0], np.cumsum(q)]) / 1e6
    zw = np.concatenate([[0.0], np.cumsum(e3t)])
    kmax = int(np.argmax(psi))
    i = next(i for i in range(kmax + 1, len(psi) - 1) if psi[i] < 0)
    z0 = zw[i - 1] + psi[i - 1] / (psi[i - 1] - psi[i]) * (zw[i] - zw[i - 1])
    return psi, zw, zw[kmax], z0


def blocks(years, values, width: int = 5):
    """Five-year block means from the first year; a final block with fewer than three
    years is dropped (as in the old script). Returns the means and the first year
    of each block."""
    yrs = np.asarray(years)
    out_t, out_y = [], []
    start = yrs[0]
    while start <= yrs[-1]:
        m = (yrs >= start) & (yrs < start + width)
        if m.sum() < 3 and out_t:
            break
        out_t.append(np.mean(values[m]))
        out_y.append(int(yrs[m][0]))
        start += width
    return np.array(out_t), out_y


def span(first: int, last: int) -> str:
    return f"{first}–{last}"


def block_index(starts, year: int) -> int:
    """Index of the 5-year block that contains the year."""
    return max(k for k, s in enumerate(starts) if s <= year)


# ------------------------------------------------------------------ main
def main() -> None:
    st.set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Section fields (GLORYS12V1), as in the old script.
    f3 = xr.open_dataset(RESULTS / "paper3r1_section3d_glorys12.nc")
    sec = xr.open_dataset(RESULTS / "paper3r1_section_glorys12.nc")
    lon = f3["lon"].values.astype(float)
    depth = f3["depth"].values
    smean = f3["s"].mean("time").values
    vbc = sec["V_bc"].mean("time").values
    sbar = sec["S_bar"].mean("time").values
    e3t = sec["e3t"].values
    psi, zw, z_max, z0 = cells_from_profile(vbc, e3t)
    q = vbc * e3t
    pos, neg = np.clip(q, 0, None), np.clip(-q, 0, None)
    s_n = (pos * sbar).sum() / pos.sum()
    s_s = (neg * sbar).sum() / neg.sum()
    row = f3["row_lon"].values.astype(float)
    row = np.where(row > 180, row - 360, row)
    rio = (row >= -58.5) & (row <= -56.9) & np.isfinite(f3["row_sss_mean"].values)
    lat0 = float(f3.attrs["section_latitude"])
    print(f"z_psimax {z_max:.0f} m  z0 {z0:.0f} m  S_N {s_n:.3f}  S_S {s_s:.3f}")

    # Reanalysis limb factors (attribution.json), as in the old script.
    att = json.loads((ANALYSIS / "attribution.json").read_text())

    fig = st.figure("two", FIG_H)

    # ============================================================ a: map
    proj = ccrs.Orthographic(central_longitude=-18, central_latitude=-24)
    globe = 1.95
    axm = add_ax(fig, 0.14, 0.34, globe, globe, projection=proj)
    axm.set_global()
    sss, n_en4 = en4_sss_mean()
    print(f"EN4 monthly files averaged: {n_en4}")
    levels = np.arange(33.8, 37.41, 0.2)
    cmap = plt.get_cmap("cividis", len(levels) - 1)
    cf = axm.contourf(
        sss["lon"],
        sss["lat"],
        sss.values,
        levels=levels,
        cmap=cmap,
        norm=colors.BoundaryNorm(levels, cmap.N),
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    axm.add_feature(cfeature.LAND, facecolor="#4d4d4d", edgecolor="none", zorder=2)
    axm.gridlines(
        linewidth=0.25,
        color="white",
        alpha=0.5,
        xlocs=range(-180, 181, 30),
        ylocs=range(-90, 91, 30),
    )
    pc = ccrs.PlateCarree()
    xsec = np.linspace(-54, 19.4, 200)
    axm.plot(xsec, np.full(xsec.size, lat0), color="white", lw=2.8, transform=pc,
             zorder=3, solid_capstyle="butt")
    axm.plot(xsec, np.full(xsec.size, lat0), color="black", lw=1.3, transform=pc,
             zorder=4, solid_capstyle="butt")
    for x in [b[0] for b in BANDS] + [BANDS[-1][1]]:
        axm.plot([x, x], [lat0 - 1.6, lat0 + 1.6], color="black", lw=0.8,
                 transform=pc, zorder=4)
    # Band labels sit clear of the section line and its band ticks (7 pt).
    for (lo, hi), lab in zip(BANDS, BAND_LABELS, strict=True):
        axm.text(0.5 * (lo + hi), lat0 + BAND_LABEL_DLAT, lab, ha="center", va="bottom",
                 fontsize=7, transform=pc, zorder=5,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    # Reference point of the excluded Rio de la Plata: a filled dot, because open markers
    # are reserved for the first block of each path in panel (d) (see the caption).
    axm.plot(row[rio].mean(), lat0, "o", ms=3.2, mfc="black", mec="white", mew=0.5,
             transform=pc, zorder=6)
    cax = add_ax(fig, 0.355, 2.55, 1.52, 0.085)
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=[34, 35, 36, 37])
    cb.set_label("Surface salinity (PSU)", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2)
    st.panel_label_pt(axm, "a", dx_pt=0.0)

    # ============================================================ b: section
    axs = add_ax(fig, 2.92, 0.34, 2.98, 2.12)
    slev = np.array([34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 34.9, 35.0, 35.25,
                     35.5, 35.75, 36.0, 36.3])
    scmap = plt.get_cmap("cividis", len(slev) - 1)
    cs = axs.contourf(lon, depth, smean, levels=slev, cmap=scmap,
                      norm=colors.BoundaryNorm(slev, scmap.N), extend="both")
    axs.contour(lon, depth, smean, levels=[34.5, 34.9, 35.5], colors="white",
                linewidths=0.3, alpha=0.8)
    axs.axhline(z_max, color="white", lw=1.0, ls=":")
    axs.axhline(z0, color="white", lw=1.0, ls="--")
    depth_axis(axs)
    axs.set_xlim(lon.min(), lon.max())
    axs.set_xticks([-50, -40, -30, -20, -10, 0, 10])
    axs.set_facecolor("#4d4d4d")
    axs.set_xlabel("Longitude (degrees east)")
    for name, x, z, c in (
        ("SACW", -25, 160, "black"),
        ("AAIW", -25, 800, "white"),
        ("NADW", -25, 2450, "black"),
        ("AABW", -38, 4700, "white"),
    ):
        axs.text(x, z, name, ha="center", va="center", fontsize=7, color=c,
                 fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=1.6,
                                             foreground="white" if c == "black" else "black")])
    axs.text(8.0, z0 - 380, "upper cell above", ha="center", va="center", fontsize=7,
             color="white", style="italic", path_effects=DARK_HALO)
    axs.text(8.0, z0 + 420, "abyssal cell below", ha="center", va="center", fontsize=7,
             color="white", style="italic", path_effects=DARK_HALO)
    axs.text(lon.min() + 1.0, z_max - 60, "base of northward limb", ha="left",
             va="bottom", fontsize=7, color="white", path_effects=DARK_HALO)
    axs.text(lon.min() + 1.0, z0 - 60, "cell boundary", ha="left", va="bottom",
             fontsize=7, color="white", path_effects=DARK_HALO)
    cb2 = fig.colorbar(cs, cax=add_ax(fig, 6.02, 0.34, 0.085, 2.12),
                       ticks=[34.2, 34.4, 34.6, 34.8, 35.0, 35.5, 36.0])
    cb2.set_label("Salinity (PSU)", fontsize=7, labelpad=2)
    cb2.ax.tick_params(labelsize=7, length=2)
    st.panel_label_pt(axs, "b")

    # ============================================================ c: profiles
    top_c, h_c = 3.38, 3.40
    ax1 = add_ax(fig, 0.50, top_c, 0.58, h_c)
    ax2 = add_ax(fig, 1.14, top_c, 0.58, h_c, sharey=ax1)
    ax3 = add_ax(fig, 1.78, top_c, 0.58, h_c, sharey=ax1)
    v_sv = vbc / 1e3  # m2 s-1 -> Sv per km of depth
    ax1.fill_betweenx(depth, 0, v_sv, where=v_sv > 0, color=st.C_NORTH, alpha=0.9, lw=0)
    ax1.fill_betweenx(depth, 0, v_sv, where=v_sv < 0, color=st.C_SOUTH, alpha=0.9, lw=0)
    ax1.plot(v_sv, depth, color="black", lw=0.5)
    ax1.axvline(0, color="black", lw=0.5)
    ax1.set_xlabel("$V_{bc}$\n(Sv km$^{-1}$)", linespacing=1.1)
    ax1.xaxis.set_major_locator(MaxNLocator(3))

    zf = np.linspace(0, zw[-1], 2000)
    pf = np.interp(zf, zw, psi)
    ax2.fill_betweenx(zf, 0, pf, where=zf <= z0, color=st.C_CELL_UP, lw=0)
    ax2.fill_betweenx(zf, 0, pf, where=zf >= z0, color=st.C_CELL_AB, lw=0)
    ax2.plot(psi, zw, color="black", lw=0.8)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.set_xlabel("$\\psi$\n(Sv)", linespacing=1.1)
    ax2.xaxis.set_major_locator(MaxNLocator(3))

    ax3.plot(sbar, depth, color="black", lw=0.8)
    ax3.axvline(s_n, color=st.C_NORTH, lw=1.0, ls="-")
    ax3.axvline(s_s, color=st.C_SOUTH, lw=1.0, ls="--")
    ax3.set_xlabel("$\\bar{S}$\n(PSU)", linespacing=1.1)
    ax3.set_xlim(34.25, 35.8)
    ax3.set_xticks([34.4, 35.0, 35.6])
    # The two vertical lines, labelled at their top ends: S-up (red, solid) is the
    # transport-weighted salinity of the northward limb, S-down (blue, dashed) of the
    # southward limb. Values are the section values computed above.
    ax3.annotate(f"S↑ {s_n:.2f}", xy=(s_n, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(1.5, 2.0), textcoords="offset points", ha="left", va="bottom",
                 fontsize=7, color=st.C_NORTH, annotation_clip=False)
    ax3.annotate(f"S↓ {s_s:.2f}", xy=(s_s, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(-1.5, 2.0), textcoords="offset points", ha="right", va="bottom",
                 fontsize=7, color=st.C_SOUTH, annotation_clip=False)
    for k, ax in enumerate((ax1, ax2, ax3)):
        ax.axhline(z_max, color="black", lw=0.6, ls=":")
        ax.axhline(z0, color="black", lw=0.6, ls="--")
        depth_axis(ax, label=(k == 0))
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
    st.panel_label_pt(ax1, "c")

    # ============================================================ d: plane
    axd = add_ax(fig, 2.80, top_c, 3.62, h_c)
    tt, dd = np.meshgrid(np.linspace(12.5, 25.0, 300), np.linspace(-0.01, 0.37, 300))
    ff = -tt * dd / S0
    # F_ovS colour: PuOr reversed, the negative half only (purple at -0.30, white at 0).
    fov_cmap = colors.ListedColormap(st.CMAP_FOVS(np.linspace(0.0, 0.46, 256)))
    fov_norm = colors.Normalize(vmin=-0.30, vmax=0.0)
    cf_d = axd.contourf(tt, dd, ff, levels=np.linspace(-0.30, 0.0, 31), cmap=fov_cmap,
                        norm=fov_norm, extend="min")
    lev = np.array([-0.20, -0.15, -0.10, -0.075, -0.05, -0.025])
    cs_d = axd.contour(tt, dd, ff, levels=lev, colors="0.30", linewidths=0.5,
                       linestyles="--")
    x_lab = 24.0
    axd.clabel(
        cs_d,
        fmt=lambda v: minus(v, "{:.3f}"),
        fontsize=7,
        inline_spacing=2,
        manual=[(x_lab, -v * S0 / x_lab) for v in lev if -v * S0 / x_lab < 0.36],
    )
    for txt in cs_d.labelTexts:
        txt.set_bbox(dict(fc="white", ec="none", pad=0.8, alpha=0.9))

    # Colour bar of the shading, at the empty lower left of the plane.
    cax_d = add_ax(fig, 2.80 + 0.14, top_c + h_c - 0.40, 1.05, 0.07)
    cbd = fig.colorbar(cf_d, cax=cax_d, orientation="horizontal",
                       ticks=[-0.30, -0.15, 0.0])
    cbd.set_ticklabels([minus(-0.30, "{:.2f}"), minus(-0.15, "{:.2f}"), "0.00"])
    cbd.ax.tick_params(labelsize=7, length=2)
    cbd.ax.set_title("$F_{\\mathrm{ovS}}$ (Sv)", fontsize=7, loc="left", pad=2)

    halo_w = [pe.withStroke(linewidth=2.0, foreground="white")]
    # Label anchors in data coordinates, (x, y, ha, va) for the start and the end
    # label. They sit in the empty parts of the plane, never on a path.
    places = {
        "oras5": ((19.0, 0.035, "right", "center"), (17.22, 0.176, "right", "center")),
        "glorys12": ((22.25, 0.072, "right", "bottom"), (23.55, 0.245, "right", "bottom")),
        "ecco": ((13.9, 0.330, "center", "bottom"), (15.0, 0.336, "left", "bottom")),
    }
    for key, name, col in (
        ("oras5", "ORAS5", st.C_ORAS5),
        ("glorys12", "GLORYS12V1", st.C_GLORYS),
        ("ecco", "ECCO-V4r4", st.C_ECCO),
    ):
        r = att[key]
        mk = PRODUCT_MARKER[key]
        tv, y0 = blocks(r["years"], np.array(r["T_limb_Sv"]))
        dv, _ = blocks(r["years"], np.array(r["dS_limb_PSU"]))
        yrs = np.asarray(r["years"])
        first_block_start = y0[0]
        last_block_start = y0[-1]
        last_block_end = int(min(yrs[-1], last_block_start + 4))
        n_last = int(np.sum((yrs >= last_block_start) & (yrs < last_block_start + 5)))
        # Path: solid line; filled markers at the intermediate blocks; an open marker
        # of the product shape at the first block; an arrowhead whose tip sits on the
        # last block.
        axd.plot(tv, dv, "-", color=col, lw=1.3, zorder=3)
        axd.plot(tv[1:-1], dv[1:-1], ls="none", marker=mk, ms=3.0, mfc=col, mec=col,
                 zorder=4)
        axd.plot(tv[0], dv[0], ls="none", marker=mk, ms=5.2, mfc="white", mec=col,
                 mew=1.1, zorder=7)
        axd.annotate(
            "", xy=(tv[-1], dv[-1]), xytext=(tv[-2], dv[-2]),
            arrowprops={"arrowstyle": "-|>,head_length=0.5,head_width=0.3", "color": col,
                        "lw": 1.3, "mutation_scale": 12, "shrinkA": 0, "shrinkB": 0},
            zorder=6,
        )
        (sx, sy, sha, sva), (ex, ey, eha, eva) = places[key]
        axd.text(sx, sy, span(first_block_start, first_block_start + 4),
                 color=col, fontsize=ENDPOINT_LABEL_FS, ha=sha, va=sva,
                 path_effects=halo_w, zorder=6)
        last_txt = span(last_block_start, last_block_end)
        if n_last < 5:
            last_txt += f" ({n_last} yr)"
        axd.text(ex, ey, f"{name}\n{last_txt}",
                 color=col, fontsize=ENDPOINT_LABEL_FS, ha=eha, va=eva,
                 linespacing=1.05, path_effects=halo_w, zorder=6)
        # Observing-system step of Eq. (stepfit), read from the JSON (2015 ORAS5, 2004 GLORYS12V1).
        step = r.get("mean_stats", {}).get("step_year_in_model")
        if step is not None:
            k = block_index(y0, int(step))
            dx, dy, sha_step = STEP_OFFSET[key]
            axd.text(tv[k] + dx, dv[k] + dy, f"step {int(step)}", color=col,
                     fontsize=ENDPOINT_LABEL_FS, ha=sha_step, va="center",
                     path_effects=halo_w, zorder=6)
            print(f"{key}: step {step} in block {y0[k]} point T {tv[k]:.3f} dS {dv[k]:.4f}")
        print(f"{key}: blocks {len(tv)} first {first_block_start} last {last_block_start}"
              f"-{last_block_end} n_last {n_last} T_end {tv[-1]:.3f} dS_end {dv[-1]:.4f}")

    # Direction of a more negative F_ovS: a stronger overturning (T up) or a wider contrast (dS up).
    ox, oy = 17.9, 0.265
    axd.annotate("", xy=(ox + 2.5, oy), xytext=(ox, oy),
                 arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.0,
                             "mutation_scale": 8})
    axd.annotate("", xy=(ox, oy + 0.050), xytext=(ox, oy),
                 arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.0,
                             "mutation_scale": 8})
    axd.text(ox + 1.25, oy - 0.010, "stronger overturning", fontsize=7,
             ha="center", va="top", path_effects=halo_w)
    axd.text(ox + 0.14, oy + 0.050, "wider salinity contrast", fontsize=7,
             ha="left", va="center", path_effects=halo_w)

    axd.set_xlim(12.5, 25.0)
    axd.set_ylim(-0.01, 0.37)
    axd.set_xticks([14, 16, 18, 20, 22, 24])
    axd.set_xlabel("Northward-limb transport $T$ (Sv)")
    axd.set_ylabel("Limb salinity contrast $\\Delta S$ (PSU)")
    # Offset 0 instead of the default: the S-down label of panel (c) sits in this strip.
    st.panel_label_pt(axd, "d", dx_pt=0.0)

    st.check_min_fontsize(fig)
    st.save(fig, str(OUT_DIR / "Fig1_schematic"))
    plt.close(fig)
    f3.close()
    sec.close()


if __name__ == "__main__":
    main()
