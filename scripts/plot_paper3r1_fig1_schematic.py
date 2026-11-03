#!/usr/bin/env python3
"""Figure 1: the section, the limbs and cells, and the workflow of the analysis.

a  Mean surface salinity of the Atlantic sector (EN4.2.2, 2005-2024) on an
   orthographic view, with the contiguous 34.5 S section, the four longitude
   bands used to locate the salinity change, and the non-contiguous water body
   that a fixed longitude box would admit.
b  Longitude-depth section of the time-mean salinity in GLORYS12V1 (1993-2025),
   with the water masses of the limbs and the depths that bound the cells.
c  The profiles that define the limbs and cells: the zonally integrated
   velocity after removal of the net transport, its streamfunction and the
   zonal-mean salinity, with the transport-weighted limb salinities.
d  The plane of the two factors of F_ovS = -T dS / S0, shaded by F_ovS, with the
   5-year-mean path of each reanalysis through it.

Writes ``PAPER_3_v2/figures_r1/Fig1_schematic.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from paper3v2_style import TWO_COL, panel_letter, set_style  # noqa: E402

RESULTS = REPO / "data" / "results"
EN4_DIR = REPO / "data" / "en4"
FIG_DIR = REPO / "PAPER_3_v2" / "figures_r1"
BANDS = [(-54.0, -45.0), (-45.0, -20.0), (-20.0, 0.0), (0.0, 19.4)]
BAND_LABELS = ["WB", "WI", "EI", "CB"]
C_NORTH = "#b2182b"
C_SOUTH = "#2166ac"
YTICKS = [0, 100, 300, 600, 1000, 2000, 3000, 4000, 5000]


def depth_fwd(z):
    return np.sqrt(np.clip(z, 0, None))


def depth_inv(z):
    return np.asarray(z) ** 2


def depth_axis(ax, label=True):
    ax.set_yscale("function", functions=(depth_fwd, depth_inv))
    ax.set_ylim(5600, 0)
    ax.set_yticks(YTICKS)
    if label:
        ax.set_ylabel("Depth (m)")


def en4_sss_mean() -> xr.DataArray:
    files = sorted(EN4_DIR.glob("EN.4.2.2.f.analysis.g10.*.nc"))
    acc = None
    for f in files:
        with xr.open_dataset(f, decode_timedelta=False) as ds:
            s = ds["salinity"].isel(time=0, depth=0).load()
        acc = s if acc is None else acc + s
    sss = acc / len(files)
    lon = sss["lon"].values
    return sss.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")


def cells_from_profile(vbc: np.ndarray, e3t: np.ndarray):
    q = vbc * e3t
    psi = np.concatenate([[0.0], np.cumsum(q)]) / 1e6
    zw = np.concatenate([[0.0], np.cumsum(e3t)])
    kmax = int(np.argmax(psi))
    i = next(i for i in range(kmax + 1, len(psi) - 1) if psi[i] < 0)
    z0 = zw[i - 1] + psi[i - 1] / (psi[i - 1] - psi[i]) * (zw[i] - zw[i - 1])
    return psi, zw, zw[kmax], z0


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

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

    fig = plt.figure(figsize=(TWO_COL, 5.3))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[0.82, 1.18],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.30,
    )

    # a: orthographic map ----------------------------------------------------
    proj = ccrs.Orthographic(central_longitude=-18, central_latitude=-24)
    axm = fig.add_subplot(gs[0, 0], projection=proj)
    axm.set_global()
    sss = en4_sss_mean()
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
    axm.plot(
        np.linspace(-54, 19.4, 100),
        np.full(100, lat0),
        color="white",
        lw=2.6,
        transform=pc,
        zorder=3,
        solid_capstyle="butt",
    )
    axm.plot(
        np.linspace(-54, 19.4, 100),
        np.full(100, lat0),
        color="black",
        lw=1.2,
        transform=pc,
        zorder=4,
        solid_capstyle="butt",
    )
    for x in [b[0] for b in BANDS] + [BANDS[-1][1]]:
        axm.plot(
            [x, x],
            [lat0 - 2.0, lat0 + 2.0],
            color="black",
            lw=0.7,
            transform=pc,
            zorder=4,
        )
    for (lo, hi), lab in zip(BANDS, BAND_LABELS, strict=False):
        axm.text(
            0.5 * (lo + hi),
            lat0 + 3.0,
            lab,
            ha="center",
            va="bottom",
            fontsize=5.5,
            transform=pc,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
        )
    axm.plot(
        row[rio].mean(),
        lat0,
        "o",
        ms=2.6,
        mfc="white",
        mec="black",
        mew=0.4,
        transform=pc,
        zorder=6,
    )
    cax = axm.inset_axes([0.18, -0.07, 0.64, 0.035])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=[34, 35, 36, 37])
    cb.set_label("Surface salinity (PSU)", fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5.5)
    panel_letter(axm, "a", x=-0.02, y=1.0)

    # b: salinity section --------------------------------------------------------
    axs = fig.add_subplot(gs[0, 1])
    slev = np.array(
        [
            34.2,
            34.3,
            34.4,
            34.5,
            34.6,
            34.7,
            34.8,
            34.9,
            35.0,
            35.25,
            35.5,
            35.75,
            36.0,
            36.3,
        ]
    )
    scmap = plt.get_cmap("cividis", len(slev) - 1)
    cs = axs.contourf(
        lon,
        depth,
        smean,
        levels=slev,
        cmap=scmap,
        norm=colors.BoundaryNorm(slev, scmap.N),
        extend="both",
    )
    axs.contour(
        lon,
        depth,
        smean,
        levels=[34.5, 34.9, 35.5],
        colors="white",
        linewidths=0.3,
        alpha=0.8,
    )
    for z, ls in ((z_max, ":"), (z0, "--")):
        axs.axhline(z, color="white", lw=0.9, ls=ls)
    depth_axis(axs)
    axs.set_xlim(lon.min(), lon.max())
    axs.set_facecolor("#4d4d4d")
    axs.set_xlabel("Longitude (degrees east)")
    for name, x, z, c in (
        ("SACW", -25, 160, "black"),
        ("AAIW", -25, 800, "white"),
        ("NADW", -25, 2450, "black"),
        ("AABW", -38, 4700, "white"),
    ):
        axs.text(
            x,
            z,
            name,
            ha="center",
            va="center",
            fontsize=5.5,
            color=c,
            fontweight="bold",
            path_effects=[
                pe.withStroke(
                    linewidth=1.6, foreground="white" if c == "black" else "black"
                )
            ],
        )
    for zlab, txt in ((z0 - 380, "upper cell above"), (z0 + 420, "abyssal cell below")):
        axs.text(
            8.0,
            zlab,
            txt,
            ha="center",
            va="center",
            fontsize=5.2,
            color="white",
            style="italic",
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
        )
    cb2 = fig.colorbar(
        cs,
        ax=axs,
        pad=0.02,
        fraction=0.035,
        ticks=[34.2, 34.4, 34.6, 34.8, 35.0, 35.5, 36.0],
    )
    cb2.set_label("Salinity (PSU)", fontsize=6)
    cb2.ax.tick_params(labelsize=5.5)
    panel_letter(axs, "b", x=-0.10, y=1.02)

    # c: profiles ------------------------------------------------------------
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0], wspace=0.14)
    ax1 = fig.add_subplot(sub[0])
    v_sv = vbc / 1e3  # m2 s-1 -> Sv per km of depth
    ax1.fill_betweenx(depth, 0, v_sv, where=v_sv > 0, color=C_NORTH, alpha=0.8, lw=0)
    ax1.fill_betweenx(depth, 0, v_sv, where=v_sv < 0, color=C_SOUTH, alpha=0.8, lw=0)
    ax1.plot(v_sv, depth, color="black", lw=0.5)
    ax1.axvline(0, color="black", lw=0.4)
    ax1.set_xlabel("$V_{bc}$ (Sv km$^{-1}$)")
    ax2 = fig.add_subplot(sub[1], sharey=ax1)
    zf = np.linspace(0, zw[-1], 2000)
    pf = np.interp(zf, zw, psi)
    ax2.fill_betweenx(zf, 0, pf, where=zf <= z0, color="#bdbdbd", lw=0)
    ax2.fill_betweenx(zf, 0, pf, where=zf >= z0, color="#636363", lw=0)
    ax2.plot(psi, zw, color="black", lw=0.8)
    ax2.axvline(0, color="black", lw=0.4)
    ax2.set_xlabel("$\\psi$ (Sv)")
    ax3 = fig.add_subplot(sub[2], sharey=ax1)
    ax3.plot(sbar, depth, color="black", lw=0.8)
    ax3.axvline(s_n, color=C_NORTH, lw=0.9, ls="-")
    ax3.axvline(s_s, color=C_SOUTH, lw=0.9, ls="--")
    ax3.set_xlabel("$\\bar{S}$ (PSU)")
    ax3.set_xlim(34.25, 35.8)
    ax3.set_xticks([34.4, 35.0, 35.6])
    for k, ax in enumerate((ax1, ax2, ax3)):
        for z, ls in ((z_max, ":"), (z0, "--")):
            ax.axhline(z, color="black", lw=0.5, ls=ls)
        depth_axis(ax, label=(k == 0))
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
    panel_letter(ax1, "c", x=-0.62, y=1.02)

    # d: the plane of the two factors ------------------------------------------
    import json

    from paper3v2_style import C_ECCO, C_GLORYS, C_ORAS5

    att = json.loads(
        (REPO / "PAPER_3_v2" / "analysis" / "attribution.json").read_text()
    )
    axd = fig.add_subplot(gs[1, 1])
    s0 = 35.0
    tt, dd = np.meshgrid(np.linspace(12.5, 25.0, 300), np.linspace(-0.01, 0.37, 300))
    ff = -tt * dd / s0
    lev = np.array([-0.30, -0.20, -0.15, -0.10, -0.075, -0.05, -0.025, 0.0])
    axd.contourf(tt, dd, ff, levels=lev, cmap="Greys_r", alpha=0.45, extend="both")
    cs = axd.contour(tt, dd, ff, levels=lev[:-1], colors="0.35", linewidths=0.4)
    # Contour labels in a column at the right edge, clear of the paths.
    x_lab = 24.45
    axd.clabel(
        cs,
        fmt=lambda v: f"{v:+.3g}",
        fontsize=5.0,
        inline_spacing=2,
        manual=[(x_lab, -v * s0 / x_lab) for v in lev[:-1] if -v * s0 / x_lab < 0.36],
    )
    axd.text(
        24.9,
        0.362,
        "$F_{\\mathrm{ovS}}$ (Sv)",
        fontsize=5.5,
        color="0.25",
        ha="right",
        va="top",
    )

    def blocks(years, values, width=5):
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

    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    # (start label offset and alignment, end label offset and alignment)
    places = {
        "oras5": ((-0.35, 0.0, "right", "center"), (-0.35, 0.0, "right", "center")),
        "glorys12": ((0.35, -0.004, "left", "center"), (0.2, -0.035, "left", "center")),
        "ecco": ((-0.35, 0.0, "right", "center"), (0.35, 0.006, "left", "center")),
    }
    handles = []
    for key, label, col in (
        ("oras5", "ORAS5", C_ORAS5),
        ("glorys12", "GLORYS12V1", C_GLORYS),
        ("ecco", "ECCO-V4r4", C_ECCO),
    ):
        r = att[key]
        tv, y0 = blocks(r["years"], np.array(r["T_limb_Sv"]))
        dv, _ = blocks(r["years"], np.array(r["dS_limb_PSU"]))
        (h,) = axd.plot(tv, dv, "-o", color=col, lw=1.2, ms=2.4, zorder=3, label=label)
        handles.append(h)
        axd.plot(tv[0], dv[0], "o", mfc="white", mec=col, mew=0.9, ms=4.0, zorder=5)
        axd.annotate(
            "",
            xy=(tv[-1], dv[-1]),
            xytext=(tv[-2], dv[-2]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": col,
                "lw": 1.2,
                "mutation_scale": 8,
            },
            zorder=5,
        )
        (sx, sy, sha, sva), (ex, ey, eha, eva) = places[key]
        axd.text(
            tv[0] + sx,
            dv[0] + sy,
            f"{y0[0]}",
            color=col,
            fontsize=5.5,
            ha=sha,
            va=sva,
            path_effects=halo,
            zorder=6,
        )
        axd.text(
            tv[-1] + ex,
            dv[-1] + ey,
            f"{r['years'][-1]}",
            color=col,
            fontsize=5.5,
            ha=eha,
            va=eva,
            path_effects=halo,
            zorder=6,
        )
    axd.legend(
        handles=handles,
        loc="lower left",
        fontsize=5.5,
        handlelength=1.6,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        title="5-year means",
        title_fontsize=5.5,
    )
    # The two routes to a more negative F_ovS.
    ox, oy = 17.2, 0.245
    axd.annotate(
        "",
        xy=(ox + 2.6, oy),
        xytext=(ox, oy),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 0.9,
            "mutation_scale": 7,
        },
    )
    axd.annotate(
        "",
        xy=(ox, oy + 0.07),
        xytext=(ox, oy),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 0.9,
            "mutation_scale": 7,
        },
    )
    axd.text(
        ox + 1.3,
        oy - 0.008,
        "stronger overturning",
        fontsize=5.5,
        ha="center",
        va="top",
        path_effects=halo,
    )
    axd.text(
        ox + 0.2,
        oy + 0.05,
        "wider salinity\ncontrast",
        fontsize=5.5,
        ha="left",
        va="center",
        linespacing=1.1,
        path_effects=halo,
    )
    axd.text(
        ox + 1.3,
        oy + 0.088,
        "two routes to a more negative $F_{\\mathrm{ovS}}$",
        fontsize=5.5,
        ha="center",
        va="bottom",
        style="italic",
        path_effects=halo,
    )
    axd.set_xlim(12.5, 25.0)
    axd.set_ylim(-0.01, 0.37)
    axd.set_xlabel("Northward-limb transport $T$ (Sv)")
    axd.set_ylabel("Limb salinity contrast $\\Delta S$ (PSU)")
    panel_letter(axd, "d", x=-0.13, y=1.02)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"Fig1_schematic.{ext}", bbox_inches="tight", dpi=600)
    print(f"z_psimax {z_max:.0f} m  z0 {z0:.0f} m  S_N {s_n:.3f}  S_S {s_s:.3f}")


if __name__ == "__main__":
    main()
