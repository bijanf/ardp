#!/usr/bin/env python3
"""Section diagnostics at 34.5 S from the archived longitude-depth fields.

Reads ``data/results/paper3r1_section3d_<product>.nc`` and recomputes every
section quantity with an explicit cell volume, so that the representation of
the vertical grid and of the bottom topography can be varied.

For each month, with v and S the meridional velocity and salinity in the wet
cells of the contiguous Atlantic section, dx the zonal cell width and dz the
vertical cell thickness,

    q_k   = sum_x v dx dz                (volume transport of level k)
    a_k   = sum_x dx dz                  (wet area of level k)
    S_k   = sum_x S dx dz / a_k          (area-weighted zonal-mean salinity)
    v_c   = sum_k q_k / sum_k a_k        (uniform velocity carrying the net transport)
    q'_k  = q_k - v_c a_k                (level transport after its removal)

so that sum_k q'_k = 0 exactly. The overturning freshwater transport and its
factors are

    F_ovS = -(1/S0) sum_k q'_k (S_k - S0) = -(1/S0) T (S_N - S_S),
    T     = sum_k max(q'_k, 0),

and the azonal component is F_az = -(1/S0) sum_k sum_x (v - q_k/a_k)(S - S_k) dx dz.

Thickness conventions:

* ``bounds`` (baseline): dz from level bounds. ECCO distributes its bounds;
  for ORAS5 and GLORYS12V1, whose distributed files carry level-centre depths
  only, the bounds are reconstructed as the midpoints between level centres,
  with the surface at zero.
* ``centres``: dz as the spacing between consecutive level centres, the
  convention of the first version of this analysis for ORAS5 and GLORYS12V1.
* ``bottom<f>``: the bounds convention with the thickness of the deepest wet
  cell of every water column multiplied by f, which brackets the effect of
  partial bottom cells that the distributed fields do not carry.

Writes ``data/results/paper3r1_section_<product>[_<variant>].nc`` in the format
of the earlier section files, so that the downstream analyses can read either.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
S0 = 35.0
SURFACE_LAYER_M = 100.0
PRODUCTS = {"oras5": "ORAS5", "glorys12": "GLORYS12V1", "ecco": "ECCO-V4r4"}


def thickness(
    depth: np.ndarray, product: str, e3t_file: np.ndarray, how: str
) -> np.ndarray:
    if how == "centres":
        return np.diff(depth, prepend=0.0)
    if product == "ecco":
        return e3t_file  # true bounds, stored at extraction time
    zw = np.concatenate([[0.0], 0.5 * (depth[1:] + depth[:-1])])
    zw = np.append(zw, depth[-1] + 0.5 * (depth[-1] - depth[-2]))
    return np.diff(zw)


def section_series(
    v: np.ndarray,
    s: np.ndarray,
    e1t: np.ndarray,
    e3t: np.ndarray,
    kbot: np.ndarray,
    bottom_factor: float = 1.0,
) -> dict:
    """Vectorised section diagnostics for all months; v, s are (time, depth, x)."""
    wet = ~np.isnan(s) & ~np.isnan(v)
    dz = np.broadcast_to(e3t[:, None], wet.shape[1:]).copy()
    if bottom_factor != 1.0:
        cols = np.where(kbot >= 0)[0]
        dz[kbot[cols], cols] *= bottom_factor
    w = np.where(wet, (e1t[None, :] * dz)[None, :, :], 0.0)
    vv = np.where(wet, v, 0.0)
    ss = np.where(wet, s, 0.0)

    q = (vv * w).sum(axis=2)
    a = w.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sbar = np.where(a > 0, (ss * w).sum(axis=2) / a, 0.0)
        vmean = np.where(a > 0, q / a, 0.0)
    vp = np.where(wet, v - vmean[:, :, None], 0.0)
    sp = np.where(wet, s - sbar[:, :, None], 0.0)
    f_az = -(1.0 / S0) * (vp * sp * w).sum(axis=(1, 2)) / 1e6

    v_net = q.sum(axis=1)
    a_tot = a.sum(axis=1)
    v_c = v_net / a_tot
    q_bc = q - v_c[:, None] * a
    f_ov = -(1.0 / S0) * (q_bc * (sbar - S0)).sum(axis=1) / 1e6
    f_raw = -(1.0 / S0) * (q * (sbar - S0)).sum(axis=1) / 1e6

    pos = np.clip(q_bc, 0, None)
    neg = np.clip(-q_bc, 0, None)
    t_n = pos.sum(axis=1)
    s_n = (pos * sbar).sum(axis=1) / t_n
    s_s = (neg * sbar).sum(axis=1) / neg.sum(axis=1)

    psi = np.cumsum(q, axis=1) / 1e6
    edges = np.cumsum(e3t)
    below = edges >= 250.0
    psi250 = np.cumsum(np.where(below[None, :], q, 0.0), axis=1) / 1e6
    surf = edges <= SURFACE_LAYER_M
    f_surf = -(1.0 / S0) * (q_bc[:, surf] * (sbar[:, surf] - S0)).sum(axis=1) / 1e6

    return {
        "F_ov": f_ov,
        "F_ov_raw": f_raw,
        "F_az": f_az,
        "V_net_Sv": v_net / 1e6,
        "T_limb_Sv": t_n / 1e6,
        "S_north": s_n,
        "S_south": s_s,
        "dS_limb": s_n - s_s,
        "psi_max_surface_Sv": psi.max(axis=1),
        "psi_max_250m_Sv": psi250.max(axis=1),
        "F_ov_surface_layer": f_surf,
        "V_c_mm_per_s": v_c * 1e3,
        # Profiles per unit thickness, the format of the earlier section files.
        "V_int": q / e3t[None, :],
        "V_bc": q_bc / e3t[None, :],
        "A_xy": a / e3t[None, :],
        "S_bar": sbar,
    }


def build(product: str, how: str, bottom_factor: float = 1.0) -> xr.Dataset:
    f3 = xr.open_dataset(RESULTS / f"paper3r1_section3d_{product}.nc")
    depth = f3["depth"].values
    e3t = thickness(depth, product, f3["e3t"].values, how)
    out = section_series(
        f3["v"].values.astype(float),
        f3["s"].values.astype(float),
        f3["e1t"].values,
        e3t,
        f3["kbot"].values,
        bottom_factor,
    )
    profiles = ("V_int", "V_bc", "A_xy", "S_bar")
    ds = xr.Dataset(
        {k: ("time", v) for k, v in out.items() if k not in profiles},
        coords={"time": f3["time"].values, "depth": depth},
    )
    for k in profiles:
        ds[k] = (("time", "depth"), out[k])
    ds["e3t"] = ("depth", e3t)
    ds["e3t"].attrs["note"] = f"vertical cell thickness, convention '{how}'"
    ds.attrs.update(
        product=PRODUCTS[product],
        section_latitude=float(f3.attrs["section_latitude"]),
        n_atlantic_points=int(f3.attrs["n_atlantic_points"]),
        reference_salinity_PSU=S0,
        thickness_convention=how,
        bottom_cell_factor=bottom_factor,
        factorisation="F_ov = -(1/S0) * T_limb * dS_limb, exact, no residual",
    )
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="ecco,glorys12,oras5")
    args = ap.parse_args()
    variants = [
        ("", "bounds", 1.0),
        ("_centres", "centres", 1.0),
        ("_bottom0p5", "bounds", 0.5),
        ("_bottom0p2", "bounds", 0.2),
    ]
    for p in args.products.split(","):
        for suffix, how, f in variants:
            ds = build(p, how, f)
            out = RESULTS / f"paper3r1_section_{p}{suffix}.nc"
            ds.to_netcdf(out)
            m = {
                k: float(ds[k].mean())
                for k in ("F_ov", "T_limb_Sv", "dS_limb", "F_az", "V_net_Sv")
            }
            print(
                f"{p:9s}{suffix:12s} F_ov {m['F_ov']:+.4f}  T {m['T_limb_Sv']:6.2f}  "
                f"dS {m['dS_limb']:.4f}  F_az {m['F_az']:+.4f}  "
                f"V_net {m['V_net_Sv']:+.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
