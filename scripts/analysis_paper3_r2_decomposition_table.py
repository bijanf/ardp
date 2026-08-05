#!/usr/bin/env python3
"""PAPER_3 round-2 WP1: companion table of every F_ovS decomposition.

Reads the canonical decomposition NetCDFs in data/results/ (plain names only,
never *.preBT / *OLD_BUGGY*) and emits one markdown table plus the same
content as JSON into revision/rev_papaer3_02/results/.

Rows are product + window. Shares are 100 * component / delta_total, so they
are signed and need not lie in [0, 100]: the three components sum to the total
by construction, but a component opposing the total gives a negative share.
"""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "data" / "results"
OUT_DIR = REPO / "revision" / "rev_papaer3_02" / "results"

SV_TO_MSV = 1e3

# (label, filename). Order is the table order.
SOURCES: list[tuple[str, str]] = [
    ("ORAS5 (pre-registered)", "fovs_decomposition_oras5.nc"),
    ("ORAS5 (paper3 epochs)", "fovs_decomposition_oras5_paper3_epochs.nc"),
    ("GLORYS12 (pre-registered)", "fovs_decomposition_glorys12.nc"),
    ("GLORYS12 (halves)", "fovs_decomposition_glorys12_paper3_halves.nc"),
    ("ECCO-V4r4", "fovs_decomposition_ecco.nc"),
    ("SODA 3.15.2", "fovs_decomposition_soda.nc"),
]


def read_one(label: str, fname: str) -> dict:
    path = RESULTS_DIR / fname
    ds = xr.open_dataset(path)
    a = ds.attrs
    ds.close()

    total = float(a["delta_total_Sv"]) * SV_TO_MSV
    d_v = float(a["delta_v_Sv"]) * SV_TO_MSV
    d_s = float(a["delta_s_Sv"]) * SV_TO_MSV
    d_x = float(a["delta_cross_Sv"]) * SV_TO_MSV

    def share(x: float) -> float | None:
        return 100.0 * x / total if total != 0 else None

    return {
        "label": label,
        "file": fname,
        "product": str(a["product"]),
        "early_period": str(a["early_period"]),
        "late_period": str(a["late_period"]),
        "section_latitude": float(a["section_latitude"]),
        "reference_salinity_PSU": float(a["reference_salinity_PSU"]),
        "F_ov_early_mSv": float(a["F_ov_early_Sv"]) * SV_TO_MSV,
        "F_ov_late_mSv": float(a["F_ov_late_Sv"]) * SV_TO_MSV,
        "delta_total_mSv": total,
        "delta_v_mSv": d_v,
        "delta_s_mSv": d_s,
        "delta_cross_mSv": d_x,
        "v_share_pct": share(d_v),
        "s_share_pct": share(d_s),
        "cross_share_pct": share(d_x),
        "residual_mSv": float(a["residual_Sv"]) * SV_TO_MSV,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [read_one(label, fname) for label, fname in SOURCES]

    (OUT_DIR / "WP1_decomposition_table.json").write_text(json.dumps(rows, indent=2))

    header = (
        "| Product (window) | Early | Late | dTotal (mSv) | dF_v (mSv) | "
        "dF_s (mSv) | dF_cross (mSv) | v share (%) | s share (%) | "
        "F_ov early (mSv) | F_ov late (mSv) |"
    )
    sep = "|" + "---|" * 11
    lines = [
        "# WP1. F_ovS decomposition companion table",
        "",
        "All values from the canonical decomposition NetCDFs in `data/results/`",
        "(plain names only). Sv converted to mSv by x1000. Shares are",
        "`100 * component / dTotal` and are signed: a component opposing the total",
        "gives a negative share, and the three shares (v, s, cross) sum to 100 %",
        "by construction. Reference salinity is 35.0 PSU for every row.",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['early_period']} | {r['late_period']} | "
            f"{r['delta_total_mSv']:+.1f} | {r['delta_v_mSv']:+.1f} | "
            f"{r['delta_s_mSv']:+.1f} | {r['delta_cross_mSv']:+.1f} | "
            f"{r['v_share_pct']:+.1f} | {r['s_share_pct']:+.1f} | "
            f"{r['F_ov_early_mSv']:+.1f} | {r['F_ov_late_mSv']:+.1f} |"
        )

    lines += [
        "",
        "## Provenance",
        "",
        "| Row | Source file | Section latitude | Cross share (%) | Residual (mSv) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | `{r['file']}` | {r['section_latitude']:.2f} | "
            f"{r['cross_share_pct']:+.1f} | {r['residual_mSv']:.2e} |"
        )
    lines.append("")

    (OUT_DIR / "WP1_decomposition_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
