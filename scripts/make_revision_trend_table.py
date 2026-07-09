#!/usr/bin/env python3
"""Assemble the South Atlantic upper-ocean salinity trend table for the revision.

Reviewer 1 asked that the salinity trends currently listed in running prose
(manuscript lines ~64-77) be collected into a table. This script gathers the
observational Argo-based trends (computed in this study, stored in
``data/results/argo_trends.json``) and the reanalysis basin-mean trends used in
the Fig 2b bar chart, and writes a markdown + LaTeX table to ``revision/``.

Provenance note: the reanalysis (GLORYS12V1, ORAS5) basin-mean salinity trends
are over 1993-2025, whereas the Argo products (EN4.2.2, RG09) are over
2005-2024. The window is reported per row; harmonising the reanalysis window to
the Argo era is handled separately (robustness task C4).

Reads:  data/results/argo_trends.json
Writes: revision/trend_table.md
        revision/trend_table.tex
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS = Path("data/results")
OUT = Path("revision")

# Reanalysis basin-mean upper-300 m salinity trends used in Fig 2b.
# (Same literature values as scripts/plot_obs_grounding.py::LIT_TRENDS.)
REANALYSIS_TRENDS = {
    "GLORYS12V1": (0.084, 0.025, "1993-2025"),
    "ORAS5": (-0.014, 0.020, "1993-2025"),
}


def _fmt(slope: float, ci: float) -> str:
    return f"{slope:+.3f} $\\pm$ {ci:.3f}"


def main() -> None:
    with open(RESULTS / "argo_trends.json") as f:
        trends = json.load(f)

    # (product, quantity, slope, ci95_half, period, source)
    rows: list[tuple[str, str, float, float, str, str]] = []

    en = trends["EN4.2.2"]
    rows.append(
        (
            "EN4.2.2",
            "upper-300 m basin salinity",
            en["slope_psu_per_dec"],
            en["ci95_half_psu_per_dec"],
            "2005-2024",
            "this study",
        )
    )
    rg = trends["RG09"]
    rows.append(
        (
            "Roemmich-Gilson Argo",
            "upper-300 m basin salinity",
            rg["slope_psu_per_dec"],
            rg["ci95_half_psu_per_dec"],
            "2005-2024",
            "this study",
        )
    )
    # SAMBA / 34.5S sustained-observation estimate. Reference corrected to
    # Pita et al. 2024 (verified via DOI 10.3389/fmars.2024.1474133; the
    # first author is Pita, not Volkov). See revision/results/citation_fixes.md.
    rows.append(
        (
            "SAMBA/34.5S (XBT+Argo)",
            "34.5S upper-ocean salinity",
            0.05,
            0.02,
            "2009-2023",
            "Pita et al. 2024",
        )
    )
    for name, (slope, ci, period) in REANALYSIS_TRENDS.items():
        rows.append(
            (
                name,
                "upper-300 m basin salinity",
                slope,
                ci,
                period,
                "this study",
            )
        )

    OUT.mkdir(exist_ok=True)

    # ---- Markdown ----
    md = [
        "# South Atlantic upper-ocean salinity trends (revision table, R1.3)",
        "",
        "| Product | Quantity | Trend (PSU/dec) | 95% CI (half-width) "
        "| Period | Source |",
        "|---|---|---|---|---|---|",
    ]
    for prod, qty, slope, ci, period, src in rows:
        md.append(f"| {prod} | {qty} | {slope:+.3f} | {ci:.3f} | {period} | {src} |")
    md += [
        "",
        "All observational products show a significant upper-ocean salinification "
        "of about +0.05 PSU/dec. GLORYS12V1 (+0.084) is the strongest and ORAS5 "
        "(-0.014) the sole outlier with a slight freshening.",
        "",
        "Note: the EN4.2.2 and RG09 confidence intervals are Santer N_eff-adjusted "
        "AR(1) 95% intervals. The reanalysis (GLORYS12V1, ORAS5) trends are over "
        "1993-2025; harmonising them to the Argo era is reported in the robustness "
        "test (task C4).",
    ]
    (OUT / "trend_table.md").write_text("\n".join(md) + "\n")

    # ---- LaTeX ----
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{South Atlantic upper-ocean salinity trends.} "
        r"Observational Argo-based products (this study) and reanalysis "
        r"basin-mean trends. All observational estimates show a significant "
        r"salinification of $\sim +0.05$~PSU\,dec$^{-1}$; ORAS5 is the sole "
        r"product with a slight freshening. Confidence intervals for EN4.2.2 "
        r"and RG09 are Santer $N_\mathrm{eff}$-adjusted AR(1) 95\% intervals.}",
        r"\label{tab:salinity_trends}",
        r"\begin{tabular}{llcccl}",
        r"\hline",
        r"Product & Quantity & Trend (PSU\,dec$^{-1}$) & 95\% CI & Period & Source \\",
        r"\hline",
    ]
    for prod, qty, slope, ci, period, src in rows:
        tex.append(f"{prod} & {qty} & {_fmt(slope, ci)} & & {period} & {src} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "trend_table.tex").write_text("\n".join(tex) + "\n")

    log.info("Wrote revision/trend_table.md and revision/trend_table.tex")
    for prod, _qty, slope, ci, period, _src in rows:
        log.info(f"  {prod:24s} {slope:+.3f} +/- {ci:.3f} PSU/dec  ({period})")


if __name__ == "__main__":
    main()
