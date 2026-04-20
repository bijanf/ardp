#!/usr/bin/env python3
"""Table S2: net-volume-transport diagnostics per product and per window.

The barotropic subtraction in the F_ovS kernel removes a
spurious signal proportional to the net volume transport V_net
across the 34.5 S section. This table reports V_net per product
for the main (1993-2005 vs 2013-2025-ish) and post-Argo
(2006-2012 vs 2018-2024-ish) decomposition windows, plus the
implied uncorrected-vs-corrected F_ov shift, so readers can see
how large the artefact would have been.

Reads:  data/results/fovs_decomposition_{product}{,_postargo}.nc
Writes: paper2/tableS2.tex  (LaTeX booktabs table)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


PRODUCTS = [
    ("ORAS5",      "oras5"),
    ("GLORYS12V1", "glorys12"),
    ("SODA 3.15.2", "soda"),
    ("ECCO-V4r4",  "ecco"),
]


def _load(results_dir: Path, key: str, tag: str = "") -> dict | None:
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"fovs_decomposition_{key}{suffix}.nc"
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    out = {
        "early": ds.attrs["early_period"],
        "late": ds.attrs["late_period"],
        "V_net_early": float(ds.attrs.get("V_net_early_Sv", float("nan"))),
        "V_net_late": float(ds.attrs.get("V_net_late_Sv", float("nan"))),
        "v_bar_early": float(ds.attrs.get("v_bar_early_ms", float("nan"))),
        "v_bar_late": float(ds.attrs.get("v_bar_late_ms", float("nan"))),
    }
    ds.close()
    return out


def _fmt(val: float) -> str:
    if val != val:                       # NaN
        return r"\textendash"
    return f"${val:+.2f}$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("paper2/tableS2.tex"))
    args = parser.parse_args()

    rows = []
    for label, key in PRODUCTS:
        m = _load(args.results_dir, key)
        p = _load(args.results_dir, key, "postargo")
        rows.append((label, m, p))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Net volume transport ($V_\mathrm{net}$) across "
        r"34.5\textdegree S per product and per decomposition window.} "
        r"$V_\mathrm{net} = \int V_\mathrm{int}(z)\,\mathrm{d}z$ is the "
        r"quantity the barotropic subtraction in our $\Fovs$ kernel removes; "
        r"its magnitude indicates how strongly the product violates "
        r"depth-integrated mass conservation over the section. Without the "
        r"subtraction a $|\Delta V_\mathrm{net}|$ drift of 1~Sv between "
        r"periods would inject approximately "
        r"$|\Delta V_\mathrm{net}|\cdot|\bar{S}-S_0|/S_0 \approx 14$~mSv of "
        r"spurious signal into $\Delta\Fovs$. All values reported here are "
        r"the period-mean $V_\mathrm{net}$ in Sv after the barotropic "
        r"correction was NOT yet applied to the period average (diagnostic "
        r"only; the F_ov computation itself is barotropic-corrected "
        r"throughout).}",
        r"\label{tab:vnet}",
        r"\small",
        r"\begin{tabular}{l|rr|rr}",
        r"\hline\hline",
        r" & \multicolumn{2}{c|}{Main window [Sv]}"
        r" & \multicolumn{2}{c}{Post-Argo window [Sv]} \\",
        r"Product & $V_\mathrm{net}$ early & $V_\mathrm{net}$ late"
        r" & $V_\mathrm{net}$ early & $V_\mathrm{net}$ late \\",
        r"\hline",
    ]
    for label, m, p in rows:
        vm0 = _fmt(m["V_net_early"]) if m else r"\textendash"
        vm1 = _fmt(m["V_net_late"]) if m else r"\textendash"
        vp0 = _fmt(p["V_net_early"]) if p else r"\textendash"
        vp1 = _fmt(p["V_net_late"]) if p else r"\textendash"
        lines.append(
            rf"{label} & {vm0} & {vm1} & {vp0} & {vp1} \\"
        )
    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")

    # Console summary
    print("\nV_net diagnostics per product:")
    print(f"{'Product':<14s} {'m_early':>9s} {'m_late':>9s}"
          f" {'p_early':>9s} {'p_late':>9s}")
    for label, m, p in rows:
        vme = f"{m['V_net_early']:+.2f}" if m else "--"
        vml = f"{m['V_net_late']:+.2f}" if m else "--"
        vpe = f"{p['V_net_early']:+.2f}" if p else "--"
        vpl = f"{p['V_net_late']:+.2f}" if p else "--"
        print(f"{label:<14s} {vme:>9s} {vml:>9s} {vpe:>9s} {vpl:>9s}")


if __name__ == "__main__":
    main()
