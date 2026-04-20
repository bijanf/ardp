#!/usr/bin/env python3
"""Table S1: F_ovS linear trends × 3 methods × 4 reanalyses.

Produces both a CSV and a publication-ready LaTeX tabular that can be
pasted into the manuscript supplementary materials.

Methods: naive OLS, Santer et al. (2000) N_eff, GLS Prais-Winsten AR(1).
Reads each reanalysis's F_ovS time series and computes the three trends.

Output:
  data/results/paper2_tableS1_trends.csv
  data/results/paper2_tableS1_trends.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amoc_reanalysis_anomalies import (  # noqa: E402
    _linear_trend_gls,
    _linear_trend_ols,
    _linear_trend_santer,
)

PRODUCTS = [
    ("ORAS5",      "oras5_f_ovs.nc"),
    ("GLORYS12V1", "glorys12_f_ovs.nc"),
    ("SODA3.15.2", "soda_f_ovs.nc"),
    ("ECCO-V4r4",  "ecco_f_ovs.nc"),
]


def _load_annual(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    var = "F_ovS" if "F_ovS" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]
    if "year" in da.dims:
        years = da["year"].values.astype(float)
        vals = da.values
    elif "time" in da.dims:
        t_year = da["time"].dt.year.values
        uniq = np.unique(t_year)
        vals = np.array([np.nanmean(da.values[t_year == y]) for y in uniq])
        years = uniq.astype(float)
    else:
        ds.close()
        return None
    ds.close()
    return years, vals


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_cell(slope_sv_dec: float, p: float, neff: int) -> str:
    """Pretty-format a trend cell: slope ± (p-value formatted) [N_eff]."""
    slope_mSv_yr = slope_sv_dec * 100  # Sv/decade → mSv/yr
    star = _stars(p)
    return f"{slope_mSv_yr:+.2f}{star}  (N_eff={neff})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output-csv", type=Path,
                        default=Path("data/results/paper2_tableS1_trends.csv"))
    parser.add_argument("--output-tex", type=Path,
                        default=Path("data/results/paper2_tableS1_trends.tex"))
    args = parser.parse_args()

    rows = []
    for label, fname in PRODUCTS:
        ts = _load_annual(args.results_dir / fname)
        if ts is None:
            print(f"  {label}: no file, skipping")
            continue
        yrs, vals = ts
        sl_ols, p_ols, n_ols = _linear_trend_ols(yrs, vals)
        sl_san, p_san, neff_san = _linear_trend_santer(yrs, vals)
        sl_gls, p_gls, neff_gls = _linear_trend_gls(yrs, vals)

        rows.append({
            "product": label,
            "period": f"{int(yrs[0])}–{int(yrs[-1])}",
            "N": len(yrs),
            "mean_Sv": float(np.mean(vals)),
            "std_Sv": float(np.std(vals)),
            "ols_trend_mSv_yr": sl_ols * 100,
            "ols_p": p_ols,
            "santer_trend_mSv_yr": sl_san * 100,
            "santer_p": p_san,
            "santer_Neff": neff_san,
            "gls_trend_mSv_yr": sl_gls * 100,
            "gls_p": p_gls,
            "ols_cell": _fmt_cell(sl_ols, p_ols, n_ols),
            "santer_cell": _fmt_cell(sl_san, p_san, neff_san),
            "gls_cell": _fmt_cell(sl_gls, p_gls, neff_gls),
        })
        print(f"  {label:<12s}  mean={float(np.mean(vals)):+.4f}  "
              f"OLS={rows[-1]['ols_cell']}  "
              f"Santer={rows[-1]['santer_cell']}  "
              f"GLS={rows[-1]['gls_cell']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")

    # LaTeX output
    def _tex_escape(s: str) -> str:
        return s.replace("±", r"$\pm$").replace("–", "--")

    tex_lines = [
        r"\begin{table}[h]",
        r"\caption{\textbf{Table S1.} Linear trends of F\textsubscript{\emph{ovS}} at 34.5\degree S across four ocean reanalyses under three significance methods: naive ordinary least squares (OLS), Santer et al.\ (2000) effective-sample-size correction, and generalised least squares with Prais-Winsten AR(1) errors. Trends are given in mSv yr$^{-1}$; starred values are significant at $p<0.05$ (\textasteriskcentered) or $p<0.01$ (\textasteriskcentered\textasteriskcentered). $N_\mathrm{eff}$ is the Santer / GLS effective sample size.}",
        r"\label{tab:S1_trends}",
        r"\centering",
        r"\begin{tabular}{llrrlll}",
        r"\hline",
        r"Product & Period & N & Mean (Sv) & OLS & Santer & GLS \\",
        r"\hline",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['product']} & {r['period']} & {r['N']} & "
            f"{r['mean_Sv']:+.3f} & "
            f"{_tex_escape(r['ols_cell'])} & "
            f"{_tex_escape(r['santer_cell'])} & "
            f"{_tex_escape(r['gls_cell'])} \\\\"
        )
    tex_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    args.output_tex.write_text("\n".join(tex_lines))
    print(f"Saved: {args.output_tex}")


if __name__ == "__main__":
    main()
