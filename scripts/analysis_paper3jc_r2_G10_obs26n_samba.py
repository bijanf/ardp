#!/usr/bin/env python3
"""SAMBA layer means against the reanalysis layer means over the SAMBA record (action A106, item 3; cluster C54).

What the script does
  1. Attempts once to retrieve the SAMBA transport file from the NOAA FTP site
     (the same file and host as scripts/download_samba.py), with a 30 s timeout,
     into a scratch directory. The outcome is recorded in the JSON. Nothing is
     written to data/.
  2. Reads the SAMBA monthly and daily files in data/external
     (samba_moc_monthly.nc, samba_moc_daily.nc) and reports their means, standard
     deviations and ranges. The files hold MOC transport anomalies (the Kersale et al.
     2020 transport anomaly file), so their time means are zero by construction.
     This is the check that decides whether a SAMBA absolute layer mean exists in
     the repository.
  3. Reads the reanalysis layer means over the 47 common months (September 2013 to
     July 2017) from PAPER_3_v2/analysis/jc_n10_samba.json (keys
     reanalysis_only_common_window and pairs). These are absolute transports, the
     values integrated by scripts/analysis_paper3jc_n10_samba.py. They are copied, not
     recomputed.
  4. Writes the SAMBA-minus-reanalysis mean differences as null with the reason,
     and the anomaly standard deviations and their ratios, which can be compared.

No random draws are made. Writes PAPER_3_v2/analysis/jc_r2_G10_obs26n_samba.json.
"""

from __future__ import annotations

import ftplib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "data" / "external"
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
OUT = ANALYSIS / "jc_r2_G10_obs26n_samba.json"

FTP_HOST = "ftp.aoml.noaa.gov"
FTP_DIR = "/phod/pub/SAM/2020_Kersale_etal_ScienceAdvances/"
TRANSPORT_FILE = "Upper_Abyssal_Transport_Anomalies.txt"


def try_download_once() -> dict:
    """One attempt, into a scratch directory; the outcome is returned for the record."""
    scratch = Path(tempfile.mkdtemp(prefix="samba_try_"))
    target = scratch / TRANSPORT_FILE
    try:
        ftp = ftplib.FTP(FTP_HOST, timeout=30)
        ftp.login()
        ftp.cwd(FTP_DIR)
        names = ftp.nlst()
        with open(target, "wb") as fh:
            ftp.retrbinary(f"RETR {TRANSPORT_FILE}", fh.write)
        ftp.quit()
        return {
            "attempted": True,
            "host": FTP_HOST,
            "directory": FTP_DIR,
            "outcome": "retrieved",
            "listing_length": len(names),
            "bytes": int(target.stat().st_size),
            "scratch_dir_note": "file kept in a scratch directory, not in data/",
        }
    except Exception as exc:  # network, timeout and protocol errors are all recorded
        return {
            "attempted": True,
            "host": FTP_HOST,
            "directory": FTP_DIR,
            "outcome": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "note": "No SAMBA data retrieved; the repository files are used as they are.",
        }


def summarise(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "n": int(x.size),
        "mean_Sv": float(x.mean()),
        "std_Sv": float(x.std(ddof=1)),
        "min_Sv": float(x.min()),
        "max_Sv": float(x.max()),
    }


def main() -> None:
    download = try_download_once()

    monthly = xr.open_dataset(EXTERNAL / "samba_moc_monthly.nc")
    daily = xr.open_dataset(EXTERNAL / "samba_moc_daily.nc")
    m_up = monthly["upper_cell"].values
    m_ab = monthly["abyssal_cell"].values
    d_up = daily["upper_cell"].values
    d_ab = daily["abyssal_cell"].values

    samba_files = {
        "monthly_span": [
            str(np.datetime_as_string(monthly["time"].values[0], unit="M")),
            str(np.datetime_as_string(monthly["time"].values[-1], unit="M")),
        ],
        "monthly_attrs_description": str(monthly.attrs.get("description", "")),
        "upper_cell_monthly": summarise(m_up),
        "abyssal_cell_monthly": summarise(m_ab),
        "upper_cell_daily": summarise(d_up),
        "abyssal_cell_daily": summarise(d_ab),
        "is_anomaly_file": bool(abs(float(np.nanmean(m_up))) < 0.5 and abs(float(np.nanmean(m_ab))) < 0.5),
        "interpretation": (
            "Time means of the stored SAMBA series are near zero, so the files hold transport "
            "anomalies, not absolute layer transports. A SAMBA layer mean is therefore not "
            "available in the repository."
        ),
    }

    jc10 = json.loads((ANALYSIS / "jc_n10_samba.json").read_text())
    window = jc10["reanalysis_only_common_window"]
    layer_means = {}
    for prod in ("ORAS5", "GLORYS12V1", "ECCO-V4r4"):
        if prod in window:
            layer_means[prod] = {
                key: {"raw_mean_Sv": float(val["raw_mean_Sv"]), "raw_std_Sv": float(val["raw_std_Sv"])}
                for key, val in window[prod].items()
                if isinstance(val, dict) and "raw_mean_Sv" in val
            }

    pair_stats = {}
    for prod, pairs in jc10["pairs"].items():
        pair_stats[prod] = {}
        for key, val in pairs.items():
            pair_stats[prod][key] = {
                "label": val.get("label"),
                "raw_mean_reanalysis_Sv": val.get("raw_mean_reanalysis_Sv"),
                "raw_mean_samba_stored_Sv": val.get("raw_mean_samba_stored_Sv"),
                "std_reanalysis_anomaly_Sv": val.get("std_x"),
                "std_samba_anomaly_Sv": val.get("std_y"),
                "std_ratio_reanalysis_over_samba": val.get("std_ratio_x_over_y"),
                "std_ratio_ci_95": val.get("std_ratio_ci_95"),
                "r": val.get("r"),
                "n_eff": val.get("n_eff"),
                "p": val.get("p"),
            }

    comparison = {}
    for prod in pair_stats:
        comparison[prod] = {}
        for key in ("up_cell__samba_up", "up_layer__samba_up", "ab_cell__samba_ab", "ab_layer__samba_ab"):
            entry = pair_stats[prod].get(key, {})
            comparison[prod][key] = {
                "reanalysis_mean_Sv": entry.get("raw_mean_reanalysis_Sv"),
                "samba_mean_Sv": None,
                "samba_minus_reanalysis_mean_Sv": None,
                "reason_null": (
                    "The repository holds SAMBA anomalies only (time mean of the stored series is "
                    "near zero); an absolute SAMBA layer mean cannot be formed."
                ),
                "std_ratio_reanalysis_over_samba": entry.get("std_ratio_reanalysis_over_samba"),
                "std_ratio_ci_95": entry.get("std_ratio_ci_95"),
                "r": entry.get("r"),
                "n_eff": entry.get("n_eff"),
            }

    result = {
        "meta": {
            "id": "jc_r2_G10_obs26n_samba",
            "group": "G10_obs26n",
            "actions": ["A106 item 3"],
            "cluster": "C54",
            "date": "2026-11-05",
            "script": "scripts/analysis_paper3jc_r2_G10_obs26n_samba.py",
            "inputs": [
                "data/external/samba_moc_monthly.nc",
                "data/external/samba_moc_daily.nc",
                "PAPER_3_v2/analysis/jc_n10_samba.json (reanalysis_only_common_window, pairs)",
                "ftp://ftp.aoml.noaa.gov/phod/pub/SAM/2020_Kersale_etal_ScienceAdvances/Upper_Abyssal_Transport_Anomalies.txt (one attempt)",
            ],
            "seeds": "none (no random draws)",
            "replicates": "none",
            "units": "Sv; correlations dimensionless; n_eff in months",
            "record": "September 2013 to July 2017, 47 months (common window)",
        },
        "download_attempt": download,
        "samba_files": samba_files,
        "reanalysis_layer_means_common_window_Sv": layer_means,
        "samba_vs_reanalysis_means": comparison,
        "anomaly_std_comparison_from_jc_n10": pair_stats,
    }

    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
