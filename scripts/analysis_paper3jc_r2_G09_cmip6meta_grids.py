#!/usr/bin/env python3
"""CMIP6 ocean grid labels of the 25 models of the section set (round-2 group G09).

The section files in data/cmip6_sections carry no grid label. For each model,
this script asks the ESGF search service for the Omon vo files of the
historical experiment and reads the grid label (gn, gr, ...) from the file
identifier. The member is not recorded in the section files, so the label is
that of the first file returned for the model; when a model publishes several
labels the set of labels is recorded as well.

Inputs: data/results/fovs_decomposition_cmip6_summary.csv (model list); the
ESGF search service at esgf-node.ornl.gov (read only, one query per model).
Output: PAPER_3_v2/analysis/jc_r2_G09_cmip6meta_grids.json.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "results"
OUT = REPO / "PAPER_3_v2" / "analysis" / "jc_r2_G09_cmip6meta_grids.json"
BASE = "https://esgf-node.ornl.gov/esgf-1-5-bridge/"
PATTERN = re.compile(r"\.Omon\.vo\.([A-Za-z0-9]+)\.")


def grid_labels(model: str) -> dict:
    query = {
        "type": "File",
        "source_id": model,
        "variable_id": "vo",
        "experiment_id": "historical",
        "table_id": "Omon",
        "limit": "20",
        "format": "application/solr+json",
    }
    url = BASE + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=180) as resp:
        doc = json.loads(resp.read().decode())
    labels = []
    members = []
    for d in doc["response"]["docs"]:
        m = PATTERN.search(d.get("master_id", ""))
        if m:
            labels.append(m.group(1))
            mem = re.search(r"historical\.([^.]+)\.Omon", d.get("master_id", ""))
            members.append(mem.group(1) if mem else None)
    if not labels:
        return {"grid_label": None, "labels_seen": [], "members_seen": []}
    return {
        "grid_label": labels[0],
        "member_first_file": members[0],
        "labels_seen": sorted(set(labels)),
        "members_seen": sorted(set(m for m in members if m)),
        "n_files_returned": len(labels),
    }


def main() -> None:
    models = pd.read_csv(RESULTS / "fovs_decomposition_cmip6_summary.csv")["model"].tolist()
    out = {}
    for model in models:
        try:
            out[model] = grid_labels(model)
        except Exception as exc:  # record the failure and continue
            out[model] = {"grid_label": None, "error": repr(exc)}
        print(model, out[model].get("grid_label"), flush=True)
    payload = {
        "meta": {
            "date": "2026-11-05",
            "group": "G09_cmip6meta",
            "actions": ["A84 (grid_name)"],
            "script": "scripts/analysis_paper3jc_r2_G09_cmip6meta_grids.py",
            "inputs": [
                "data/results/fovs_decomposition_cmip6_summary.csv (model list)",
                "ESGF search service esgf-node.ornl.gov (read only, Omon vo historical)",
            ],
            "seed": None,
            "sieve_resamples": None,
            "replicates": {"queries": len(models)},
            "units": {},
            "note": "The member of the section files is not recorded; the label is that of the first file returned.",
        },
        "grids": out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)


if __name__ == "__main__":
    main()
