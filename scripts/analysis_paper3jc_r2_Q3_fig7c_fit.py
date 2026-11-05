"""Slope and correlation of the 17 common CMIP6 models, as plotted in Fig. 7c.

The panel plots the change at 34.5 S (y) against the change at 26.5 N (x), so the
printed slope is the ordinary least-squares slope of y on x over the models with
both changes. The per-model values are read from r1_cmip6.json, the same table the
figure script reads. The reverse regression (x on y) is kept as a separate key so
that the two directions cannot be confused.
"""
import json
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "PAPER_3_v2" / "analysis"
rows = json.loads((ANALYSIS / "r1_cmip6.json").read_text())["per_model"]
sub = [r for r in rows
       if r.get("dpsi34") is not None and r.get("damoc26") is not None
       and np.isfinite(r["dpsi34"]) and np.isfinite(r["damoc26"])]
x = np.array([r["damoc26"] for r in sub], dtype=float)
y = np.array([r["dpsi34"] for r in sub], dtype=float)
slope_y_on_x = float(np.polyfit(x, y, 1)[0])
slope_x_on_y = float(np.polyfit(y, x, 1)[0])
r = float(np.corrcoef(x, y)[0, 1])
out = {
    "id": "Q3_fig7c_fit",
    "script": "scripts/analysis_paper3jc_r2_Q3_fig7c_fit.py",
    "source": "PAPER_3_v2/analysis/r1_cmip6.json (per_model)",
    "n_models": len(sub),
    "models": [r_["model"] for r_ in sub],
    "units": {"slope": "Sv per Sv", "r": "dimensionless"},
    "plotted_fit": {"x": "damoc26 (change at 26.5 N)", "y": "dpsi34 (change at 34.5 S)",
                    "slope_ols_y_on_x": slope_y_on_x, "r": r},
    "reverse_fit_not_plotted": {"slope_ols_x_on_y": slope_x_on_y},
}
(ANALYSIS / "jc_r2_Q3_fig7c_fit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["plotted_fit"], indent=1), slope_x_on_y)
