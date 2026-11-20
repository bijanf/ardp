"""Selection of the Atlantic basin in the published CMIP6 overturning streamfunction (msftmz, msftyz).

The order of the basin dimension differs between models: most store atlantic_arctic_ocean first, MPI-ESM1-2
stores global_ocean first. The basin is therefore chosen by its label (a string variable such as `sector`) or by
the `requested` attribute of the basin coordinate ("atlantic_arctic_ocean=0, ..."), and the selection fails if
neither is present.
"""
from __future__ import annotations

import re

import numpy as np
import xarray as xr


def _labels(ds: xr.Dataset, bdim: str) -> list[str] | None:
    for name in ("sector", "basin", "region", bdim):
        if name not in ds:
            continue
        val = np.asarray(ds[name].values)
        if val.dtype.kind == "S" and val.ndim == 2:  # character array (basin, strlen)
            return [b"".join(r).decode(errors="ignore").strip().lower() for r in val]
        if val.dtype.kind in "SUO":
            return [(x.decode(errors="ignore") if isinstance(x, bytes) else str(x)).strip().lower()
                    for x in np.atleast_1d(val)]
    for name in (bdim, "basin"):  # e.g. IPSL: long_name "Sub-basin mask (1=Global 2=Atlantic 3=Indo-Pacific)"
        if name not in ds:
            continue
        for key in ("long_name", "description", "flag_meanings"):
            txt = str(ds[name].attrs.get(key, "")).lower()
            pairs = dict((int(i), n) for i, n in re.findall(r"(\d+)\s*=\s*([a-z][a-z_-]*)", txt))
            if len(pairs) == ds.sizes[bdim]:
                vals = np.asarray(ds[name].values).astype(int).tolist() if ds[name].dtype.kind in "iuf" else []
                if sorted(vals) == sorted(pairs):
                    return [pairs[v] for v in vals]
                return [pairs.get(i + min(pairs), "") for i in range(ds.sizes[bdim])]
    for name in (bdim, "basin"):
        if name in ds and "requested" in ds[name].attrs:
            req = str(ds[name].attrs["requested"]).lower()
            pairs = dict((int(i), n) for n, i in re.findall(r"([a-z_]+)\s*=\s*(\d+)", req))
            if pairs:
                return [pairs.get(i, "") for i in range(ds.sizes[bdim])]
    return None


def atlantic_basin(ds: xr.Dataset, da: xr.DataArray) -> tuple[xr.DataArray, int, str]:
    """Return the Atlantic slice of `da`, its basin index and its label. Raises if the basin cannot be identified."""
    bdim = next((d for d in da.dims if "basin" in d), None)
    if bdim is None:
        return da, -1, "no basin dimension"
    labels = _labels(ds, bdim)
    if labels and all(len(n) == 1 for n in labels):  # e.g. CMCC: char sector(basin, strlen=1) holds initials only
        full = {"a": "atlantic_arctic_ocean", "i": "indian_pacific_ocean", "g": "global_ocean"}
        labels = [full.get(n, n) for n in labels]
    if not labels:
        raise ValueError(f"basin dimension '{bdim}' has no labels and no 'requested' attribute")
    idx = next((i for i, n in enumerate(labels) if "atlantic" in n), None)
    if idx is None:
        raise ValueError(f"no Atlantic basin among {labels}")
    return da.isel({bdim: idx}), idx, labels[idx]

