"""Abstract base class for reanalysis data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr
import xgcm

from ardp.ingestion.nemo_grid import build_nemo_grid
from ardp.ingestion.variable_maps import VarNames


class ReanalysisLoader(ABC):
    """Abstract loader for NEMO-based reanalysis products."""

    def __init__(self, data_dir: str | Path, mesh_mask_path: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.mesh_mask_path = Path(mesh_mask_path) if mesh_mask_path else None

    @property
    @abstractmethod
    def var_names(self) -> VarNames:
        """Return the variable name mapping for this product."""

    @abstractmethod
    def load_dataset(
        self,
        variables: list[str] | None = None,
        time_range: tuple[str, str] | None = None,
        chunks: dict[str, int] | None = None,
    ) -> xr.Dataset:
        """Load and return a dataset with canonical variable names.

        Parameters
        ----------
        variables : list[str] or None
            Canonical variable names to load. None = load all.
        time_range : tuple[str, str] or None
            (start, end) date strings for time slicing.
        chunks : dict or None
            Dask chunk sizes. Default: {"time": 1}.
        """

    def build_grid(self, ds: xr.Dataset, **kwargs: object) -> xgcm.Grid:
        """Build an xgcm Grid from the loaded dataset."""
        return build_nemo_grid(ds, **kwargs)

    def rename_to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename product-specific variable names to canonical names."""
        vn = self.var_names
        rename_map: dict[str, str] = {}

        canonical_map = {
            vn.temperature: "temperature",
            vn.salinity: "salinity",
            vn.u_velocity: "u_velocity",
            vn.v_velocity: "v_velocity",
            vn.ssh: "ssh",
        }

        for product_name, canon_name in canonical_map.items():
            if product_name in ds and product_name != canon_name:
                rename_map[product_name] = canon_name

        if rename_map:
            ds = ds.rename(rename_map)
        return ds
