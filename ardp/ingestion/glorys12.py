"""GLORYS12V1 reanalysis loader."""

from __future__ import annotations

import xarray as xr

from ardp.ingestion._base import ReanalysisLoader
from ardp.ingestion.variable_maps import GLORYS12, VarNames


class GLORYS12Loader(ReanalysisLoader):
    """Loader for Mercator GLORYS12V1 reanalysis."""

    @property
    def var_names(self) -> VarNames:
        return GLORYS12

    def load_dataset(
        self,
        variables: list[str] | None = None,
        time_range: tuple[str, str] | None = None,
        chunks: dict[str, int] | None = None,
    ) -> xr.Dataset:
        if chunks is None:
            chunks = {"time": 1}

        pattern = str(self.data_dir / "*.nc")
        ds = xr.open_mfdataset(pattern, chunks=chunks, combine="by_coords")

        # Merge mesh mask if available
        if self.mesh_mask_path and self.mesh_mask_path.exists():
            mesh = xr.open_dataset(self.mesh_mask_path)
            if "time" in mesh.dims:
                mesh = mesh.isel(time=0, drop=True)
            ds = xr.merge([ds, mesh], compat="override")

        # Standardize dimension names
        rename_dims: dict[str, str] = {}
        if "depth" in ds.dims and "z" not in ds.dims:
            rename_dims["depth"] = "z"
        if rename_dims:
            ds = ds.rename(rename_dims)

        ds = self.rename_to_canonical(ds)

        if time_range:
            ds = ds.sel(time=slice(*time_range))

        return ds
