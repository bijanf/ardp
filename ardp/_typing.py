"""Type aliases for the ARDP package."""

from __future__ import annotations

from typing import Union

import numpy as np
import xarray as xr

XrDataset = xr.Dataset
XrDataArray = xr.DataArray
NpArray = np.ndarray[tuple[int, ...], np.dtype[np.floating[object]]]
Latitude = Union[float, np.floating[object]]
Longitude = Union[float, np.floating[object]]
