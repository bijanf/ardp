"""Build xgcm.Grid from NEMO C-grid datasets."""

from __future__ import annotations

import xarray as xr
import xgcm


def build_nemo_grid(
    ds: xr.Dataset,
    x_dim: str = "x",
    y_dim: str = "y",
    z_dim: str = "z",
    periodic: list[str] | None = None,
) -> xgcm.Grid:
    """Create an xgcm Grid for NEMO C-grid staggering.

    NEMO uses an Arakawa C-grid with T/U/V/F/W stagger points.
    T-points are at cell centers; U-points are offset +1/2 in X;
    V-points are offset +1/2 in Y; W-points are offset +1/2 in Z.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing NEMO grid coordinates and metrics.
    x_dim, y_dim, z_dim : str
        Names of the spatial dimensions in the dataset.
    periodic : list[str] or None
        Axes that are periodic (e.g., ["X"] for global ocean).

    Returns
    -------
    xgcm.Grid
    """
    if periodic is None:
        periodic = ["X"]

    coords = {
        "X": {"center": x_dim},
        "Y": {"center": y_dim},
        "Z": {"center": z_dim},
    }

    grid = xgcm.Grid(
        ds,
        coords=coords,
        periodic=periodic,
        autoparse_metadata=False,
    )
    return grid


def add_nemo_metrics(
    ds: xr.Dataset,
    grid: xgcm.Grid,
    e1t: str = "e1t",
    e2t: str = "e2t",
    e3t: str = "e3t",
) -> xgcm.Grid:
    """Attach horizontal and vertical metrics to an xgcm Grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with metric arrays.
    grid : xgcm.Grid
        The xgcm Grid object to add metrics to.
    e1t, e2t, e3t : str
        Names of metric variables in ds.

    Returns
    -------
    xgcm.Grid
        Grid with metrics set (returns same object for convenience).
    """
    metrics: dict[tuple[str, ...], list[str]] = {}

    if e1t in ds:
        metrics[("X",)] = [e1t]
    if e2t in ds:
        metrics[("Y",)] = [e2t]
    if e3t in ds:
        metrics[("Z",)] = [e3t]

    if metrics:
        periodic_axes = [ax for ax, axis in grid.axes.items() if getattr(axis, "_boundary", None) == "periodic"]
        grid = xgcm.Grid(
            ds,
            coords=grid._coords,
            periodic=periodic_axes,
            metrics=metrics,
            autoparse_metadata=False,
        )
    return grid
