"""Cartopy-based map plotting for AMOC diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_atlantic_field(
    field: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot a 2D field on an Atlantic-centered map projection.

    Parameters
    ----------
    field : xr.DataArray
        2D field to plot (y, x).
    lon, lat : xr.DataArray
        2D longitude and latitude arrays.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    vmin, vmax : float or None
        Color limits.
    ax : matplotlib Axes or None
        Existing axes with cartopy projection. Created if None.

    Returns
    -------
    matplotlib Axes
    """
    import cartopy.crs as ccrs

    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(10, 8),
        )

    im = ax.pcolormesh(
        lon.values, lat.values, field.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.coastlines()
    ax.set_extent([-80, 30, -60, 70], crs=ccrs.PlateCarree())
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7, label=field.attrs.get("units", ""))
    return ax


def plot_timeseries(
    ts: xr.DataArray,
    title: str = "",
    ylabel: str = "",
    ax: Any | None = None,
) -> Any:
    """Plot a time series.

    Parameters
    ----------
    ts : xr.DataArray
        1D time series with time coordinate.
    title, ylabel : str
        Labels.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ts.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel or ts.attrs.get("units", ""))
    ax.grid(True, alpha=0.3)
    return ax


def plot_timeseries_with_trend(
    ts: xr.DataArray,
    title: str = "",
    ylabel: str = "",
    ax: Any | None = None,
) -> Any:
    """Plot a time series with a linear regression trend overlay.

    Parameters
    ----------
    ts : xr.DataArray
        1D time series with time coordinate.
    title, ylabel : str
        Labels.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ts.plot(ax=ax, label="data")

    # Compute linear trend
    time = ts["time"]
    if hasattr(time.values[0], "year"):
        years = np.array([
            t.year + (t.month - 1) / 12.0 + (t.day - 1) / 365.25
            for t in time.values
        ])
    else:
        years = time.values.astype(float) / 365.25

    values = ts.values.ravel()
    valid = np.isfinite(values)

    if valid.sum() >= 2:
        coeffs = np.polyfit(years[valid], values[valid], 1)
        trend_line = np.polyval(coeffs, years)
        ax.plot(time.values, trend_line, "r--", linewidth=1.5, label="trend")
        slope_per_year = coeffs[0]
        ax.annotate(
            f"Trend: {slope_per_year:.4f}/yr",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=9, verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
        )
        ax.legend(loc="lower left", fontsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel or ts.attrs.get("units", ""))
    ax.grid(True, alpha=0.3)
    return ax


def plot_region_box(
    ax: Any,
    bounds: tuple[float, float, float, float],
    color: str = "red",
    linewidth: float = 2.0,
) -> None:
    """Draw a rectangular region boundary on a cartopy axes.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes with a cartopy projection.
    bounds : tuple
        (lon_min, lon_max, lat_min, lat_max).
    color : str
        Line color.
    linewidth : float
        Line width.
    """
    import cartopy.crs as ccrs

    lon_min, lon_max, lat_min, lat_max = bounds
    lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(lons, lats, color=color, linewidth=linewidth, transform=ccrs.PlateCarree())


def plot_multi_panel_fingerprints(
    fingerprints: dict[str, xr.DataArray],
) -> Any:
    """Create a 2x2 summary panel of AMOC fingerprint time series.

    Parameters
    ----------
    fingerprints : dict[str, xr.DataArray]
        Mapping of label -> time series. Up to 4 entries.

    Returns
    -------
    matplotlib Figure
    """
    n = len(fingerprints)
    nrows = 2 if n > 2 else 1
    ncols = 2 if n > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for i, (label, ts) in enumerate(fingerprints.items()):
        plot_timeseries_with_trend(ts, title=label, ylabel=label, ax=axes[i])

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


def save_figure(fig: Any, path: str | Path, dpi: int = 150) -> None:
    """Save a figure and print confirmation.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save.
    path : str or Path
        Output file path.
    dpi : int
        Resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)
