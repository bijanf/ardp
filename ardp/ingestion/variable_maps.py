"""Variable name mappings between reanalysis products and canonical names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VarNames:
    """Mapping from canonical variable names to product-specific names."""

    temperature: str
    salinity: str
    u_velocity: str
    v_velocity: str
    ssh: str
    e1t: str  # x-spacing at T-points
    e2t: str  # y-spacing at T-points
    e1u: str
    e2u: str
    e1v: str
    e2v: str
    e3t: str  # cell thickness at T-points
    e3u: str
    e3v: str
    depth: str
    lat_t: str  # latitude at T-points
    lon_t: str  # longitude at T-points
    lat_v: str
    lon_v: str
    tmask: str
    umask: str
    vmask: str


ORAS5 = VarNames(
    temperature="votemper",
    salinity="vosaline",
    u_velocity="vozocrtx",
    v_velocity="vomecrty",
    ssh="sossheig",
    e1t="e1t",
    e2t="e2t",
    e1u="e1u",
    e2u="e2u",
    e1v="e1v",
    e2v="e2v",
    e3t="e3t",
    e3u="e3u",
    e3v="e3v",
    depth="deptht",
    lat_t="nav_lat",
    lon_t="nav_lon",
    lat_v="nav_lat",
    lon_v="nav_lon",
    tmask="tmask",
    umask="umask",
    vmask="vmask",
)

GLORYS12 = VarNames(
    temperature="thetao",
    salinity="so",
    u_velocity="uo",
    v_velocity="vo",
    ssh="zos",
    e1t="e1t",
    e2t="e2t",
    e1u="e1u",
    e2u="e2u",
    e1v="e1v",
    e2v="e2v",
    e3t="e3t",
    e3u="e3u",
    e3v="e3v",
    depth="depth",
    lat_t="latitude",
    lon_t="longitude",
    lat_v="latitude",
    lon_v="longitude",
    tmask="tmask",
    umask="umask",
    vmask="vmask",
)

CGLORS = VarNames(
    temperature="votemper",
    salinity="vosaline",
    u_velocity="vozocrtx",
    v_velocity="vomecrty",
    ssh="sossheig",
    e1t="e1t",
    e2t="e2t",
    e1u="e1u",
    e2u="e2u",
    e1v="e1v",
    e2v="e2v",
    e3t="e3t",
    e3u="e3u",
    e3v="e3v",
    depth="deptht",
    lat_t="nav_lat",
    lon_t="nav_lon",
    lat_v="nav_lat",
    lon_v="nav_lon",
    tmask="tmask",
    umask="umask",
    vmask="vmask",
)

PRODUCT_MAP: dict[str, VarNames] = {
    "oras5": ORAS5,
    "glorys12": GLORYS12,
    "cglors": CGLORS,
}
