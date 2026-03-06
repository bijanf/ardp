"""Download functions for reanalysis products from Copernicus and CDS APIs."""

from __future__ import annotations

from pathlib import Path


def download_glorys12(
    output_dir: str | Path,
    start: str,
    end: str,
    lon_min: float = -80.0,
    lon_max: float = 30.0,
    lat_min: float = -60.0,
    lat_max: float = 70.0,
    variables: list[str] | None = None,
) -> Path:
    """Download GLORYS12V1 monthly reanalysis from Copernicus Marine.

    Downloads year-by-year to avoid timeouts and enable resumption.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory. Files saved to ``output_dir/glorys12/``.
    start, end : str
        Start/end dates, e.g. "1993-01" and "2023-12".
    lon_min, lon_max, lat_min, lat_max : float
        Spatial bounds.
    variables : list[str] or None
        Variables to download. Default: thetao, so, uo, vo, zos.

    Returns
    -------
    Path
        Directory containing downloaded files.
    """
    import copernicusmarine

    if variables is None:
        variables = ["thetao", "so", "uo", "vo", "zos"]

    dest = Path(output_dir) / "glorys12"
    dest.mkdir(parents=True, exist_ok=True)

    # Parse start/end years
    start_year = int(start.split("-")[0])
    end_year = int(end.split("-")[0])

    for year in range(start_year, end_year + 1):
        outfile = dest / f"glorys12_{year}.nc"
        if outfile.exists():
            print(f"Already exists: {outfile}")
            continue

        y_start = f"{year}-01"
        y_end = f"{year}-12"

        print(f"Downloading GLORYS12V1 year {year}...")
        print(f"  Variables: {variables}")
        print(f"  Region: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

        try:
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
                variables=variables,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=f"{y_start}-01T00:00:00",
                end_datetime=f"{y_end}-28T23:59:59",
                output_filename=outfile.name,
                output_directory=str(dest),
            )
            print(f"  Saved: {outfile}")
        except Exception as e:
            print(f"  Failed for {year}: {e}")
            # Clean up partial file
            for f in dest.glob(f"glorys12_{year}.nc*"):
                f.unlink()

    return dest


def download_oras5(
    output_dir: str | Path,
    start_year: int,
    end_year: int,
    lon_min: float = -80.0,
    lon_max: float = 30.0,
    lat_min: float = -60.0,
    lat_max: float = 70.0,
    variables: list[str] | None = None,
) -> Path:
    """Download ORAS5 monthly reanalysis from CDS API.

    Downloads year-by-year to stay within CDS request size limits.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory. Files saved to ``output_dir/oras5/``.
    start_year, end_year : int
        Year range (inclusive).
    lon_min, lon_max, lat_min, lat_max : float
        Spatial bounds.
    variables : list[str] or None
        CDS variable names. Default: temperature, salinity, velocities, SSH.

    Returns
    -------
    Path
        Directory containing downloaded files.
    """
    import cdsapi

    if variables is None:
        variables = [
            "potential_temperature",
            "salinity",
            "meridional_velocity",
            "zonal_velocity",
            "sea_surface_height",
        ]

    dest = Path(output_dir) / "oras5"
    dest.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    months = [f"{m:02d}" for m in range(1, 13)]

    for year in range(start_year, end_year + 1):
        # Check if any file for this year already exists
        existing = list(dest.glob(f"*{year}*"))
        if existing:
            print(f"Already exists for {year}: {existing[0].name}")
            continue

        product = "consolidated" if year <= 2021 else "operational"
        print(f"Downloading ORAS5 year {year} ({product})...")
        outfile_zip = dest / f"oras5_{year}.zip"
        try:
            client.retrieve(
                "reanalysis-oras5",
                {
                    "product_type": [product],
                    "vertical_resolution": "all_levels",
                    "variable": variables,
                    "year": [str(year)],
                    "month": months,
                },
                str(outfile_zip),
            )
        except Exception as e:
            print(f"  Failed for {year}: {e}")
            outfile_zip.unlink(missing_ok=True)
            continue

        # CDS delivers ORAS5 as zip — extract NetCDF files
        import zipfile
        if outfile_zip.exists() and zipfile.is_zipfile(outfile_zip):
            with zipfile.ZipFile(outfile_zip, "r") as zf:
                zf.extractall(dest)
            outfile_zip.unlink()
            print(f"  Extracted to: {dest}")
        elif outfile_zip.exists():
            # If it's actually a NetCDF, just rename
            outfile_zip.rename(dest / f"oras5_{year}.nc")
            print(f"  Saved: oras5_{year}.nc")

    return dest


def download_cglors(
    output_dir: str | Path,
    start: str,
    end: str,
    lon_min: float = -80.0,
    lon_max: float = 30.0,
    lat_min: float = -60.0,
    lat_max: float = 70.0,
    variables: list[str] | None = None,
) -> Path:
    """Download C-GLORS monthly reanalysis from Copernicus Marine.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory. Files saved to ``output_dir/cglors/``.
    start, end : str
        Start/end dates, e.g. "1993-01" and "2023-12".
    lon_min, lon_max, lat_min, lat_max : float
        Spatial bounds.
    variables : list[str] or None
        Variables to download. Default: votemper, vosaline, vozocrtx,
        vomecrty, sossheig.

    Returns
    -------
    Path
        Directory containing downloaded files.
    """
    import copernicusmarine

    if variables is None:
        variables = ["votemper", "vosaline", "vozocrtx", "vomecrty", "sossheig"]

    dest = Path(output_dir) / "cglors"
    dest.mkdir(parents=True, exist_ok=True)

    outfile = dest / f"cglors_{start}_{end}.nc"
    if outfile.exists():
        print(f"Already exists: {outfile}")
        return dest

    print(f"Downloading C-GLORS: {start} to {end}")
    print(f"  Variables: {variables}")
    print(f"  Region: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1M-m",
        variables=variables,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=f"{start}-01T00:00:00",
        end_datetime=f"{end}-28T23:59:59",
        output_filename=outfile.name,
        output_directory=str(dest),
    )

    print(f"Saved: {outfile}")
    return dest
