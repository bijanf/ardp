#!/usr/bin/env python3
"""SMILE AMOC at 26.5°N — DKRZ ESGF download/process/delete pipeline.

Streams `msftmz` (zonally-integrated meridional overturning
streamfunction) from DKRZ ESGF for each available initial-condition
member of MPI-ESM1-2-LR (50 members), extracts the 26.5°N annual-mean
AMOC time series (max of streamfunction below 500 m, the standard
RAPID-equivalent definition), and writes one NPZ with all members.

Used to give Main Fig.~2\\textbf{(b)} (mechanism-conditional AMOC
trajectories) a 50-member SMILE band, complementing the F_ovS-only
SMILE result in Supp Fig.~S7.

Reads:  ESGF DKRZ via Solr (variable=msftmz, source_id=MPI-ESM1-2-LR)
Writes: data/results/smile_amoc26n_mpi_lr.npz
        with keys: members[], years[member], amoc[member]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import requests
import xarray as xr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOURCE = "MPI-ESM1-2-LR"
VARIABLE = "msftmz"
TABLE = "Omon"
GRID = "gn"
RAPID_LAT = 26.5  # standard RAPID array latitude
DEPTH_MIN = 500.0   # ignore shallow recirculation; AMOC is below 500 m
SCRATCH = Path("/tmp/smile_amoc")
DEFAULT_OUTPUT = Path("data/results/smile_amoc26n_mpi_lr.npz")
REPLICA_HOSTS = ["esgf3.dkrz.de", "esgf-data1.llnl.gov", "esgf.ceda.ac.uk"]


def _esgf_files(experiment_id: str) -> dict[str, list[dict]]:
    """Return {member: [{filename, download_url}]} via direct Solr."""
    base = "https://esgf-data.dkrz.de/esg-search/search"
    params = {
        "format": "application/solr+json",
        "type": "File",
        "project": "CMIP6",
        "source_id": SOURCE,
        "experiment_id": experiment_id,
        "variable": VARIABLE,
        "table_id": TABLE,
        "grid_label": GRID,
        "distrib": "false",
        "limit": 10000,
    }
    r = requests.get(base, params=params, timeout=120)
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    members: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    for d in docs:
        member = d.get("variant_label")
        if isinstance(member, list):
            member = member[0]
        urls = d.get("url", [])
        download_url = None
        for u in urls:
            parts = u.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                download_url = parts[0]
                break
        if download_url is None:
            continue
        fname = d.get("title") or download_url.split("/")[-1]
        seen.setdefault(member, set())
        if fname in seen[member]:
            continue
        seen[member].add(fname)
        members.setdefault(member, []).append({
            "filename": fname,
            "download_url": download_url,
        })
    return members


def _download(url: str, dest: Path, timeout: int = 600) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    candidates = [url]
    for host in REPLICA_HOSTS:
        for src in REPLICA_HOSTS:
            if f"//{src}/" in url and src != host:
                candidates.append(url.replace(f"//{src}/", f"//{host}/"))
    for u in candidates:
        try:
            with requests.get(u, stream=True, timeout=timeout,
                              allow_redirects=True) as r:
                if r.status_code != 200:
                    continue
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                return True
        except Exception as exc:  # noqa: BLE001
            log.debug(f"  {u}: {exc}")
            continue
    return False


def _extract_amoc(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, amoc_max_below500_at_26.5N)."""
    da = ds[VARIABLE]
    # Identify dims
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords), None)
    lev_name = next((n for n in ("lev", "depth", "olevel")
                     if n in ds.coords), None)
    if lat_name is None or lev_name is None:
        raise RuntimeError(f"missing lat/lev coords in {list(ds.coords)}")

    lat_vals = ds[lat_name].values
    j_idx = int(np.abs(lat_vals - RAPID_LAT).argmin())
    sec = da.isel({lat_name: j_idx})
    depth = ds[lev_name].values.astype(float)
    deep = depth >= DEPTH_MIN
    sec_deep = sec.isel({lev_name: np.where(deep)[0]})
    # If basin index is present (msftmz often has 'basin' dim), pick the
    # Atlantic-Arctic basin (basin index 0 in MPI's convention).
    if "basin" in sec_deep.dims:
        sec_deep = sec_deep.isel(basin=0)
    # AMOC = max of streamfunction over depth (in Sv = 1e9 kg/s
    # for MPI-OM convention, OR already in Sv depending on file)
    psi_max = sec_deep.max(dim=lev_name)
    psi_annual = psi_max.groupby("time.year").mean(dim="time")
    years = psi_annual["year"].values.astype(float)
    vals = psi_annual.values
    # Convert kg s^-1 -> Sv if needed (msftmz is in kg s^-1; 1 Sv = 1e9 kg/s)
    if np.nanmean(np.abs(vals)) > 1e6:
        vals = vals / 1e9
    return years, vals


def process_member(member: str,
                   files_hist: list[dict],
                   files_ssp: list[dict],
                   scratch: Path) -> tuple[np.ndarray, np.ndarray] | None:
    member_dir = scratch / member
    member_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths_hist = []
        for f in files_hist:
            dest = member_dir / f["filename"]
            if not _download(f["download_url"], dest):
                log.error(f"  {member}: failed to download {f['filename']}")
                return None
            paths_hist.append(dest)
        paths_ssp = []
        for f in files_ssp:
            dest = member_dir / f["filename"]
            if not _download(f["download_url"], dest):
                log.error(f"  {member}: failed to download {f['filename']}")
                return None
            paths_ssp.append(dest)

        ds_h = xr.open_mfdataset(
            sorted(paths_hist), combine="by_coords", parallel=False,
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            engine="netcdf4")
        years_h, amoc_h = _extract_amoc(ds_h)
        ds_h.close()

        ds_s = xr.open_mfdataset(
            sorted(paths_ssp), combine="by_coords", parallel=False,
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            engine="netcdf4")
        years_s, amoc_s = _extract_amoc(ds_s)
        ds_s.close()

        # Concat hist + ssp deduplicating
        combined_y = np.concatenate([years_h, years_s])
        combined_a = np.concatenate([amoc_h, amoc_s])
        order = np.argsort(combined_y)
        combined_y = combined_y[order]
        combined_a = combined_a[order]
        _, unique = np.unique(combined_y, return_index=True)
        years = combined_y[sorted(unique)]
        amoc = combined_a[sorted(unique)]
        log.info(
            f"  {member}: {len(years)} years; "
            f"AMOC 1950-1980 mean = {np.nanmean(amoc[(years>=1950)&(years<=1980)]):+.1f} Sv,"
            f"  2080-2100 mean = {np.nanmean(amoc[(years>=2080)&(years<=2100)]):+.1f} Sv"
        )
        return years, amoc
    except Exception as exc:  # noqa: BLE001
        log.error(f"  {member}: FAILED ({exc})")
        return None
    finally:
        shutil.rmtree(member_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--members", nargs="+", default=None)
    args = parser.parse_args()

    log.info(f"SMILE AMOC at 26.5°N for {SOURCE} via DKRZ ESGF (msftmz)...")
    hist = _esgf_files("historical")
    ssp = _esgf_files("ssp585")
    log.info(f"  historical: {len(hist)} members")
    log.info(f"  ssp585:     {len(ssp)} members")
    common = sorted(set(hist) & set(ssp))
    log.info(f"  members in both: {len(common)}")
    if args.members is not None:
        common = [m for m in common if m in args.members]
        log.info(f"  filtered to: {common}")

    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    members_done: list[str] = []
    for m in common:
        result = process_member(m, hist[m], ssp[m], args.scratch)
        if result is None:
            continue
        years, amoc = result
        out[f"{m}_years"] = years
        out[f"{m}_amoc"] = amoc
        members_done.append(m)

    if not members_done:
        log.error("No members processed successfully.")
        return

    out["members"] = np.array(members_done, dtype=object)
    np.savez(args.output, **out)
    log.info(f"Saved: {args.output}  ({len(members_done)} members)")

    # Quick summary
    weakening = []
    for m in members_done:
        y = out[f"{m}_years"]
        a = out[f"{m}_amoc"]
        base = float(np.nanmean(a[(y >= 1950) & (y <= 1980)]))
        end = float(np.nanmean(a[(y >= 2081) & (y <= 2100)]))
        if base > 0:
            weakening.append(100 * (base - end) / base)
    arr = np.array(weakening)
    log.info(
        f"--- SMILE AMOC weakening summary ---\n"
        f"  mean weakening by 2100: {arr.mean():+.1f}%\n"
        f"  spread (1σ):            {arr.std():.1f}%\n"
        f"  range:                  [{arr.min():+.1f}%, {arr.max():+.1f}%]"
    )


if __name__ == "__main__":
    main()
