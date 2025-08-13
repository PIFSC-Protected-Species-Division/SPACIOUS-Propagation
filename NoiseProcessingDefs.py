# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 05:44:47 2025

@author: pam_user
"""


import os, re, tempfile, pathlib
import numpy as np
import xarray as xr
from google.cloud import storage
from tqdm import tqdm
import pyhydrophone as pyhy
import pypam
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------------------
# Helpers
# -------------------------------
def parse_gs_uri(gs_uri: str):
    """Accepts 'bucket/prefix', 'bucket', or 'gs://bucket/prefix'."""
    if gs_uri.startswith("gs://"):
        gs_uri = gs_uri[5:]
    parts = gs_uri.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix

# Extract start time from filenames like ..._YYMMDD-HHMMSS.mmm.wav
_fname_dt = re.compile(r".*_(\d{6})-(\d{6})(?:\.(\d{3}))?\.wav$", re.IGNORECASE)
def start_dt_from_name(name: str):
    import datetime as _dt
    m = _fname_dt.match(os.path.basename(name))
    if not m:
        return None
    yymmdd, hhmmss, msec = m.groups()
    yy = int(yymmdd[0:2]); mm = int(yymmdd[2:4]); dd = int(yymmdd[4:6])
    hh = int(hhmmss[0:2]); mi = int(hhmmss[2:4]); ss = int(hhmmss[4:6])
    year = 2000 + yy  # 22 -> 2022
    usec = int(msec) * 1000 if msec else 0
    return _dt.datetime(year, mm, dd, hh, mi, ss, usec, tzinfo=_dt.timezone.utc)

def iter_wav_blobs_sorted(bucket, prefix):
    """Return .wav blobs in chronological order (by filename time, fallback updated)."""
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.lower().endswith(".wav")]
    def _key(b):
        dt = start_dt_from_name(b.name)
        return dt or b.updated or 0
    blobs.sort(key=_key)
    return blobs

def _calibrate_band_levels(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert density dB to band dB by adding 10*log10(BW).
    Uses 'frequency_increment' if present, otherwise tries band edges.
    """
    if "millidecade_bands" not in ds:
        return ds

    psd = ds["millidecade_bands"]  # dB (density)
    inc_da = None

    if "frequency_increment" in ds:
        inc = ds["frequency_increment"]
        if isinstance(inc, xr.DataArray):
            inc_da = inc
        elif isinstance(inc, xr.Dataset) and "inc_value" in inc:
            inc_da = inc["inc_value"]

    if inc_da is None:
        if "band_edges" in ds:
            edges = ds["band_edges"]  # (band, edge=2) [Hz]
            bw = (edges.sel(edge=1) - edges.sel(edge=0)).astype("float64")
            inc_da = 10.0 * np.log10(bw)
        elif all(k in ds for k in ("f_low", "f_high")):
            bw = (ds["f_high"] - ds["f_low"]).astype("float64")
            inc_da = 10.0 * np.log10(bw)

    if inc_da is None:
        out = ds.copy()
        out.attrs["warning"] = "No frequency increment found; band dB not applied."
        return out

    inc_da = inc_da.squeeze()
    calibrated = psd + inc_da  # broadcast over band dimension
    out = ds.copy()
    out["millidecade_bands"] = calibrated
    out["millidecade_bands"].attrs["description"] = "Hybrid millidecade band level (dB re 1 µPa^2)"
    out["millidecade_bands"].attrs["note"] = "Density dB + 10*log10(BW)"
    return out

def _encode_and_write(ds: xr.Dataset, path: str):
    """Write NetCDF; compress numerics; leave strings/objects alone."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    encoding = {}
    for v, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            encoding[v] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        elif np.issubdtype(da.dtype, np.integer):
            encoding[v] = {"zlib": True, "complevel": 4}
        # strings/object: leave unencoded (no compression)
    ds.to_netcdf(path, mode="w", encoding=encoding)

# -------------------------------
# Core: GCS -> temp -> ASA -> split-by-day -> write NetCDF
# -------------------------------
def process_bucket_audio_daily(
    gs_uri,
    drifter_Cal,
    binsize,
    nfft,
    include_dirs,
    zipped_files,
    dc_subtract,
    band,
    out_dir,
    max_files=None,
):
    bucket_name, prefix = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    current_day = None           # numpy.datetime64[D]
    day_parts = []               # list[Dataset] for the current day
    processed = 0

    def _flush_day():
        nonlocal current_day, day_parts
        if current_day is None or not day_parts:
            return
        day_ds = xr.concat(day_parts, dim="time").sortby("time")
        date_str = np.datetime_as_string(current_day, unit="D").replace("-", "")
        out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
        day_ds = day_ds.assign_attrs({
            **day_ds.attrs,
            "provenance": "GCS streamed -> ASA.hybrid_millidecade_bands -> calibrated -> daily split",
            "bucket": bucket_name,
            "prefix": prefix,
            "binsize_s": binsize,
            "nfft": nfft,
            "band_hz": list(band),
            "timezone": "UTC",
        })
        _encode_and_write(day_ds, out_path)
        print(f"[write] {out_path}  (time bins: {day_ds.dims.get('time', 'NA')})")
        current_day = None
        day_parts = []

    blobs = iter_wav_blobs_sorted(bucket, prefix)
    for blob in tqdm(blobs, desc="GCS → ASA → daily NetCDF"):
        if max_files is not None and processed >= max_files:
            break

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)

                # Run ASA on the temp folder (ASA scans directory)
                asa_cal = pypam.ASA(
                    hydrophone=drifter_Cal,
                    folder_path=tmpdir,
                    binsize=binsize,
                    nfft=nfft,
                    timezone="UTC",
                    include_dirs=include_dirs,
                    zipped=zipped_files,
                    dc_subtract=dc_subtract,
                    calibration=-1,
                )
                ds = asa_cal.hybrid_millidecade_bands(
                    db=True, method="density", band=band, percentiles=None
                )

                # Optional provenance per time bin
                if "time" in ds.coords:
                    ds["source_file"] = xr.DataArray(
                        np.array([blob.name] * ds.sizes["time"]),
                        dims=["time"],
                        coords={"time": ds["time"]},
                    )

                # Calibrate density dB -> band dB
                ds = _calibrate_band_levels(ds)

                # --- daily split ---
                if "time" not in ds.coords:
                    # Fallback: assign whole file to start-day
                    dt0 = start_dt_from_name(blob.name) or (blob.updated or None)
                    if dt0 is None:
                        raise RuntimeError(f"Cannot determine day for {blob.name}")
                    day_key = np.datetime64(dt0.date())
                    if current_day is None:
                        current_day = day_key
                        day_parts = [ds]
                    elif day_key == current_day:
                        day_parts.append(ds)
                    else:
                        _flush_day()
                        current_day = day_key
                        day_parts = [ds]
                else:
                    # Add a day coordinate (UTC midnight) and group
                    ds = ds.assign_coords(day=ds["time"].dt.floor("D"))
                    for day_key, ds_day in ds.groupby("day"):
                        day_key = np.datetime64(day_key, "D")
                        if current_day is None:
                            current_day = day_key
                            day_parts = [ds_day]
                        elif day_key == current_day:
                            day_parts.append(ds_day)
                        else:
                            _flush_day()
                            current_day = day_key
                            day_parts = [ds_day]

            processed += 1

        except Exception as e:
            print(f"[skip] {blob.name}: {e}")

    # Flush the trailing day
    try:
        _flush_day()
    except Exception as e:
        print(f"[warn] final flush failed: {e}")

    print(f"Done. Processed {processed} file(s).")
