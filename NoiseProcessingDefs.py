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
import bisect
import datetime as _dt
# -------------------------------
# Helpers
# -------------------------------

def _add_bandwidth_to_density(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert PSD density dB → per-band dB by adding 10*log10(BW).
    Uses 'frequency_increment' (inc_value = 10*log10(Hz)) or band edges if present.
    """
    if "millidecade_bands" not in ds:
        return ds

    psd = ds["millidecade_bands"]
    inc_da = None

    if "frequency_increment" in ds:
        inc = ds["frequency_increment"]
        if isinstance(inc, xr.Dataset) and "inc_value" in inc:
            inc_da = inc["inc_value"]
        elif isinstance(inc, xr.DataArray):
            inc_da = inc
    if inc_da is None and "band_edges" in ds:
        edges = ds["band_edges"]  # (band, edge=2) [Hz]
        bw = (edges.sel(edge=1) - edges.sel(edge=0)).astype("float64")
        inc_da = 10.0 * np.log10(bw)

    if inc_da is None:
        out = ds.copy()
        out.attrs["warning"] = "No frequency increment found; band dB not applied."
        return out

    inc_da = inc_da.squeeze().astype("float64")
    out = ds.copy()
    out["millidecade_bands"] = psd + inc_da
    out["millidecade_bands"].attrs.update({
        "description": "Hybrid millidecade band level",
        "units": "dB re 1 µPa² (band)"
    })
    out.attrs["bandwidth_added_db"] = True
    return out

def _apply_freq_response(ds: xr.Dataset, drifter_Cal) -> xr.Dataset:
    """
    Apply hydrophone frequency-dependent increment (dB) to 'millidecade_bands'.
    Safe to call multiple times; guarded by ds.attrs['freq_cal_applied_db'].
    Also stores the curve as 'freq_cal_increment_db' for provenance.
    """
    var = "millidecade_bands"
    if var not in ds:
        return ds
    if ds.attrs.get("freq_cal_applied_db", False):
        return ds

    if "frequency_bins" not in ds.coords:
        raise KeyError("frequency_bins coord missing; cannot apply freq response")

    # 1-D increment (len = n_bands), from your calibration file via pyhydrophone
    freq = ds["frequency_bins"].values
    inc = drifter_Cal.freq_cal_inc(freq)["inc_value"].astype("float64")  # dB
    
    # # Plot the incement -giggle test
    # plt.plot(freq, inc)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('dB')
    # plt.title('End-to-End Sensitivity (hyd, preamp, voltage)')

    # Apply (+) in dB
    ds = ds.copy()
    ds[var] = ds[var] + xr.DataArray(inc, dims=["frequency_bins"])
    

    # Label
    a = dict(ds[var].attrs)
    a.setdefault("standard_name", "Power spectral density")
    a["units"] = "dB re 1 µPa²/Hz (density)"
    a["calibration_note"] = "Applied hydrophone+chain frequency response (dB) via drifter_Cal.freq_cal_inc"
    ds[var].attrs = a

    # Keep the curve for provenance
    ds["freq_cal_increment_db"] = xr.DataArray(
        inc, dims=["frequency_bins"], coords={"frequency_bins": ds["frequency_bins"]}
    )
    ds["freq_cal_increment_db"].attrs.update({
        "description": "Hydrophone end-to-end frequency response increment",
        "units": "dB"
    })

    ds.attrs["freq_cal_applied_db"] = True
    return ds



def as_day64(x):
    """Normalize to numpy datetime64[D]."""
    if isinstance(x, np.datetime64):
        return np.datetime64(x, "D")
    if isinstance(x, _dt.datetime):
        return np.datetime64(x.date(), "D")
    if isinstance(x, _dt.date):
        return np.datetime64(x, "D")
    # string "YYYY-MM-DD"
    return np.datetime64(str(x), "D")

def blob_start_day(blob):
    """Prefer filename start; else blob.updated. Returns np.datetime64[D]."""
    dt = start_dt_from_name(blob.name) or blob.updated
    if dt is None:
        raise RuntimeError(f"Cannot determine start time for {blob.name}")
    return as_day64(dt)

def list_sorted_wav_blobs(bucket, prefix):
    """Return (.blobs, .days) where days[i] = np.datetime64[D] for blob i, sorted chronologically."""
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.lower().endswith(".wav")]
    blobs.sort(key=lambda b: start_dt_from_name(b.name) or b.updated or 0)
    days = [blob_start_day(b) for b in blobs]
    return blobs, days

def find_first_index_on_or_after(blobs_days, target_day64):
    """bisect to the first blob whose start day >= target_day64"""
    # blobs_days must be a non-decreasing list of np.datetime64[D]
    i = bisect.bisect_left(blobs_days, target_day64)
    return i

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

# # -------------------------------
# # Core: GCS -> temp -> ASA -> split-by-day -> write NetCDF
# # -------------------------------
# def process_bucket_audio_daily(
#     gs_uri,
#     drifter_Cal,
#     binsize,
#     nfft,
#     include_dirs,
#     zipped_files,
#     dc_subtract,
#     band,
#     out_dir,
#     max_files=None,
# ):
#     bucket_name, prefix = parse_gs_uri(gs_uri)
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)

#     current_day = None           # numpy.datetime64[D]
#     day_parts = []               # list[Dataset] for the current day
#     processed = 0

#     def _flush_day():
#         nonlocal current_day, day_parts
#         if current_day is None or not day_parts:
#             return
#         day_ds = xr.concat(day_parts, dim="time").sortby("time")
#         date_str = np.datetime_as_string(current_day, unit="D").replace("-", "")
#         out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
#         day_ds = day_ds.assign_attrs({
#             **day_ds.attrs,
#             "provenance": "GCS streamed -> ASA.hybrid_millidecade_bands -> calibrated -> daily split",
#             "bucket": bucket_name,
#             "prefix": prefix,
#             "binsize_s": binsize,
#             "nfft": nfft,
#             "band_hz": list(band),
#             "timezone": "UTC",
#         })
#         _encode_and_write(day_ds, out_path)
#         print(f"[write] {out_path}  (time bins: {day_ds.dims.get('time', 'NA')})")
#         current_day = None
#         day_parts = []

#     blobs = iter_wav_blobs_sorted(bucket, prefix)
#     for blob in tqdm(blobs, desc="GCS → ASA → daily NetCDF"):
#         if max_files is not None and processed >= max_files:
#             break

#         try:
#             with tempfile.TemporaryDirectory() as tmpdir:
#                 local_path = os.path.join(tmpdir, os.path.basename(blob.name))
#                 blob.download_to_filename(local_path)

#                 # Run ASA on the temp folder (ASA scans directory)
#                 asa_cal = pypam.ASA(
#                     hydrophone=drifter_Cal,
#                     folder_path=tmpdir,
#                     binsize=binsize,
#                     nfft=nfft,
#                     timezone="UTC",
#                     include_dirs=include_dirs,
#                     zipped=zipped_files,
#                     dc_subtract=dc_subtract,
#                     calibration=-1,
#                 )
#                 ds = asa_cal.hybrid_millidecade_bands(
#                     db=True, method="density", band=band, percentiles=None
#                 )

#                 # Optional provenance per time bin
#                 if "time" in ds.coords:
#                     ds["source_file"] = xr.DataArray(
#                         np.array([blob.name] * ds.sizes["time"]),
#                         dims=["time"],
#                         coords={"time": ds["time"]},
#                     )

#                 # Calibrate density dB -> band dB
#                 ds = _calibrate_band_levels(ds)

#                 # --- daily split ---
#                 if "time" not in ds.coords:
#                     # Fallback: assign whole file to start-day
#                     dt0 = start_dt_from_name(blob.name) or (blob.updated or None)
#                     if dt0 is None:
#                         raise RuntimeError(f"Cannot determine day for {blob.name}")
#                     day_key = np.datetime64(dt0.date())
#                     if current_day is None:
#                         current_day = day_key
#                         day_parts = [ds]
#                     elif day_key == current_day:
#                         day_parts.append(ds)
#                     else:
#                         _flush_day()
#                         current_day = day_key
#                         day_parts = [ds]
#                 else:
#                     # Add a day coordinate (UTC midnight) and group
#                     ds = ds.assign_coords(day=ds["time"].dt.floor("D"))
#                     for day_key, ds_day in ds.groupby("day"):
#                         day_key = np.datetime64(day_key, "D")
#                         if current_day is None:
#                             current_day = day_key
#                             day_parts = [ds_day]
#                         elif day_key == current_day:
#                             day_parts.append(ds_day)
#                         else:
#                             _flush_day()
#                             current_day = day_key
#                             day_parts = [ds_day]

#             processed += 1

#         except Exception as e:
#             print(f"[skip] {blob.name}: {e}")

#     # Flush the trailing day
#     try:
#         _flush_day()
#     except Exception as e:
#         print(f"[warn] final flush failed: {e}")

#     print(f"Done. Processed {processed} file(s).")


# def process_bucket_audio_daily(
#     gs_uri,
#     drifter_Cal,
#     binsize,
#     nfft,
#     include_dirs,
#     zipped_files,
#     dc_subtract,
#     band,
#     out_dir,
#     max_files=None,
#     start_date=None,   # "YYYY-MM-DD" or None
#     end_date=None,     # "YYYY-MM-DD" or None
#     skip_if_exists=True,
#     verbose=True,
# ):
#     bucket_name, prefix = parse_gs_uri(gs_uri)
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)

#     # ---- enumerate and pre-filter by day ----
#     blobs, blob_days = list_sorted_wav_blobs(bucket, prefix)

#     # Build the list of unique days present in the bucket (sorted)
#     unique_days = np.unique(np.array(blob_days))
#     # Apply date window
#     if start_date is not None:
#         start_d = as_day64(start_date)
#         unique_days = unique_days[unique_days >= start_d]
#     if end_date is not None:
#         end_d = as_day64(end_date)
#         unique_days = unique_days[unique_days <= end_d]

#     if skip_if_exists:
#         keep = []
#         for d in unique_days:
#             date_str = np.datetime_as_string(d, unit="D").replace("-", "")
#             out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
#             if not os.path.exists(out_path):
#                 keep.append(d)
#             elif verbose:
#                 print(f"[skip-day] {str(d)} exists in output; skipping")
#         unique_days = np.array(keep)

#     if unique_days.size == 0:
#         print("[info] Nothing to do for given date window / skip rules.")
#         return

#     # Find first blob index on/after the first wanted day
#     first_day = unique_days[0]
#     start_idx = find_first_index_on_or_after(blob_days, first_day)

#     # Prepare a set for quick membership tests when grouping inside files
#     days_to_process = set(unique_days.tolist())

#     # ---- state for day buffering ----
#     current_day = None           # numpy.datetime64[D]
#     day_parts = []               # list[Dataset] for the current day
#     processed_files = 0

#     os.makedirs(out_dir, exist_ok=True)

#     def _flush_day():
#         nonlocal current_day, day_parts
#         if current_day is None or not day_parts:
#             return
#         date_str = np.datetime_as_string(current_day, unit="D").replace("-", "")
#         out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")

#         day_ds = xr.concat(day_parts, dim="time").sortby("time")
#         day_ds = day_ds.assign_attrs({
#             **day_ds.attrs,
#             "provenance": "GCS streamed -> ASA.hybrid_millidecade_bands -> calibrated -> daily split",
#             "bucket": bucket_name,
#             "prefix": prefix,
#             "binsize_s": binsize,
#             "nfft": nfft,
#             "band_hz": list(band),
#             "timezone": "UTC",
#         })
#         _encode_and_write(day_ds, out_path)
#         if verbose:
#             print(f"[write] {out_path}  (time bins: {day_ds.dims.get('time', 'NA')})")
#         current_day = None
#         day_parts = []

#     # ---- main loop (start from start_idx) ----
#     for blob, bday in tqdm(list(zip(blobs[start_idx:], blob_days[start_idx:])),
#                            desc="GCS → ASA → daily NetCDF"):
#         if max_files is not None and processed_files >= max_files:
#             break

#         # Fast skip whole file if its start day is outside window or already complete
#         if bday not in days_to_process and (end_date is not None or start_date is not None or skip_if_exists):
#             # BUT: file could cross midnight; we still need to check groups after running ASA.
#             # To truly fast-skip, only skip if bday is not within [min(days_to_process), max(days_to_process)].
#             if bday < unique_days.min() or bday > unique_days.max():
#                 continue

#         try:
#             with tempfile.TemporaryDirectory() as tmpdir:
#                 local_path = os.path.join(tmpdir, os.path.basename(blob.name))
#                 blob.download_to_filename(local_path)

#                 asa_cal = pypam.ASA(
#                     hydrophone=drifter_Cal,
#                     folder_path=tmpdir,
#                     binsize=binsize,
#                     nfft=nfft,
#                     timezone="UTC",
#                     include_dirs=include_dirs,
#                     zipped=zipped_files,
#                     dc_subtract=dc_subtract,
#                     calibration=-1,
#                 )
#                 ds = asa_cal.hybrid_millidecade_bands(
#                     db=True, method="density", band=band, percentiles=None
#                 )

#                 # provenance per time bin (optional)
#                 if "time" in ds.coords:
#                     ds["source_file"] = xr.DataArray(
#                         np.array([blob.name] * ds.sizes["time"]),
#                         dims=["time"],
#                         coords={"time": ds["time"]},
#                     )

#                 ds = _calibrate_band_levels(ds)

#                 if "time" not in ds.coords:
#                     # Assign entire file to its start day; only add if wanted.
#                     k = bday
#                     if k not in days_to_process:
#                         continue
#                     if current_day is None:
#                         current_day = k; day_parts = [ds]
#                     elif k == current_day:
#                         day_parts.append(ds)
#                     else:
#                         _flush_day(); current_day = k; day_parts = [ds]
#                 else:
#                     # Proper split; only keep groups whose day is requested
#                     ds = ds.assign_coords(day=ds["time"].dt.floor("D"))
#                     for day_key, ds_day in ds.groupby("day"):
#                         k = as_day64(day_key)
#                         if k not in days_to_process:
#                             continue
#                         if current_day is None:
#                             current_day = k; day_parts = [ds_day]
#                             if verbose:
#                                 print(f"[day start] {str(current_day)}")
#                         elif k == current_day:
#                             day_parts.append(ds_day)
#                         else:
#                             if verbose:
#                                 print(f"[day flip] {str(current_day)} -> {str(k)}")
#                             _flush_day(); current_day = k; day_parts = [ds_day]

#             processed_files += 1

#         except Exception as e:
#             print(f"[skip] {blob.name}: {e}")

#     # Always flush the trailing day (if it’s one of the requested days)
#     try:
#         if current_day in days_to_process:
#             _flush_day()
#         else:
#             # drop partial accumulation for a non-requested day
#             pass
#     except Exception as e:
#         print(f"[warn] final flush failed: {e}")

#     print(f"Done. Processed {processed_files} file(s).")

# def process_bucket_audio_daily(
#     gs_uri,
#     drifter_Cal,
#     binsize,
#     nfft,
#     include_dirs,
#     zipped_files,
#     dc_subtract,
#     band,
#     out_dir,
#     max_files=None,
#     start_date=None,   # "YYYY-MM-DD" or None
#     end_date=None,     # "YYYY-MM-DD" or None
#     skip_if_exists=True,
#     verbose=True,
#     output_units: str = "density",
# ):
#     bucket_name, prefix = parse_gs_uri(gs_uri)
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)

#     # ---- enumerate and pre-filter by day ----
#     blobs, blob_days = list_sorted_wav_blobs(bucket, prefix)  # requires helpers from previous message

#     unique_days = np.unique(np.array(blob_days))
#     if start_date is not None:
#         start_d = as_day64(start_date)
#         unique_days = unique_days[unique_days >= start_d]
#     if end_date is not None:
#         end_d = as_day64(end_date)
#         unique_days = unique_days[unique_days <= end_d]

#     if skip_if_exists:
#         keep = []
#         for d in unique_days:
#             date_str = np.datetime_as_string(d, unit="D").replace("-", "")
#             out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
#             if not os.path.exists(out_path):
#                 keep.append(d)
#             elif verbose:
#                 print(f"[skip-day] {str(d)} exists in output; skipping")
#         unique_days = np.array(keep)

#     if unique_days.size == 0:
#         print("[info] Nothing to do for given date window / skip rules.")
#         return

#     first_day = unique_days[0]
#     start_idx = find_first_index_on_or_after(blob_days, first_day)  # helper

#     days_to_process = set(unique_days.tolist())

#     # ---- state ----
#     os.makedirs(out_dir, exist_ok=True)
#     current_day = None           # np.datetime64[D]
#     day_parts = []               # list of per-day xarray Datasets
#     processed_files = 0

#     def _flush_day():
#         nonlocal current_day, day_parts
#         if current_day is None or not day_parts:
#             return
#         date_str = np.datetime_as_string(current_day, unit="D").replace("-", "")
#         out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
#         day_ds = xr.concat(day_parts, dim="time").sortby("time")
#         day_ds = day_ds.assign_attrs({
#             **day_ds.attrs,
#             "provenance": "GCS streamed -> ASA.hybrid_millidecade_bands -> calibrated -> daily split",
#             "bucket": bucket_name,
#             "prefix": prefix,
#             "binsize_s": binsize,
#             "nfft": nfft,
#             "band_hz": list(band),
#             "timezone": "UTC",
#         })
#         _encode_and_write(day_ds, out_path)
#         if verbose:
#             print(f"[write] {out_path}  (time bins: {day_ds.dims.get('time', 'NA')})")
#         current_day = None
#         day_parts = []

#     # ---- main loop (with FRONT-FLUSH) ----
#     seq = list(zip(blobs[start_idx:], blob_days[start_idx:]))
#     for blob, bday in tqdm(seq, desc="GCS → ASA → daily NetCDF"):
#         if max_files is not None and processed_files >= max_files:
#             break

#         # FRONT-FLUSH: if we already buffered a day and the next blob starts a new day, write the old one now
#         if current_day is not None and day_parts and bday != current_day:
#             if verbose:
#                 print(f"[day flip] {str(current_day)} -> {str(bday)} (pre-file flush)")
#             _flush_day()

#         # Optional fast skip: if blob starts far outside target window
#         if (bday < unique_days.min()) or (bday > unique_days.max()):
#             continue

#         try:
#             with tempfile.TemporaryDirectory() as tmpdir:
#                 local_path = os.path.join(tmpdir, os.path.basename(blob.name))
#                 blob.download_to_filename(local_path)

#                 asa_cal = pypam.ASA(
#                     hydrophone=drifter_Cal,
#                     folder_path=tmpdir,
#                     binsize=binsize,
#                     nfft=nfft,
#                     timezone="UTC",
#                     include_dirs=include_dirs,
#                     zipped=zipped_files,
#                     dc_subtract=dc_subtract,
#                     calibration=-1,
#                 )
#                 ds = asa_cal.hybrid_millidecade_bands(
#                     db=True, method="density", band=band, percentiles=None
#                 )

#                 if "time" in ds.coords:
#                     ds["source_file"] = xr.DataArray(
#                         np.array([blob.name] * ds.sizes["time"]),
#                         dims=["time"], coords={"time": ds["time"]},
#                     )

#                 ds = _calibrate_band_levels(ds)

#                 # group by target days and accumulate only requested days
#                 if "time" not in ds.coords:
#                     k = bday
#                     if k not in days_to_process:
#                         # nothing to do for this file
#                         pass
#                     else:
#                         if current_day is None:
#                             current_day = k; day_parts = [ds]
#                             if verbose: print(f"[day start] {str(current_day)} (no time coord)")
#                         elif k == current_day:
#                             day_parts.append(ds)
#                         else:
#                             if verbose: print(f"[day flip] {str(current_day)} -> {str(k)} (no time coord)")
#                             _flush_day(); current_day = k; day_parts = [ds]
#                 else:
#                     ds = ds.assign_coords(day=ds["time"].dt.floor("D"))
#                     for day_key, ds_day in ds.groupby("day"):
#                         k = np.datetime64(day_key, "D")
#                         if k not in days_to_process:
#                             continue
#                         if current_day is None:
#                             current_day = k; day_parts = [ds_day]
#                             if verbose: print(f"[day start] {str(current_day)}")
#                         elif k == current_day:
#                             day_parts.append(ds_day)
#                         else:
#                             if verbose: print(f"[day flip] {str(current_day)} -> {str(k)}")
#                             _flush_day(); current_day = k; day_parts = [ds_day]

#             processed_files += 1

#         except Exception as e:
#             print(f"[skip] {blob.name}: {e}")

#     # trailing flush (only if it’s one of the requested days)
#     try:
#         if current_day in days_to_process:
#             _flush_day()
#     except Exception as e:
#         print(f"[warn] final flush failed: {e}")

#     print(f"Done. Processed {processed_files} file(s).")

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
    start_date=None,   # "YYYY-MM-DD" or None
    end_date=None,     # "YYYY-MM-DD" or None
    skip_if_exists=True,
    verbose=True,
    output_units: str = "density",  # "density" (dB re 1 µPa²/Hz) or "band" (dB re 1 µPa²)
):
    bucket_name, prefix = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # ---- enumerate and pre-filter by day ----
    blobs, blob_days = list_sorted_wav_blobs(bucket, prefix)

    unique_days = np.unique(np.array(blob_days))
    if start_date is not None:
        start_d = as_day64(start_date)
        unique_days = unique_days[unique_days >= start_d]
    if end_date is not None:
        end_d = as_day64(end_date)
        unique_days = unique_days[unique_days <= end_d]

    if skip_if_exists:
        keep = []
        for d in unique_days:
            date_str = np.datetime_as_string(d, unit="D").replace("-", "")
            out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
            if not os.path.exists(out_path):
                keep.append(d)
            elif verbose:
                print(f"[skip-day] {str(d)} exists in output; skipping")
        unique_days = np.array(keep)

    if unique_days.size == 0:
        print("[info] Nothing to do for given date window / skip rules.")
        return

    first_day = unique_days[0]
    start_idx = find_first_index_on_or_after(blob_days, first_day)

    days_to_process = set(unique_days.tolist())

    # ---- state ----
    os.makedirs(out_dir, exist_ok=True)
    current_day = None           # np.datetime64[D]
    day_parts = []               # list of per-day xarray Datasets
    processed_files = 0

    def _flush_day():
        nonlocal current_day, day_parts
        if current_day is None or not day_parts:
            return
        date_str = np.datetime_as_string(current_day, unit="D").replace("-", "")
        out_path = os.path.join(out_dir, f"HybridMilli_{date_str}.nc")
        day_ds = xr.concat(day_parts, dim="time").sortby("time")
        day_ds = day_ds.assign_attrs({
            **day_ds.attrs,
            "provenance": "GCS streamed -> ASA.hybrid_millidecade_bands -> freq-calibrated -> daily split",
            "bucket": bucket_name,
            "prefix": prefix,
            "binsize_s": binsize,
            "nfft": nfft,
            "band_hz": list(band),
            "timezone": "UTC",
            "output_units": output_units.lower(),
        })
        _encode_and_write(day_ds, out_path)
        if verbose:
            print(f"[write] {out_path}  (time bins: {day_ds.dims.get('time', 'NA')})")
        current_day = None
        day_parts = []

    # ---- main loop (with FRONT-FLUSH) ----
    seq = list(zip(blobs[start_idx:], blob_days[start_idx:]))
    for blob, bday in tqdm(seq, desc="GCS → ASA → daily NetCDF"):
        if max_files is not None and processed_files >= max_files:
            break

        # FRONT-FLUSH: if next blob starts a new day, write the old one now
        if current_day is not None and day_parts and bday != current_day:
            if verbose:
                print(f"[day flip] {str(current_day)} -> {str(bday)} (pre-file flush)")
            _flush_day()

        # Optional fast skip when clearly outside the target window
        if (bday < unique_days.min()) or (bday > unique_days.max()):
            continue

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
                    calibration=-1,   # use hydrophone info for chain constants
                )
                ds = asa_cal.hybrid_millidecade_bands(
                    db=True, method="density", band=band, percentiles=None
                )

                # Per-time-bin provenance (optional)
                if "time" in ds.coords:
                    ds["source_file"] = xr.DataArray(
                        np.array([blob.name] * ds.sizes["time"]),
                        dims=["time"], coords={"time": ds["time"]},
                    )

                # >>> NEW: apply hydrophone frequency-dependent response (guarded)
                ds = _apply_freq_response(ds, drifter_Cal)

                # >>> OPTIONAL: convert density → band levels
                if output_units.lower() == "band":
                    ds = _add_bandwidth_to_density(ds)

                # ---- split by day and accumulate only requested days ----
                if "time" not in ds.coords:
                    # Assign entire file to its start day; only add if wanted.
                    k = bday
                    if k in days_to_process:
                        if current_day is None:
                            current_day = k; day_parts = [ds]
                            if verbose: print(f"[day start] {str(current_day)} (no time coord)")
                        elif k == current_day:
                            day_parts.append(ds)
                        else:
                            if verbose: print(f"[day flip] {str(current_day)} -> {str(k)} (no time coord)")
                            _flush_day(); current_day = k; day_parts = [ds]
                else:
                    # Proper split; only keep groups whose day is requested
                    ds = ds.assign_coords(day=ds["time"].dt.floor("D"))
                    for day_key, ds_day in ds.groupby("day"):
                        k = np.datetime64(day_key, "D")
                        if k not in days_to_process:
                            continue
                        if current_day is None:
                            current_day = k; day_parts = [ds_day]
                            if verbose: print(f"[day start] {str(current_day)}")
                        elif k == current_day:
                            day_parts.append(ds_day)
                        else:
                            if verbose: print(f"[day flip] {str(current_day)} -> {str(k)}")
                            _flush_day(); current_day = k; day_parts = [ds_day]

            processed_files += 1

        except Exception as e:
            print(f"[skip] {blob.name}: {e}")

    # trailing flush (only if it’s one of the requested days)
    try:
        if current_day in days_to_process:
            _flush_day()
    except Exception as e:
        print(f"[warn] final flush failed: {e}")

    print(f"Done. Processed {processed_files} file(s).")
