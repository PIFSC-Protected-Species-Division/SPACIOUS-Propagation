# -*- coding: utf-8 -*-
"""
H5ArrivalsBridge.py — small helpers to read your arrivals from HDF5
and package them for plotting/processing.

Targets the layout produced by your save_dive_frequency():
  drift_{drift_id}/dive_{dive_id}/frequency_{freq_khz}/
    - lat, lon, depth, tl, valid_len
    - arrivals/pt_00000, pt_00001, ... (each holding DataFrame-like columns)

Functions:
- list_points(...)
- load_point_by_index(...)
- load_point_near(...)
"""

import h5py
import numpy as np
from geopy.distance import geodesic

# ---- minimal Thorp for optional path length estimation guardrails ----
def _thorp_alpha_db_per_km(f_hz):
    f = np.maximum(np.asarray(f_hz, float), 1.0)/1000.0
    return 0.11*(f**2/(1+f**2)) + 44*(f**2/(4100+f**2)) + 2.75e-4*f**2 + 0.003

# ---- internal utils ---------------------------------------------------------

def _try_get(ds_group, names, default=None):
    """Return first dataset in 'names' that exists, else default."""
    for n in names:
        if n in ds_group:
            return ds_group[n][()]
    return default

def _to_complex(maybe_complex, grp):
    """
    If 'maybe_complex' already complex, return as-is.
    Else try to build from ('amp_real','amp_imag') fallbacks.
    """
    arr = maybe_complex
    if arr is not None and np.iscomplexobj(arr):
        return np.asarray(arr)
    # try split real/imag
    re = _try_get(grp, ["amp_real", "arrival_amp_real", "amplitude_real"])
    im = _try_get(grp, ["amp_imag", "arrival_amp_imag", "amplitude_imag"])
    if re is not None and im is not None:
        return np.asarray(re) + 1j*np.asarray(im)
    # last resort: treat as real-only
    if arr is not None:
        return np.asarray(arr, dtype=float) + 0j
    return None

def _normalize_vector(x):
    if x is None:
        return None
    x = np.asarray(x)
    return x.ravel() if x.ndim > 1 else x

def _point_group(h5, drift_id, dive_id, freq_khz, pt_index):
    base = f"drift_{drift_id}/dive_{dive_id}/frequency_{freq_khz}/arrivals"
    return h5[f"{base}/pt_{pt_index:05d}"]

def _freq_group(h5, drift_id, dive_id, freq_khz):
    return h5[f"drift_{drift_id}/dive_{dive_id}/frequency_{freq_khz}"]

# ---- public API -------------------------------------------------------------

def list_points(h5_path, dive_id, drift_id="01", freq_khz=35000):
    """
    Returns a dict with index→(lat,lon,n_rows) and also the lat/lon vectors.
    Useful to browse/choose a point.
    """
    with h5py.File(h5_path, "r") as h5:
        grp = _freq_group(h5, drift_id, dive_id, freq_khz)
        lat = np.asarray(grp["lat"])
        lon = np.asarray(grp["lon"])
        arr_root = grp["arrivals"]

        out = {}
        for name in arr_root.keys():
            if not name.startswith("pt_"):
                continue
            i = int(name.split("_")[1])
            nrows = arr_root[name].attrs.get("n_rows", None)
            out[i] = (float(lat[i]), float(lon[i]), int(nrows) if nrows is not None else None)
    return {"index_map": out, "lat": lat, "lon": lon}

def load_point_by_index(h5_path, 
                        dive_id, 
                        pt_index= 'pt_00001',
                        drift_id="01", 
                        freq_khz='frequency_35000',
                        c_eff_m_s=1480.0, 
                        estimate_pathlen_if_missing=True):
    """
    Load one arrivals point and package as:
      dict(time_of_arrival, arrival_amplitude, path_length_m[opt],
           rx_depth_ndx[opt], rx_range_ndx[opt], tx_depth_ndx[opt])
    """
    with h5py.File(h5_path, "r") as h5:
        grp = h5[drift_id][dive_id][freq_khz]

        # time fields found in your pipeline
        toa = np.array(h5[drift_id][dive_id][freq_khz]['arrivals'][pt_index]['time_of_arrival'])
        amp = np.array(h5[drift_id][dive_id][freq_khz]['arrivals'][pt_index]['arrival_amplitude'])

        # optional indices (handy for filtering/QA)
        rx_d = np.array(h5[drift_id][dive_id][freq_khz]['arrivals'][pt_index]['rx_depth'])
        rx_r = np.array(h5[drift_id][dive_id][freq_khz]['arrivals'][pt_index]['rx_range'])
        tx_d = np.array(h5[drift_id][dive_id][freq_khz]['arrivals'][pt_index]['tx_depth'])

        # optional direct path length
        Rm = np.array(h5[drift_id][dive_id][freq_khz]['arrivals'][pt_index]['rx_range'])

        toa = _normalize_vector(toa)
        amp = _normalize_vector(amp)
        rx_d = _normalize_vector(rx_d)
        rx_r = _normalize_vector(rx_r)
        tx_d = _normalize_vector(tx_d)
        Rm   = _normalize_vector(Rm)

        if toa is None or amp is None or toa.size == 0 or amp.size == 0:
            # empty point
            return {
                "time_of_arrival": np.array([]),
                "arrival_amplitude": np.array([], dtype=complex),
                "path_length_m": None
            }

        if Rm is None and estimate_pathlen_if_missing:
            Rm = toa * float(c_eff_m_s)

        out = {
            "time_of_arrival": np.asarray(toa, float),
            "arrival_amplitude": np.asarray(amp, complex),
            "path_length_m": np.asarray(Rm, float) if Rm is not None else None
        }
        # pass-through indices if present (non-essential for plotting)
        if rx_d is not None: out["rx_depth_ndx"] = rx_d
        if rx_r is not None: out["rx_range_ndx"] = rx_r
        if tx_d is not None: out["tx_depth_ndx"] = tx_d
        return out

def load_point_near(h5_path, dive_id, target_lat, target_lon,
                    drift_id="01", freq_khz=35000, **kwargs):
    """
    Choose the nearest grid point by (lat,lon) and load its arrivals.
    Returns (arrivals_dict, pt_index, nearest_lat, nearest_lon, distance_km).
    """
    with h5py.File(h5_path, "r") as h5:
        grp = _freq_group(h5, drift_id, dive_id, freq_khz)
        lat = np.asarray(grp["lat"])
        lon = np.asarray(grp["lon"])

    # brute force nearest (N usually manageable; otherwise KD-tree)
    dists = np.array([geodesic((target_lat, target_lon), (float(lat[i]), float(lon[i]))).km
                      for i in range(len(lat))])
    idx = int(np.nanargmin(dists))
    arr = load_point_by_index(h5_path, dive_id, idx, drift_id=drift_id, freq_khz=freq_khz, **kwargs)
    return arr, idx, float(lat[idx]), float(lon[idx]), float(dists[idx])
