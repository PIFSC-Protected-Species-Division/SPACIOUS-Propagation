# -*- coding: utf-8 -*-
"""
PlottingDefs.py — utilities for peak-to-peak grids, plotting, and detection stats.

Key points:
- Windows uses threads by default for CreateOutputCSVs so you can call it at top-level
  (no __main__ guard required). On POSIX you can opt into processes.
- Workers open HDF5 inside the worker (no unpicklables passed).
"""

import os
import sys
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata, NearestNDInterpolator
from skimage import measure
from geopy.distance import geodesic
from pyproj import Transformer, Geod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Optional: tqdm for nicer progress bars (safe if not installed)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs): return it

# ---- BLAS env sanity (keep lightweight at import time) -----------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")

# ---- Globals (lightweight) ---------------------------------------------------
geod = Geod(ellps="WGS84")

# =============================================================================
# Core helpers
# =============================================================================

def scaleP2P(segment: np.ndarray, outP2P: float = 220.0) -> np.ndarray:
    """Scale 'segment' so its peak-to-peak is 'outP2P' dB (re linear units)."""
    init_p2pdB = 20.0 * np.log10(np.ptp(segment))
    gain = 10 ** ((outP2P - init_p2pdB) / 20.0)
    return segment * gain


def arrivals_to_impulse_response(arrivals: dict, fs: int, abs_time: bool = False) -> np.ndarray:
    """Convert sparse arrivals (toa, amp) into a complex-valued impulse response."""
    toa = arrivals["time_of_arrival"]
    amp = arrivals["arrival_amplitude"]
    t0 = 0 if abs_time else np.min(toa)
    irlen = int(np.ceil((np.max(toa) - t0) * fs)) + 1
    ir = np.zeros(irlen, dtype=np.complex128)
    for i in range(len(toa)):
        ndx = int(np.round((toa[i].real - t0) * fs))
        if 0 <= ndx < irlen:
            ir[ndx] = amp[i]
    return ir


def p2pArrivalSNR(arrivals: dict, segment: np.ndarray, fs: int):
    """Convolve segment with IR → return peak-to-peak level (dB) and the waveform."""
    ir = arrivals_to_impulse_response(arrivals, fs=fs, abs_time=False)
    out_sig = np.convolve(segment, ir)[:len(segment)]
    out_real = np.real(out_sig)
    p2p_db = 20.0 * np.log10(np.ptp(out_real))
    return np.round(p2p_db, 1), out_real


# =============================================================================
# Parallel CSV building
# =============================================================================

def _enumerate_dives(h5_path: str):
    """List dive IDs under drift_01."""
    with h5py.File(h5_path, "r") as hf:
        return list(hf["drift_01"].keys())


def process_run(run_id: str, run_index: int, depth_row: np.ndarray,
                segment: np.ndarray, fs: int, h5_path: str, dive_id: str):
    """
    Worker: compute p2p outputs for one run across all depth indices present in depth_row.
    Returns (run_index, [(depth_idx, p2p_db), ...])
    """
    required = ("time_of_arrival", "arrival_amplitude", "tx_depth_ndx", "rx_depth_ndx", "rx_range_ndx")
    results = []

    with h5py.File(h5_path, "r") as hf:
        path = f"drift_01/{dive_id}/frequency_35000/arrivals/{run_id}"
        arr0 = hf[path]
        data = {}
        for name in required:
            if name not in arr0:
                continue
            arr = arr0[name][()]
            # handle masked arrays defensively
            try:
                arr = arr.data if hasattr(arr, "data") and getattr(arr, "mask", None) is not None else arr
            except Exception:
                pass
            data[name] = arr.ravel() if np.ndim(arr) > 1 else arr

        if len(data.get("rx_depth_ndx", [])) == 0:
            return run_index, []

        # Only evaluate for depths that exist in this run's row
        depth_idxs = np.where(depth_row > 0)[0]
        for d_idx in depth_idxs:
            mask = data["rx_depth_ndx"] == d_idx
            if not np.any(mask):
                results.append((d_idx, np.nan))
                continue
            hyd = {k: v[mask] for k, v in data.items()}
            ptp_out, _ = p2pArrivalSNR(hyd, segment, fs)
            results.append((d_idx, ptp_out))

    return run_index, results


def _choose_executor(prefer_processes: bool):
    """
    Return (ExecutorClass, mode_label).
    - On Windows, default to threads to avoid spawn re-import issues when callers
      invoke at top-level.
    - On POSIX, allow either.
    """
    if os.name == "nt":
        # Keep simple: threads by default even if prefer_processes=True,
        # unless the caller knows to use a proper __main__ launcher.
        return ThreadPoolExecutor, "thread"
    return (ProcessPoolExecutor if prefer_processes else ThreadPoolExecutor,
            "process" if prefer_processes else "thread")


def CreateOutputCSVs(h5_path: str,
                     segment: np.ndarray,
                     samplerate: int,
                     out_path: str,
                     nWorkers: int = 10,
                     prefer_processes: bool = False):
    """
    Build per-dive peak-to-peak CSVs in parallel over RUNS.

    Windows: defaults to threads (safe for top-level calls).
    POSIX: set prefer_processes=True if CPU-bound and you want processes.
    """
    # Defensive: never do heavy work from a spawned child
    if mp.current_process().name != "MainProcess":
        return

    os.makedirs(out_path, exist_ok=True)
    dive_ids = _enumerate_dives(h5_path)

    for dive_id in dive_ids:
        print(f"Processing dive: {dive_id}")
        # Read lightweight metadata once in parent
        with h5py.File(h5_path, "r") as hf:
            dive_grp = hf[f"drift_01/{dive_id}/frequency_35000"]
            run_ids = list(dive_grp["arrivals"].keys())
            depth_grid = np.array(dive_grp["depth"])
            drifter_depth = float(dive_grp.parent.attrs["drifter_depth"])

        p2p_grid = np.full_like(depth_grid, np.nan, dtype=np.float64)

        Executor, mode = _choose_executor(prefer_processes)

        with Executor(max_workers=nWorkers) as pool:
            futures = []
            for run_index, run_id in enumerate(run_ids):
                depth_row = depth_grid[run_index, :]
                futures.append(pool.submit(
                    process_run, run_id, run_index, depth_row,
                    segment, samplerate, h5_path, dive_id
                ))

            for fut in tqdm(futures, desc=f"Dive {dive_id}"):
                run_index, pairs = fut.result()
                for depth_idx, val in pairs:
                    p2p_grid[run_index, depth_idx] = val

        out_name = f"PeakToPeak_{dive_id}_GliderDepth_{int(round(drifter_depth))}m.csv"
        out_file = os.path.join(out_path, out_name)
        np.savetxt(out_file, p2p_grid, delimiter=",")
        print(f"Saved: {out_file}")


# =============================================================================
# Frequency-dependent absorption
# =============================================================================

def calc_seawater_absorption(frequency, distance=1000, temperature=27,
                             salinity=35, pressure=10, pH=8.1, formula_source="AM"):
    """Return sea absorption in dB/m for various formulas (AM default)."""
    if formula_source == "FG":
        f = frequency / 1000.0
        d = distance / 1000.0
        c = 1412.0 + 3.21 * temperature + 1.19 * salinity + 0.0167 * pressure
        A1 = 8.86 / c * 10 ** (0.78 * pH - 5)
        P1 = 1.0
        f1 = 2.8 * np.sqrt(salinity / 35) * 10 ** (4 - 1245 / (temperature + 273))
        A2 = 21.44 * salinity / c * (1 + 0.025 * temperature)
        P2 = 1.0 - 1.37e-4 * pressure + 6.2e-9 * pressure * pressure
        f2 = 8.17 * 10 ** (8 - 1990 / (temperature + 273)) / (1 + 0.0018 * (salinity - 35))
        P3 = 1.0 - 3.83e-5 * pressure + 4.9e-10 * pressure * pressure
        if temperature < 20:
            A3 = (4.937e-4 - 2.59e-5 * temperature + 9.11e-7 * temperature ** 2 - 1.5e-8 * temperature ** 3)
        else:
            A3 = 3.964e-4 - 1.146e-5 * temperature + 1.45e-7 * temperature ** 2 - 6.5e-10 * temperature ** 3
        a = (A1 * P1 * f1 * f * f / (f1 * f1 + f * f)
             + A2 * P2 * f2 * f * f / (f2 * f2 + f * f)
             + A3 * P3 * f * f)
        sea_abs = -20 * np.log10(10 ** (-a * d / 20.0)) / 1000  # dB/m from dB/km
    elif formula_source == "AM":
        freq = frequency / 1000.0
        D = pressure / 1000.0
        f1 = 0.78 * np.sqrt(salinity / 35) * np.exp(temperature / 26)
        f2 = 42 * np.exp(temperature / 17)
        a1 = 0.106 * (f1 * (freq ** 2)) / ((f1 ** 2) + (freq ** 2)) * np.exp((pH - 8) / 0.56)
        a2 = (0.52 * (1 + temperature / 43) * (salinity / 35)
              * (f2 * (freq ** 2)) / ((f2 ** 2) + (freq ** 2)) * np.exp(-D / 6))
        a3 = 0.00049 * (freq ** 2) * np.exp(-(temperature / 27 + D))
        sea_abs = (a1 + a2 + a3) / 1000.0
    elif formula_source == "AZFP":
        temp_k = temperature + 273.0
        f1 = 1320.0 * temp_k * np.exp(-1700 / temp_k)
        f2 = 1.55e7 * temp_k * np.exp(-3052 / temp_k)
        k = 1 + pressure / 10.0
        a = 8.95e-8 * (1 + temperature * (2.29e-2 - 5.08e-4 * temperature))
        b = (salinity / 35.0) * 4.88e-7 * (1 + 0.0134 * temperature) * (1 - 0.00103 * k + 3.7e-7 * (k * k))
        c = (4.86e-13 * (1 + temperature * ((-0.042) + temperature * (8.53e-4 - temperature * 6.23e-6)))
             * (1 + k * (-3.84e-4 + k * 7.57e-8)))
        if salinity == 0:
            sea_abs = c * frequency ** 2
        else:
            sea_abs = ((a * f1 * (frequency ** 2)) / ((f1 * f1) + (frequency ** 2))
                       + (b * f2 * (frequency ** 2)) / ((f2 * f2) + (frequency ** 2))
                       + c * (frequency ** 2))
    else:
        raise ValueError("Unknown formula source")
    return sea_abs


def alphaAdjustment(bellhopFreq: float = 35000, newFreq: float = 2000) -> float:
    """Difference in absorption coefficient (dB/km) between two frequencies."""
    a1 = calc_seawater_absorption(bellhopFreq)
    a2 = calc_seawater_absorption(newFreq)
    return (a1 - a2) * 1000.0  # dB/km


def apply_alpha_correction(h5_path: str, RLdata: np.ndarray, alpha_db_per_km: float,
                           diveId: str = "dive_42") -> np.ndarray:
    """Apply slant-range alpha correction to RL grid."""
    with h5py.File(h5_path, "r") as hf:
        grp = hf[f"drift_01/{diveId}/frequency_35000"]
        depth_grid = np.array(grp["depth"])
        lat = np.array(grp["lat"])
        lon = np.array(grp["lon"])
        drifter_lat = float(grp.parent.attrs["start_lat"])
        drifter_lon = float(grp.parent.attrs["start_lon"])
        drifter_depth = float(grp.parent.attrs["drifter_depth"])

    # Align shapes defensively
    nloc, ndep = RLdata.shape
    if lat.shape[0] > nloc:
        lat = lat[:nloc]; lon = lon[:nloc]; depth_grid = depth_grid[:nloc, :]

    corrected = RLdata.copy()
    for i in range(nloc):
        horiz_km = geodesic((drifter_lat, drifter_lon), (lat[i], lon[i])).km
        for j in range(ndep):
            val = RLdata[i, j]
            if val == -500 or not np.isfinite(val):
                continue
            vertical_km = abs(depth_grid[i, j] - drifter_depth) / 1000.0
            slant_km = np.hypot(horiz_km, vertical_km)
            corrected[i, j] = val + alpha_db_per_km * slant_km
    return corrected


# =============================================================================
# Plotting
# =============================================================================

def plot_peak2peak_isosurfaces(h5_path: str, pp_grid: np.ndarray, diveId: str = "dive_42",
                               iso_levels=(90,), xy_res=200, cmap=cm.viridis,
                               seabed_color="0.6", elev=25, azim=-45):
    """Marching-cubes 3D visualization of p2p iso-levels over a local UTM grid."""
    with h5py.File(h5_path, "r") as hf:
        grp = hf[f"drift_01/{diveId}/frequency_35000"]
        depth_grid = np.array(grp["depth"])
        lat = np.array(grp["lat"])
        lon = np.array(grp["lon"])
        drifter_lat = float(grp.parent.attrs["start_lat"])
        drifter_lon = float(grp.parent.attrs["start_lon"])

    # Local UTM centered at drifter
    utm_zone = int((drifter_lon + 180) // 6) + 1
    hemisphere = "north" if drifter_lat >= 0 else "south"
    transformer = Transformer.from_crs(
        "epsg:4326", f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    x0, y0 = transformer.transform(drifter_lon, drifter_lat)
    x, y = x - x0, y - y0

    xi = np.linspace(x.min(), x.max(), xy_res)
    yi = np.linspace(y.min(), y.max(), xy_res)
    X2d, Y2d = np.meshgrid(xi, yi)

    z_vec = depth_grid[0, :]
    Z = len(z_vec)
    PP_vol = np.full((Z, xy_res, xy_res), np.nan, dtype=float)

    for k in range(Z):
        valid = np.isfinite(pp_grid[:, k])
        if valid.sum() < 3:
            continue
        pts = np.column_stack((x[valid], y[valid]))
        vals = pp_grid[valid, k]
        try:
            PP_slice = griddata(pts, vals, (X2d, Y2d), method="cubic")
        except Exception:
            PP_slice = NearestNDInterpolator(pts, vals)(X2d, Y2d)
        PP_vol[k] = PP_slice

    finite_vals = PP_vol[np.isfinite(PP_vol)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite peak-to-peak values – check inputs")
    real_min, real_max = finite_vals.min(), finite_vals.max()
    nan_fill = real_min - 1.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for level in iso_levels:
        if not (real_min < level < real_max):
            print(f"⚠️  Skipping {level} dB – outside [{real_min:.1f}, {real_max:.1f}] dB")
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(np.nan_to_num(PP_vol, nan=nan_fill), level=level)
        except RuntimeError as e:
            print(f"⚠️  marching_cubes failed for {level} dB: {e}")
            continue

        idx_x = np.clip(np.round(verts[:, 2]).astype(int), 0, xi.size - 1)
        idx_y = np.clip(np.round(verts[:, 1]).astype(int), 0, yi.size - 1)
        idx_z = np.clip(np.round(verts[:, 0]).astype(int), 0, Z - 1)

        verts_xyz = np.empty_like(verts)
        verts_xyz[:, 0] = xi[idx_x]
        verts_xyz[:, 1] = yi[idx_y]
        verts_xyz[:, 2] = z_vec[idx_z]

        face_color = cmap((level - real_min) / (real_max - real_min))
        ax.add_collection3d(
            Poly3DCollection(verts_xyz[faces], facecolor=face_color, edgecolor="none", alpha=0.4)
        )

    # Seabed surface (simple max depth per location)
    seabed_raw = np.nanmax(depth_grid, axis=1)
    valid_cols = np.isfinite(seabed_raw)
    seabed_grid = griddata((x[valid_cols], y[valid_cols]), seabed_raw[valid_cols], (X2d, Y2d),
                           method="linear", fill_value=np.nan)
    ax.plot_surface(X2d, Y2d, np.ma.masked_invalid(seabed_grid),
                    color=seabed_color, alpha=0.6, linewidth=0, antialiased=False)

    ax.set_xlabel("East–West range (m)")
    ax.set_ylabel("North–South range (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_zlim(np.nanmax(seabed_raw), 0)
    ax.set_title(f"Peak-to-Peak Iso-Surfaces ({iso_levels} dB re 1 µPa p-p)")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def plot_detection_probability(h5_path: str, RLdata: np.ndarray, threshold_db: float,
                               cmap="viridis", diveId: str = "dive_42", vmin=0, vmax=1,
                               title=None, s=40):
    """Scatter map of detection probability per location."""
    with h5py.File(h5_path, "r") as hf:
        grp = hf[f"drift_01/{diveId}/frequency_35000"]
        lat = np.array(grp["lat"]); lon = np.array(grp["lon"])
        drifter_lat = float(grp.parent.attrs["start_lat"])
        drifter_lon = float(grp.parent.attrs["start_lon"])

    is_valid = RLdata > -500
    is_detected = RLdata > threshold_db
    detection_prob = (is_detected & is_valid).sum(axis=1) / np.maximum(is_valid.sum(axis=1), 1)
    detection_prob = np.nan_to_num(detection_prob, nan=0.0)

    utm_zone = int((drifter_lon + 180) // 6) + 1
    hemisphere = "north" if drifter_lat >= 0 else "south"
    transformer = Transformer.from_crs(
        "epsg:4326", f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    x0, y0 = transformer.transform(drifter_lon, drifter_lat)
    x, y = x - x0, y - y0

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=detection_prob, cmap=cmap, vmin=vmin, vmax=vmax, s=s)
    fig.colorbar(sc, ax=ax, label="Detection Probability")
    ax.set_xlabel("East–West Range (m)")
    ax.set_ylabel("North–South Range (m)")
    ax.set_title(title or f"Detection Probability (Threshold: {threshold_db} dB)")
    ax.grid(True); ax.axis("equal")
    plt.tight_layout(); plt.show()
    return detection_prob


def plot_detection_vs_range(h5_path: str, RLdata: np.ndarray, diveId: str = "dive_42",
                            threshold_db: float = 120.0, bin_width_km: float = 1.0,
                            min_range_km: float = 0.0, max_range_km: float = 20,
                            color="tab:blue", title=None) -> pd.DataFrame:
    """Median detection probability vs range with 95% CI."""
    with h5py.File(h5_path, "r") as hf:
        grp = hf[f"drift_01/{diveId}/frequency_35000"]
        lat = np.array(grp["lat"]); lon = np.array(grp["lon"])
        drifter_lat = float(grp.parent.attrs["start_lat"])
        drifter_lon = float(grp.parent.attrs["start_lon"])

    is_valid = RLdata > -500
    is_detected = RLdata > threshold_db
    dp = (is_detected & is_valid).sum(axis=1) / np.maximum(is_valid.sum(axis=1), 1)
    dp = np.nan_to_num(dp, nan=0.0)

    ranges_km = np.array([geodesic((drifter_lat, drifter_lon), (lat[i], lon[i])).km for i in range(len(lat))])
    if max_range_km is None:
        max_range_km = float(np.nanmax(ranges_km))

    bins = np.arange(min_range_km, max_range_km + bin_width_km, bin_width_km)
    centers = (bins[:-1] + bins[1:]) / 2.0

    med, lo95, hi95 = [], [], []
    for i in range(len(bins) - 1):
        in_bin = (ranges_km >= bins[i]) & (ranges_km < bins[i + 1])
        vals = dp[in_bin]
        if vals.size == 0:
            med.append(np.nan); lo95.append(np.nan); hi95.append(np.nan)
        else:
            med.append(np.nanmedian(vals))
            lo95.append(np.nanpercentile(vals, 2.5))
            hi95.append(np.nanpercentile(vals, 97.5))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(centers, med, color=color, label="Median")
    ax.fill_between(centers, lo95, hi95, color=color, alpha=0.3, label="95% CI")
    ax.set_xlabel("Range from Drifter (km)")
    ax.set_ylabel("Detection Probability")
    ax.set_ylim(0, 1); ax.set_xlim(min_range_km, max_range_km)
    ax.grid(True); ax.legend()
    ax.set_title(title or f"Detection Probability vs Range (Threshold: {threshold_db} dB)")
    plt.tight_layout(); plt.show()

    return pd.DataFrame({"range_km": centers, "median": med, "lower_95": lo95, "upper_95": hi95})


def plot_detection_by_bearing(RLdata: np.ndarray, h5_path: str, diveId: str = "dive_42",
                              threshold_db: float = 120.0, bearing_bin_width: float = 10.0,
                              range_bin_width_km: float = 0.2, min_range_km: float = 0.0,
                              max_range_km: float =20, cmap="turbo", alpha=0.6,
                              title=None):
    """Median detection probability vs range for each bearing sector."""
    from matplotlib.cm import get_cmap

    with h5py.File(h5_path, "r") as hf:
        grp = hf[f"drift_01/{diveId}/frequency_35000"]
        lat = np.array(grp["lat"]); lon = np.array(grp["lon"])
        drifter_lat = float(grp.parent.attrs["start_lat"])
        drifter_lon = float(grp.parent.attrs["start_lon"])

    is_valid = RLdata > -500
    is_detected = RLdata > threshold_db
    dp = (is_detected & is_valid).sum(axis=1) / np.maximum(is_valid.sum(axis=1), 1)
    dp = np.nan_to_num(dp, nan=0.0)

    N = len(lat)
    bearings = np.zeros(N); ranges_km = np.zeros(N)
    for i in range(N):
        az12, _, dist_m = geod.inv(drifter_lon, drifter_lat, lon[i], lat[i])
        bearings[i] = az12 % 360.0
        ranges_km[i] = dist_m / 1000.0

    if max_range_km is None:
        max_range_km = float(np.nanmax(ranges_km))

    bearing_bins = np.arange(0, 360, bearing_bin_width)
    range_bins = np.arange(min_range_km, max_range_km + range_bin_width_km, range_bin_width_km)
    centers = (range_bins[:-1] + range_bins[1:]) / 2.0

    stats_by_bearing = {}
    cmap_obj = get_cmap(cmap, len(bearing_bins))

    fig, ax = plt.subplots(figsize=(10, 6))
    for b_idx, b0 in enumerate(bearing_bins):
        b1 = b0 + bearing_bin_width
        in_slice = (bearings >= b0) & (bearings < b1)
        if np.sum(in_slice) < 5:
            continue
        dp_slice = dp[in_slice]; r_slice = ranges_km[in_slice]

        dp_binned = []
        for i in range(len(range_bins) - 1):
            in_bin = (r_slice >= range_bins[i]) & (r_slice < range_bins[i + 1])
            vals = dp_slice[in_bin]
            dp_binned.append(np.nanmedian(vals) if vals.size > 0 else np.nan)

        df = pd.DataFrame({"range_km": centers, "detection_prob": dp_binned})
        stats_by_bearing[b0] = df
        ax.plot(df["range_km"], df["detection_prob"], label=f"{b0:.0f}–{b1:.0f}°",
                color=cmap_obj(b_idx), alpha=alpha)

    ax.set_xlabel("Range from Drifter (km)")
    ax.set_ylabel("Detection Probability")
    ax.set_ylim(0, 1); ax.set_xlim(min_range_km, max_range_km)
    ax.set_title(title or f"Detection Probability vs Range by Bearing\n(Threshold: {threshold_db} dB)")
    ax.grid(True)
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(); plt.show()
    return stats_by_bearing


# =============================================================================
# Hazard-rate fitting
# =============================================================================

def hazard_rate(r, sigma, b):
    return 1 - np.exp(-(r / sigma) ** -b)

def hazard_rate_log(r, sigma, b):
    r = np.maximum(r, 0.01)
    return 1 - np.exp(-(np.log(r) / sigma) ** -b)

def fit_and_plot_hazard_rate_by_location(my_data: np.ndarray,
                                         lat: np.ndarray,
                                         lon: np.ndarray,
                                         depth_grid: np.ndarray,
                                         drifter_lat: float, drifter_lon: float,
                                         threshold_db: float = 120.0,
                                         n_boot: int = 500,
                                         smooth_range: np.ndarray = np.linspace(0.05, 20, 200),
                                         ci_alpha: float = 0.3,
                                         use_log_scale: bool = True,
                                         seed: int = 42) -> pd.DataFrame :
    """Fit hazard-rate curve to location-level detection proportions; bootstrap CI."""
    from scipy.optimize import curve_fit

    np.random.seed(seed)
    N = len(lat)
    ranges_km = np.array([geodesic((drifter_lat, drifter_lon), (lat[i], lon[i])).km for i in range(N)])

    is_valid = my_data > -500
    is_detected = my_data > threshold_db
    num_valid = is_valid.sum(axis=1)
    num_detected = (is_valid & is_detected).sum(axis=1)
    proportions = np.divide(num_detected, np.maximum(num_valid, 1), where=num_valid > 0)

    mask = (num_valid > 0) & np.isfinite(proportions)
    X = np.clip(ranges_km[mask], 2, None)
    Y = proportions[mask]

    model_fn = hazard_rate_log if use_log_scale else hazard_rate

    try:
        popt, _ = curve_fit(model_fn, X, Y, p0=(0.5, 2.0),
                            bounds=([0.01, 0.1], [10.0, 10.0]), maxfev=10000)
    except Exception as e:
        print(f"⚠️ Initial fit failed: {e}")
        return None

    y_fit = model_fn(smooth_range, *popt)

    # Bootstrap at binomial level
    boot_preds = []
    Xo = X.copy()
    Yo = Y.copy()
    n_valid_o = num_valid[mask]
    for _ in range(n_boot):
        boot_detected = [np.random.binomial(n, p) for n, p in zip(n_valid_o, Yo)]
        Yb = np.array(boot_detected) / np.maximum(n_valid_o, 1)
        try:
            bopt, _ = curve_fit(model_fn, Xo, Yb, p0=(0.5, 2.0),
                                bounds=([0.01, 0.1], [10.0, 10.0]), maxfev=10000)
            boot_preds.append(model_fn(smooth_range, *bopt))
        except Exception:
            continue

    boot_preds = np.array(boot_preds)
    boot_preds = boot_preds[~np.isnan(boot_preds).all(axis=1)]
    print(f"✅ {boot_preds.shape[0]} successful bootstrap fits")

    if boot_preds.shape[0] < 10:
        lower = np.full_like(smooth_range, np.nan)
        upper = np.full_like(smooth_range, np.nan)
    else:
        lower = np.nanpercentile(boot_preds, 2.5, axis=0)
        upper = np.nanpercentile(boot_preds, 97.5, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(smooth_range, y_fit, label="Fitted Hazard-Rate", color="blue")
    valid_ci = np.isfinite(lower) & np.isfinite(upper)
    if np.any(valid_ci):
        ax.fill_between(smooth_range[valid_ci], lower[valid_ci], upper[valid_ci],
                        color="blue", alpha=ci_alpha, label="95% CI")
    ax.scatter(Xo, Yo, color="black", s=20, zorder=3, label="Location Proportions")
    ax.set_xlabel("Range from Drifter (km)")
    ax.set_ylabel("Detection Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hazard-Rate Model (Threshold: {threshold_db} dB, {'log' if use_log_scale else 'linear'})")
    ax.grid(True); ax.legend()
    plt.tight_layout(); plt.show()

    print(f"✅ Fitted params: σ = {popt[0]:.3f}, b = {popt[1]:.3f}")

    return pd.DataFrame({"range_km": smooth_range, "mean": y_fit, "lower_95": lower, "upper_95": upper})
