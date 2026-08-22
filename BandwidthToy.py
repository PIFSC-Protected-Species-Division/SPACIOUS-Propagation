
"""
Created on Wed Jul  8 20:50:39 2026

@author: pam_user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:22:01 2026

@author: pam_user
"""

# -*- coding: utf-8 -*-
"""
Single-bearing Bellhop peak-to-peak TL diagnostic.

This is a small, standalone version of the larger propagation workflow. It
follows the same single-bearing bathymetry and SSP setup as
SingleAnglePropagationPlots.py, but it computes transmission loss from the
received waveform peak-to-peak level rather than from Bellhop coherent TL.

Per source depth, the script:
1. Loads and scales the source click waveform.
2. Runs Bellhop arrivals on a dense receiver grid.
3. Places the source click in a finite, guarded native-FFT buffer.
4. Builds each receiver transfer function directly on that FFT grid.
5. Multiplies the native source FFT by that transfer function.
6. Converts the received waveform to peak-to-peak TL:
       TL_p2p = SL_p2p - RL_p2p
7. Calculates PAMGuard-style 10 dB bandwidth and plots both grids.

The original SingleAnglePropagationPlots.py file is left unchanged.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import wavfile
from scipy.signal import resample_poly
from pyproj import Geod
import arlpy.uwapm as pm

from PlottingDefs import (
    arrivals_to_impulse_response,
    scaleP2P,
    thorp_alpha_db_per_km,
)


# =============================================================================
# User inputs
# =============================================================================

drift_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_CTD.csv"
)

gebco_nc = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\bathymetry\GEBCO_28_Jul_2025_937903cf24aa"
    r"\gebco_2024_n44.6_s40.2_w-126.3_e-124.0.nc"
)

drift_ends_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv"
)

source_wav_path = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\ExampleData\WHICEAS_click.wav"
)

SITE_END_INDEX = 15
BEARING_DEG = 270

MAX_RANGE_M = 20_000
BATHY_INTERVAL_M = 200
RX_RANGE_STEP_M = 200
RX_DEPTH_STEP_M = 100

SOURCE_DEPTHS_M = [500]
FREQ_HZ = 12_000

BOTTOM_SOUND_SPEED = 1575
BOTTOM_DENSITY = 1700
BOTTOM_ABSORPTION = 1

NBEAMS = 0
MIN_ANGLE = -90
MAX_ANGLE = 90
SOUNDSPEED_INTERP = "pchip"

CLICK_TARGET_FS_HZ = 500_000
SOURCE_P2P_DB = 220.0
COHERENT_FMIN_HZ = 1_000.0
COHERENT_FMAX_HZ = 24_000.0
C_EFF_M_S = 1480.0

# Native-FFT synthesis buffer. The click is placed after a guard interval so
# band-limited pre-ringing does not wrap to the end of the inverse FFT record.
SYNTHESIS_GUARD_TIME_S = 0.005

# PAMGuard-style click bandwidth settings.
# Set PAMGUARD_CLICK_SAMPLES to the saved click length configured in PAMGuard
# for the closest possible match. None uses the complete synthesized waveform.
PAMGUARD_CLICK_SAMPLES = 512       # e.g. 256, 512, or 1024
PAMGUARD_PRE_PEAK_FRACTION = 0.25  # fraction of the saved clip before the peak
PAMGUARD_USE_HANN = True
PAMGUARD_PEAK_SEARCH_HZ = (1_000.0, 24_000.0)
PAMGUARD_DB_DROP = 10.0
PAMGUARD_SMOOTHING_BINS = 5

MAX_ARRIVAL_DELAY = .01

OUT_DIR = "single_bearing_p2p_TL_test_outputs"

# Debug controls for Spyder step-through.
DEBUG_STOP_AFTER_FIRST_SOURCE = False
DEBUG_LIMIT_RX_RANGES = None   # e.g., 5
DEBUG_LIMIT_RX_DEPTHS = None   # e.g., 8


# =============================================================================
# Helper functions
# =============================================================================

GEOD = Geod(ellps="WGS84")


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in km for scalar or vector inputs."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def make_bathymetry_dataframe(gebco_path):
    """Load GEBCO NetCDF into a flat DataFrame matching the full script."""
    ds = xr.open_dataset(gebco_path)
    return pd.DataFrame({
        "depth": ds["elevation"].values.flatten(),
        "lat": np.repeat(ds["lat"].values, len(ds["lon"])),
        "lon": np.tile(ds["lon"].values, len(ds["lat"])),
    })


def extract_bathymetry_along_bearing(
    bathymetry_df,
    start_lat,
    start_lon,
    bearing_deg,
    max_range_m=20_000,
    interval_m=200,
):
    """Extract interpolated bathymetry along a fixed bearing."""
    ranges_m = np.arange(0, max_range_m + interval_m, interval_m, dtype=float)

    lons, lats, _ = GEOD.fwd(
        np.full_like(ranges_m, start_lon, dtype=float),
        np.full_like(ranges_m, start_lat, dtype=float),
        np.full_like(ranges_m, bearing_deg, dtype=float),
        ranges_m,
    )

    points = bathymetry_df[["lat", "lon"]].values
    values = bathymetry_df["depth"].values
    path_points = np.column_stack((lats, lons))

    elev = griddata(points, values, path_points, method="linear")
    if np.any(np.isnan(elev)):
        nan_mask = np.isnan(elev)
        elev[nan_mask] = griddata(points, values, path_points[nan_mask], method="nearest")

    depth_m = np.maximum(-elev, 1.0)
    path_df = pd.DataFrame({
        "range_m": ranges_m,
        "lat": lats,
        "lon": lons,
        "elevation_m": elev,
        "depth_m": depth_m,
    })
    path_df = path_df.drop_duplicates(subset="range_m").sort_values("range_m")
    path_df.loc[path_df.index.min(), "range_m"] = 0.0

    bathy = path_df[["range_m", "depth_m"]].values.tolist()
    return bathy, path_df


def select_dive_and_make_ssp(drift_ctd, drift_ends, site_end_index, max_depth_m):
    """Pick the CTD profile associated with one target/end index and build the SSP."""
    drift_ctd = drift_ctd.copy()
    drift_ctd["DiveID"] = drift_ctd["DiveNumber"].astype(str)

    dive_num = drift_ends.loc[site_end_index, "closestDive"]
    group = drift_ctd[drift_ctd["DiveNumber"] == dive_num].copy()
    if group.empty:
        raise ValueError(f"No CTD rows found for closestDive={dive_num}")

    depth_diff = np.diff(group["Depth_m"], prepend=np.nan)
    group["Direction"] = np.where(depth_diff > 0, "dec", "asc")
    if len(depth_diff) > 1 and depth_diff[1] > 0:
        group.loc[group.index[0], "Direction"] = "dec"
    else:
        group.loc[group.index[0], "Direction"] = "asc"

    group_dec = group[group["Direction"] == "dec"].reset_index(drop=True)
    if group_dec.empty:
        raise ValueError(f"No descending rows found for dive {dive_num}")

    start_lat = float(group_dec["Latitude"].iloc[0])
    start_lon = float(group_dec["Longitude"].iloc[0])

    profile = pd.DataFrame({
        "depth": group_dec["Depth_m"].to_numpy(),
        "ss": group_dec["SoundSpeed_m_s"].to_numpy(),
    })
    profile = profile.dropna().sort_values("depth").reset_index(drop=True)
    profile.loc[0, "depth"] = 0.0

    if profile.empty:
        raise ValueError(f"No valid SSP points found for dive {dive_num}")

    last_depth = float(profile["depth"].iloc[-1])
    last_ss = float(profile["ss"].iloc[-1])
    if max_depth_m > last_depth + 10:
        z_ext = np.arange(last_depth + 10, max_depth_m + 50, 50)
        ext = pd.DataFrame({"depth": z_ext, "ss": np.repeat(last_ss, len(z_ext))})
        profile = pd.concat([profile, ext], ignore_index=True)

    profile["ss"] = np.abs(profile["ss"])
    profile = profile.sort_values("depth").drop_duplicates(subset="depth")

    ssp = profile[["depth", "ss"]].values.tolist()
    return dive_num, start_lat, start_lon, profile, ssp


def load_click_waveform(wav_path, target_fs_hz):
    """Load the source click, mix to mono, and resample to the target rate."""
    fs, audio = wavfile.read(wav_path)
    audio = np.asarray(audio)

    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float64) / float(scale)
    else:
        audio = audio.astype(np.float64)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if fs != target_fs_hz:
        gcd = int(np.gcd(int(fs), int(target_fs_hz)))
        up = int(target_fs_hz // gcd)
        down = int(fs // gcd)
        audio = resample_poly(audio, up, down)
        fs = target_fs_hz

    return audio, int(fs)


def arrivals_to_dataframe(arrivals_raw):
    """Convert Bellhop arrivals output to a flat DataFrame."""
    if isinstance(arrivals_raw, pd.DataFrame):
        df = arrivals_raw.copy()
    elif hasattr(arrivals_raw, "to_dataframe"):
        df = arrivals_raw.to_dataframe().reset_index()
    else:
        df = pd.DataFrame(arrivals_raw)

    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()

    return df.reset_index(drop=True)


def first_existing_column(df, candidates):
    """Return the first column from candidates that exists in df."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def normalize_arrivals_table(arrivals_raw, max_time=None):
    """Standardize Bellhop arrivals table columns for receiver filtering.

    Parameters
    ----------
    arrivals_raw : array-like or DataFrame
        Raw Bellhop arrivals structure.
    max_time : float or None
        If given, retain only arrivals whose time_of_arrival is within
        *max_time* seconds of the earliest arrival for each receiver
        (depth/range pair).  ``None`` (default) keeps all arrivals.
    """
    df = arrivals_to_dataframe(arrivals_raw)
    if df.empty:
        return df, "index", "index"

    amp_col = first_existing_column(
        df,
        ["arrival_amplitude", "arrival_amp", "amplitude"],
    )
    if amp_col is None:
        real_col = first_existing_column(df, ["amp_real", "arrival_amp_real", "amplitude_real"])
        imag_col = first_existing_column(df, ["amp_imag", "arrival_amp_imag", "amplitude_imag"])
        if real_col is None or imag_col is None:
            raise KeyError(f"Could not identify arrival amplitude columns: {list(df.columns)}")
        df["arrival_amplitude"] = df[real_col].to_numpy(dtype=float) + 1j * df[imag_col].to_numpy(dtype=float)
    else:
        df["arrival_amplitude"] = np.asarray(df[amp_col].to_numpy(), dtype=complex)

    time_col = first_existing_column(df, ["time_of_arrival", "arrival_time", "toa"])
    if time_col is None:
        raise KeyError(f"Could not identify time-of-arrival column: {list(df.columns)}")
    df["time_of_arrival"] = df[time_col].to_numpy(dtype=float)

    depth_idx_col = first_existing_column(df, ["rx_depth_ndx", "rx_depth_index"])
    depth_val_col = first_existing_column(df, ["rx_depth", "receiver_depth"])
    range_idx_col = first_existing_column(df, ["rx_range_ndx", "rx_range_index"])
    range_val_col = first_existing_column(df, ["rx_range", "receiver_range"])

    if depth_idx_col is not None:
        df["rx_depth_key"] = df[depth_idx_col].to_numpy(dtype=int)
        depth_mode = "index"
    elif depth_val_col is not None:
        df["rx_depth_key"] = df[depth_val_col].to_numpy(dtype=float)
        depth_mode = "value"
    else:
        raise KeyError(f"Could not identify receiver depth column: {list(df.columns)}")

    if range_idx_col is not None:
        df["rx_range_key"] = df[range_idx_col].to_numpy(dtype=int)
        range_mode = "index"
    elif range_val_col is not None:
        df["rx_range_key"] = df[range_val_col].to_numpy(dtype=float)
        range_mode = "value"
    else:
        raise KeyError(f"Could not identify receiver range column: {list(df.columns)}")

    path_col = first_existing_column(df, ["path_length_m", "path_length", "ray_length_m"])
    if path_col is not None:
        df["path_length_m"] = df[path_col].to_numpy(dtype=float)

    if max_time is not None:
        first_toa = df.groupby(["rx_depth_key", "rx_range_key"])["time_of_arrival"].transform("min")
        df = df[df["time_of_arrival"] <= first_toa + max_time].reset_index(drop=True)

    return df, depth_mode, range_mode


def export_grids_to_csv(tl_grid, bw10_grid, rx_range, rx_depth, src_depth, out_dir):
    """Convert 2D grids to a ggplot-compatible long-format CSV.
    
    Parameters
    ----------
    tl_grid : ndarray
        2D array of transmission loss values (rows=depth, cols=range)
    bw10_grid : ndarray
        2D array of 10dB bandwidth values (rows=depth, cols=range)
    rx_range : ndarray
        1D array of receiver range values (in meters)
    rx_depth : ndarray
        1D array of receiver depth values (in meters)
    src_depth : float
        Source depth for naming the output file
    out_dir : str
        Output directory path
    
    Returns
    -------
    None (writes CSV file to disk)
    """
    # Create meshgrid
    range_mesh, depth_mesh = np.meshgrid(rx_range, rx_depth)
    
    # Flatten all arrays
    ranges_flat = range_mesh.flatten()
    depths_flat = depth_mesh.flatten()
    tl_flat = tl_grid.flatten()
    bw10_flat = bw10_grid.flatten()
    
    # Create DataFrame with ggplot-compatible column names
    export_df = pd.DataFrame({
        "horizontal_m": ranges_flat,
        "vertical_m": depths_flat,
        "ppTL": tl_flat,
        "bw10db": bw10_flat,
    })
    
    # Save to CSV
    out_csv = os.path.join(out_dir, f"grids_ggplot_source_{int(src_depth):04d}m.csv")
    export_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


def receiver_arrivals_from_table(
    arrivals_df,
    depth_mode,
    range_mode,
    depth_index,
    range_index,
    rx_depth,
    rx_range,
):
    """Extract one receiver's arrivals as the dict expected by the synthesis helpers."""
    if arrivals_df.empty:
        return None

    if depth_mode == "index":
        depth_mask = arrivals_df["rx_depth_key"].to_numpy(dtype=int) == int(depth_index)
    else:
        depth_mask = np.isclose(
            arrivals_df["rx_depth_key"].to_numpy(dtype=float),
            float(rx_depth[depth_index]),
            atol=max(RX_DEPTH_STEP_M * 0.25, 1e-6),
        )

    if range_mode == "index":
        range_mask = arrivals_df["rx_range_key"].to_numpy(dtype=int) == int(range_index)
    else:
        range_mask = np.isclose(
            arrivals_df["rx_range_key"].to_numpy(dtype=float),
            float(rx_range[range_index]),
            atol=max(RX_RANGE_STEP_M * 0.25, 1e-6),
        )

    subset = arrivals_df.loc[depth_mask & range_mask]
    if subset.empty:
        return None

    arrivals = {
        "time_of_arrival": subset["time_of_arrival"].to_numpy(dtype=float),
        "arrival_amplitude": np.asarray(subset["arrival_amplitude"].to_numpy(), dtype=complex),
    }
    if "path_length_m" in subset.columns:
        arrivals["path_length_m"] = subset["path_length_m"].to_numpy(dtype=float)
    return arrivals



def prepare_source_fft(
    source_click,
    fs,
    max_delay_s=0.01,
    guard_time_s=0.005,
):
    """Prepare the source click on one native rFFT frequency grid.

    The FFT record is long enough to contain the source click, the retained
    relative multipath delay spread, and guard intervals on both sides. The
    click is placed after the first guard interval to prevent band-limited
    pre-ringing from wrapping around to the end of the inverse FFT record.
    """
    source_click = np.asarray(source_click, dtype=float).reshape(-1)
    source_click = np.nan_to_num(source_click)

    if source_click.size == 0:
        raise ValueError("The source click waveform is empty.")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    guard_samples = int(np.ceil(float(guard_time_s) * fs))
    delay_samples = int(np.ceil(float(max_delay_s) * fs))

    required_samples = (
        guard_samples
        + source_click.size
        + delay_samples
        + guard_samples
    )
    n_time = 1 << int(np.ceil(np.log2(max(required_samples, 2))))

    padded_source = np.zeros(n_time, dtype=float)
    click_start_sample = guard_samples
    click_stop_sample = click_start_sample + source_click.size
    padded_source[click_start_sample:click_stop_sample] = source_click

    source_fft = np.fft.rfft(padded_source, n=n_time)
    fft_freqs_hz = np.fft.rfftfreq(n_time, d=1.0 / fs)

    return source_fft, fft_freqs_hz, n_time, click_start_sample


def compute_transfer_function_on_fft_grid(
    arrivals,
    fft_freqs_hz,
    f_ref_hz,
    synthesis_fmin_hz,
    synthesis_fmax_hz,
    c_eff_m_s=1480.0,
    arrivals_include_absorption=True,
):
    """Construct the coherent transfer function on the native rFFT grid."""
    tau = np.asarray(arrivals["time_of_arrival"], dtype=float).reshape(-1)
    amp = np.asarray(arrivals["arrival_amplitude"], dtype=complex).reshape(-1)

    transfer = np.zeros_like(fft_freqs_hz, dtype=np.complex128)
    if tau.size == 0 or amp.size == 0:
        return transfer
    if tau.size != amp.size:
        raise ValueError("Arrival-time and arrival-amplitude arrays differ in length.")

    tau_rel = tau - np.min(tau)

    if arrivals.get("path_length_m") is not None:
        path_length_m = np.asarray(
            arrivals["path_length_m"], dtype=float
        ).reshape(-1)
    else:
        path_length_m = tau * float(c_eff_m_s)

    active = (
        (fft_freqs_hz >= float(synthesis_fmin_hz))
        & (fft_freqs_hz <= float(synthesis_fmax_hz))
    )
    active_freqs = fft_freqs_hz[active]
    if active_freqs.size == 0:
        return transfer

    alpha_db_per_km = thorp_alpha_db_per_km(active_freqs)
    alpha_ref_db_per_km = float(
        thorp_alpha_db_per_km(np.array([f_ref_hz], dtype=float))[0]
    )

    if arrivals_include_absorption:
        delta_db_per_km = alpha_db_per_km - alpha_ref_db_per_km
    else:
        delta_db_per_km = alpha_db_per_km

    delta_np_per_m = (
        (delta_db_per_km / 20.0)
        * np.log(10.0)
        / 1000.0
    )

    active_transfer = np.zeros(active_freqs.size, dtype=np.complex128)
    for arrival_index in range(tau_rel.size):
        absorption = np.exp(
            -delta_np_per_m * path_length_m[arrival_index]
        )
        phase = np.exp(
            -1j
            * 2.0
            * np.pi
            * active_freqs
            * tau_rel[arrival_index]
        )
        active_transfer += (
            amp[arrival_index]
            * absorption
            * phase
        )

    transfer[active] = active_transfer
    return transfer


def synthesize_received_waveform(
    arrivals,
    source_fft,
    fft_freqs_hz,
    fs,
    n_time,
    f_ref_hz,
    synthesis_fmin_hz,
    synthesis_fmax_hz,
    c_eff_m_s=1480.0,
):
    """Synthesize one received click directly on the native rFFT grid."""
    impulse_response = arrivals_to_impulse_response(
        arrivals,
        fs=fs,
        abs_time=False,
    )

    transfer_function = compute_transfer_function_on_fft_grid(
        arrivals=arrivals,
        fft_freqs_hz=fft_freqs_hz,
        f_ref_hz=f_ref_hz,
        synthesis_fmin_hz=synthesis_fmin_hz,
        synthesis_fmax_hz=synthesis_fmax_hz,
        c_eff_m_s=c_eff_m_s,
        arrivals_include_absorption=True,
    )

    received_fft = source_fft * transfer_function
    received = np.fft.irfft(received_fft, n=n_time)

    return np.real(received), impulse_response, transfer_function

def extract_fixed_click_window(
    waveform,
    window_samples,
    pre_peak_fraction=0.25,
):
    """
    Extract a fixed-length click window around the largest absolute sample.

    Samples outside the available waveform are filled with zeros. This mirrors
    a fixed saved-click window without adding artificial noise.
    """
    waveform = np.asarray(waveform, dtype=float).reshape(-1)
    waveform = np.nan_to_num(waveform)

    if waveform.size == 0:
        raise ValueError("The source waveform is empty.")

    window_samples = int(window_samples)

    if window_samples < 2:
        raise ValueError("window_samples must be at least 2.")

    if not 0.0 <= pre_peak_fraction <= 1.0:
        raise ValueError("pre_peak_fraction must be between 0 and 1.")

    peak_index = int(np.argmax(np.abs(waveform)))

    pre_peak_samples = int(
        round(window_samples * pre_peak_fraction)
    )

    source_start = peak_index - pre_peak_samples
    source_end = source_start + window_samples

    click_window = np.zeros(
        window_samples,
        dtype=float,
    )

    valid_source_start = max(0, source_start)
    valid_source_end = min(waveform.size, source_end)

    destination_start = valid_source_start - source_start
    destination_end = (
        destination_start
        + valid_source_end
        - valid_source_start
    )

    if valid_source_end > valid_source_start:
        click_window[destination_start:destination_end] = waveform[
            valid_source_start:valid_source_end
        ]

    return click_window, peak_index


def next_power_of_two(value):
    """Return the smallest power of two greater than or equal to value."""
    value = int(value)

    if value < 1:
        return 1

    return 1 << int(np.ceil(np.log2(value)))


def smooth_linear_spectrum(values, smoothing_bins=1):
    """
    Apply an optional centered moving average in linear spectral units.

    A value of 1 performs no smoothing.
    """
    values = np.asarray(values, dtype=float)
    smoothing_bins = int(smoothing_bins)

    if smoothing_bins <= 1:
        return values.copy()

    # Use an odd window so that smoothing is centered.
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1

    half_width = smoothing_bins // 2

    padded = np.pad(
        values,
        pad_width=half_width,
        mode="edge",
    )

    kernel = (
        np.ones(smoothing_bins, dtype=float)
        / smoothing_bins
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def calculate_pamguard_style_bandwidth(
    click_waveform,
    sampling_rate_hz,
    search_range_hz=(1_000.0, 24_000.0),
    db_drop=10.0,
    use_hann=True,
    smoothing_bins=1,
):
    """
    Calculate a discrete-bin PAMGuard-style peak-frequency width.

    Processing:
      1. FFT length is the smallest power of two containing the click.
      2. A Hann window is optionally applied to the complete saved click.
      3. FFT magnitude is calculated.
      4. The largest bin inside the search range is selected.
      5. The search moves outward until the magnitude is at least 10 dB
         below the peak.
      6. No interpolation is performed between FFT bins.

    smoothing_bins=1 gives the raw single-channel result. Larger values are
    included only as a diagnostic for classifier-style spectral smoothing.
    """
    click_waveform = np.asarray(
        click_waveform,
        dtype=float,
    ).reshape(-1)

    click_waveform = np.nan_to_num(click_waveform)

    if click_waveform.size < 2:
        raise ValueError(
            "The click waveform must contain at least two samples."
        )

    n_fft = next_power_of_two(click_waveform.size)

    if use_hann:
        fft_input = (
            click_waveform
            * np.hanning(click_waveform.size)
        )
    else:
        fft_input = click_waveform.copy()

    fft_values = np.fft.rfft(
        fft_input,
        n=n_fft,
    )

    magnitude = np.abs(fft_values)

    frequencies_hz = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sampling_rate_hz,
    )

    # Exclude the Nyquist bin for closer correspondence with the usual
    # half-spectrum representation.
    magnitude = magnitude[:n_fft // 2]
    frequencies_hz = frequencies_hz[:n_fft // 2]

    magnitude_for_width = smooth_linear_spectrum(
        magnitude,
        smoothing_bins=smoothing_bins,
    )

    search_low_hz, search_high_hz = search_range_hz

    search_indices = np.flatnonzero(
        (frequencies_hz >= search_low_hz)
        & (frequencies_hz <= search_high_hz)
    )

    if search_indices.size == 0:
        raise ValueError(
            "No FFT bins fall inside the requested peak-search range."
        )

    peak_index = search_indices[
        np.argmax(
            magnitude_for_width[search_indices]
        )
    ]

    peak_magnitude = magnitude_for_width[peak_index]

    if peak_magnitude <= 0:
        raise ValueError(
            "The spectral peak has zero magnitude."
        )

    # A 10 dB amplitude reduction corresponds to division by 10^(10/20).
    threshold_magnitude = (
        peak_magnitude
        / 10.0 ** (db_drop / 20.0)
    )

    low_index = peak_index
    high_index = peak_index

    while (
        low_index > search_indices[0]
        and magnitude_for_width[low_index] > threshold_magnitude
    ):
        low_index -= 1

    while (
        high_index < search_indices[-1]
        and magnitude_for_width[high_index] > threshold_magnitude
    ):
        high_index += 1

    f_low_hz = frequencies_hz[low_index]
    f_high_hz = frequencies_hz[high_index]
    bandwidth_hz = f_high_hz - f_low_hz

    return {
        "bandwidth_hz": float(bandwidth_hz),
        "f_low_hz": float(f_low_hz),
        "f_high_hz": float(f_high_hz),
        "peak_frequency_hz": float(
            frequencies_hz[peak_index]
        ),
        "peak_index": int(peak_index),
        "low_index": int(low_index),
        "high_index": int(high_index),
        "threshold_magnitude": float(threshold_magnitude),
        "frequencies_hz": frequencies_hz,
        "raw_magnitude": magnitude,
        "magnitude_for_width": magnitude_for_width,
        "n_fft": int(n_fft),
    }




def compute_pamguard_bandwidth_metrics(waveform, sampling_rate_hz):
    """Apply the exact same click extraction and bandwidth method to any waveform."""
    click_clip, waveform_peak_index = extract_fixed_click_window(
        waveform=waveform,
        window_samples=PAMGUARD_CLICK_SAMPLES,
        pre_peak_fraction=PAMGUARD_PRE_PEAK_FRACTION,
    )

    result = calculate_pamguard_style_bandwidth(
        click_waveform=click_clip,
        sampling_rate_hz=sampling_rate_hz,
        search_range_hz=PAMGUARD_PEAK_SEARCH_HZ,
        db_drop=PAMGUARD_DB_DROP,
        use_hann=PAMGUARD_USE_HANN,
        smoothing_bins=PAMGUARD_SMOOTHING_BINS,
    )

    result["click_clip"] = click_clip
    result["waveform_peak_index"] = int(waveform_peak_index)
    return result


def plot_source_bandwidth_diagnostic(source_click, source_fs, out_dir):
    """Plot and save the source spectrum used by the receiver bandwidth metric."""
    result = compute_pamguard_bandwidth_metrics(source_click, source_fs)

    magnitude = result["magnitude_for_width"]
    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, np.finfo(float).tiny)
    )
    magnitude_db -= np.nanmax(magnitude_db)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        result["frequencies_hz"] / 1000.0,
        magnitude_db,
        linewidth=1.2,
        label=(
            "Smoothed magnitude spectrum "
            f"({PAMGUARD_SMOOTHING_BINS} bins)"
        ),
    )
    ax.axhline(
        -PAMGUARD_DB_DROP,
        linestyle="--",
        linewidth=1.2,
        label=f"-{PAMGUARD_DB_DROP:g} dB",
    )
    ax.axvline(
        result["f_low_hz"] / 1000.0,
        linestyle=":",
        linewidth=1.5,
        label="10 dB limits",
    )
    ax.axvline(
        result["f_high_hz"] / 1000.0,
        linestyle=":",
        linewidth=1.5,
    )
    ax.axvline(
        result["peak_frequency_hz"] / 1000.0,
        linestyle="-.",
        linewidth=1.2,
        label="Spectral peak",
    )

    upper_hz = min(PAMGUARD_PEAK_SEARCH_HZ[1], source_fs / 2.0)
    ax.set_xlim(PAMGUARD_PEAK_SEARCH_HZ[0] / 1000.0, upper_hz / 1000.0)
    ax.set_ylim(-50, 3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude relative to peak (dB)")
    ax.set_title(
        "Source click: "
        f"{PAMGUARD_DB_DROP:g} dB bandwidth = "
        f"{result['bandwidth_hz'] / 1000.0:.2f} kHz"
    )
    ax.legend()

    fig.tight_layout()
    out_png = os.path.join(out_dir, "source_click_10dB_bandwidth.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    plt.show()

    print(
        "Source click bandwidth: "
        f"{result['bandwidth_hz']:.1f} Hz "
        f"({result['f_low_hz']:.1f}-{result['f_high_hz']:.1f} Hz); "
        f"peak={result['peak_frequency_hz']:.1f} Hz; "
        f"FFT={result['n_fft']}; "
        f"smoothing={PAMGUARD_SMOOTHING_BINS} bins"
    )
    return result


def peak_to_peak_db(signal):
    """Peak-to-peak level in dB re linear amplitude units."""
    return 20.0 * np.log10(np.ptp(np.real(signal)) + 1e-18)


def compute_receiver_metrics(received_waveform, sampling_rate_hz):
    """Compute P2P level and the shared PAMGuard-style bandwidth metric."""
    p2p_db = peak_to_peak_db(received_waveform)
    bandwidth = compute_pamguard_bandwidth_metrics(
        received_waveform,
        sampling_rate_hz,
    )

    return {
        "p2p_db": float(p2p_db),
        "bw_10db_hz": float(bandwidth["bandwidth_hz"]),
        "f_low_hz": float(bandwidth["f_low_hz"]),
        "f_high_hz": float(bandwidth["f_high_hz"]),
        "peak_frequency_hz": float(bandwidth["peak_frequency_hz"]),
    }


def run_single_source_depth(
    src_depth,
    max_water_depth,
    bathy,
    ssp,
    rx_range,
    rx_depth,
    source_fs,
    source_p2p_db,
    source_fft,
    fft_freqs_hz,
    n_time,
    click_start_sample,
):
    """Run one source-depth case and return grids plus debug artifacts."""
    if src_depth >= max_water_depth:
        print(
            f"Skipping {src_depth} m; source is deeper than bathymetry "
            "along this bearing."
        )
        return None

    print(
        "Running arrivals-based peak-to-peak TL for source depth "
        f"{src_depth} m..."
    )

    env = pm.create_env2d(
        depth=bathy,
        soundspeed=ssp,
        bottom_soundspeed=BOTTOM_SOUND_SPEED,
        bottom_density=BOTTOM_DENSITY,
        bottom_absorption=BOTTOM_ABSORPTION,
        tx_depth=src_depth,
        frequency=FREQ_HZ,
        nbeams=NBEAMS,
        max_angle=MAX_ANGLE,
        min_angle=MIN_ANGLE,
        soundspeed_interp=SOUNDSPEED_INTERP,
    )
    env["rx_range"] = rx_range
    env["rx_depth"] = rx_depth

    arrivals_raw = pm.compute_arrivals(env)
    arrivals_df, depth_mode, range_mode = normalize_arrivals_table(
        arrivals_raw,
        MAX_ARRIVAL_DELAY,
    )

    tl_grid = np.full((len(rx_depth), len(rx_range)), np.nan, dtype=float)
    bw10_grid = np.full((len(rx_depth), len(rx_range)), np.nan, dtype=float)
    sample_bundle = None

    if arrivals_df.empty:
        return {
            "arrivals_df": arrivals_df,
            "tl_grid": tl_grid,
            "bw10_grid": bw10_grid,
            "sample_bundle": sample_bundle,
        }

    n_ranges = (
        len(rx_range)
        if DEBUG_LIMIT_RX_RANGES is None
        else min(len(rx_range), int(DEBUG_LIMIT_RX_RANGES))
    )
    n_depths = (
        len(rx_depth)
        if DEBUG_LIMIT_RX_DEPTHS is None
        else min(len(rx_depth), int(DEBUG_LIMIT_RX_DEPTHS))
    )

    for range_index in range(n_ranges):
        for depth_index in range(n_depths):
            receiver_arrivals = receiver_arrivals_from_table(
                arrivals_df,
                depth_mode,
                range_mode,
                depth_index,
                range_index,
                rx_depth,
                rx_range,
            )
            if receiver_arrivals is None:
                continue

            received, impulse_response, transfer_function = (
                synthesize_received_waveform(
                    arrivals=receiver_arrivals,
                    source_fft=source_fft,
                    fft_freqs_hz=fft_freqs_hz,
                    fs=source_fs,
                    n_time=n_time,
                    f_ref_hz=FREQ_HZ,
                    synthesis_fmin_hz=COHERENT_FMIN_HZ,
                    synthesis_fmax_hz=COHERENT_FMAX_HZ,
                    c_eff_m_s=C_EFF_M_S,
                )
            )

            metrics = compute_receiver_metrics(received, source_fs)
            tl_grid[depth_index, range_index] = (
                source_p2p_db - metrics["p2p_db"]
            )
            bw10_grid[depth_index, range_index] = metrics["bw_10db_hz"]

            if sample_bundle is None:
                sample_bundle = {
                    "rx_depth_m": float(rx_depth[depth_index]),
                    "rx_range_m": float(rx_range[range_index]),
                    "impulse_response": impulse_response,
                    "transfer_function": transfer_function,
                    "frequency_hz": fft_freqs_hz,
                    "source_fft": source_fft,
                    "received_waveform": received,
                    "received_p2p_db": metrics["p2p_db"],
                    "received_bw10_hz": metrics["bw_10db_hz"],
                    "received_bw10_low_hz": metrics["f_low_hz"],
                    "received_bw10_high_hz": metrics["f_high_hz"],
                    "received_peak_frequency_hz": metrics["peak_frequency_hz"],
                    "sampling_rate_hz": float(source_fs),
                    "n_time": int(n_time),
                    "click_start_sample": int(click_start_sample),
                }

    return {
        "arrivals_df": arrivals_df,
        "tl_grid": tl_grid,
        "bw10_grid": bw10_grid,
        "sample_bundle": sample_bundle,
    }

def plot_peak_to_peak_tl(tl_grid, rx_range, rx_depth, bathy_df, src_depth, out_dir):
    """Plot peak-to-peak transmission loss on the receiver grid."""
    finite_vals = tl_grid[np.isfinite(tl_grid)]
    if finite_vals.size == 0:
        raise ValueError("No valid peak-to-peak TL values were computed for plotting.")

    level_min = 2.0 * np.floor(np.nanmin(finite_vals) / 2.0)
    level_max = 2.0 * np.ceil(np.nanmax(finite_vals) / 2.0)
    if level_max <= level_min:
        level_max = level_min + 2.0
    levels = np.arange(level_min, level_max + 2.0, 2.0)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    cf = ax.contourf(
        rx_range / 1000.0,
        rx_depth,
        tl_grid,
        levels=levels,
        cmap="viridis_r",
        extend="both",
    )

    ax.plot(bathy_df["range_m"] / 1000.0, bathy_df["depth_m"], "k-", lw=1.5)
    ax.fill_between(
        bathy_df["range_m"] / 1000.0,
        bathy_df["depth_m"],
        np.nanmax(bathy_df["depth_m"]) + 200.0,
        color="0.75",
        alpha=0.6,
    )
    ax.scatter([0.0], [src_depth], marker="*", s=140, c="white", edgecolors="k", zorder=5)

    ax.set_ylim(np.nanmax(bathy_df["depth_m"]) + 50.0, 0.0)
    ax.set_xlim(0.0, np.nanmax(rx_range) / 1000.0)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Peak-to-peak TL, source depth = {src_depth} m")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Peak-to-peak transmission loss (dB)")

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"peak_to_peak_TL_source_{int(src_depth):04d}m.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    plt.show()
    
    
def plot_10dB_bw(bw10_grid, rx_range, rx_depth, bathy_df, src_depth, out_dir):
    """Plot received 10 dB bandwidth on the receiver range-depth grid."""
    finite_vals = bw10_grid[np.isfinite(bw10_grid)]
    if finite_vals.size == 0:
        raise ValueError("No valid 10 dB bandwidth values were computed for plotting.")

    level_min = 250.0 * np.floor(np.nanmin(finite_vals) / 250.0)
    level_max = 250.0 * np.ceil(np.nanmax(finite_vals) / 250.0)
    if level_max <= level_min:
        level_max = level_min + 250.0
    levels = np.arange(level_min, level_max + 250.0, 250.0)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    cf = ax.contourf(
        rx_range / 1000.0,
        rx_depth,
        bw10_grid,
        levels=levels,
        cmap="viridis_r",
        extend="both",
    )

    ax.plot(bathy_df["range_m"] / 1000.0, bathy_df["depth_m"], "k-", lw=1.5)
    ax.fill_between(
        bathy_df["range_m"] / 1000.0,
        bathy_df["depth_m"],
        np.nanmax(bathy_df["depth_m"]) + 200.0,
        color="0.75",
        alpha=0.6,
    )
    ax.scatter([0.0], [src_depth], marker="*", s=140, c="white", edgecolors="k", zorder=5)

    ax.set_ylim(np.nanmax(bathy_df["depth_m"]) + 50.0, 0.0)
    ax.set_xlim(0.0, np.nanmax(rx_range) / 1000.0)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Received 10 dB BW, source depth = {src_depth} m")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("10 dB Bandwidth (Hz)")

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"10dB_BW_source_{int(src_depth):04d}m.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    plt.show()
    


# =============================================================================
# Main
# =============================================================================


def main():
    
    
    
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading source click, CTD, target, and bathymetry files...")
    source_click, source_fs = load_click_waveform(source_wav_path, CLICK_TARGET_FS_HZ)
    source_click = scaleP2P(source_click, outP2P=SOURCE_P2P_DB)
    source_p2p_db = peak_to_peak_db(source_click)
    plot_source_bandwidth_diagnostic(
        source_click,
        source_fs,
        OUT_DIR,
    )
    result = compute_pamguard_bandwidth_metrics(source_click, source_fs)
    print(result['bandwidth_hz'])
    
    
    source_fft, fft_freqs_hz, n_time, click_start_sample = prepare_source_fft(
        source_click=source_click,
        fs=source_fs,
        max_delay_s=MAX_ARRIVAL_DELAY,
        guard_time_s=SYNTHESIS_GUARD_TIME_S,
    )

    drift_ctd = pd.read_csv(drift_csv)
    drift_ends = pd.read_csv(drift_ends_csv)
    bathymetry_df = make_bathymetry_dataframe(gebco_nc)

    rough_start_lat = float(drift_ends.loc[SITE_END_INDEX, "lat"])
    rough_start_lon = float(drift_ends.loc[SITE_END_INDEX, "lon"])

    bathymetry_df["distance_km"] = haversine(
        rough_start_lon,
        rough_start_lat,
        bathymetry_df["lon"],
        bathymetry_df["lat"],
    )
    bathy_subset = bathymetry_df[
        (bathymetry_df["distance_km"] <= 25.0)
        & (bathymetry_df["depth"] < -50.0)
    ].copy()
    if bathy_subset.empty:
        raise ValueError("Bathymetry subset is empty. Check SITE_END_INDEX and GEBCO file bounds.")

    _, rough_path_df = extract_bathymetry_along_bearing(
        bathy_subset,
        rough_start_lat,
        rough_start_lon,
        BEARING_DEG,
        max_range_m=MAX_RANGE_M,
        interval_m=BATHY_INTERVAL_M,
    )
    max_bathy_depth = float(np.nanmax(rough_path_df["depth_m"]))

    dive_num, start_lat, start_lon, profile, ssp = select_dive_and_make_ssp(
        drift_ctd,
        drift_ends,
        SITE_END_INDEX,
        max_depth_m=max_bathy_depth,
    )

    print(f"Using SITE_END_INDEX={SITE_END_INDEX}")
    print(f"Using closestDive={dive_num}")
    print(f"Source location: lat={start_lat:.5f}, lon={start_lon:.5f}")
    print(f"Bearing: {BEARING_DEG} deg")
    print(f"Source click P2P level: {source_p2p_db:.1f} dB")
    print(
        "Native FFT synthesis: "
        f"n_time={n_time}, duration={n_time/source_fs:.4f} s, "
        f"df={source_fs/n_time:.2f} Hz, "
        f"band={COHERENT_FMIN_HZ:g}-{COHERENT_FMAX_HZ:g} Hz"
    )
    print(
        "PAMGuard bandwidth settings: "
        f"click_samples={PAMGUARD_CLICK_SAMPLES}, "
        f"hann={PAMGUARD_USE_HANN}, "
        f"drop={PAMGUARD_DB_DROP:g} dB, "
        f"smoothing={PAMGUARD_SMOOTHING_BINS} bins, "
        f"peak_search={PAMGUARD_PEAK_SEARCH_HZ} Hz"
    )

    bathymetry_df["distance_km"] = haversine(
        start_lon,
        start_lat,
        bathymetry_df["lon"],
        bathymetry_df["lat"],
    )
    bathy_subset = bathymetry_df[
        (bathymetry_df["distance_km"] <= 25.0)
        & (bathymetry_df["depth"] < -50.0)
    ].copy()

    bathy, path_df = extract_bathymetry_along_bearing(
        bathy_subset,
        start_lat,
        start_lon,
        BEARING_DEG,
        max_range_m=MAX_RANGE_M,
        interval_m=BATHY_INTERVAL_M,
    )

    max_water_depth = float(np.nanmax(path_df["depth_m"]))
    rx_range = np.arange(RX_RANGE_STEP_M, MAX_RANGE_M + RX_RANGE_STEP_M, RX_RANGE_STEP_M)
    rx_depth = np.arange(100, max_water_depth, RX_DEPTH_STEP_M)

    path_df.to_csv(os.path.join(OUT_DIR, "single_bearing_bathymetry.csv"), index=False)
    profile.to_csv(os.path.join(OUT_DIR, "ssp_profile_used.csv"), index=False)

    print(f"Receiver grid: {len(rx_range)} ranges x {len(rx_depth)} depths")
    print(f"Max bathymetry depth along bearing: {max_water_depth:.1f} m")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(path_df["range_m"] / 1000.0, path_df["depth_m"], "k-")
    ax.set_ylim(max_water_depth + 50.0, 0.0)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Water depth (m)")
    ax.set_title("Bathymetry along selected bearing")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bathymetry_along_bearing.png"), dpi=200)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(profile["ss"], profile["depth"], "k-")
    ax.set_ylim(max_water_depth + 50.0, 0.0)
    ax.set_xlabel("Sound speed (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"SSP used, dive {dive_num}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ssp_profile_used.png"), dpi=200)
    plt.show()

    for src_depth in SOURCE_DEPTHS_M:
        depth_result = run_single_source_depth(
            src_depth=src_depth,
            max_water_depth=max_water_depth,
            bathy=bathy,
            ssp=ssp,
            rx_range=rx_range,
            rx_depth=rx_depth,
            source_fs=source_fs,
            source_p2p_db=source_p2p_db,
            source_fft=source_fft,
            fft_freqs_hz=fft_freqs_hz,
            n_time=n_time,
            click_start_sample=click_start_sample,
        )
        if depth_result is None:
            continue

        arrivals_df = depth_result["arrivals_df"]
        arrivals_csv = os.path.join(OUT_DIR, f"arrivals_source_{int(src_depth):04d}m.csv")
        arrivals_df.to_csv(arrivals_csv, index=False)
        print(f"Saved {arrivals_csv}")

        tl_grid = depth_result["tl_grid"]
        bw10_grid = depth_result["bw10_grid"]
        sample_bundle = depth_result["sample_bundle"]

        if arrivals_df.empty:
            print(f"No arrivals returned for source depth {src_depth} m.")

        out_npy = os.path.join(OUT_DIR, f"peak_to_peak_tl_source_{int(src_depth):04d}m.npy")
        np.save(out_npy, tl_grid)
        print(f"Saved {out_npy}")

        out_bw_npy = os.path.join(OUT_DIR, f"peak_to_peak_bw10_source_{int(src_depth):04d}m.npy")
        np.save(out_bw_npy, bw10_grid)
        print(f"Saved {out_bw_npy}")

        export_grids_to_csv(tl_grid, bw10_grid, rx_range, rx_depth, src_depth, OUT_DIR)

        if sample_bundle is not None:
            sample_npz = os.path.join(OUT_DIR, f"sample_pipeline_source_{int(src_depth):04d}m.npz")
            np.savez(sample_npz, **sample_bundle)
            print(f"Saved {sample_npz}")

        if np.isfinite(bw10_grid).any():
            plot_10dB_bw(bw10_grid, rx_range, rx_depth, path_df, src_depth, OUT_DIR)
            
        if np.isfinite(tl_grid).any():
            plot_peak_to_peak_tl(tl_grid, rx_range, rx_depth, path_df, src_depth, OUT_DIR)

        if DEBUG_STOP_AFTER_FIRST_SOURCE:
            print("DEBUG_STOP_AFTER_FIRST_SOURCE=True -> stopping after first source depth.")
            break

    print("Done.")


if __name__ == "__main__":
    main()