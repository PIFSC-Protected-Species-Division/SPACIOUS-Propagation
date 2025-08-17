# -*- coding: utf-8 -*-
"""
Long-term underwater noise processing pipeline with:
- Google Cloud Storage (GCS) input support
- Memory-safe block streaming and daily HDF5 rotation
- Hybrid milli-decade band metrics + broadband and 1/3-octave levels
- Optional fast timestamp storage (epoch) and start-date filtering
- Heavily documented for clarity and maintenance
"""

from __future__ import annotations

import os
import re
import gc
import glob
import math
import shutil
import tempfile
from datetime import datetime, time, timedelta, date
from urllib.parse import urlparse
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sps

# Plotting utils (used by the helper plot functions at bottom)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Optional GCS support
try:
    from google.cloud import storage  # type: ignore
    _HAS_GCS = True
except Exception:
    _HAS_GCS = False


# ---------------------------------------------------------------------------
# Utilities for log-spaced milli-decade bands
# ---------------------------------------------------------------------------

def get_band_table(
    fft_bin_size: float,
    bin1_center_frequency: float = 0.0,
    fs: float = 64000.0,
    base: float = 10.0,
    bands_per_division: int = 1000,
    first_output_band_center_frequency: float = 435.0,
    use_fft_res_at_bottom: bool = False,
) -> np.ndarray:
    """
    Build a 3-column array of [f_low, f_center, f_high] defining logarithmically
    spaced frequency bands, optionally using linear-FFT resolution at the low end.

    Parameters
    ----------
    fft_bin_size : float
        Frequency resolution of the FFT (Hz).
    bin1_center_frequency : float
        Center frequency of the first (linear) FFT bin if using linear bottom.
    fs : float
        Sample rate (Hz).
    base : float
        Log base (10 for decades, 2 for octaves).
    bands_per_division : int
        Bands per factor-of-`base`.
        For millidecades: base=10, bands_per_division=1000.
    first_output_band_center_frequency : float
        Frequency (Hz) where the log bands begin.
    use_fft_res_at_bottom : bool
        If True, emit linear bins until the log bands exceed FFT resolution.

    Returns
    -------
    np.ndarray
        (n_bands, 3) array of [f_low, f_center, f_high].
    """
    band_count = 0
    max_freq = fs / 2.0
    low_mult = base ** (-1.0 / (2.0 * bands_per_division))
    high_mult = base ** (1.0 / (2.0 * bands_per_division))
    center_freq = 0.0
    linear_bin_count = 0
    log_bin_count = 0

    # Optional linear region at bottom (one band per FFT bin)
    if use_fft_res_at_bottom:
        bin_width = 0.0
        while bin_width < fft_bin_size:
            band_count += 1
            center_freq = first_output_band_center_frequency * (base ** (band_count / bands_per_division))
            bin_width = (high_mult * center_freq) - (low_mult * center_freq)

        center_freq = first_output_band_center_frequency * (base ** (band_count / bands_per_division))
        linear_bin_count = int(np.ceil(center_freq / fft_bin_size))

        while (linear_bin_count * fft_bin_size - center_freq) > 0.0:
            band_count += 1
            linear_bin_count += 1
            center_freq = first_output_band_center_frequency * (base ** (band_count / bands_per_division))

        if fft_bin_size * linear_bin_count > max_freq:
            linear_bin_count = int(max_freq / fft_bin_size) + 1

    log_band1 = band_count

    # Count number of log-spaced bands
    while max_freq > center_freq:
        band_count += 1
        log_bin_count += 1
        center_freq = first_output_band_center_frequency * (base ** (band_count / bands_per_division))

    # Allocate
    bands = np.zeros((linear_bin_count + log_bin_count, 3), dtype=float)

    # Fill linear bin bands
    for i in range(linear_bin_count):
        center_freq = bin1_center_frequency + i * fft_bin_size
        bands[i, 1] = center_freq
        bands[i, 0] = center_freq - 0.5 * fft_bin_size
        bands[i, 2] = center_freq + 0.5 * fft_bin_size

    # Fill log bands
    for i in range(log_bin_count):
        out_idx = linear_bin_count + i
        m_num = log_band1 + i + 1
        center_freq = first_output_band_center_frequency * (base ** ((m_num - 1) / bands_per_division))
        bands[out_idx, 1] = center_freq
        bands[out_idx, 0] = center_freq * low_mult
        bands[out_idx, 2] = center_freq * high_mult

    # Ensure the last band's high edge hits Nyquist
    if log_bin_count > 0:
        bands[out_idx, 2] = max_freq

    return bands


# ---------------------------------------------------------------------------
# Filename date parsing patterns
# ---------------------------------------------------------------------------

DATE_FORMATS = {
    r"\d{8}_\d{6}_\d{3}": "%Y%m%d_%H%M%S_%f",  # 20230101_123456_789
    r"\d{8}_\d{6}": "%Y%m%d_%H%M%S",          # 20230101_123456
    r"\d{8}T\d{6}": "%Y%m%dT%H%M%S",          # 20230101T123456
    r"\d{9}\.\d{12}": "%Y%m%d%H%M%S%f",       # 123456789.20230101234567
    r"\d{4}\.\d{12}": "%Y%m%d%H%M%S%f",       # 2023.20230101234567
    r"\d{6}-\d{6}\.\d{3}": "%y%m%d-%H%M%S.%f" # 220408-160025.415 (glider-like)
}


# ---------------------------------------------------------------------------
# Core processing class
# ---------------------------------------------------------------------------

class NoiseApp:
    """
    Long-term noise processing orchestrator.

    Features
    --------
    - Accepts local paths or `gs://bucket/prefix` for audio sources.
    - Streams through files block-by-block to control memory usage.
    - Rotates output into one HDF5 file per *day* (UTC date by default).
    - Computes:
        * Hybrid milli-decade APSD levels (dB re 1 µPa²/Hz)
        * Broadband SPL (dB re 1 µPa)
        * 1/3-octave band levels (dB re 1 µPa)
        * Decade-band levels (dB re 1 µPa)
    - Writes appends safely (open/append/close) to keep memory pressure low.
    - Optional epoch timestamp storage for faster writes.
    - Start-date filtering (skip earlier files).

    Parameters
    ----------
    soundFilePath : str
        Local folder or GCS URL (gs://bucket/prefix).
    ProjName : str
        Identifier used in output filenames.
    DepName : str
        HDF5 top-level group name (e.g., hydrophone ID).
    DatabaseLoc : str
        Output folder for daily HDF5s.
    Si : float | str | pandas.DataFrame
        Sensitivity. If float: dB re 1 V/µPa (e.g., -184).
        If str: CSV path with two cols [Hz, dB].
        If DataFrame: same two columns already loaded.
    clipFileSec : float
        Seconds to clip at the head of each file (e.g., 3 for SoundTrap).
    channel : int
        0-based audio channel index to process.
    r : float
        Window overlap fraction for spectrogram (0..1).
    winname : str
        Window type name (currently Hann only).
    lcut, hcut : float | None
        Analysis band limits (Hz). Defaults to [0, Nyquist].
    aveSec : float
        Time-averaging bin in seconds for metrics.
    pref : float
        Reference pressure (1 µPa underwater).
    rmDC : bool
        Remove mean from each block before FFT.
    Si_units : str
        'V/µPa' (default) or 'V/Pa' if your CSV is in Pa units.
    time_storage : {'str','epoch'}
        'str' → ISO strings (easy plotting), 'epoch' → float seconds (faster I/O).
    tmp_root : Optional[str]
        Where to place temporary downloads/extractions. If None, system temp is used.
    gcs_chunk_mb : int
        GCS download chunk size in MiB.
    tmp_max_gb : float
        If set > 0, guard against writing files larger than this to temp.
    """

    # -------------------------
    # Constructor / properties
    # -------------------------
    def __init__(
        self,
        soundFilePath: str,
        ProjName: str,
        DepName: str,
        DatabaseLoc: str,
        Si: float | str | pd.DataFrame = -184.0,
        clipFileSec: float = 0.0,
        channel: int = 0,
        r: float = 0.5,
        winname: str = "Hann",
        lcut: Optional[float] = None,
        hcut: Optional[float] = None,
        aveSec: float = 60.0,
        pref: float = 1.0,
        rmDC: bool = True,
        Si_units: str = "V/µPa",
        time_storage: str = "str",
        tmp_root: Optional[str] = None,
        gcs_chunk_mb: int = 16,
        tmp_max_gb: float = 0.0,
    ):
        # I/O settings
        self.soundFilePath = soundFilePath
        self.ProjName = ProjName
        self.DepName = DepName
        self.DatabaseLoc = DatabaseLoc
        os.makedirs(self.DatabaseLoc, exist_ok=True)

        # Sensitivity handling
        self.Si = Si
        self.Si_units = Si_units

        # Analysis params
        self.clipFileSec = clipFileSec
        self.channel = channel
        self.r = float(r)
        self.winname = winname
        self.lcut = lcut
        self.hcut = hcut
        self.aveSec = float(aveSec)
        self.rmDC = rmDC
        self.pref = float(pref)

        # Derived during prep
        self.fs: Optional[int] = None
        self.N: Optional[int] = None
        self.overlap: int = 0
        self.step: int = 0
        self.f: Optional[np.ndarray] = None          # frequency grid for rFFT
        self.M_uPa: Optional[np.ndarray] = None      # V/µPa linear scale over self.f
        self.flowInd: int = 0
        self.fhighInd: int = 0
        self.welch: Optional[float] = None           # Merchant-style compress factor

        # Time format for HDF5 DateTime
        self.time_storage = time_storage  # 'str' or 'epoch'

        # File discovery and parsing
        self.DatePattern: Optional[str] = None
        self.DateFormat: Optional[str] = None
        self.audiofiles: Optional[List[str]] = None

        # Temp dir control
        self.tmp_root = tmp_root
        self.gcs_chunk_mb = int(gcs_chunk_mb)
        self.tmp_max_gb = float(tmp_max_gb)

        # GCS client cache
        self._gcs_storage_client = None

        # Metadata
        self.DateRun = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.fullPath: Optional[str] = None  # path to current day's HDF5

        # Metric caches (computed lazily)
        self.decPrms = None   # decade band indices
        self.TolPrms = None   # third-octave indices
        self.HbrdMlDec = None # hybrid milli-decade band edges

        # If Si is a CSV path, load it now for convenience
        if isinstance(self.Si, str):
            self.Si = pd.read_csv(self.Si)

    # ---------------
    # GCS utilities
    # ---------------
    @staticmethod
    def _is_gcs_path(path: str) -> bool:
        """Return True if `path` looks like a gs:// URL."""
        return isinstance(path, str) and path.startswith("gs://")

    @staticmethod
    def _parse_gs_uri(uri: str) -> Tuple[str, str]:
        """
        Split gs://bucket/prefix into (bucket, key/prefix).
        """
        p = urlparse(uri)
        return p.netloc, p.path.lstrip("/")

    def _get_gcs_client(self):
        """Create or return a cached google.cloud.storage.Client."""
        if not _HAS_GCS:
            raise ImportError("google-cloud-storage not installed. pip install google-cloud-storage")
        if self._gcs_storage_client is None:
            self._gcs_storage_client = storage.Client()
        return self._gcs_storage_client

    # ------------------------
    # Input discovery / filter
    # ------------------------
    def _list_audio_inputs(self) -> List[str]:
        """
        Enumerate audio files under local folder or GCS prefix.
        For local folders, all files matching the extension of the first file are returned.
        For GCS, all blobs with known audio extensions are returned.
        """
        exts = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".caf"}

        if self._is_gcs_path(self.soundFilePath):
            bucket, prefix = self._parse_gs_uri(self.soundFilePath)
            client = self._get_gcs_client()
            blobs = client.list_blobs(bucket, prefix=prefix)
            uris = [
                f"gs://{bucket}/{b.name}"
                for b in blobs
                if os.path.splitext(b.name)[1].lower() in exts and not b.name.endswith("/")
            ]
            if not uris:
                raise FileNotFoundError(f"No audio files found under {self.soundFilePath}")
            return sorted(uris)

        # Local directory
        if not os.path.isdir(self.soundFilePath):
            raise FileNotFoundError(f"Local folder not found: {self.soundFilePath}")
        entries = [
            f for f in os.listdir(self.soundFilePath)
            if os.path.isfile(os.path.join(self.soundFilePath, f))
        ]
        if not entries:
            raise FileNotFoundError(f"No files in {self.soundFilePath}")
        _, ext = os.path.splitext(entries[0])
        files = glob.glob(os.path.join(self.soundFilePath, f"*{ext}"))
        if not files:
            raise FileNotFoundError(f"No *{ext} files in {self.soundFilePath}")
        return sorted(files)

    # ------------------------
    # Temp download management
    # ------------------------
    def _ensure_tmp_space(self, path_for_disk: str, bytes_needed: int) -> None:
        """
        Guardrail: make sure there's enough space on disk before downloading.
        Raises OSError if limits would be exceeded.

        Parameters
        ----------
        path_for_disk : str
            A path on the target filesystem (parent of temp dir).
        bytes_needed : int
            Estimated file size in bytes.
        """
        # Limit by tmp_max_gb if configured
        if self.tmp_max_gb and self.tmp_max_gb > 0:
            if bytes_needed > self.tmp_max_gb * (1024 ** 3):
                raise OSError(
                    f"Refusing to download ~{bytes_needed/1024**2:.1f} MiB: exceeds tmp_max_gb={self.tmp_max_gb}."
                )

        total, used, free = shutil.disk_usage(os.path.dirname(path_for_disk))
        if bytes_needed > free:
            raise OSError(
                f"Insufficient disk space: need {bytes_needed/1024**2:.1f} MiB, only {free/1024**2:.1f} MiB free."
            )

    def _download_to_temp(self, uri_or_path: str, tmpdir: str) -> str:
        """
        Download gs:// object to a temp file (chunked) and return local path.
        If `uri_or_path` is already local, return it unchanged.

        Notes
        -----
        - Uses the class' `gcs_chunk_mb` to set chunk size.
        - Validates space with `_ensure_tmp_space` using blob size.
        """
        if not self._is_gcs_path(uri_or_path):
            return uri_or_path

        bucket, key = self._parse_gs_uri(uri_or_path)
        client = self._get_gcs_client()
        blob = client.bucket(bucket).blob(key)

        # File size check / space guard
        blob.reload()  # ensure size is populated
        size = blob.size or 0
        local = os.path.join(tmpdir, os.path.basename(key))
        self._ensure_tmp_space(local, size)

        # Larger chunks help throughput on high-latency links
        blob.chunk_size = max(1, int(self.gcs_chunk_mb)) * 1024 * 1024

        # Perform the download
        blob.download_to_filename(local)
        return local

    # ------------------------
    # Date parsing utilities
    # ------------------------
    def get_datetime_format(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Inspect a filename and pick a matching regex and datetime format for timestamp parsing.

        Returns
        -------
        (pattern, strptime_format) or (None, None) if no pattern matched.
        """
        for pat, fmt in DATE_FORMATS.items():
            if re.search(pat, filename):
                self.DatePattern = pat
                self.DateFormat = fmt
                return pat, fmt
        return None, None

    def _date_key_from_name(self, path_or_name: str) -> Tuple[Optional[date], Optional[datetime]]:
        """
        Extract a UTC date (YYYYMMDD) and a datetime from a filename using the learned pattern.
        If parsing fails, we fall back to file modification time (UTC).
        """
        base = os.path.basename(path_or_name)
        if self.DatePattern and self.DateFormat:
            m = re.search(self.DatePattern, base)
            if m:
                dt = datetime.strptime(m.group(0), self.DateFormat)
                return dt.date(), dt
        # Fallback: filesystem mtime → UTC datetime
        try:
            ts = os.path.getmtime(path_or_name)
            dt = datetime.utcfromtimestamp(ts)
            return dt.date(), dt
        except Exception:
            return None, None

    # ------------------------
    # Sensitivity calibration
    # ------------------------
    def _interp_sensitivity_db_uPa(self) -> np.ndarray:
        """
        Interpolate (or broadcast) sensitivity in dB re 1 V/µPa onto self.f.

        Returns
        -------
        np.ndarray
            dB re 1 V/µPa across the analysis frequency grid.
        """
        assert self.f is not None and self.fs is not None
        if isinstance(self.Si, pd.DataFrame):
            f_col = self.Si.columns[0]
            s_col = self.Si.columns[1]
            sens_db = np.interp(
                self.f,
                np.concatenate(([0.0], self.Si[f_col].values, [self.fs / 2.0])),
                np.concatenate(([self.Si[s_col].iloc[0]], self.Si[s_col].values, [self.Si[s_col].iloc[-1]])),
            )
        else:
            sens_db = np.full_like(self.f, float(self.Si), dtype=float)
        return sens_db

    def _build_M_uPa(self) -> np.ndarray:
        """
        Build linear sensitivity magnitude M(f) in V/µPa given the configured units.
        """
        sens_db = self._interp_sensitivity_db_uPa()
        u = str(self.Si_units).lower().replace("u", "µ")
        if u in ("v/µpa", "v/μpa", "v per µpa", "v per μpa"):
            return 10.0 ** (sens_db / 20.0)
        elif u in ("v/pa", "v per pa"):
            return (10.0 ** (sens_db / 20.0)) / 1e6
        else:
            raise ValueError(f"Unknown Si_units='{self.Si_units}'. Use 'V/µPa' or 'V/Pa'.")

    # ------------------------
    # Blocked reading utility
    # ------------------------
    def _read_blocks_from_file(
        self,
        path: str,
        block_sec: float = 30.0,
        max_block_bytes: int = 64 * 1024**2,
    ) -> Iterable[Tuple[np.ndarray, int]]:
        """
        Yield consecutive blocks from a file as (mono_float32_signal, start_sample_index).

        - Ensures each block has at least one full FFT window (self.N).
        - Always returns channel-selected mono data.

        Parameters
        ----------
        path : str
            Local path to an audio file.
        block_sec : float
            Target block duration (seconds).
        max_block_bytes : int
            Upper bound on the per-block allocation to keep memory in check.
        """
        assert self.N is not None, "Call prep_audio() before reading blocks."
        with sf.SoundFile(path, "r") as f:
            fs = int(f.samplerate)
            ch = int(f.channels)
            bps = np.dtype("float32").itemsize
            max_frames_by_mem = max(1, max_block_bytes // (bps * ch))
            frames_per_block = max(self.N, min(int(block_sec * fs), int(max_frames_by_mem)))

            total_frames = len(f)
            start = 0
            while start < total_frames:
                frames = min(frames_per_block, total_frames - start)
                f.seek(start)
                yb2d = f.read(frames=frames, dtype="float32", always_2d=True)
                yb = yb2d[:, self.channel] if yb2d.shape[1] > 1 else yb2d[:, 0]
                yield yb, start
                start += frames

    # ------------------------
    # One-time preparation
    # ------------------------
    def prep_audio(self) -> None:
        """
        Discover inputs, infer filename date pattern, probe sample rate/FFT settings,
        and build the sensitivity grid. This does **not** create any HDF5 files.
        """
        inputs = self._list_audio_inputs()
        self.audiofiles = inputs

        # Train filename date parsing on the first file's basename
        first_name = os.path.basename(inputs[0])
        self.get_datetime_format(first_name)

        # Probe fs/N/etc using one temporary local copy if needed
        with tempfile.TemporaryDirectory(dir=self.tmp_root, prefix="noiseapp_probe_") as td:
            probe_path = self._download_to_temp(inputs[0], td)
            info = sf.info(probe_path)
            self.fs = int(info.samplerate)

        # FFT grid and overlap
        self.N = min(self.fs, 2**15)
        self.overlap = int(np.ceil(self.N * self.r))
        self.step = self.N - self.overlap

        # Analysis band defaults
        if self.lcut is None:
            self.lcut = 0.0
        if self.hcut is None:
            self.hcut = self.fs / 2.0

        # Welch compression factor (Merchant)
        self.welch = self.aveSec * (self.fs / self.N) / (1.0 - self.r)

        # rFFT frequency grid and calibration
        self.f = np.fft.rfftfreq(self.N, d=1.0 / self.fs)
        self.M_uPa = self._build_M_uPa()

        # Band indices
        self.flowInd = int(np.searchsorted(self.f, self.lcut, side="left"))
        self.fhighInd = int(np.searchsorted(self.f, min(self.hcut, self.fs / 2.0), side="right") - 1)

    # ------------------------
    # HDF5 management
    # ------------------------
    def initilize_HDF5(self, fullPath: str, projName: str) -> None:
        """
        Create/overwrite the HDF5 for the current day and write run/parameter metadata.
        """
        self.fullPath = fullPath
        metaVals = {
            "channel": self.channel,
            "r": self.r,
            "fs": self.fs,
            "N": self.N,
            "winname": self.winname,
            "lcut": self.lcut,
            "hcut": self.hcut,
            "overlap": self.overlap,
            "step": self.step,
            "rmDc": self.rmDC,
            "aveSec": self.aveSec,
            "welch": self.welch,
            "rmDCoffset": self.rmDC,
            "DateRun": self.DateRun,
            "time_storage": self.time_storage,
        }
        print(f"Creating HDF5 File {projName}")
        with h5py.File(fullPath, "w") as f:
            instrument_group = f.create_group(self.DepName)
            params_group = instrument_group.create_group("Parameters")
            for k, v in metaVals.items():
                params_group.attrs[k] = v

    def writeDatatoHDF5(
        self,
        new_data: np.ndarray | List | str,
        data_type: str,
        data_start: int = 0,
        max_rows: Optional[int] = None,
        storage_mode: str = "float64",
        fill_value=np.nan,
    ) -> None:
        """
        Append-or-create dataset and auto-resize as needed.

        Conventions
        -----------
        - Strings: 1-D variable length UTF-8 array
        - Vectors: stored as 2-D (rows, 1)
        - Matrices: stored as 2-D (rows, cols)
        - File is opened and **closed on every call** so memory stays low.
        """
        assert self.fullPath is not None, "No HDF5 path set."
        if new_data is None:
            raise ValueError("new_data cannot be None.")

        arr = np.asarray(new_data)
        if arr.ndim == 0:
            arr = arr[None]

        is_string = (storage_mode == "str")
        is_vector = (arr.ndim == 1)
        nrows = int(arr.shape[0])
        ncols = 1 if is_vector else int(arr.shape[1])

        with h5py.File(self.fullPath, "a") as hdf:
            grp = hdf.require_group(self.DepName)

            # Create dataset if missing
            if data_type not in grp:
                init_rows = int(max((max_rows or 0), data_start + nrows))

                if is_string:
                    dt = h5py.string_dtype(encoding="utf-8")
                    chunk_len = max(1024, min(16384, nrows))
                    dset = grp.create_dataset(
                        data_type,
                        shape=(init_rows,),
                        maxshape=(None,),
                        chunks=(chunk_len,),
                        dtype=dt,
                        fillvalue="0000-00-00 00:00:00",
                    )
                else:
                    chunk_rows = max(64, min(4096, nrows))
                    dset = grp.create_dataset(
                        data_type,
                        shape=(init_rows, ncols),
                        maxshape=(None, ncols),
                        chunks=(chunk_rows, ncols),
                        dtype=storage_mode,
                        fillvalue=fill_value,
                    )
            else:
                dset = grp[data_type]
                if not is_string and (dset.ndim != 2 or int(dset.shape[1]) != ncols):
                    raise ValueError(
                        f"Column mismatch for '{data_type}': incoming {ncols}, dataset shape {dset.shape}"
                    )

            # Grow rows if needed
            need_rows = int(data_start + nrows)
            if dset.shape[0] < need_rows:
                if dset.maxshape[0] is not None and need_rows > dset.maxshape[0]:
                    raise ValueError(
                        f"Dataset '{data_type}' not resizable. Existing {dset.shape}, need rows {need_rows}."
                    )
                if dset.ndim == 1:
                    dset.resize((need_rows,))
                else:
                    dset.resize((need_rows, dset.shape[1]))

            # Write slice
            if dset.ndim == 1:
                dset[data_start:data_start + nrows] = arr.astype(str)
            else:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                dset[data_start:data_start + nrows, :] = arr

    # ------------------------
    # Metric computations
    # ------------------------
    def calcBroadband(self, PssCropped: np.ndarray, delf: float) -> np.ndarray:
        """
        Broadband SPL from calibrated PSD (µPa²/Hz).
        PssCropped shape: (T, F)
        """
        delf = float(delf)
        total_power = np.sum(PssCropped, axis=1) * delf
        rms = np.sqrt(np.maximum(total_power, 0.0))
        return 20.0 * np.log10(np.maximum(rms, 1e-30) / self.pref)

    def calcDecadeband(self, PssCropped: np.ndarray) -> np.ndarray:
        """
        Decade-band levels (sum power across every decade).
        Returns (T, n_decades) in dB re 1 µPa.
        """
        assert self.f is not None and self.lcut is not None and self.hcut is not None
        if self.decPrms is None:
            decade_edges = np.logspace(
                np.floor(np.log10(self.lcut + 1.0)),
                np.ceil(np.log10(self.hcut) - 1.0),
                num=int(np.ceil(np.log10(self.hcut)) - np.floor(np.log10(self.lcut + 1.0))),
            )
            idxVals = np.zeros([len(decade_edges), 2])
            for ii in range(len(decade_edges)):
                idxVals[ii, 0] = np.searchsorted(self.f, decade_edges[ii], side="left")
                idxVals[ii, 1] = np.searchsorted(self.f, decade_edges[ii] * 10.0, side="right")

            self.decPrms = dict(decade_edges=decade_edges, idxVals=idxVals)

        decade_bands = np.zeros([self.decPrms["idxVals"].shape[0], PssCropped.shape[0]], dtype=float)
        for ii in range(self.decPrms["idxVals"].shape[0]):
            lo = int(self.decPrms["idxVals"][ii, 0])
            hi = int(self.decPrms["idxVals"][ii, 1])
            band_sum = np.sum(PssCropped[:, lo:hi], axis=1)
            decade_bands[ii, :] = 10.0 * np.log10(np.maximum(band_sum, 1e-30) / (self.pref ** 2))
        return decade_bands.T

    def calc13Octave(self, PssCropped: np.ndarray, B: float) -> np.ndarray:
        """
        1/3-octave band levels.

        Parameters
        ----------
        PssCropped : (T, F) calibrated PSD in µPa²/Hz
        B : float
            Bandwidth normalization constant (use 1.0)
        """
        assert self.f is not None and self.lcut is not None and self.hcut is not None
        if self.TolPrms is None:
            low13band = max(25.0, self.lcut)
            lobandf = np.floor(np.log10(low13band))
            hibandf = np.ceil(np.log10(self.hcut))
            nband = int(10 * (hibandf - lobandf) + 1)

            fc = np.zeros(nband)
            fc[0] = 10 ** lobandf
            for i in range(1, nband):
                fc[i] = fc[i - 1] * (10 ** 0.1)
            fc = fc[(fc >= low13band) & (fc <= self.hcut)]
            nfc = len(fc)

            fb = fc * (10 ** -0.05)
            fb = np.append(fb, fc[-1] * (10 ** 0.05))
            if fb[-1] > self.hcut:
                fc = fc[:-1]
                fb = fb[:-1]
                nfc = len(fc)

            fli = np.zeros(nfc)
            fui = np.zeros(nfc)
            for i in range(nfc):
                fli[i] = np.searchsorted(self.f, fb[i], side="left")
                fui[i] = np.searchsorted(self.f, fb[i + 1], side="right") - 1

            self.TolPrms = dict(nfc=nfc, fli=fli, fui=fui, fc=fc)

        P13 = np.zeros((PssCropped.shape[0], self.TolPrms["nfc"]), dtype=float)
        for i in range(self.TolPrms["nfc"]):
            lo = int(self.TolPrms["fli"][i])
            hi = int(self.TolPrms["fui"][i]) + 1
            if hi > lo:
                P13[:, i] = np.sum(PssCropped[:, lo:hi], axis=1)

        a13 = 10.0 * np.log10(np.maximum((1.0 / B) * P13, 1e-30) / (self.pref ** 2))
        return a13

    def calcHybridMilidecades(self, apsd: np.ndarray) -> np.ndarray:
        """
        Hybrid milli-decade band-averaged spectral density from APSD.

        Parameters
        ----------
        apsd : (T, F)
            APSD in dB re 1 µPa²/Hz aligned with self.f.

        Returns
        -------
        (T, nBands) in dB re 1 µPa²/Hz
        """
        assert self.f is not None and self.fs is not None and self.N is not None
        df = (self.f[1] - self.f[0]) if len(self.f) > 1 else (self.fs / self.N)

        if self.HbrdMlDec is None:
            # Linear-per-bin region up to ~435 Hz
            k_hi = int(np.searchsorted(self.f, 435.0, side="right"))
            k_hi = max(1, k_hi)
            lo = np.maximum(self.f[:k_hi] - 0.5 * df, 0.0)
            hi = self.f[:k_hi] + 0.5 * df
            bands_lin = np.column_stack([lo, self.f[:k_hi], hi])

            # Log millidecades above the break
            aa = get_band_table(
                fft_bin_size=df,
                bin1_center_frequency=0.0,
                fs=self.fs,
                base=10.0,
                bands_per_division=1000,
                first_output_band_center_frequency=max(435.0, self.f[min(k_hi, len(self.f) - 1)]),
                use_fft_res_at_bottom=False,
            )
            aa = aa[aa[:, 1] > 435.0]
            self.HbrdMlDec = {"freqLims": np.vstack([bands_lin, aa])}

        bands = self.HbrdMlDec["freqLims"]
        T = apsd.shape[0]
        out = np.empty((T, bands.shape[0]), dtype=float)

        for i, (flo, fcen, fhi) in enumerate(bands):
            idx = np.where((self.f >= flo) & (self.f < fhi))[0]
            if idx.size == 0:
                # Degenerate narrow band: take nearest bin
                idx0 = int(np.argmin(np.abs(self.f - fcen)))
                out[:, i] = apsd[:, idx0]
            elif idx.size == 1:
                out[:, i] = apsd[:, idx[0]]
            else:
                p_lin = np.nansum(10.0 ** (apsd[:, idx] / 10.0), axis=1) * df  # µPa² over band
                bw = max(fhi - flo, df)
                avg_density = p_lin / bw  # µPa²/Hz
                out[:, i] = 10.0 * np.log10(np.maximum(avg_density, 1e-30) / (self.pref ** 2))

        return np.round(out, 2)

    # ------------------------
    # Main processing
    # ------------------------
    def run_analysis(self, start_date: Optional[str | date] = None) -> None:
        """
        End-to-end processing:
        - Discovers inputs (or uses already discovered list)
        - Downloads each file to a temp folder (if GCS)
        - Streams blocks to compute PSD → metrics → HDF5 appends
        - Rotates HDF5 per UTC day, keeping memory usage bounded

        Parameters
        ----------
        start_date : str | datetime.date | None
            Skip files earlier than this date. If str, use 'YYYY-MM-DD'.
        """
        # Prep once
        self.prep_audio()
        assert self.audiofiles is not None
        assert self.f is not None and self.fs is not None and self.N is not None
        window = np.hanning(self.N).astype(np.float32)

        # Parse the filter date if any
        start_dt_date: Optional[date] = None
        if start_date is not None:
            if isinstance(start_date, str):
                start_dt_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            elif isinstance(start_date, date):
                start_dt_date = start_date
            else:
                raise ValueError("start_date must be 'YYYY-MM-DD' or datetime.date")

        current_date_key: Optional[str] = None  # YYYYMMDD string per current file's day
        data_start = 0                           # row cursor within current day's HDF5

        # Create a temp root for this run; it is auto-deleted on success/exception.
        with tempfile.TemporaryDirectory(prefix="noiseapp_", dir=self.tmp_root) as tmproot:
            print("Temp dir:", tmproot)

            for inp in self.audiofiles:
                # Date filter (based on filename if possible)
                file_date_guess, _ = self._date_key_from_name(inp)
                if start_dt_date and file_date_guess and (file_date_guess < start_dt_date):
                    # Skip early files quickly without downloading them
                    continue

                # Download if needed (GCS → local)
                local_path = self._download_to_temp(inp, tmproot)

                # Determine UTC date for rotation from the local filename (fallback: mtime)
                file_date, file_ts_dt = self._date_key_from_name(local_path)
                date_key = file_date.strftime("%Y%m%d") if file_date else "unknown"

                # Switch to a new HDF5 when the day flips
                if date_key != current_date_key:
                    current_date_key = date_key
                    projName = f"{self.ProjName}_{date_key}.h5"
                    self.fullPath = os.path.join(self.DatabaseLoc, projName)
                    self.initilize_HDF5(self.fullPath, projName)
                    data_start = 0
                    # Reset metric caches to force headers to be written once/day
                    self.decPrms = None
                    self.TolPrms = None
                    self.HbrdMlDec = None

                print(os.path.basename(local_path))

                all_t: List[np.ndarray] = []
                all_psd: List[np.ndarray] = []

                # Stream ~30-s blocks (memory-capped)
                for yb, start_samp in self._read_blocks_from_file(
                    local_path, block_sec=30.0, max_block_bytes=32 * 1024**2
                ):
                    # Optional clip at file start
                    extra_offset = 0.0
                    if self.clipFileSec and start_samp == 0:
                        clip_frames = int(self.clipFileSec * self.fs)
                        if clip_frames < len(yb):
                            yb = yb[clip_frames:]
                            extra_offset = self.clipFileSec
                        else:
                            # File is shorter than clip; skip this block
                            continue

                    if self.rmDC:
                        yb = yb - np.mean(yb)

                    if len(yb) < self.N:
                        # Need at least one full window
                        continue

                    # PSD (density) using scipy.signal.spectrogram
                    f_bins, t_rel, Sxx = sps.spectrogram(
                        yb,
                        fs=self.fs,
                        window=window,
                        nperseg=self.N,
                        noverlap=self.overlap,
                        nfft=self.N,
                        detrend=False,
                        scaling="density",
                        mode="psd",
                    )

                    # Absolute seconds from file start (float seconds)
                    t_abs = t_rel + (start_samp / self.fs) + extra_offset

                    # If for some reason FFT grid changes, sync sensitivity grid
                    if len(self.f) != len(f_bins) or (not np.allclose(self.f, f_bins)):
                        self.f = f_bins.copy()
                        self.M_uPa = self._build_M_uPa()

                    # V²/Hz -> µPa²/Hz using sensitivity M (V/µPa)
                    newPss_V2Hz = Sxx.T.astype(np.float32, copy=False)  # (T, F)
                    M = self.M_uPa[None, :].astype(np.float32, copy=False)
                    newPss_cal = newPss_V2Hz / (M ** 2)

                    all_t.append(t_abs.astype(np.float32))
                    all_psd.append(newPss_cal.astype(np.float32))

                # If no data accumulated for this file, continue
                if not all_t:
                    # Free any residuals just in case
                    del all_t, all_psd
                    gc.collect()
                    continue

                # Concatenate into a single time/PSD array
                Tsec = np.concatenate(all_t)      # (Ncols,)
                PSD = np.vstack(all_psd)          # (Ncols, F)
                del all_t, all_psd                # free immediately
                gc.collect()

                # Bin to aveSec by grouping columns into time bins
                delf = (self.f[1] - self.f[0]) if len(self.f) > 1 else (self.fs / self.N)
                t0_sec = float(Tsec.min())
                t_anchor = t0_sec - (t0_sec % self.aveSec)
                bin_idx = ((Tsec - t_anchor) // self.aveSec).astype(np.int64)

                uniq = np.unique(bin_idx)
                PSD_bin = np.zeros((len(uniq), PSD.shape[1]), dtype=np.float32)
                dt_bins: List[datetime] = []

                for j, b in enumerate(uniq):
                    m = (bin_idx == b)
                    PSD_bin[j, :] = np.nanmean(PSD[m, :], axis=0, dtype=np.float64)  # stable mean
                    tc = (int(b) + 0.5) * self.aveSec + t_anchor
                    if file_ts_dt is None:
                        dt_bins.append(datetime.utcfromtimestamp(0) + timedelta(seconds=float(tc)))
                    else:
                        # Attach to the file's date; keep time-of-day relative within that date
                        day_start = datetime.combine(file_ts_dt.date(), time())
                        dt_bins.append(day_start + timedelta(seconds=float(tc)))
                del PSD
                gc.collect()

                # Metrics
                apsd_60 = 10.0 * np.log10(np.maximum(PSD_bin, 1e-30) / (self.pref ** 2)).astype(np.float32)
                milidec = self.calcHybridMilidecades(apsd_60).astype(np.float32)
                Broadband = self.calcBroadband(PSD_bin, float(delf)).astype(np.float32)
                TOL = self.calc13Octave(PSD_bin, B=1.0).astype(np.float32)
                decadeBands = self.calcDecadeband(PSD_bin).astype(np.float32)

                # Serialize timestamps
                if self.time_storage == "epoch":
                    # seconds since UNIX epoch (float64 for precision)
                    dt_vals = np.array([dt.timestamp() for dt in dt_bins], dtype="float64")
                    self.writeDatatoHDF5(dt_vals, "DateTime", data_start=data_start, storage_mode="float64")
                else:
                    # ISO strings for plotting convenience
                    ttISO = np.array([dt.strftime("%Y%m%dT%H%M%S") for dt in dt_bins])
                    self.writeDatatoHDF5(ttISO, "DateTime", data_start=data_start, storage_mode="str")

                # Data arrays
                self.writeDatatoHDF5(milidec, "hybridMiliDecLevels", data_start=data_start, storage_mode="float32")
                self.writeDatatoHDF5(Broadband, "broadband", data_start=data_start, storage_mode="float32")
                self.writeDatatoHDF5(TOL, "thirdoct", data_start=data_start, storage_mode="float32")
                self.writeDatatoHDF5(decadeBands, "decadeLevels", data_start=data_start, storage_mode="float32")

                # Write headers on first chunk of the day
                if data_start == 0:
                    self.writeDatatoHDF5(self.HbrdMlDec["freqLims"], "hybridDecFreqHz",
                                         data_start=0, max_rows=len(self.HbrdMlDec["freqLims"]), storage_mode="float32")
                    self.writeDatatoHDF5(self.TolPrms["fc"], "thirdOctFreqHz",
                                         data_start=0, max_rows=len(self.TolPrms["fc"]), storage_mode="float32")
                    self.writeDatatoHDF5(self.decPrms["decade_edges"], "decadeFreqHz",
                                         data_start=0, max_rows=len(self.decPrms["decade_edges"]), storage_mode="float32")

                # Advance the row cursor
                data_start += len(dt_bins)

                # Aggressive cleanup for long runs
                del PSD_bin, apsd_60, milidec, Broadband, TOL, decadeBands, dt_bins
                gc.collect()

        # When we exit the with-block, the temp directory is removed.
        # All HDF5 files are closed after each append (open/close per write).
        return

    # ------------------------
    # Optional calibration self-test
    # ------------------------
    def selftest_calibration(
        self,
        file_index: int = 0,
        max_duration_sec: Optional[float] = None,
        waveform_is_pressure: bool = False,
        expect_tolerance_db: float = 2.0,
        synthetic_sens_db: float = 6.0,
        autoguess_units: bool = True,
    ) -> dict:
        """
        Lightweight single-file sanity check of the calibration path.

        See the earlier versions for more detailed commentary.
        """
        # Discover one file if not already
        if self.audiofiles is None:
            self.prep_audio()
        assert self.audiofiles is not None
        test_file = self.audiofiles[file_index]

        with sf.SoundFile(test_file, "r") as f:
            fs0 = f.samplerate
            if max_duration_sec is None:
                max_duration_sec = min(60.0, len(f) / fs0)
            frames_to_read = int(max_duration_sec * fs0)
            data = f.read(frames=frames_to_read, dtype="float32", always_2d=True)

        yy = data[:, self.channel] if data.ndim > 1 else data
        if self.rmDC:
            yy = yy - np.mean(yy)

        if self.fs is None:
            self.fs = fs0
            self.N = min(self.fs, 2**15)
            self.overlap = int(np.ceil(self.N * self.r))
            if self.lcut is None or self.hcut is None:
                self.lcut = 0.0
                self.hcut = self.fs / 2.0
            self.welch = self.aveSec * (self.fs / self.N) / (1 - self.r)
            self.f = np.fft.rfftfreq(self.N, d=1.0 / self.fs)

            # Build M_uPa (with optional CSV unit autoguess)
            if isinstance(self.Si, pd.DataFrame):
                f_col = self.Si.columns[0]
                s_col = self.Si.columns[1]
                sens_all_db = self.Si[s_col].astype(float).values
                mean_db = float(np.nanmean(sens_all_db))
                si_is_v_per_uPa = (mean_db < -120.0) if autoguess_units else True
                sens_db = np.interp(
                    self.f,
                    np.concatenate(([0.0], self.Si[f_col].values, [self.fs / 2.0])),
                    np.concatenate(([self.Si[s_col].iloc[0]], self.Si[s_col].values, [self.Si[s_col].iloc[-1]])),
                )
                if si_is_v_per_uPa:
                    M_uPa = 10.0 ** (sens_db / 20.0)
                    csv_units = "V/µPa"
                else:
                    M_uPa = (10.0 ** (sens_db / 20.0)) / 1e6
                    csv_units = "V/Pa"
            else:
                sens_db = float(self.Si)
                M_uPa = np.full_like(self.f, 10.0 ** (sens_db / 20.0), dtype=float)
                csv_units = "V/µPa (scalar)"
            self.M_uPa = M_uPa
        else:
            csv_units = "cached"

        # PSD
        window = np.hanning(self.N).astype(np.float32)
        f_spec, t_spec, Sxx = sps.spectrogram(
            yy, fs=self.fs, window=window, nperseg=self.N, noverlap=self.overlap,
            nfft=self.N, detrend=False, scaling="density", mode="psd"
        )
        delf = (f_spec[1] - f_spec[0]) if len(f_spec) > 1 else (self.fs / self.N)
        tt = np.linspace(0, len(yy) / self.fs, num=Sxx.shape[1], dtype=float)
        newPss_V2Hz = Sxx.T

        M_uPa_aligned = np.interp(f_spec, self.f, self.M_uPa,
                                  left=self.M_uPa[0], right=self.M_uPa[-1])
        newPss_cal = newPss_V2Hz / (M_uPa_aligned[None, :] ** 2)

        total_power_t = np.sum(newPss_cal, axis=1) * delf
        rms_psd = float(np.sqrt(np.mean(total_power_t)))
        Lp_psd_db = 20.0 * np.log10(max(rms_psd, 1e-30) / self.pref)

        if waveform_is_pressure:
            rms_td = float(np.sqrt(np.mean(yy ** 2)))
            Lp_time_db = 20.0 * np.log10(max(rms_td, 1e-30) / self.pref)
            delta_db = Lp_time_db - Lp_psd_db
        else:
            Lp_time_db = np.nan
            delta_db = np.nan

        M_gain = M_uPa_aligned * (10.0 ** (synthetic_sens_db / 20.0))
        newPss_cal_gain = newPss_V2Hz / (M_gain[None, :] ** 2)
        total_power_t_gain = np.sum(newPss_cal_gain, axis=1) * delf
        rms_psd_gain = float(np.sqrt(np.mean(total_power_t_gain)))
        Lp_psd_gain_db = 20.0 * np.log10(max(rms_psd_gain, 1e-30) / self.pref)
        shift_db = Lp_psd_gain_db - Lp_psd_db

        print("\n=== Calibration Self-Test ===")
        print(f"File: {os.path.basename(test_file)}")
        print(f"CSV units assumed: {csv_units}")
        print(f"fs={self.fs:.1f} Hz, N={self.N}, delf={delf:.6f} Hz, "
              f"T={newPss_cal.shape[0]}, F={newPss_cal.shape[1]}")
        if waveform_is_pressure:
            print(f"Time-domain SPL: {Lp_time_db:.2f} dB re 1 µPa")
            print(f"PSD-integrated SPL: {Lp_psd_db:.2f} dB re 1 µPa")
            print(f"Δ(TD - PSD): {delta_db:+.2f} dB (tol ±{expect_tolerance_db:.1f} dB)")
        else:
            print("Waveform treated as VOLTAGE (skipping TD vs PSD).")
            print(f"PSD-integrated SPL: {Lp_psd_db:.2f} dB re 1 µPa")
        print(f"Sensitivity +{synthetic_sens_db:.1f} dB → SPL shift {shift_db:+.2f} dB "
              f"(expect ≈ -{synthetic_sens_db:.1f} dB).")

        return dict(
            file=test_file, fs=int(self.fs), N=int(self.N), delf=float(delf),
            csv_units=csv_units, waveform_is_pressure=bool(waveform_is_pressure),
            Lp_time_db=float(Lp_time_db) if np.isfinite(Lp_time_db) else np.nan,
            Lp_psd_db=float(Lp_psd_db),
            delta_db=float(delta_db) if np.isfinite(delta_db) else np.nan,
            Lp_psd_gain_db=float(Lp_psd_gain_db),
            shift_db=float(shift_db)
        )


# ---------------------------------------------------------------------------
# Plot helpers (optional, for inspecting outputs)
# ---------------------------------------------------------------------------

def plot_milidecade_statistics(instrument_group, pBands = [5, 25, 50, 75, 95]):
    """
    Plot empirical probability density (SPD) with percentile curves and RMS
    from the hybrid milli-decade dataset in an HDF5 instrument group.
    """
    # Time stamps (strings) and valid range
    time_stamps = instrument_group["DateTime"][:].astype(str)
    max_data_len = min(
        np.argmax(time_stamps == "0000-00-00 00:00:00") if "0000-00-00 00:00:00" in time_stamps else len(time_stamps),
        len(time_stamps),
    )
    ff = instrument_group["hybridDecFreqHz"][:, 1]  # band centers

    PSD = instrument_group["hybridMiliDecLevels"][0:max_data_len, :]
    RMSlevel = 10 * np.log10(np.mean(10 ** (PSD / 10.0), axis=0))

    p = np.percentile(PSD, pBands, axis=0)
    mindB = np.floor(np.min(PSD) / 10.0) * 10.0
    maxdB = np.ceil(np.max(PSD) / 10.0) * 10.0

    # SPD histogram
    hind = 0.1
    dbvec = np.arange(mindB, maxdB + hind, hind)
    M = PSD.shape[0] - 1
    d = np.zeros((len(dbvec) - 1, PSD.shape[1]))
    for i in range(PSD.shape[1]):
        d[:, i] = np.histogram(PSD[:, i], bins=dbvec, density=True)[0]
    d /= (hind * M)
    d[d == 0] = np.nan

    X, Y = np.meshgrid(ff + 1, dbvec[:-1])

    fig, ax0 = plt.subplots(1, 1, figsize=(9, 6))
    c = ax0.pcolor(X, Y, d, shading="auto")
    ax0.set_xscale("log")
    plt.colorbar(c, ax=ax0, label="Empirical Probability Density")
    ax0.set_xlabel("Frequency (Hz)")
    ax0.set_ylabel("PSD (dB re 1 µPa²/Hz)")
    ax0.set_title("Empirical Probability Density (SPD)")

    grays = [[k/10.0]*3 for k in range(0, 5)]
    for i, p_band in enumerate(pBands):
        ax0.semilogx(ff + 1, p[i, :], label=f"L{p_band}", color=grays[i])
    ax0.semilogx(ff + 1, RMSlevel, label="RMS Level", color="m", linewidth=2)
    ax0.set_xlim(ff.min(), ff.max())
    ax0.set_ylim(Y.min(), Y.max())
    ax0.legend(loc="upper right", fontsize=9)
    plt.show()
    return fig


def plot_ltsa(instrument_group, averaging_period="5min", titleText=""):
    """
    Long-Term Spectral Average heatmap using median (or RMS-equivalent) within bins.

    Notes
    -----
    - Expects datasets 'hybridDecFreqHz' and 'hybridMiliDecLevels' and 'DateTime'.
    - Uses ISO strings in DateTime; for epoch storage adapt accordingly.
    """
    freq_tab = instrument_group["hybridDecFreqHz"][:]
    time_stamps = instrument_group["DateTime"][:].astype(str)
    time_stamps = pd.to_datetime(time_stamps, errors="coerce")
    time_stamps = time_stamps.dropna()

    # Trim to valid data range (if padded)
    max_data_len = np.where(time_stamps.astype(str) == "0000-00-00 00:00:00")[0]
    if len(max_data_len) > 0:
        time_stamps = time_stamps[:max_data_len[0]]

    data = instrument_group["hybridMiliDecLevels"][0:len(time_stamps), :]

    median_spacing = pd.to_timedelta(averaging_period)
    time_chunks = pd.date_range(
        start=time_stamps.iloc[0].floor(freq=averaging_period),
        end=time_stamps.iloc[-1].ceil(freq=averaging_period),
        freq=median_spacing,
    )

    # Prepare grid
    X, Y = np.meshgrid(time_chunks[:-1], freq_tab[:, 1])
    nlVals = np.zeros(X.shape) / 0.0

    # Bin by time and compute RMS-equivalent
    for i in range(len(time_chunks) - 1):
        idx = np.where((time_stamps >= time_chunks[i]) & (time_stamps < time_chunks[i + 1]))[0]
        chunk = data[idx, :]
        if chunk.size > 0:
            # Equivalent to RMS over the period in dB space
            med_vals = 10.0 * np.log10(np.mean(10.0 ** (chunk / 10.0), axis=0))
            nlVals[:, i] = np.round(med_vals, 2)

    nlVals = nlVals[:, ~np.isnan(nlVals).all(axis=0)]

    fig, ax = plt.subplots(figsize=(10, 6))
    cbar = sns.heatmap(np.flipud(nlVals), cmap="cubehelix", ax=ax, cbar=True).collections[0].colorbar
    cbar.set_label(r"RMS SPL (dB re 1 $\mu$Pa)", fontsize=8)

    ax.set_ylim([70, 150])
    ax.set_ylabel(r"RMS SPL (dB re 1 $\mu$Pa)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(direction="out", top=True, right=True, labelsize=12)
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})

    ax.set_yticks(np.linspace(0, len(freq_tab[:, 1]) - 1, num=6))
    ax.set_yticklabels(np.flip(freq_tab[:, 1].astype(int)))
    ax.set_title("Long-Term Spectral Average (LTSA)" + (f" — {titleText}" if titleText else ""))

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # GCS source (bucket/prefix)
    gsCloudLoc = "gs://pifsc-1/glider/sg680_MHI_Apr2022/recordings/wav"
    out_dir = r"C:\Users\pam_user\Documents\HybridMilliDaily"
    calib_csv = r"C:\Users\pam_user\Downloads\sg680_CalCurCEAS_Sep2024_sensitivity_2025-07-29.csv"

    app = NoiseApp(
        Si=calib_csv,
        soundFilePath=gsCloudLoc,
        ProjName="sg680_MHI_Apr2022",
        DepName="SG650",
        DatabaseLoc=out_dir,
        rmDC=True,
        Si_units="V/µPa",
        time_storage="str",   # 'epoch' for faster writes
        tmp_root=out_dir,     # keep temp folders on the same drive as outputs
        gcs_chunk_mb=16,      # tune for your network
        tmp_max_gb=0.0,       # set >0 to enforce per-file temp size guard
    )

    # Start at a specific date (skip earlier files)
    app.run_analysis(start_date="2022-04-08")

    # Example to plot later:
    h5_name = r"C:\Users\pam_user\Documents\HybridMilliDaily\sg680_MHI_Apr2022_20220408.h5"
    with h5py.File(h5_name, "r") as hdf:
        grp = hdf["SG650"]
        _ = plot_milidecade_statistics(grp)
        _ = plot_ltsa(grp, averaging_period="60min", titleText="Apr 8")
