# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:29:19 2026

@author: pam_user
"""

# -*- coding: utf-8 -*-
"""
Clean parallel export of Bellhop HDF5 → CSV

Run from terminal:
python ExportSensitivityCSVs_clean.py
"""

# -----------------------------------------------------------------------------
# 1. Make local modules importable (MUST be before imports)
# -----------------------------------------------------------------------------
import sys
sys.path.append(r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation")

# -----------------------------------------------------------------------------
# 2. Imports
# -----------------------------------------------------------------------------
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import glob
import os

import numpy as np
import librosa

from PlottingDefs import scaleP2P, CreateOutputCSVs_long_worker


# -----------------------------------------------------------------------------
# 3. Worker function (must be top-level for multiprocessing)
# -----------------------------------------------------------------------------
def process_h5file(h5file, base_in, base_out, click_waveform, samplerate):
    hfLoc = base_in / h5file
    print(f"Starting: {h5file}")

    # classify sediment
    name_lower = h5file.lower()
    if "gravel" in name_lower:
        subdir = "gravel"
    elif "silt" in name_lower:
        subdir = "silt"
    else:
        subdir = "basalt"

    out_path = base_out / subdir
    out_path.mkdir(parents=True, exist_ok=True)


    CreateOutputCSVs_long_worker(
            h5_path=str(hfLoc),
            segment=click_waveform,         # 1-D np.ndarray
            samplerate=samplerate,
            out_path=str(out_path),

            nWorkers=4,
            f_ref_hz=12000,
            prefer_processes=False,         # threads are safer on Windows top-level
            fmin_hz=100, 
            fmax_hz=samplerate/2, 
            df_hz=100
        )
    h5_path,
    segment,
    samplerate,
    out_path,
    f_ref_hz=12000,
    return f"Done: {h5file}"


# -----------------------------------------------------------------------------
# 4. Main execution block (REQUIRED on Windows)
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # File locations
    # -------------------------------------------------------------------------
    base_in  = Path(r"X:\Kaitlin_Palmer\CalCurCEAS_propagation_hdf5s")
    base_out = Path(r"X:\Kaitlin_Palmer\BotSensitivityCSVs")

    # -------------------------------------------------------------------------
    # Load click waveform ONCE (shared input)
    # -------------------------------------------------------------------------
    wav_path = r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes\ExampleData\WHICEAS_click.wav"

    audiodata, samplerate = librosa.load(wav_path, sr=60000, mono=False)

    click_waveform = scaleP2P(audiodata, outP2P=220)

    # -------------------------------------------------------------------------
    # Get file list
    # -------------------------------------------------------------------------
    h5_files = sorted([f.name for f in base_in.glob("*.h5")])

    print(f"\nFound {len(h5_files)} files\n")

    # -------------------------------------------------------------------------
    # Parallel setup
    # -------------------------------------------------------------------------
    n_cores = mp.cpu_count()
    n_jobs = min(6, n_cores - 1)  # avoid disk bottleneck

    print(f"Using {n_jobs} workers\n")

    # -------------------------------------------------------------------------
    # Run parallel processing
    # -------------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:

        futures = [
            executor.submit(
                process_h5file,
                h5file,
                base_in,
                base_out,
                click_waveform,
                samplerate
            )
            for h5file in h5_files
        ]

        for f in as_completed(futures):
            try:
                print(f.result())
            except Exception as e:
                print("ERROR:", e)

    print("\nAll files processed.\n")