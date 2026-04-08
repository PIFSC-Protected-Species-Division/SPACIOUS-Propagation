# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 05:45:46 2025

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
from NoiseProcessingDefs import process_bucket_audio_daily

# 1) Reset matplotlib (optional visual hygiene)
mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False
plt.style.use("default")

# 2) Set up hydrophone/recording system
model = "Whisper?"
name = "sg679_CalCurCEAS_Aug2024_sensitivity_2025"
serial_number = 411
calFileLoc = r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes\ExampleData\sg680_CalCurCEAS_Sep2024_sensitivity_2025-07-29.nc"

drifter_Cal = pyhy.custom(
    name=name,
    model=model,
    sensitivity=0,
    serial_number=serial_number,
    preamp_gain=0,
    Vpp=10,
    calibration_file=calFileLoc,
)
drifter_Cal.end_to_end_calibration()


# 3) ASA parameters
band = [0, 65000]           # Hz (<= Nyquist)
nfft = band[1] * 2          # simple rule: 2x top freq
binsize = 60.0              # seconds
include_dirs = False
zipped_files = False
dc_subtract = 0

#%%
# 4) IO locations
gsCloudLoc = "pifsc-1/glider/sg680_MHI_Apr2022/recordings/wav"
out_dir = r"C:\Users\pam_user\Documents\HybridMilliDaily"

process_bucket_audio_daily(
    gs_uri="pifsc-1/glider/sg680_MHI_Apr2022/recordings/wav",
    drifter_Cal=drifter_Cal,
    binsize=binsize,
    nfft=nfft,
    include_dirs=include_dirs,
    zipped_files=zipped_files,
    dc_subtract=dc_subtract,
    band=band,
    out_dir=out_dir,
    start_date="2022-04-06",   # first day to process
)
#%% Load/plot daily averages
# Plot the spectrum mean with the standard deviation

import pypam
import xarray as xr, pypam, matplotlib.pyplot as plt

ds = xr.open_dataset(r"C:\Users\pam_user\Documents\HybridMilliDaily\HybridMilli_20220407.nc")
var = "millidecade_bands"

# Reduce over 'id' and RESTORE attributes for pypam's ylabel
orig_attrs = ds[var].attrs
if "standard_name" not in orig_attrs:
    orig_attrs["standard_name"] = "Power spectral density"
if "units" not in orig_attrs:
    orig_attrs["units"] = "dB re 1 µPa²/Hz"

ds2 = ds.copy()
ds2[var] = ds2[var].median(dim="id", skipna=True).assign_attrs(orig_attrs)
ds2[var] = ds2[var].transpose("time", "frequency_bins")  # explicit order

pypam.plots.plot_spectrum_median(
    ds2, data_var=var, frequency_coord="frequency_bins", time_coord="time"
)
plt.show()



