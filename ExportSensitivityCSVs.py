# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 04:07:27 2025

@author: pam_user

This is to be used with PlottingDefs.py this should be an example, plottingDefs
should be the begining of a package
"""
import sys
sys.path.append(r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation")
from scipy.io import wavfile
import os
#from PlottingDefs import CreateOutputCSVs, CreateOutputCSVs_Spherical, export_long_tables_spherical
from PlottingDefs import scaleP2P # for source level scaling
from PlottingDefs import plot_peak2peak_isosurfaces, plot_detection_probability
from PlottingDefs import plot_detection_vs_range, fig_signal_ir_output, fit_and_plot_hazard_rate_by_location
from PlottingDefs import CreateOutputCSVs_long
import librosa
import numpy as np
import matplotlib.pyplot as plt




#%% Create the CSV's of the arrival RLs
# File locations for the HDF5 from the bellhop models, audio file to convolve
# and where to save th exported csvs
h5_path = 'X:\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\BottomSenExperiment\\Spacious_CalCurses_Sensitivity_PCHIP_12kHz_20km_200m_BotSensitivity.h5'
wav_path = "C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\WHICEAS_click.wav"
out_path = "X:\Kaitlin_Palmer\CalCurCEAS_propagation_csvs\BottomSenExperiment"

# h5_path = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5'
# wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
# out_path = "C:\\Users\\kaity\\Desktop\\TestCoherentBellhopCSVS"

# --- Signal Setup ---
#samplerate, audiodata = wavfile.read(wav_path)
audiodata, samplerate = librosa.load(wav_path, sr=60000,    mono= False)
#t_start, t_end, chan = 32.58, 32.60, 4
#segment = audiodata[chan, int(round(t_start * samplerate)):int(round(t_end * samplerate))]
tt = np.linspace(0, len(audiodata)/samplerate, len(audiodata))
# Adjust figure size and DPI if needed
plt.figure(figsize=(11, 5), dpi=100)

# Plot raw click for a giggle test
plt.figure(1)
plt.plot(tt*1000, audiodata)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Raw Sperm Whale Click', fontsize=14)

# Scale the segment using the custom function so it's 220 dB 
# peak-to-peak prior to convolution with the impulse response
click_waveform = scaleP2P(audiodata, outP2P= 220)

plt.plot(tt*1000, click_waveform)


#%% Use convolution of the signal of interest to calculate the peak to peak
# RL at each of the sensor locations

# Calculate the received arrays and export to csv- this takes a while

import os
import glob
import os
from pathlib import Path

os.chdir('X:\\Kaitlin_Palmer\\\CalCurCEAS_propagation_hdf5s\\')
result = glob.glob('*.{}'.format('h5'))
print(result)

base_in  = Path(r"X:\Kaitlin_Palmer\CalCurCEAS_propagation_hdf5s")
base_out = Path(r"X:\Kaitlin_Palmer\BotSensitivityCSVs")



for h5file in result:
    # Full path to input HDF5
    hfLoc = base_in / h5file

    # Decide subfolder based on filename (case-insensitive)
    name_lower = h5file.lower()
    if "gravel" in name_lower:
        subdir = "gravel"
    elif "silt" in name_lower:
        subdir = "silt"
    else:
        subdir = "basalt"  # keep base_out; or use "other" if you prefer a catch-all

    # Build output path and ensure it exists
    out_path = (base_out / subdir) if subdir else base_out
    out_path.mkdir(parents=True, exist_ok=True)

    # Optional: construct an explicit CSV filename (not used by CreateOutputCSVs_long)
    fname = Path(h5file).stem + "On_axis"
    out_csv = out_path / f"{fname}.csv"

    # Run export
    CreateOutputCSVs_long(
        h5_path=str(hfLoc),
        segment=click_waveform,         # 1-D np.ndarray
        samplerate=samplerate,
        out_path=str(out_path),
        coherent=True,
        nWorkers=15,
        f_ref_hz=12000,
        prefer_processes=False,         # threads are safer on Windows top-level
        fmin_hz=100, 
        fmax_hz=samplerate/2, 
        df_hz=100
    )

    print(f"{h5file} -> {out_path}")


# Export the metadata for each gird
#%% Same thing as above but parallelized on the outer loop

# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing as mp

# from pathlib import Path

# def process_h5file(h5file, base_in, base_out, click_waveform, samplerate):
#     hfLoc = base_in / h5file
    
#     print("Starting worker:", h5file)

#     name_lower = h5file.lower()
#     if "gravel" in name_lower:
#         subdir = "gravel"
#     elif "silt" in name_lower:
#         subdir = "silt"
#     else:
#         subdir = "basalt"

#     out_path = (base_out / subdir) if subdir else base_out
#     out_path.mkdir(parents=True, exist_ok=True)

#     CreateOutputCSVs_long(
#         h5_path=str(hfLoc),
#         segment=click_waveform,
#         samplerate=samplerate,
#         out_path=str(out_path),
#         coherent=True,
#         nWorkers=1,                  # 🔴 critical change
#         f_ref_hz=12000,
#         prefer_processes=False,
#         fmin_hz=100,
#         fmax_hz=samplerate/2,
#         df_hz=100
#     )

#     return f"{h5file} -> {out_path}"

# if __name__ == "__main__":
#     base_in  = Path(r"X:\Kaitlin_Palmer\CalCurCEAS_propagation_hdf5s")
#     base_out = Path(r"X:\Kaitlin_Palmer\BotSensitivityCSVs")

#     n_cores = mp.cpu_count()
#     n_jobs = min(12, n_cores - 1)
#     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#         futures = [
#             executor.submit(
#                 process_h5file,
#                 h5file,
#                 base_in,
#                 base_out,
#                 click_waveform,
#                 samplerate
#             )
#             for h5file in result
#         ]

#         for f in as_completed(futures):
#             print(f.result())
# #%% Should we model Pdet as a function of RL?

# detThreshs = [130,135,140]

# for thresh in detThreshs:
#     # Plot the new iso-surface
#     plot_peak2peak_isosurfaces(
#                     h5_path, RLdata, diveId ='dive_42',
#                     iso_levels=(thresh,),
#                     xy_res=200,
#                     seabed_color='0.6',
#                     elev=40, azim=-90)
     
#     plot_detection_probability(h5_path,
#         RLdata, thresh,
#         cmap='viridis',
#         diveId ='dive_42', 
#         vmin=0, vmax=1, 
#         title=None, s=40)

#     stats_df = plot_detection_vs_range(h5_path=h5_path,
#                     RLdata=RLdata,
#                     threshold_db=thresh,
#                     bin_width_km= .1)
#     fit_and_plot_hazard_rate_by_location(RLdata= RLdata,
#                                              h5_path= h5_path, 
#                                              diveId=  "dive_42",
#                                              threshold_db = thresh) 

    
#%% Restrict to sperm whale depths

# # Get the depth values from the HDF5
# import h5py

# # Now get the depths
# hf = h5py.File(h5_path, 'r')
# diveId ='dive_42'
# dive_grp = hf[f'drift_01/{diveId}/frequency_35000']
# run_ids = list(dive_grp['arrivals'].keys())
# depth_grid = np.array(dive_grp['depth'])

# # Say 500m to 1200m depth that's column 5 on
# np.nanmax(RLdata)
# RLdata[:, 1:4] = -500
# RLdata[:, 13:27] = -500


# # Plot the new iso-surface
# plot_peak2peak_isosurfaces(
#                 h5_path, RLdata, 
#                 diveId ='dive_42',
#                 iso_levels=(140,),
#                 xy_res=200,
#                 seabed_color='0.6',
#                 elev=25, azim=-90)

# plot_detection_probability(h5_path,
#     RLdata, thresh,
#     cmap='viridis',diveId ='dive_42', vmin=0, vmax=1, 
#     title=None, s=40)


#%% Pipeline Examples

# This section is intended 
from H5ArrivalsBridge import list_points, load_point_by_index, load_point_near
from PlottingDefs import (
    build_freq_grid, fig_source_and_grid, fig_absorption,
    fig_transfer_and_received, fig_compare_legacy_vs_coherent
)

h5 = h5_path
drift_id='drift_01'
dive_id = "dive_42"      # whatever you have
fs = samplerate                   # your click sample rate
click = click_waveform     # np.ndarray


# 1) pick a point by index
arrivals =  load_point_by_index(h5_path, 
                        dive_id, 
                        pt_index ='pt_03051',
                        drift_id=drift_id, 
                        freq_khz='frequency_35000',
                        c_eff_m_s=1480.0, 
                        estimate_pathlen_if_missing=True)


# 3) build frequency grid and make figures
freqs = build_freq_grid(2000, 65000, 200, fs=fs)
fig_source_and_grid(click, fs, freqs)
fig_absorption(freqs, f_ref_hz=35000.0)

fig3, r = fig_transfer_and_received(click, fs, arrivals, freqs,
                                    f_ref_hz=35000.0,
                                    c_eff_m_s=1480.0,
                                    arrivals_include_absorption=True)


fig3, r = fig_signal_ir_output(click, fs, arrivals, freqs,
                                    f_ref_hz=35000.0,
                                    c_eff_m_s=1480.0,
                                    arrivals_include_absorption=True)

fig4, stats = fig_compare_legacy_vs_coherent(click, fs, arrivals, freqs,
                                             f_ref_hz=35000.0,
                                             c_eff_m_s=1480.0,
                                             arrivals_include_absorption=True)




