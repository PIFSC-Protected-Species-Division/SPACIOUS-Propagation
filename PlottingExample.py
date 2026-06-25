# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 04:07:27 2025

@author: pam_user

This is to be used with PlottingDefs.py this should be an example, plottingDefs
should be the begining of a package
"""

from scipy.io import wavfile
import os
#from PlottingDefs import CreateOutputCSVs, CreateOutputCSVs_Spherical, export_long_tables_spherical
from PlottingDefs import scaleP2P # for source level scaling
from PlottingDefs import plot_peak2peak_isosurfaces, plot_detection_probability, plot_peak2peak_isosurfaces_long
from PlottingDefs import plot_detection_vs_range, fig_signal_ir_output, fit_and_plot_hazard_rate_by_location
from PlottingDefs import CreateOutputCSVs_long, alphaAdjustment, apply_alpha_correction, plot_detection_by_bearing
import librosa
import numpy as np
import matplotlib.pyplot as plt




import matplotlib as mpl
import matplotlib.pyplot as plt

# Turn off LaTeX text rendering (this is what's failing)
mpl.rcParams["text.usetex"] = False

# Optional: make sure mathtext is used instead of TeX
mpl.rcParams["mathtext.default"] = "regular"

#%% Creat the CSV's of the arrival RLs
# File locations for the HDF5 from the bellhop models, audio file to convolve
# and where to save th exported csvs
h5_path = 'X:\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\BottomSenExperiment\\Spacious_CalCurses_Silt_PCHIP_12kHz_20km_500m_BotSensitivity.h5'
wav_path = "C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\1705_20171028_010934_441.wav"
out_path = "X:\Kaitlin_Palmer\CalCurCEAS_propagation_csvs\BottomSenExperiment"

# h5_path = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5'
# wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
# out_path = "C:\\Users\\kaity\\Desktop\\TestCoherentBellhopCSVS"

# --- Signal Setup ---
#samplerate, audiodata = wavfile.read(wav_path)
audiodata, samplerate = librosa.load(wav_path, sr=65000,    mono= False)
t_start, t_end, chan = 32.58, 32.60, 4
segment = audiodata[chan, int(round(t_start * samplerate)):int(round(t_end * samplerate))]
tt = np.linspace(0, len(segment)/samplerate, len(segment))
# Adjust figure size and DPI if needed
plt.figure(figsize=(11, 5), dpi=100)

# Plot raw click for a giggle test
plt.figure(1)
plt.plot(tt*1000, segment)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Raw Sperm Whale Click', fontsize=14)

# Scale the segment using the custom function so it's 220 dB 
# peak-to-peak prior to convolution with the impulse response
click_waveform = scaleP2P(segment, outP2P= 220)

plt.plot(tt*1000, segment)


#%% Use convolution of the signal of interest to calculate the peak to peak
# RL at each of the sensor locations

# Calculate the received arrays and export to csv- this takes a while

import os
import glob

os.chdir('X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\')
result = glob.glob('*.{}'.format('h5'))
print(result)

for h5file in result:
    hfLoc = (os.path.join('X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\',
                       h5file))
    
    # Export metadata
    fname = os.path.splitext(h5file)[0]
    out_csv = os.path.join('X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_csvs', 
                           fname+ '.csv')
    
    out_path = 'X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_csvs'

    
    CreateOutputCSVs_long(
        h5_path=hfLoc,
        segment=click_waveform,                 # 1-D np.ndarray
        samplerate=samplerate,
        out_path=out_path,
        coherent=False,
        nWorkers=60,
        f_ref_hz =35000,
        prefer_processes=False,           # threads are safer on Windows top-level
        fmin_hz=1000, 
        fmax_hz=20000, 
        df_hz=200
    )
    print(h5file)


# Export the metadata for each gird

#%% Load the RL grid in the previous section and make plots 

rLlOC = 'X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_csvs\\PeakToPeak_dive_42_GliderDepth_500m.csv'
rLlOC = 'X:\\Kaitlin_Palmer\\BotSensitivityCSVs\\si'

RLdata = np.genfromtxt(rLlOC, delimiter=',')


np.nanmax(RLdata)

# The impulse response was created using a bellhop model at 35khz and the
# amplitude is based on 35khz attenuation. If we are interested more in 4khz
# or similar then we need to increase the amplitude of the impulse response 
# (less attenuation)

# Simple calcuation to get the change in alpha (db/km) between two frequencies
# High frequency is absorbed in sea water at higher levels than low frequency
alphachange = alphaAdjustment(bellhopFreq=35000, newFreq =2000)

# The change in alpha coefficient in absorption is 5.2 dB/km between
# 35 khz and 2khz so we need to add that back into the RL grids accounting
# for beam angle. Does not completely account for beam lenght (e.g. bounces) 
# or change in ssp

# Create the adjusted implse response 
corrected_data = apply_alpha_correction(h5_path= h5_path, 
                                        RLdata =RLdata,
                                        alpha_db_per_km=alphachange, 
                                        diveId ='dive_42')

import pandas as pd

rLlOC = 'X:\\Kaitlin_Palmer\\BotSensitivityCSVs_WHICEAS_clip\\BotSensitivityCSVs\\silt\\PeakToPeak_dive_167_GliderDepth_500m_0_29khz_long.csv'
rl_df = pd.read_csv(rLlOC)

h5_path = 'X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\Spacious_CalCurses_silt_PCHIP_12kHz_20km_500m_BotSensitivity.h5'




# Plot the new iso-surface
plot_peak2peak_isosurfaces(
                h5_path, 
                rl_df, 
                diveId ='dive_167',
                title = 'Site 163 135 dB Isopleth',
                iso_levels=(140,),  
                xy_res=200,
                source_depth_m=100,
                interp_method='cubic', # default linear, faster
                source_x_m=0,
                source_y_m=0,
                seabed_color='0.6',
                elev=26, 
                azim=-75,
                render_mode='publication') # default 'fast')


# Quick preview
plot_peak2peak_isosurfaces(..., render_mode="fast")

# Balanced quality (default)
plot_peak2peak_isosurfaces(..., render_mode="balanced")

# Publication quality (higher-res, cubic interpolation, larger figure)
plot_peak2peak_isosurfaces(..., render_mode="publication", 
                           save_path="figure.png", save_dpi=600)




 
plot_detection_probability(h5_path,
    RLdata, 80,
    cmap='viridis',diveId ='dive_42', vmin=0, vmax=1, 
    title=None, s=40)

stats_df = plot_detection_vs_range(h5_path=h5_path,
                RLdata=corrected_data,
                threshold_db=80,
                bin_width_km= .1)

stats_dict = plot_detection_by_bearing( 
                h5_path= h5_path,
                RLdata = corrected_data,
                threshold_db=80,
                diveId ='dive_42')

#%% Should we model Pdet as a function of RL?


import pandas as pd

rLlOC = 'X:\\Kaitlin_Palmer\\BotSensitivityCSVs_WHICEAS_clip\\BotSensitivityCSVs\\silt\\PeakToPeak_dive_167_GliderDepth_500m_0_29khz_long.csv'
rl_df = pd.read_csv(rLlOC)


h5_path = 'X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\Spacious_CalCurses_silt_PCHIP_12kHz_20km_500m_BotSensitivity.h5'



# rl_long is a pandas DataFrame with columns: lat, lon, depth_m, RL, drifterlat, drifterlon, ...
fig, ax = plot_peak2peak_isosurfaces_long(
        h5_path,
        rl_long=rl_df,
        diveId="dive_167",
        iso_levels=(135),
        xy_res=200,
        z_mode="data"   # or "h5" if you want to force the h5 depth grid
    )





print("normalized:", list(rl_df.columns))


detThreshs = [110,115,120]

for thresh in detThreshs:
    # Plot the new iso-surface
    # rl_long is a pandas DataFrame with columns: lat, lon, depth_m, RL, drifterlat, drifterlon, ...


    plot_peak2peak_isosurfaces_long(
                    h5_path, rl_df, diveId ='dive_167',
                    iso_levels= [thresh],
                    xy_res=200,
                    seabed_color='0.6',
                    elev=40, azim=-90)
     
    # plot_detection_probability(h5_path,
    #     RLdata, thresh,
    #     cmap='viridis',
    #     diveId ='dive_42', 
    #     vmin=0, vmax=1, 
    #     title=None, s=40)

    # stats_df = plot_detection_vs_range(h5_path=h5_path,
    #                 RLdata=RLdata,
    #                 threshold_db=thresh,
    #                 bin_width_km= .1)
    # fit_and_plot_hazard_rate_by_location(RLdata= RLdata,
    #                                          h5_path= h5_path, 
    #                                          diveId=  "dive_42",
    #                                          threshold_db = thresh) 

    
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




