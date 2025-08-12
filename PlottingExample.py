# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 04:07:27 2025

@author: pam_user

This is to be used with PlottingDefs.py this should be an example, plottingDefs
should be the begining of a package
"""

from scipy.io import wavfile
import os
from PlottingDefs import CreateOutputCSVs
from PlottingDefs import scaleP2P, CreateOutputCSVs, alphaAdjustment
from PlottingDefs import apply_alpha_correction, plot_peak2peak_isosurfaces, plot_detection_probability
from PlottingDefs import plot_detection_vs_range, plot_detection_by_bearing


import numpy as np
import matplotlib.pyplot as plt

#%% Creat the CSV's of the arrival RLs
# File locations for the HDF5 from the bellhop models, audio file to convolve
# and where to save th exported csvs
h5_path = 'C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\Spacious_CalCurses_Sensitivity_PCHIP_35khz_20km_500m.h5'
wav_path = "C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
out_path = "X:\Kaitlin_Palmer\CalCurCEAS_propagation_csvs"


# --- Signal Setup ---
samplerate, audiodata = wavfile.read(wav_path)
t_start, t_end, chan = 32.58, 32.60, 4
segment = audiodata[int(round(t_start * samplerate)):int(round(t_end * samplerate)), chan]
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
segment = scaleP2P(segment, outP2P= 220)

plt.plot(tt*1000, segment)

#%% Use convolution of the signal of interest to calculate the peak to peak
# RL at each of the sensor locations

# Calculate the received arrays and export to csv- this takes a while
CreateOutputCSVs(h5_path, segment, samplerate, out_path, nWorkers=56)


#%% Load the RL grid in the previous section and make plots 
RLdata = np.genfromtxt('X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_csvs\\PeakToPeak_dive_42_GliderDepth_800m.csv', delimiter=',')
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

# Plot the new iso-surface
plot_peak2peak_isosurfaces(
                h5_path, corrected_data, diveId ='dive_42',
                iso_levels=(80,),
                xy_res=200,
                seabed_color='0.6',
                elev=25, azim=-90)
 
plot_detection_probability(h5_path,
    RLdata, 80,
    cmap='viridis',diveId ='dive_42', vmin=0, vmax=1, 
    title=None, s=40)

stats_df = plot_detection_vs_range(h5_path=h5_path,
                RLdata=RLdata,
                threshold_db=5,
                bin_width_km= .1)

stats_dict = plot_detection_by_bearing( 
                h5_path= h5_path,
                RLdata = RLdata,
                threshold_db=20,
                diveId ='dive_42')

#%% Should we model Pdet as a function of RL?

detThreshs = [20,40,80]

for thresh in detThreshs:
    # Plot the new iso-surface
    plot_peak2peak_isosurfaces(
                    h5_path, corrected_data, diveId ='dive_42',
                    iso_levels=(thresh,),
                    xy_res=200,
                    seabed_color='0.6',
                    elev=25, azim=-90)
     
    plot_detection_probability(h5_path,
        RLdata, thresh,
        cmap='viridis',diveId ='dive_42', vmin=0, vmax=1, 
        title=None, s=40)

    stats_df = plot_detection_vs_range(h5_path=h5_path,
                    RLdata=corrected_data,
                    threshold_db=thresh,
                    bin_width_km= .1)
    
#%% Restrict to sperm whale depths


    my_data = genfromtxt('PeakToPeakDive_dive_24_dec.csv', delimiter=',')

# Get the depth values from the HDF5
import h5py

# Now get the depths
hf = h5py.File(h5_path, 'r')
diveId ='dive_42'
dive_grp = hf[f'drift_01/{diveId}/frequency_35000']
run_ids = list(dive_grp['arrivals'].keys())
depth_grid = np.array(dive_grp['depth'])

# Say 500m to 1200m depth that's column 5 on
np.nanmax(corrected_data)
corrected_data[:, 1:4] = -500
corrected_data[:, 13:27] = -500


# Plot the new iso-surface
plot_peak2peak_isosurfaces(
                h5_path, corrected_data, diveId ='dive_42',
                iso_levels=(80,),
                xy_res=200,
                seabed_color='0.6',
                elev=25, azim=-90)

plot_detection_probability(h5_path,
    corrected_data, thresh,
    cmap='viridis',diveId ='dive_42', vmin=0, vmax=1, 
    title=None, s=40)









