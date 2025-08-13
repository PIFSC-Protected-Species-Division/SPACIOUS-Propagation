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

h5_path = 'C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\Spacious_Hawaii_diveDepth_ArrArray_PCHIP_35khz_20km - Copy.h5'
wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
out_path = "C:\\Users\\kaity\\Desktop\\TestCoherentBellhopCSVS"




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
click_waveform = scaleP2P(segment, outP2P= 220)

plt.plot(tt*1000, segment)

#%% Use convolution of the signal of interest to calculate the peak to peak
# RL at each of the sensor locations

# Calculate the received arrays and export to csv- this takes a while

CreateOutputCSVs(
    h5_path=h5_path,
    segment=click_waveform,          # your recorded click
    samplerate=samplerate,
    out_path=out_path,
    coherent=True,                   # enable wideband coherent p2p
    nWorkers=4,
    fmin_hz=20000.0,
    fmax_hz=90000.0,
    df_hz=200.0,
    c_eff_m_s=1480.0,
    f_ref_hz=35000.0,                # your Bellhop run frequency
    arrivals_include_absorption=True # typical with arlpy/bellhop
)



#%% Pipeline Examples
from H5ArrivalsBridge import list_points, load_point_by_index, load_point_near
from PlottingDefs import (
    build_freq_grid, fig_source_and_grid, fig_absorption,
    fig_transfer_and_received, fig_compare_legacy_vs_coherent
)

h5 = h5_path
drift_id='drift_01'
dive_id = "dive_24_dec"      # whatever you have
fs = samplerate                   # your click sample rate
click = click_waveform     # np.ndarray

RLdata = np.genfromtxt('C:\\Users\\kaity\\Desktop\\TestCoherentBellhopCSVS\\PeakToPeak_dive_24_dec_GliderDepth_100m.csv', delimiter=',')


# 1) pick a point by index
arrivals =  load_point_by_index(h5_path, 
                        dive_id, 
                        pt_index ='pt_00001',
                        drift_id=drift_id, 
                        freq_khz='frequency_35000',
                        c_eff_m_s=1480.0, 
                        estimate_pathlen_if_missing=True)

# 3) build frequency grid and make figures
freqs = build_freq_grid(2000, 25000, 200, fs=fs)
fig_source_and_grid(click, fs, freqs)
fig_absorption(freqs, f_ref_hz=35000.0)

fig3, r = fig_transfer_and_received(click, fs, arrivals, freqs,
                                    f_ref_hz=35000.0,
                                    c_eff_m_s=1480.0,
                                    arrivals_include_absorption=True)

fig4, stats = fig_compare_legacy_vs_coherent(click, fs, arrivals, freqs,
                                             f_ref_hz=35000.0,
                                             c_eff_m_s=1480.0,
                                             arrivals_include_absorption=True)

thresh =30

plot_peak2peak_isosurfaces(
                h5_path, RLdata, diveId ='dive_24_dec',
                iso_levels=(thresh,),
                xy_res=200,
                seabed_color='0.6',
                elev=25, azim=-90)

plot_detection_probability(h5_path,
    RLdata, thresh,
    cmap='viridis',diveId ='dive_24_dec', vmin=0, vmax=1, 
    title=None, s=40)
#%% Pipeline Plots
from PlottingDefs import build_freq_grid, fig_pipeline_overview
from H5ArrivalsBridge import load_point_by_index

h5 = "Spacious_CalCurses_Sensitivity_PCHIP_35khz_20km.h5"
dive_id = "dive_24_dec"
arrivals =  load_point_by_index(h5_path, 
                        dive_id, 
                        pt_index ='pt_00001',
                        drift_id=drift_id, 
                        freq_khz='frequency_35000',
                        c_eff_m_s=1480.0, 
                        estimate_pathlen_if_missing=True)
# your click + fs
click = click_waveform
fs = samplerate

freqs = build_freq_grid(20000, 90000, 200, fs=fs)  # will clamp visually to Nyquist if needed
fig, artifacts = fig_pipeline_overview(click, fs, arrivals, freqs,
                                       f_ref_hz=35000.0,
                                       arrivals_include_absorption=True,
                                       title="Coherent Pipeline Overview",
                                       save_path=None)  # or "pipeline.svg"


