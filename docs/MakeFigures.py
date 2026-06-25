# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 21:30:37 2025

@author: kaity
"""



# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:33:29 2026

@author: pam_user
"""

# -*- coding: utf-8 -*-
"""
Two-panel figure for sg680 California Current Survey

LEFT PANEL
-----------
Sound speed profiles (SSPs)
- all descending dives shown in gray
- selected example dives highlighted

RIGHT PANEL
------------
Bathymetry map
- contour bathymetry
- shallow water (<10 m) shown as land
- glider track overlaid
- selected dives marked with stars

Author: Kaity + cleaned/restructured
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patheffects as pe
import xarray as xr


# =============================================================================
# USER INPUTS
# =============================================================================

# --- CTD data ---------------------------------------------------------------
ctd_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_CTD.csv"
)

# --- Track / endpoint data --------------------------------------------------
track_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv"
)

# --- Bathymetry -------------------------------------------------------------
bathy_nc = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\bathymetry\GEBCO_28_Jul_2025_937903cf24aa"
    r"\gebco_2024_n44.6_s40.2_w-126.3_e-124.0.nc"
)

# Selected dive indices
selected_indices = [13, 5, 21]

# Highlight colors
highlight_colors = ['tab:blue', 'tab:orange', 'tab:green']


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading CTD data...")
driftCTD = pd.read_csv(ctd_csv)

print("Loading track data...")
driftEnds = pd.read_csv(track_csv)

print("Loading bathymetry...")
ds = xr.open_dataset(bathy_nc)

lon = ds['lon'].values
lat = ds['lat'].values
elevation = ds['elevation'].values

# Flatten for triangulation plotting
bathymetry_df = pd.DataFrame({
    'depth': elevation.flatten(),
    'lat': np.repeat(lat, len(lon)),
    'lon': np.tile(lon, len(lat))
})


# =============================================================================
# DETERMINE ASCENDING / DESCENDING
# =============================================================================

depth_diff = np.diff(driftCTD['Depth_m'], prepend=np.nan)

driftCTD['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')

# Fix first row
if depth_diff[1] > 0:
    driftCTD.at[0, 'Direction'] = 'dec'
else:
    driftCTD.at[0, 'Direction'] = 'asc'

# Dive IDs
driftCTD['DiveID'] = (
    driftCTD['DiveNumber'].astype(str)
    + '_'
    + driftCTD['Direction']
)


# =============================================================================
# SSP INTERPOLATION FUNCTION
# =============================================================================

def interpolate_sound_speed(dive_data, max_depth=1000):

    dive_data = dive_data.sort_values('Depth_m').copy()

    dive_data.dropna(
        subset=['Depth_m', 'SoundSpeed_m_s'],
        inplace=True
    )

    if len(dive_data) < 10:
        return None

    depth_range = np.arange(0, max_depth)

    interp_ss = np.interp(
        depth_range,
        dive_data['Depth_m'],
        dive_data['SoundSpeed_m_s']
    )

    return pd.DataFrame({
        'Depth_m': depth_range,
        'SoundSpeed_m_s': interp_ss
    })


# =============================================================================
# BUILD INTERPOLATED SSP DATASET
# =============================================================================

print("Interpolating SSPs...")

all_interpolated = []

# Use descending dives only
dec_dives = driftCTD[driftCTD['Direction'] == 'dec']

for dive_id, group in dec_dives.groupby('DiveID'):

    # Skip tiny profiles
    if len(group) < 200:
        continue

    interp_df = interpolate_sound_speed(group, max_depth=1000)

    if interp_df is None:
        continue

    interp_df['DiveID'] = dive_id

    all_interpolated.append(interp_df)

all_interpolated = pd.concat(
    all_interpolated,
    ignore_index=True
)


# =============================================================================
# SELECTED DIVES
# =============================================================================

selected_dive_numbers = (
    driftEnds.iloc[selected_indices]['closestDive']
    .astype(int)
    .tolist()
)

print("Selected dives:", selected_dive_numbers)

selected_profiles = {}

for dive_num in selected_dive_numbers:

    dive_id = f"{dive_num}_dec"

    prof = all_interpolated[
        all_interpolated['DiveID'] == dive_id
    ].copy()

    selected_profiles[dive_num] = prof


# =============================================================================
# FIGURE SETUP
# =============================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

fig = plt.figure(figsize=(14, 7))

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.15, 1.25],
    wspace=0.22
)

ax_ssp = fig.add_subplot(gs[0, 0])
ax_map = fig.add_subplot(gs[0, 1])


# =============================================================================
# LEFT PANEL — SSPS
# =============================================================================

print("Plotting SSP panel...")

# All dives in gray
for dive_id, group in all_interpolated.groupby('DiveID'):

    ax_ssp.plot(
        group['SoundSpeed_m_s'],
        group['Depth_m'],
        color='0.75',
        linewidth=0.7,
        alpha=0.35,
        zorder=1
    )

# Highlight selected dives
for color, dive_num in zip(
    highlight_colors,
    selected_dive_numbers
):

    prof = selected_profiles[dive_num]

    ax_ssp.plot(
        prof['SoundSpeed_m_s'],
        prof['Depth_m'],
        color=color,
        linewidth=2.8,
        label=f'Dive {dive_num}',
        zorder=10
    )

# Formatting
ax_ssp.invert_yaxis()

ax_ssp.set_xlim(1478, 1508)

ax_ssp.set_xlabel('Sound Speed (m s$^{-1}$)')
ax_ssp.set_ylabel('Depth (m)')

ax_ssp.set_title(
    'Example Sound Speed Profiles\n'
    'California Current Survey (sg680)',
    fontsize=14,
    fontweight='bold'
)

ax_ssp.grid(alpha=0.25)

ax_ssp.legend(
    frameon=True,
    fontsize=10,
    loc='lower right'
)


# =============================================================================
# RIGHT PANEL — BATHYMETRY MAP
# =============================================================================

print("Plotting bathymetry panel...")

# Triangulation
triang = tri.Triangulation(
    bathymetry_df['lon'],
    bathymetry_df['lat']
)

# -------------------------------------------------------------------------
# Filled bathymetry contours
# -------------------------------------------------------------------------

contour_levels = np.arange(-4000, 1, 250)

cf = ax_map.tricontourf(
    triang,
    bathymetry_df['depth'],
    levels=contour_levels,
    cmap='Blues_r',
    extend='both',
    zorder=1
)

# Contour lines
ax_map.tricontour(
    triang,
    bathymetry_df['depth'],
    levels=np.arange(-4000, 0, 500),
    colors='0.4',
    linewidths=0.45,
    alpha=0.4,
    zorder=2
)

# -------------------------------------------------------------------------
# LAND MASK (<10 m depth)
# -------------------------------------------------------------------------

land_vals = np.where(
    bathymetry_df['depth'] > -10,
    1,
    0
)

ax_map.tricontourf(
    triang,
    land_vals,
    levels=[0.5, 1.5],
    colors=['saddlebrown'],
    zorder=20
)

# -------------------------------------------------------------------------
# GLIDER TRACK
# -------------------------------------------------------------------------

track_lons = driftEnds['lon'].values
track_lats = driftEnds['lat'].values

# White underlay for visibility
ax_map.plot(
    track_lons[5:],
    track_lats[5:],
    color='white',
    linewidth=4.5,
    zorder=30
)

# Black track line
ax_map.plot(
    track_lons[5:],
    track_lats[5:],
    color='k',
    linewidth=2,
    zorder=31,
    label='Glider Track'
)

# -------------------------------------------------------------------------
# SELECTED DIVES
# -------------------------------------------------------------------------

selected_lons = track_lons[selected_indices]
selected_lats = track_lats[selected_indices]

ax_map.scatter(
    selected_lons,
    selected_lats,
    marker='*',
    s=260,
    facecolor='yellow',
    edgecolor='k',
    linewidth=1.3,
    zorder=40,
    label='Example Dives'
)

# Labels
for x, y, dive_num in zip(
    selected_lons,
    selected_lats,
    selected_dive_numbers
):

    txt = ax_map.text(
        x,
        y,
        str(dive_num),
        fontsize=10,
        fontweight='bold',
        color='white',
        ha='left',
        va='bottom',
        zorder=50
    )

    txt.set_path_effects([
        pe.withStroke(
            linewidth=3,
            foreground='black'
        )
    ])


# =============================================================================
# MAP FORMATTING
# =============================================================================

ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')

ax_map.set_title(
    'California Current Survey Track (sg680)',
    fontsize=14,
    fontweight='bold'
)

ax_map.set_aspect('equal', adjustable='box')

ax_map.grid(alpha=0.15)

# Zoom to track region
pad = 0.15

ax_map.set_xlim(
    np.nanmin(track_lons) - pad,
    np.nanmax(track_lons) + pad
)

ax_map.set_ylim(
    np.nanmin(track_lats) - pad,
    np.nanmax(track_lats) + pad
)

# Colorbar
cbar = fig.colorbar(
    cf,
    ax=ax_map,
    shrink=0.88,
    pad=0.02
)

cbar.set_label('Bathymetric Depth (m)')

ax_map.legend(
    loc='lower left',
    frameon=True
)


# =============================================================================
# PANEL LABELS
# =============================================================================

ax_ssp.text(
    0.02,
    0.98,
    '(a)',
    transform=ax_ssp.transAxes,
    fontsize=14,
    fontweight='bold',
    va='top'
)

ax_map.text(
    0.02,
    0.98,
    '(b)',
    transform=ax_map.transAxes,
    fontsize=14,
    fontweight='bold',
    va='top'
)


# =============================================================================
# FINALIZE
# =============================================================================

plt.tight_layout()

# Optional save
# plt.savefig(
#     'sg680_ssp_bathymetry_figure.png',
#     dpi=300,
#     bbox_inches='tight'
# )

plt.show()

print("Done.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load a drift
ctd_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_CTD.csv"
)
driftCTD = pd.read_csv(ctd_csv)

# Determine if the glider is ascending or descending
depth_diff = np.diff(driftCTD['Depth_m'], prepend=np.nan)

# Define 'asc' for ascending and 'dec' for descending
driftCTD['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')

# Correcting the first entry if needed
if depth_diff[1] > 0:
    driftCTD.at[0, 'Direction'] = 'dec'
else:
    driftCTD.at[0, 'Direction'] = 'asc'



# Define 'asc' for ascending and 'dec' for descending
driftCTD['DiveID'] = driftCTD['DiveNumber'].astype(str) + '_' + driftCTD['Direction']


# Giggle test the dives
# Choose a specific dive number to plot
dive_number = "37"  # You can change this to any dive number you want to inspect

# Filter the DataFrame for the chosen dive and both ascending and descending
dive_data = driftCTD[driftCTD['DiveNumber'].astype(str) == dive_number]

# Plotting
plt.figure(figsize=(3, 10))
for direction, color in [('asc', 'blue'), ('dec', 'red')]:
    segment_data = dive_data[dive_data['DiveID'].str.contains(direction)]
    plt.plot(segment_data['SoundSpeed_m_s'], segment_data['Depth_m'], label=f'Dive {dive_number} {direction}', color=color, marker='o', linestyle='-')

plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
plt.xlabel('Sound Speed (m/s)')
plt.ylabel('Depth (m)')
plt.title(f'Sound Speed Profile for Dive {dive_number}')
plt.legend()
plt.show()


# Use splines to interpolate the dive data
from scipy.interpolate import UnivariateSpline


# Function to perform spline interpolation and plot
def plot_spline_interpolation(dive_data, dive_id, plot = False):
    # Sorting by depth might be necessary if not already sorted
    dive_data_sorted = dive_data.sort_values('Depth_m')

    # Drop the NA values
    dive_data_sorted.dropna(inplace = True, subset = ['SoundSpeed_m_s'])

    # Set up the spline with sorted data
    spline = UnivariateSpline(dive_data_sorted['Depth_m'], 
                              dive_data_sorted['SoundSpeed_m_s'])

    
    # Create an array of depths at 1m intervals
    depth_range = np.arange(dive_data_sorted['Depth_m'].min(), dive_data_sorted['Depth_m'].max())
    
    # Predict sound speed at these depths using the spline
    sound_speed_interp = spline(depth_range)

    if plot:
        # Plotting
        plt.figure(figsize=(3, 10))
        plt.plot(dive_data_sorted['SoundSpeed_m_s'],dive_data_sorted['Depth_m'],  'ro', label=f'Original Data ({dive_id})')
        plt.plot(sound_speed_interp, depth_range,  'b-', label=f'Interpolated Spline ({dive_id})')
        plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
        plt.xlabel('Depth (m)')
        plt.ylabel('Sound Speed (m/s)')
        plt.title(f'Sound Speed Profile for {dive_id}')
        plt.legend()
        plt.show()

    return depth_range, sound_speed_interp

# Example usage for a specific DiveID
dive_id = '1_asc'  # Replace this with any DiveID you want to analyze
selected_dive_data = driftCTD[driftCTD['DiveID'] == dive_id]
depths, interp_speeds = plot_spline_interpolation(selected_dive_data, dive_id)



# Function to interpolate sound speed for each meter of depth
def interpolate_sound_speed(dive_data, maxDepth, plot =False):
    # Sorting by depth might be necessary if not already sorted
    dive_data_sorted = dive_data.sort_values('Depth_m')
    
    # Drop the NA values
    dive_data_sorted.dropna(inplace = True, subset = ['SoundSpeed_m_s'])

    # Create an array of depths at 1m intervals
    depth_range = np.arange(0, maxDepth)
    
    # Predict sound speed at these depths using the spline
    sound_speed_interp = np.interp(depth_range, dive_data_sorted['Depth_m'],
                                   dive_data_sorted['SoundSpeed_m_s'])
    if plot:
        # Plotting
        plt.figure(figsize=(3, 10))
        plt.plot(dive_data_sorted['SoundSpeed_m_s'], 
                 dive_data_sorted['Depth_m'],  'ro', 
                 label=f'Original Data ({dive_id})')
        plt.plot(sound_speed_interp, depth_range,  'b-', 
                 label=f'Interpolated Spline ({dive_id})')
        plt.gca().invert_yaxis()  # Inverts the y-axis so depth increases downwards
        plt.xlabel('Depth (m)')
        plt.ylabel('Sound Speed (m/s)')
        plt.title(f'Sound Speed Profile for {dive_id}')
        plt.xlim(1480, 1540)
        plt.legend()
        plt.show()
        
    return pd.DataFrame({'Depth_m': depth_range, 'SoundSpeed_m_s': sound_speed_interp})


# Collect all interpolated data
all_interpolated_data = pd.DataFrame()

# Use just the descending dives
decDives = driftCTD[driftCTD['Direction']=='dec']

for dive_id, group in decDives.groupby('DiveID'):
    if len(group)>200:
        interpolated_data = interpolate_sound_speed(group,1000, plot =False)
        interpolated_data['DiveID']=dive_id
        all_interpolated_data = pd.concat([all_interpolated_data, 
                                       interpolated_data], ignore_index=True)


# Calculate percentiles for each depth across all dives
percentiles_by_depth = all_interpolated_data.groupby('Depth_m')['SoundSpeed_m_s'].quantile([0.05, 0.95]).unstack()
percentiles_by_depth.columns = ['5th_percentile', '95th_percentile']



# plot these results
plt.figure(figsize=(3, 10))
for dive_id, group in all_interpolated_data.groupby('DiveID'):
    plt.plot( group['SoundSpeed_m_s'],group['Depth_m'], 
             color='lightgray', label=f'{dive_id}' if dive_id == list(all_interpolated_data['DiveID'].unique())[0] else "")


plt.plot(percentiles_by_depth['5th_percentile'], percentiles_by_depth.index,  label='5th Percentile')
plt.plot(percentiles_by_depth['95th_percentile'], percentiles_by_depth.index, label='95th Percentile')
plt.gca().invert_yaxis()  # Depth increases downwards
plt.ylabel('Depth (m)')
plt.xlabel('Sound Speed (m/s)')
plt.title('5th and 95th Percentiles of Sound Speed by Depth Across All Dives')
plt.legend()
plt.show()

percentiles_by_depth['Diff'] = percentiles_by_depth['95th_percentile']-percentiles_by_depth['5th_percentile']

np.max(percentiles_by_depth['Diff'])
np.min(percentiles_by_depth['Diff'])
np.median(percentiles_by_depth['Diff'])
np.mean(percentiles_by_depth['Diff'])




import pandas as pd
from PlottingDefs import plot_peak2peak_isosurfaces_long, plot_detection_probability_long


# File paths
h5_path = 'X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\Spacious_CalCurses_Silt_PCHIP_12kHz_20km_50m_BotSensitivity.h5'
rLlOC = 'X:\\Kaitlin_Palmer\\BotSensitivityCSVs_WHICEAS_clip\\BotSensitivityCSVs\\silt\\PeakToPeak_dive_167_GliderDepth_50m_0_29khz_long.csv'

# Load long-format RL table (lat, lon, depth_m, RL, ...)
rl_df = pd.read_csv(rLlOC)

# Plot the new iso-surface (long CSV version)
plot_peak2peak_isosurfaces_long(
                h5_path, rl_df,
                diveId='dive_167',
                iso_levels=(135,),
                xy_res=200,
                source_depth_m =150,
                elev=25, azim=-100)


# Plot detection probability from the same long CSV
plot_detection_probability_long(
                h5_path, rl_df,
                threshold_db=135,
                diveId='dive_167',
                cmap='viridis',
                vmin=0, vmax=1,
                s=40)

driftCTD = pd.read_csv(r"\modelling\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv")

