# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:33:18 2026

@author: pam_user
"""


"""
Two-panel figure for sg680 California Current Survey

LEFT PANEL
-----------
Sound speed profiles (SSPs)
- all descending dives shown in light gray
- selected dives 12, 86, and 167 shown in black
- selected dives use different line styles
- 5th and 95th percentiles shown in blue

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
# GLOBAL FONT / STYLE SETTINGS
# =============================================================================

TITLE_FONT_SIZE = 14
PANEL_LABEL_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 10

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": TITLE_FONT_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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


# Selected dives
selected_dive_numbers = [42, 86, 167]


# Indices in driftEnds corresponding to the selected dives on the map
selected_indices = [5, 13, 21]


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

depth_diff = np.diff(
    driftCTD['Depth_m'],
    prepend=np.nan
)

driftCTD['Direction'] = np.where(
    depth_diff > 0,
    'dec',
    'asc'
)


# Correct first entry

if depth_diff[1] > 0:
    driftCTD.at[0, 'Direction'] = 'dec'
else:
    driftCTD.at[0, 'Direction'] = 'asc'


# =============================================================================
# CREATE DIVE IDs
# =============================================================================

driftCTD['DiveID'] = (
    driftCTD['DiveNumber'].astype(str)
    + '_'
    + driftCTD['Direction']
)


# =============================================================================
# SSP INTERPOLATION FUNCTION
# =============================================================================

def interpolate_sound_speed(dive_data, max_depth=1000):

    """
    Interpolate sound speed onto a 1 m depth grid.
    """

    dive_data = dive_data.sort_values(
        'Depth_m'
    ).copy()

    dive_data.dropna(
        subset=[
            'Depth_m',
            'SoundSpeed_m_s'
        ],
        inplace=True
    )

    if len(dive_data) < 10:
        return None

    depth_range = np.arange(
        0,
        max_depth
    )

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

dec_dives = driftCTD[
    driftCTD['Direction'] == 'dec'
]


for dive_id, group in dec_dives.groupby('DiveID'):

    # Skip very small profiles

    if len(group) < 200:
        continue

    interp_df = interpolate_sound_speed(
        group,
        max_depth=1000
    )

    if interp_df is None:
        continue

    interp_df['DiveID'] = dive_id

    all_interpolated.append(
        interp_df
    )


# Make sure there are usable dives

if len(all_interpolated) == 0:

    raise ValueError(
        "No descending dives met the minimum data requirement."
    )


# Combine all dives

all_interpolated = pd.concat(
    all_interpolated,
    ignore_index=True
)


# =============================================================================
# CALCULATE PERCENTILES
# =============================================================================

print("Calculating SSP percentiles...")

percentiles_by_depth = (
    all_interpolated
    .groupby('Depth_m')['SoundSpeed_m_s']
    .quantile([0.05, 0.95])
    .unstack()
)

percentiles_by_depth.columns = [
    '5th_percentile',
    '95th_percentile'
]


# =============================================================================
# CALCULATE PERCENTILE WIDTH
# =============================================================================

percentiles_by_depth['Diff'] = (
    percentiles_by_depth['95th_percentile']
    -
    percentiles_by_depth['5th_percentile']
)


print(
    "Maximum 5th-95th percentile difference:",
    np.max(percentiles_by_depth['Diff'])
)

print(
    "Minimum 5th-95th percentile difference:",
    np.min(percentiles_by_depth['Diff'])
)

print(
    "Median 5th-95th percentile difference:",
    np.median(percentiles_by_depth['Diff'])
)

print(
    "Mean 5th-95th percentile difference:",
    np.mean(percentiles_by_depth['Diff'])
)


# =============================================================================
# CHECK THAT SELECTED DIVES EXIST
# =============================================================================

available_dive_ids = set(
    all_interpolated['DiveID']
)


for dive_num in selected_dive_numbers:

    dive_id = f"{dive_num}_dec"

    if dive_id not in available_dive_ids:

        print(
            f"WARNING: Selected dive {dive_num} "
            f"was not found in the interpolated descending dives."
        )


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(
    figsize=(14, 7)
)


gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.15, 1.25],
    wspace=0.22
)


ax_ssp = fig.add_subplot(
    gs[0, 0]
)


ax_map = fig.add_subplot(
    gs[0, 1]
)


# =============================================================================
# LEFT PANEL — SOUND SPEED PROFILES
# =============================================================================

print("Plotting SSP panel...")


# -------------------------------------------------------------------------
# ALL DESCENDING DIVES — LIGHT GRAY
# -------------------------------------------------------------------------

for dive_id, group in all_interpolated.groupby('DiveID'):

    ax_ssp.plot(
        group['SoundSpeed_m_s'],
        group['Depth_m'],
        color='lightgray',
        linewidth=0.8,
        alpha=0.65,
        zorder=1
    )


# -------------------------------------------------------------------------
# SELECTED DIVES — BLACK WITH DIFFERENT LINE STYLES
# -------------------------------------------------------------------------

selected_line_styles = {
    42: '-',
    86: '--',
    167: '-.'
}


for dive_num in selected_dive_numbers:

    dive_id = f"{dive_num}_dec"

    prof = all_interpolated[
        all_interpolated['DiveID'] == dive_id
    ].copy()

    # Only plot if the dive exists

    if len(prof) == 0:
        continue

    ax_ssp.plot(
        prof['SoundSpeed_m_s'],
        prof['Depth_m'],
        color='black',
        linewidth=2.0,
        linestyle=selected_line_styles[dive_num],
        label=f'Dive {dive_num}',
        zorder=5
    )


# -------------------------------------------------------------------------
# 5TH PERCENTILE — BLUE
# -------------------------------------------------------------------------

ax_ssp.plot(
    percentiles_by_depth['5th_percentile'],
    percentiles_by_depth.index,
    color='blue',
    linewidth=2.5,
    linestyle='-',
    label='5th percentile',
    zorder=10
)


# -------------------------------------------------------------------------
# 95TH PERCENTILE — BLUE
# -------------------------------------------------------------------------

ax_ssp.plot(
    percentiles_by_depth['95th_percentile'],
    percentiles_by_depth.index,
    color='blue',
    linewidth=2.5,
    linestyle='-',
    label='95th percentile',
    zorder=10
)


# -------------------------------------------------------------------------
# SSP FORMATTING
# -------------------------------------------------------------------------

ax_ssp.invert_yaxis()


ax_ssp.set_xlim(
    1470,
    1508
)


ax_ssp.set_xlabel(
    'Sound Speed (m s$^{-1}$)'
)


ax_ssp.set_ylabel(
    'Depth (m)'
)


ax_ssp.set_title(
    'Sound Speed Profiles'
)


ax_ssp.grid(
    alpha=0.25
)


ax_ssp.legend(
    frameon=True,
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
    edgecolor='yellow',
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
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight='bold',
        # color='white',
        ha='left',
        va='bottom',
        zorder=50
    )




# =============================================================================
# MAP FORMATTING
# =============================================================================

ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')

ax_map.set_title(
    'Glider Track'
)

ax_map.set_aspect('equal', adjustable='box')

ax_map.grid(alpha=0.15)

# Zoom to track region
pad = 0.1

ax_map.set_xlim(
    np.nanmin(track_lons) - pad,
    np.nanmax(track_lons) + pad
)

ax_map.set_ylim(
    np.nanmin(track_lats) - pad,
    np.nanmax(track_lats) -.2
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
    fontsize=PANEL_LABEL_FONT_SIZE,
    fontweight='bold',
    va='top'
)

ax_map.text(
    0.02,
    0.98,
    '(b)',
    transform=ax_map.transAxes,
    fontsize=PANEL_LABEL_FONT_SIZE,
    fontweight='bold',
    va='top'
)


# =============================================================================
# FINALIZE
# =============================================================================

plt.tight_layout()

#Optional save
plt.savefig(
    'C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation\\Figure1.pdf',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Done.")




# =============================================================================
# Figure 2
# =============================================================================

 # Change to SPATIOUS-propagation to run 
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
                elev=25, azim=-100,
                title = r'Iso-surface 35 dB$_{pp}$')


# Plot detection probability from the same long CSV
plot_detection_probability_long(
                h5_path, rl_df,
                threshold_db=135,
                diveId='dive_167',
                cmap='viridis',
                vmin=0, vmax=1,
                s=40,                 
                title = r'Detection Proportion RL >135 dB$_{pp}$')

import pandas as pd
from PlottingDefs import plot_isosurface_and_detection_combined

h5_path = 'X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s\\Spacious_CalCurses_Silt_PCHIP_12kHz_20km_50m_BotSensitivity.h5'
rLlOC = 'X:\\Kaitlin_Palmer\\BotSensitivityCSVs_WHICEAS_clip\\BotSensitivityCSVs\\silt\\PeakToPeak_dive_167_GliderDepth_50m_0_29khz_long.csv'

rl_df = pd.read_csv(rLlOC)

fig, (ax_iso, ax_det) = plot_isosurface_and_detection_combined(
                h5_path, rl_df,
                threshold_db=135,
                diveId='dive_167',
                iso_levels=(135,),
                xy_res=200,
                source_depth_m=150,
                elev=25, azim=-100,
                cmap_det='viridis',
                vmin=0, vmax=1,
                s=40,
                isoSurfTitle =r'Iso-surface 35 dB$_{pp}$',
                pdetTitle= r'Detection Proportion RL >135 dB$_{pp}$',
                save_path='C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation\\Figure 2 Propagation.tiff',
                save_dpi=300)




driftCTD = pd.read_csv(r"\modelling\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv")

