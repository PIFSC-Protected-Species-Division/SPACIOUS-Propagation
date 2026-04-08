# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 18:24:55 2026

@author: pam_user
"""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import pandas as pd
import xarray as xr
import matplotlib.tri as tri
import os, time

from PropagationDefs import haversine, ProcessPoolExecutor, _safe_worker, as_completed,save_dive_frequency
from concurrent.futures import ProcessPoolExecutor, as_completed
from PropagationDefs import haversine, _safe_worker, save_dive_frequency

if __name__ == '__main__':
    
    #%% Set up the enviornmental parameters
    propagationDepth = [50, 200, 350,500, 650, 800]
    freq_hz =12000
    
    # Bottom Characteristics
    Sediments = ['silt', 'gravel', 'basalt']
    bottom_soundspeed =[1575,1800,5250]
    bottom_density =[1700,2000,2700]
    bottom_absorption= [1,0.6,0.1]
    
    #%% Load the bottm data and dive locations
    drift_csv = r"C:\\Users\\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\modelling\\sg680_CalCurCEAS_Sep2024_CTD.csv"
    gebco_nc  = r"C:\\Users\\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_28_Jul_2025_937903cf24aa\\gebco_2024_n44.6_s40.2_w-126.3_e-124.0.nc"
    driftEnds = r'C:\\Users\\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\modelling\\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv'
    out_h5    = "X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s"
    
    # Number of CPUs to dedicate to the process
    nWorkers = max(1, mp.cpu_count() -1)
    
    # ------------------------------------------------------------------- setup
    
    # Load the CDT data and the drift ends
    driftCTD = pd.read_csv(drift_csv)
    driftEnds= pd.read_csv(driftEnds)
    
    # Extract the bathymetry from the nc file
    ds = xr.open_dataset(gebco_nc)
    bathymetry_df = pd.DataFrame({
        'depth': ds['elevation'].values.flatten(),
        'lat':   np.repeat(ds['lat'].values, len(ds['lon'])),
        'lon':   np.tile(  ds['lon'].values, len(ds['lat']))})
    
    
    # Define DiveID (part of larger code that discriminates between ascending and
    # descending)
    driftCTD['DiveID'] = driftCTD['DiveNumber'].astype(str)
    unique_ids = driftCTD['DiveID'].drop_duplicates().to_numpy()
    
    # Lat and lons of the drift ends
    lats = driftEnds['lat'].values
    lons = driftEnds['lon'].values
    
    # The dives we want
    ends = [13,5,21]
    end_dive_nums = driftEnds['closestDive'][ends]
    
    # Create a subset of the dives
    driftCTDsub = driftCTD[driftCTD['DiveNumber'].isin(end_dive_nums)]
    
    print("Transect-end indices:", ends)
    
    # Optional: plot to verify
    fig, ax = plt.subplots(figsize=(8,6))
    triang = tri.Triangulation(bathymetry_df['lon'], bathymetry_df['lat'])
    ax.tricontourf(triang, bathymetry_df['depth'], 
                    levels=100, cmap='viridis')
     
     # Plot stars for selected ends
    ax.scatter(
         lons[ends], lats[ends],
         marker='*', s=150, facecolor='yellow', edgecolor='k',
         zorder=10, label='Detected Ends'
     )
     
     # Add DiveID labels to the stars
    for x, y, dive_id in zip(lons[ends], lats[ends], driftCTDsub['DiveID'].unique()):
         ax.text(
             x, y, str(dive_id),
             fontsize=10, fontweight='bold',
             ha='left', va='bottom', color='white',
             path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")]
         )
     
     # Plot track
    ax.plot(lons, lats, '-k', zorder=5, label='Track')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(); plt.show()
        
    
    completed_dives= []
    #%% Run the loop calling the worker to do the propagation in parallel 
    # 2) loop throught the dives and run the propagation models and depths
    
    for p in range(len(Sediments)):
        
        
        # Bottom characteristics
        print(Sediments[p])
        sedimentName = Sediments[p]
        bot_ssp = bottom_soundspeed[p]
        bot_rho = bottom_density[p]
        bot_alpha = bottom_absorption[p]
        
        
        for depth in propagationDepth:
               
               for dive_id in driftCTDsub['DiveID'].unique():
                    if f"dive_{dive_id}" in completed_dives:
                        print(f"Skipping completed dive {dive_id}")
                        continue
                       
                   
                   
                    # 3) pull out the corresponding group “on the fly”
                    group = driftCTD[driftCTD['DiveID'] == dive_id]
                    
                    
                    # Determine if the glider is ascending or descending
                    depth_diff = np.diff(group['Depth_m'], prepend=np.nan)
                    group['Direction'] = np.where(depth_diff > 0, 'dec', 'asc')
                    if depth_diff[1] > 0:
                        group.at[0, 'Direction'] = 'dec'
                    else:
                        group.at[0, 'Direction'] = 'asc'
                    
                    print(dive_id)
                    
                    # Subset only to descending direction
                    group = group[group['Direction'] == 'dec'].reset_index(drop=True)
                    drifter_lat = group['Latitude'].iloc[0]
                    drifter_lon = group['Longitude'].iloc[0]
                    
                    # If the propagation depth is set to DrifterDepth then run propagation 
                    # at whatever depth that is
                    if propagationDepth== 'DrifterDepth':
                        drifter_depth = np.max([-group['Depth_m'].iloc[0], 100])
                    else:
                        drifter_depth = depth
                   
                    
                    # Create the SSP profile
                    profile = pd.DataFrame({
                        'depth': group['Depth_m'],
                        'ss':    group['SoundSpeed_m_s'] })
                    ()
                    profile.sort_values('depth', inplace=True)
                    profile.dropna(inplace=True)
                    profile.reset_index(drop=True, inplace=True)
                    profile.loc[0, 'depth'] = 0
                    
                    # Only use the dive if the profile depth is more than 200m
                    if np.max(profile['depth'])>depth:
                        
                        base_filename = 'Spacious_CalCursesV2_'+sedimentName+ '_PCHIP_'+str(int(freq_hz/1000))+'kHz_20km_'+str(depth)+'m_BotSensitivity.h5'
        
                        fullfileOut = os.path.join(out_h5, base_filename)
                        
                        # If the code crashes this code will pull the dives that have already been
                        # completed
                        #completed_dives = get_completed_dive_ids(fullfileOut)
                        
                        
                        results = {}
                        results[dive_id] = []
                        
                        bathymetry_df['distance_km'] = haversine(
                            drifter_lon, drifter_lat,
                            bathymetry_df['lon'], bathymetry_df['lat']
                        )
                        
                        # Pull out datapoints within 40km of the sensor and the water is deeper than 150 m
                        subset_df = bathymetry_df[
                            (bathymetry_df['distance_km'] <= 20) &
                            (bathymetry_df['depth'] < -150)]
                        
                        
                        # This is where we will get the propagation data from  
                        bathy_full = subset_df        # set once in main
                
                        # Downsample the datapoints by 1/20th
                        #subset_df = subset_df[subset_df.index % 20 != 0] 
                        subset_df.reset_index(drop=True, inplace=True)
                        
                    
                        total_rows = len(subset_df)
                        
                        print(f'Running dive Id {dive_id}  at {freq_hz} kHz')
                        max_depth = np.max(np.abs(subset_df['depth']))
                        last_ss = profile.iloc[-1]['ss']
                        
                        # Soundspeed profile, split the dive into ascending and descending
                        # Use only ascending for SSP calculations
                        
                        expanedProfile = pd.DataFrame(
                            {'depth': np.arange(profile.iloc[-1]['depth']+10, max_depth+50, step =50),
                                'ss': np.repeat(last_ss,
                                                len(np.arange(profile.iloc[-1]['depth']+10, max_depth+50, step =50)))})
                        
                        
                        profile = pd.concat([profile, expanedProfile])
                        profile['ss'] = np.abs(profile['ss'])
                        profile.sort_values('depth', inplace=True)
                        ssp = profile.apply(lambda row: [row['depth'], row['ss']], axis=1).tolist()
                        
                        
                        # Dictionary with keys 'start_lat', 'start_lon', and 'drifter_depth'.
                        metadata = {'start_lat': drifter_lat,
                                        'start_lon': drifter_lon,
                                        'drifter_depth': drifter_depth,
                                        'Sediment_Name': sedimentName,
                                        'bottom_soundspeed':bot_ssp,
                                        'bottom_density':bot_rho,
                                        'bottom_absorption':bot_alpha }
                        
    
                        
            
                        # Parallelize the Bellhop TL computations
                        tasks = [
                            (ii, subset_df, drifter_lat, drifter_lon, 
                             freq_hz, ssp, drifter_depth, bot_ssp, bot_rho,
                             bot_alpha)
                            for ii in np.arange(0, len(subset_df))]
                        
                        
                        t = time.time()
                        
                        
                        # Start the parallel processor
                        with ProcessPoolExecutor(max_workers=nWorkers) as pool:
                            futures = [pool.submit(_safe_worker, task) for task in tasks]
                            
                            for future in as_completed(futures):
                                status, ii, payload = future.result()
                        
                                if status == 'fail':
                                    print(f"❌  error at index {ii}: {payload}")
                                    continue
                        
                                _, tlosDb, arr, rx_depths = payload
                        
                                results[dive_id].append({
                                    'lat':  subset_df['lat'].iloc[ii],
                                    'lon':  subset_df['lon'].iloc[ii],
                                    'arr':  arr,
                                    'transmission_loss': tlosDb,
                                    'tl_depths': rx_depths
                                })
                        
                                print(f"Processed {ii} of {total_rows} points at {depth} m in {sedimentName}.")
            
                        
                                
                                  
                        save_dive_frequency(
                        h5_path      = fullfileOut,
                        drift_id     = "01",
                        dive_id      = dive_id,
                        freq_khz     = freq_hz,
                        metadata     = metadata,
                        grid_results = results[dive_id])
                        
                        elapsed = time.time() - t
                        print(f'Dive {dive_id} completed in {elapsed:.1f} s')
