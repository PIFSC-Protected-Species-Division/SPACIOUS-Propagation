# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 02:07:34 2025

@author: pam_user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:24:37 2025

@author: kaity

Code to run propagation at multiple depths and locations for sensitivity analysis


This is the most recent version includes all values in loop 2026-03-25
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # or 1, 8 … anything ≤ 24


###############################################################################
# 1) ---- move env-vars to the top (before NumPy) -----------------------------
import os, multiprocessing as mp
os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS":      "1",
    "OMP_NUM_THREADS":      "1",
})


import ssl

# Safely bypass loading corrupted Windows registry certificates
ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: None

import geopy
from geopy.distance import geodesic

print(f"geopy version {geopy.__version__} loaded successfully!")
###############################################################################
# 2) ---- real processes, coarser chunks, no chatty prints --------------------
from multiprocessing import Pool
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from geopy.distance import geodesic
from geopy.point import Point
import matplotlib.pyplot as plt
from geopy.point import Point
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
import arlpy.uwapm as pm
import arlpy.plot as arlplt
import matplotlib.tri as tri
from pyproj import Geod
import h5py
from multiprocessing.dummy import Pool as ThreadPool
import time
from multiprocessing import Pool




#---------------------------------------------------------------------------
# Create a Geod instance for vectorized geodesic computations.
geod = Geod(ellps='WGS84')
bathy_full = None        # set once in main
subset_df = None
subsetBathy= None



###############################################################################
# 2)  ––– original helper fns (unchanged) ------------------------------------
###############################################################################
# haversine, calculate_initial_compass_bearing, extract_bathymetry_from_subset
# extract_bathymetry_from_subset_vectorized, tl_incoherent_from_arrivals …
#    ↳ (copy the full originals here – omitted for brevity)   

# At global level
_subset_df = None



def get_completed_dive_ids(h5_path):
    """
    Returns a set of dive IDs already stored in the HDF5 file.
    """
    if not os.path.exists(h5_path):
        return set()

    with h5py.File(h5_path, "r") as hf:
        if "drift_01" not in hf:
            return set()
        drift_grp = hf["drift_01"]
        return set(drift_grp.keys())  # e.g. ['dive_001_asc', 'dive_002_dec']


def save_dive_frequency(h5_path, drift_id, dive_id, freq_khz,
                        metadata, grid_results, gzip_level=4):

    # ─── unchanged pre-amble (lat/lon/TL matrices) ───
    n_pts = len(grid_results)
    max_N = max(len(np.asarray(g["tl_depths"]).reshape(-1)) for g in grid_results)

    lat   = np.empty(n_pts, np.float32)
    lon   = np.empty(n_pts, np.float32)
    dmat  = np.full((n_pts, max_N), np.nan, np.float32)
    tlmat = np.full((n_pts, max_N), np.nan, np.float32)
    vlen  = np.empty(n_pts, np.uint16)

    for i, g in enumerate(grid_results):
        lat[i] = g["lat"]
        lon[i] = g["lon"]

        depths = np.asarray(g["tl_depths"]).reshape(-1)
        tlvals = np.asarray(g["transmission_loss"]).reshape(-1)

        k = depths.size
        dmat[i, :k]  = depths
        tlmat[i, :k] = np.round(tlvals, 2)
        vlen[i] = k

    # ─── open/create file ───
    if not os.path.exists(h5_path):
        print(f"[save_dive_frequency] creating new HDF5 file {h5_path}")

    with h5py.File(h5_path, "a") as hf:
        base = (
            hf.require_group(f"drift_{drift_id}")
              .require_group(f"dive_{dive_id}")
              .require_group(f"frequency_{freq_khz}")
        )

        # one-time metadata
        for k, v in metadata.items():
            base.parent.attrs[k] = v

        def _save(name, data, chunks=None):
            if name in base:
                del base[name]
            base.create_dataset(name, data=data,
                                compression="gzip",
                                compression_opts=gzip_level,
                                chunks=chunks)

        row_chunk = min(256, n_pts)
        _save("lat",        lat)
        _save("lon",        lon)
        _save("valid_len",  vlen)
        _save("depth",      dmat, (row_chunk, max_N))
        _save("tl",         tlmat, (row_chunk, max_N))

        # ─── arrivals mini-tables ───
        if "arrivals" in base:
            del base["arrivals"]
        arrivals_grp = base.create_group("arrivals")

        for i, g in enumerate(grid_results):
            arr_obj = g.get("arr", None)

            # ---------- new handling ----------
            if isinstance(arr_obj, pd.DataFrame) and not arr_obj.empty:
                pt_grp = arrivals_grp.create_group(f"pt_{i:05d}")
                for col in arr_obj.columns:
                    pt_grp.create_dataset(
                        name=col,
                        data=arr_obj[col].to_numpy(copy=False),
                        compression="gzip",
                        compression_opts=gzip_level,
                    )
                pt_grp.attrs["row_index_name"] = arr_obj.index.name or ""
                pt_grp.attrs["n_rows"] = len(arr_obj)
            else:
                # empty branch; keeps the mapping but stores no data
                pt_grp = arrivals_grp.create_group(f"pt_{i:05d}")
                pt_grp.attrs["n_rows"] = 0
            # -----------------------------------

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using the haversine formula.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculate the initial compass bearing in degrees between two points.
    """
    lat1, lon1 = map(np.radians, pointA)
    lat2, lon2 = map(np.radians, pointB)
    diffLong = lon2 - lon1
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(diffLong)
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360) % 360

def extract_bathymetry_from_subset(subset_df, start_lat, start_lon, stop_lat, stop_lon, interval):
    """
    Extracts interpolated bathymetry along the great‐circle path between two points
    using only the data from a scattered subset DataFrame.
    """
    start_point = Point(start_lat, start_lon)
    stop_point  = Point(stop_lat, stop_lon)
    total_distance_km = geodesic(start_point, stop_point).kilometers
    interval_km       = interval / 1000.0
    num_points        = max(int(total_distance_km / interval_km), 1)
    bearing           = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    path_lats = np.zeros(num_points + 1)
    path_lons = np.zeros(num_points + 1)
    for i in range(num_points + 1):
        current_distance = min(i * interval_km, total_distance_km)
        new_point = geodesic(kilometers=current_distance).destination(start_point, bearing)
        path_lats[i] = new_point.latitude
        path_lons[i] = new_point.longitude
    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    path_points   = np.vstack((path_lats, path_lons)).T
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    range_km = np.array([
        geodesic(start_point, (path_lats[i], path_lons[i])).kilometers
        for i in range(len(path_lats))
    ])
    return bathymetry_values, path_lons, path_lats, range_km

def extract_bathymetry_from_subset_vectorized(
    subset_df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    stop_lat: float,
    stop_lon: float,
    interval: float):
    """
    Compute bathymetry along the path using vectorized geodesic.
    """
    
    total_distance_km = geodesic((start_lat, start_lon), (stop_lat, stop_lon)).kilometers
    interval_km       = interval / 1000.0
    actual_distance =total_distance_km
    # We need a minimum of 1.1 kms for bellhop
    if total_distance_km<1.1:
        total_distance_km = 1.1

    num_points        = max(int(total_distance_km / interval_km), 1)
    bearing           = calculate_initial_compass_bearing((start_lat, start_lon), (stop_lat, stop_lon))
    distances_m = np.linspace(0, total_distance_km * 1000, num_points + 1)
    
    
    
    lons, lats, _ = geod.fwd(
        np.full_like(distances_m, start_lon),
        np.full_like(distances_m, start_lat),
        np.full_like(distances_m, bearing),
        distances_m
    )
    start_point = Point(start_lat, start_lon)
    range_km = np.array([geodesic(start_point, (lat, lon)).kilometers for lat, lon in zip(lats, lons)])
    subset_points = subset_df[['lat', 'lon']].values
    subset_depths = subset_df['depth'].values
    path_points   = np.column_stack((lats, lons))
    bathymetry_values = griddata(subset_points, subset_depths, path_points, method='linear')
    if np.any(np.isnan(bathymetry_values)):
        nan_mask = np.isnan(bathymetry_values)
        bathymetry_values[nan_mask] = griddata(
            subset_points, subset_depths, path_points[nan_mask], method='nearest'
        )
    return bathymetry_values, lons, lats, range_km, actual_distance

def sanitize_ssp_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Return a strictly increasing SSP profile with no duplicate depths."""
    if profile is None or profile.empty:
        raise ValueError("SSP profile is empty.")

    profile = profile.copy()
    profile = profile[['depth', 'ss']].dropna().copy()
    profile['depth'] = pd.to_numeric(profile['depth'], errors='coerce')
    profile['ss'] = pd.to_numeric(profile['ss'], errors='coerce')
    profile = profile[np.isfinite(profile['depth']) & np.isfinite(profile['ss'])].copy()

    if profile.empty:
        raise ValueError("SSP profile contains no valid depth/sound-speed pairs.")

    profile = profile.sort_values('depth').reset_index(drop=True)

    # Force the surface to be exactly zero, then collapse any duplicate depths by averaging.
    if profile.iloc[0]['depth'] > 0:
        profile.loc[0, 'depth'] = 0.0
    profile = profile.groupby('depth', as_index=False)['ss'].mean()
    profile = profile.sort_values('depth').reset_index(drop=True)

    # Bellhop requires strictly increasing depth samples. If any duplicates remain from
    # floating-point noise, nudge only the later values upward by a tiny epsilon.
    for i in range(1, len(profile)):
        if profile.iloc[i]['depth'] <= profile.iloc[i - 1]['depth']:
            profile.at[i, 'depth'] = profile.iloc[i - 1]['depth'] + 1e-6

    profile = profile.sort_values('depth').reset_index(drop=True)

    if len(profile) < 2:
        return profile

    if np.any(np.diff(profile['depth']) <= 0):
        raise ValueError("SSP profile could not be made strictly monotonic in depth.")

    return profile


def interpolate_sound_speed(dive_data, maxDepth, plot=False):
    dive_data_sorted = dive_data.sort_values('Depth_m')
    dive_data_sorted.dropna(inplace=True, subset=['SoundSpeed_m_s'])
    depth_range = np.arange(0, maxDepth)
    sound_speed_interp = np.interp(
        depth_range,
        dive_data_sorted['Depth_m'],
        dive_data_sorted['SoundSpeed_m_s']
    )
    return pd.DataFrame({'Depth_m': depth_range, 'SoundSpeed_m_s': sound_speed_interp})



def _worker(task):
    (ii, subset_df, drifter_lat, drifter_lon, freq_hz, ssp, drifter_depth,
     bot_ssp, bot_rho, bot_alpha) = task

    bathy_vals, path_lon, path_lat, cumulative_distance, actual_distance = extract_bathymetry_from_subset_vectorized(
        subset_df=subset_df,
        start_lat=drifter_lat,
        start_lon=drifter_lon,
        stop_lat=subset_df['lat'].iloc[ii],
        stop_lon=subset_df['lon'].iloc[ii],
        interval=200
    )

    bathy_grid = pd.DataFrame({
        'range': cumulative_distance * 1000,
        'depth_m': -bathy_vals
    })
    bathy_grid.drop_duplicates(inplace=True)
    bathy_grid.sort_values('range', inplace=True)
    bathy_grid.loc[bathy_grid.index.min(), 'range'] = 0

    bathy = bathy_grid.apply(lambda row: [row['range'], row['depth_m']], axis=1).tolist()

    env = pm.create_env2d(
        depth=bathy,
        soundspeed=ssp,
        bottom_soundspeed=bot_ssp,
        bottom_density=bot_rho,
        bottom_absorption=bot_alpha,
        tx_depth=drifter_depth,
        frequency=freq_hz,
        nbeams=0,
        max_angle=90,
        min_angle=-90,
        soundspeed_interp='pchip'
    )

    if actual_distance < 1.1:
        env['rx_range'] = actual_distance * 1000
        env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[0], 100)
    else:
        env['rx_range'] = bathy_grid['range'].iloc[-1]
        env['rx_depth'] = np.arange(0, bathy_grid['depth_m'].iloc[-1], 100)

    arr = pm.compute_arrivals(env)
    tlosDb = np.full(len(env['rx_depth']), np.nan)

    return ii, tlosDb, arr, env['rx_depth']
from multiprocessing.pool import ThreadPool
import traceback   # optional, if you want full stack traces

def _safe_worker(args):
    """
    Run _worker(args) but never let an exception kill the pool.
    If _worker succeeds     → return ('ok',   ii, result_tuple)
    If _worker raises error → return ('fail', ii, exc)
    """
    ii = args[0]           # first element is your index
    try:
        # _worker should return (ii, tlosDb, rx_depths)
        res = _worker(args)
        return ('ok', ii, res)
    except Exception as exc:
        # Uncomment next line if you want the full traceback printed
        traceback.print_exc()
        return ('fail', ii, exc)

if __name__ == "__main__":
   # ------------------------------------------------------------------ paths
   # Parameters for the propagaton model
   
   import matplotlib.patheffects as pe
   propagationDepth = [500]
   # Bottom Characteristics
   Sediments = ['silt']
   bottom_soundspeed =[1575]
   bottom_density =[1700]
   bottom_absorption= [1,]
   freq_hz =12000

   drift_csv = r"C:\\Users\\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\modelling\\sg680_CalCurCEAS_Sep2024_CTD.csv"
   gebco_nc  = r"C:\\Users\\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\bathymetry\\GEBCO_04_Aug_2026_51f782cc88d3\\gebco_2026_n46.0_s40.0_w-127.0_e-124.0.nc"
   driftEnds = r'C:\\Users\\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\modelling\\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv'
   out_h5    = "X:\\Kaitlin_Palmer\\CalCurCEAS_propagation_hdf5s"

   # Number of CPUs to dedicate to the process
   nWorkers = max(1, mp.cpu_count() - 4)

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
   ends = [0,1,2,3,4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21,22,23 ,24,25,26]
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
        zorder=10, label='Selected Dives'
    )
    
    # Add DiveID labels to the stars
   for x, y, dive_id in zip(lons[np.sort(ends)], lats[np.sort(ends)], driftCTDsub['DiveID'].unique()):
        ax.text(
            x, y, str(dive_id),
            fontsize=10, fontweight='bold',
            ha='left', va='bottom', color='white',
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")]
        )
    
    # Plot track
   ax.plot(lons[5:,], lats[5:,], '-k', zorder=5, label='Track')
   ax.set_aspect('equal', adjustable='box')
   ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
   ax.legend(); plt.show()
       
   
   depth =propagationDepth[0]  
   # 2) loop throught the dives and run the propagation models and depths
    
   for p in range(len(Sediments)):
        
        
        
        # Bottom characteristics
        print(Sediments[p])
        sedimentName = Sediments[p]
        bot_ssp = bottom_soundspeed[p]
        bot_rho = bottom_density[p]
        bot_alpha = bottom_absorption[p]  
        
        for depth in propagationDepth:
            base_filename = 'Spacious_CalCursesV2_'+sedimentName+ '_PCHIP_'+str(int(freq_hz/1000))+'kHz_20km_'+str(depth)+'m_BotSensitivity.h5'
            fullfileOut = os.path.join(out_h5, base_filename)
            completed_dives = get_completed_dive_ids(fullfileOut)
            
            for dive_id in driftCTDsub['DiveID'].unique():
                if f"dive_{dive_id}" in completed_dives:
                    print(f"Skipping completed dive {dive_id}")
                    continue
                   
               
               
                # 3) pull out the corresponding group “on the fly”
                group = driftCTD[driftCTD['DiveID'] == dive_id]
                
                
                # 1. Ensure 'group' is an explicit copy if it came from a groupby operation
                group = group.copy()
                
                # 2. Compute depth differences
                depth_diff = np.diff(group["Depth_m"], prepend=np.nan)
                
                # 3. Assign the 'Direction' column safely using assignment
                group["Direction"] = np.where(depth_diff > 0, "dec", "asc")
                
                # 4. Use .loc to modify individual elements safely
                first_dir = "dec" if depth_diff[1] > 0 else "asc"
                group.loc[group.index[0], "Direction"] = first_dir
                
                print(dive_id)
                diveName = 'dive_'+dive_id
                
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
               
                
                # Create the SSP profile and collapse duplicate depths before handing it to Bellhop.
                profile = pd.DataFrame({
                    'depth': group['Depth_m'],
                    'ss':    group['SoundSpeed_m_s'] })
                profile = sanitize_ssp_profile(profile)
                
                # Only use the dive if the profile depth is more than 200m
                if np.max(profile['depth'])>depth:
                    
                    # base_filename = 'Spacious_CalCursesV2_'+sedimentName+ '_PCHIP_'+str(int(freq_hz/1000))+'kHz_20km_'+str(depth)+'m_BotSensitivity.h5'
                    # fullfileOut = os.path.join(out_h5, base_filename)
                    
                    # If the code crashes this code will pull the dives that have already been
                    # completed
                    completed_dives = get_completed_dive_ids(fullfileOut)
                    print("Already completed:", completed_dives)
                    
                    # skip dives already processed
                    if diveName in completed_dives:
                        print(f"{diveName} already completed for {sedimentName}, moving on")
                        continue
                    else:
                        print(f"{diveName} not found running")
                    
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
                        
                        
                        profile = pd.concat([profile, expanedProfile], ignore_index=True)
                        profile = sanitize_ssp_profile(profile)
                        profile['ss'] = np.abs(profile['ss'])
                        ssp = [[float(row['depth']), float(row['ss'])] for _, row in profile.iterrows()]
                        
                        
                        # Dictionary with keys 'start_lat', 'start_lon', and 'drifter_depth'.
                        metadata = {'start_lat': drifter_lat,
                                        'start_lon': drifter_lon,
                                        'drifter_depth': drifter_depth,
                                        'Sediment_Name': sedimentName,
                                        'bottom_soundspeed':bot_ssp,
                                        'bottom_density':bot_rho,
                                        'bottom_absorption':bot_alpha }
                        txDepth = metadata['drifter_depth']
                        
            
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
                                if ii % 50 == 0:
                                    print(f"Processed {ii} of {total_rows} points at {depth} m in {sedimentName}.", flush=True)
                                #print(f"Processed {ii} of {total_rows} points at {depth} m.")
            
                        
                                
                                  
                        save_dive_frequency(
                        h5_path      = fullfileOut,
                        drift_id     = "01",
                        dive_id      = dive_id,
                        freq_khz     = freq_hz,
                        metadata     = metadata,
                        grid_results = results[dive_id])
                        
                        elapsed = time.time() - t
                        print(f'Dive {dive_id} completed in {elapsed:.1f} s')
                
    
