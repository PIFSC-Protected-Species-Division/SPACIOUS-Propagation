# -*- coding: utf-8 -*-
"""
Single-bearing Bellhop coherent TL diagnostic

Purpose
-------
This script is a lightweight diagnostic version of the larger parallel grid run.
It chooses one glider/dive site, extracts bathymetry along one bearing out to
20 km, runs Bellhop once per source depth using a dense receiver grid, and plots
coherent transmission loss with Matplotlib.

It does NOT loop over every lat/lon point and does NOT use multiprocessing.

Run from Spyder or Anaconda Prompt:
    python Bellhop_single_bearing_coherent_TL_test.py

Requirements
------------
    numpy, pandas, xarray, scipy, geopy, pyproj, matplotlib, arlpy
    Bellhop must already be installed and visible to arlpy.
"""

# =============================================================================
# Imports
# =============================================================================

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from geopy.distance import geodesic
from pyproj import Geod
import arlpy.uwapm as pm


# =============================================================================
# User inputs
# =============================================================================

# --- files from the full workflow ---
drift_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_CTD.csv"
)

gebco_nc = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\bathymetry\GEBCO_28_Jul_2025_937903cf24aa"
    r"\gebco_2024_n44.6_s40.2_w-126.3_e-124.0.nc"
)

drift_ends_csv = (
    r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-Propagation-Modes"
    r"\modelling\sg680_CalCurCEAS_Sep2024_final_targets_distances_withDate.csv"
)

# --- diagnostic choice ---
# These follow the earlier script logic where ends = [13, 5, 21].
# SITE_END_INDEX picks one row from drift_ends_csv.
SITE_END_INDEX = 5

# Bearing in degrees clockwise from north.
# Change this if you want to test different radial directions.
BEARING_DEG = 270

MAX_RANGE_M = 20_000
BATHY_INTERVAL_M = 200

# Dense receiver grid. Increase/decrease if Bellhop is slow.
RX_RANGE_STEP_M = 100
RX_DEPTH_STEP_M = 10

SOURCE_DEPTHS_M = [50, 200, 350, 500, 650, 800]
FREQ_HZ = 12_000

# Bottom type does not matter for this diagnostic; using silt-like values.
BOTTOM_SOUND_SPEED = 1575
BOTTOM_DENSITY = 1700
BOTTOM_ABSORPTION = 1

# Keep this as 0 initially to match the full run.
# If you want to test beam-density sensitivity, change to e.g. 5000 or 10000.
NBEAMS = 2000

MIN_ANGLE = -90
MAX_ANGLE = 90
SOUNDSPEED_INTERP = "pchip"

OUT_DIR = "single_bearing_TL_test_outputs"


# =============================================================================
# Helper functions
# =============================================================================

GEOD = Geod(ellps="WGS84")


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in km for scalar or vector inputs."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def make_bathymetry_dataframe(gebco_path):
    """Load GEBCO NetCDF into a flat DataFrame matching the full script."""
    ds = xr.open_dataset(gebco_path)
    return pd.DataFrame({
        "depth": ds["elevation"].values.flatten(),
        "lat": np.repeat(ds["lat"].values, len(ds["lon"])),
        "lon": np.tile(ds["lon"].values, len(ds["lat"])),
    })


def extract_bathymetry_along_bearing(
    bathymetry_df,
    start_lat,
    start_lon,
    bearing_deg,
    max_range_m=20_000,
    interval_m=200,
):
    """
    Extract interpolated bathymetry along a fixed bearing.

    Returns
    -------
    bathy : list of [range_m, positive_depth_m] pairs for Bellhop
    path_df : DataFrame with range_m, lat, lon, elevation_m, depth_m
    """
    ranges_m = np.arange(0, max_range_m + interval_m, interval_m, dtype=float)

    lons, lats, _ = GEOD.fwd(
        np.full_like(ranges_m, start_lon, dtype=float),
        np.full_like(ranges_m, start_lat, dtype=float),
        np.full_like(ranges_m, bearing_deg, dtype=float),
        ranges_m,
    )

    points = bathymetry_df[["lat", "lon"]].values
    values = bathymetry_df["depth"].values
    path_points = np.column_stack((lats, lons))

    elev = griddata(points, values, path_points, method="linear")
    if np.any(np.isnan(elev)):
        nan_mask = np.isnan(elev)
        elev[nan_mask] = griddata(points, values, path_points[nan_mask], method="nearest")

    # GEBCO elevation is negative underwater. Bellhop wants positive water depth.
    depth_m = np.maximum(-elev, 1.0)

    path_df = pd.DataFrame({
        "range_m": ranges_m,
        "lat": lats,
        "lon": lons,
        "elevation_m": elev,
        "depth_m": depth_m,
    })

    path_df = path_df.drop_duplicates(subset="range_m").sort_values("range_m")
    path_df.loc[path_df.index.min(), "range_m"] = 0.0

    bathy = path_df[["range_m", "depth_m"]].values.tolist()
    return bathy, path_df


def select_dive_and_make_ssp(drift_ctd, drift_ends, site_end_index, max_depth_m):
    """
    Pick the CTD profile associated with one target/end index and build the SSP
    in the same style as the full script.
    """
    drift_ctd = drift_ctd.copy()
    drift_ctd["DiveID"] = drift_ctd["DiveNumber"].astype(str)

    dive_num = drift_ends.loc[site_end_index, "closestDive"]
    group = drift_ctd[drift_ctd["DiveNumber"] == dive_num].copy()
    if group.empty:
        raise ValueError(f"No CTD rows found for closestDive={dive_num}")

    depth_diff = np.diff(group["Depth_m"], prepend=np.nan)
    group["Direction"] = np.where(depth_diff > 0, "dec", "asc")
    if len(depth_diff) > 1 and depth_diff[1] > 0:
        group.loc[group.index[0], "Direction"] = "dec"
    else:
        group.loc[group.index[0], "Direction"] = "asc"

    # Match the larger run: use descending profile for SSP and source location.
    group_dec = group[group["Direction"] == "dec"].reset_index(drop=True)
    if group_dec.empty:
        raise ValueError(f"No descending rows found for dive {dive_num}")

    start_lat = float(group_dec["Latitude"].iloc[0])
    start_lon = float(group_dec["Longitude"].iloc[0])

    profile = pd.DataFrame({
        "depth": group_dec["Depth_m"].to_numpy(),
        "ss": group_dec["SoundSpeed_m_s"].to_numpy(),
    })
    profile = profile.dropna().sort_values("depth").reset_index(drop=True)
    profile.loc[0, "depth"] = 0.0

    if profile.empty:
        raise ValueError(f"No valid SSP points found for dive {dive_num}")

    # Extend the last measured sound speed to the deepest bathymetry.
    last_depth = float(profile["depth"].iloc[-1])
    last_ss = float(profile["ss"].iloc[-1])

    if max_depth_m > last_depth + 10:
        z_ext = np.arange(last_depth + 10, max_depth_m + 50, 50)
        ext = pd.DataFrame({"depth": z_ext, "ss": np.repeat(last_ss, len(z_ext))})
        profile = pd.concat([profile, ext], ignore_index=True)

    profile["ss"] = np.abs(profile["ss"])
    profile = profile.sort_values("depth").drop_duplicates(subset="depth")

    ssp = profile[["depth", "ss"]].values.tolist()
    return dive_num, start_lat, start_lon, profile, ssp


def tl_to_numpy(tl):
    """
    Convert common ARLPy TL output forms to a 2D numpy array.

    ARLPy commonly returns an xarray-like object. This also handles plain arrays.
    """
    if hasattr(tl, "values"):
        arr = np.asarray(tl.values)
    else:
        arr = np.asarray(tl)

    # Coherent TL may be complex pressure. Convert to dB loss if needed.
    if np.iscomplexobj(arr):
        arr = -20 * np.log10(np.maximum(np.abs(arr), 1e-30))
    else:
        arr = np.asarray(arr, dtype=float)

    # Squeeze singleton source-depth dimensions if present.
    arr = np.squeeze(arr)
    return arr


def orient_tl_for_plot(tl_db, rx_depth, rx_range):
    """Ensure TL array is shaped as [depth, range] for contourf."""
    tl_db = np.asarray(tl_db)
    nz = len(rx_depth)
    nr = len(rx_range)

    if tl_db.shape == (nz, nr):
        return tl_db
    if tl_db.shape == (nr, nz):
        return tl_db.T

    raise ValueError(
        f"Unexpected TL shape {tl_db.shape}; expected {(nz, nr)} or {(nr, nz)}."
    )


def plot_tl(tl_db, rx_range, rx_depth, bathy_df, src_depth, out_dir):
    """Matplotlib-only TL plot."""
    tl_plot = orient_tl_for_plot(tl_db, rx_depth, rx_range)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    levels = np.arange(40, 141, 2)

    cf = ax.contourf(
        rx_range / 1000,
        rx_depth,
        tl_plot,
        levels=levels,
        cmap="viridis_r",
        extend="both",
    )

    ax.plot(bathy_df["range_m"] / 1000, bathy_df["depth_m"], "k-", lw=1.5)
    ax.fill_between(
        bathy_df["range_m"] / 1000,
        bathy_df["depth_m"],
        np.nanmax(bathy_df["depth_m"]) + 200,
        color="0.75",
        alpha=0.6,
    )
    ax.scatter([0], [src_depth], marker="*", s=140, c="white", edgecolors="k", zorder=5)

    ax.set_ylim(np.nanmax(bathy_df["depth_m"]) + 50, 0)
    ax.set_xlim(0, np.nanmax(rx_range) / 1000)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Coherent TL, source depth = {src_depth} m")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Transmission loss (dB)")

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"coherent_TL_source_{int(src_depth):04d}m.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")
    plt.show()


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CTD, target, and bathymetry files...")
    drift_ctd = pd.read_csv(drift_csv)
    drift_ends = pd.read_csv(drift_ends_csv)
    bathymetry_df = make_bathymetry_dataframe(gebco_nc)

    # Initial site location is based on the CTD profile, but we need max bathy depth
    # along the bearing before extending the SSP. First get rough site from driftEnds.
    rough_start_lat = float(drift_ends.loc[SITE_END_INDEX, "lat"])
    rough_start_lon = float(drift_ends.loc[SITE_END_INDEX, "lon"])

    # Limit bathymetry interpolation to nearby points for speed.
    bathymetry_df["distance_km"] = haversine(
        rough_start_lon,
        rough_start_lat,
        bathymetry_df["lon"],
        bathymetry_df["lat"],
    )
    bathy_subset = bathymetry_df[
        (bathymetry_df["distance_km"] <= 25)
        & (bathymetry_df["depth"] < -50)
    ].copy()

    if bathy_subset.empty:
        raise ValueError("Bathymetry subset is empty. Check SITE_END_INDEX and GEBCO file bounds.")

    # Create preliminary bathymetry along bearing from rough point.
    rough_bathy, rough_path_df = extract_bathymetry_along_bearing(
        bathy_subset,
        rough_start_lat,
        rough_start_lon,
        BEARING_DEG,
        max_range_m=MAX_RANGE_M,
        interval_m=BATHY_INTERVAL_M,
    )
    max_bathy_depth = float(np.nanmax(rough_path_df["depth_m"]))

    # Now get the actual source location and SSP from the associated dive.
    dive_num, start_lat, start_lon, profile, ssp = select_dive_and_make_ssp(
        drift_ctd,
        drift_ends,
        SITE_END_INDEX,
        max_depth_m=max_bathy_depth,
    )

    print(f"Using SITE_END_INDEX={SITE_END_INDEX}")
    print(f"Using closestDive={dive_num}")
    print(f"Source location: lat={start_lat:.5f}, lon={start_lon:.5f}")
    print(f"Bearing: {BEARING_DEG} deg")

    # Recompute bathymetry from actual source location.
    bathymetry_df["distance_km"] = haversine(
        start_lon,
        start_lat,
        bathymetry_df["lon"],
        bathymetry_df["lat"],
    )
    bathy_subset = bathymetry_df[
        (bathymetry_df["distance_km"] <= 25)
        & (bathymetry_df["depth"] < -50)
    ].copy()

    bathy, path_df = extract_bathymetry_along_bearing(
        bathy_subset,
        start_lat,
        start_lon,
        BEARING_DEG,
        max_range_m=MAX_RANGE_M,
        interval_m=BATHY_INTERVAL_M,
    )

    max_water_depth = float(np.nanmax(path_df["depth_m"]))
    rx_range = np.arange(RX_RANGE_STEP_M, MAX_RANGE_M + RX_RANGE_STEP_M, RX_RANGE_STEP_M)
    rx_depth = np.arange(5, max_water_depth, RX_DEPTH_STEP_M)

    # Save diagnostic inputs so the exact test can be reproduced.
    path_df.to_csv(os.path.join(OUT_DIR, "single_bearing_bathymetry.csv"), index=False)
    profile.to_csv(os.path.join(OUT_DIR, "ssp_profile_used.csv"), index=False)

    print(f"Receiver grid: {len(rx_range)} ranges x {len(rx_depth)} depths")
    print(f"Max bathymetry depth along bearing: {max_water_depth:.1f} m")

    # Quick input plot.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(path_df["range_m"] / 1000, path_df["depth_m"], "k-")
    ax.set_ylim(max_water_depth + 50, 0)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Water depth (m)")
    ax.set_title("Bathymetry along selected bearing")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bathymetry_along_bearing.png"), dpi=200)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(profile["ss"], profile["depth"], "k-")
    ax.set_ylim(max_water_depth + 50, 0)
    ax.set_xlabel("Sound speed (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"SSP used, dive {dive_num}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ssp_profile_used.png"), dpi=200)
    plt.show()

    for src_depth in SOURCE_DEPTHS_M:
        if src_depth >= max_water_depth:
            print(f"Skipping {src_depth} m; source is deeper than bathymetry along this bearing.")
            continue

        print(f"Running coherent TL for source depth {src_depth} m...")

        env = pm.create_env2d(
            depth=bathy,
            soundspeed=ssp,
            bottom_soundspeed=BOTTOM_SOUND_SPEED,
            bottom_density=BOTTOM_DENSITY,
            bottom_absorption=BOTTOM_ABSORPTION,
            tx_depth=src_depth,
            frequency=FREQ_HZ,
            nbeams=NBEAMS,
            max_angle=MAX_ANGLE,
            min_angle=MIN_ANGLE,
            soundspeed_interp=SOUNDSPEED_INTERP,
        )

        env["rx_range"] = rx_range
        env["rx_depth"] = rx_depth

        # The key diagnostic: one Bellhop run over the whole dense receiver grid.
        tl = pm.compute_transmission_loss(env, mode="coherent")
        tl_db = tl_to_numpy(tl)

        np.save(os.path.join(OUT_DIR, f"tl_db_source_{int(src_depth):04d}m.npy"), tl_db)
        plot_tl(tl_db, rx_range, rx_depth, path_df, src_depth, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
