"""
emma/postprocessing.py

Physics-based Filtering and Post-Processing for MCS Tracking.

This module implements the final stage of the EMMA (Era-5 and IMERG MCS Algorithm) pipeline.
It refines the raw tracking output by applying rigorous physical constraints to identify
robust Mesoscale Convective Systems (MCSs).

Methodology:
1.  **Metric Extraction**: For every timestep of every track, key physical properties are extracted:
    -   **Location**: Centroid coordinates directly from tracking output.
    -   **Area**: Precise physical area (km²) calculated from the grid geometry.
    -   **Instability**: Mean Lifted Index (LI) from ERA5 data within the MCS mask.
2.  **Trajectory Analysis**: Tracks are aggregated to calculate lifetime statistics:
    -   **Straightness**: Ratio of net displacement to total path length.
    -   **Volatility**: Variance in area growth/decay to remove artifacts.
    -   **Environment**: Lifetime-mean instability conditions.
3.  **Filtering**: Tracks failing defined thresholds are discarded.
4.  **Output Generation**: New NetCDF files are written containing only valid MCS tracks,
    preserving CF-compliance and compression.

References:
    Kneidinger et al. (2025)
"""

import os
import glob
import logging
import concurrent.futures
from typing import List, Set, Dict, Optional, Any

import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import great_circle

# Import project-specific helpers
from .input_output import (
    load_lifted_index_data,
    save_dataset_to_netcdf
)
from .tracking_helper_func import calculate_grid_area_map

logger = logging.getLogger(__name__)


def get_grid_dict_from_ds(ds: xr.Dataset) -> Dict[str, np.ndarray]:
    """
    Extracts coordinate arrays from a dataset to support precise area calculation.

    This function robustly handles different NetCDF coordinate naming conventions
    (e.g., 'lat'/'lon', 'latitude'/'longitude', 'rlat'/'rlon') and constructs
    2D meshgrids for regular grids if explicit 2D coordinates are missing.

    Args:
        ds (xr.Dataset): The tracking dataset containing coordinate variables.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - 'lat': 1D latitude array.
            - 'lon': 1D longitude array.
            - 'lat2d': 2D latitude grid.
            - 'lon2d': 2D longitude grid.

    Raises:
        ValueError: If valid grid coordinates cannot be determined.
    """
    lat_1d, lon_1d, lat_2d, lon_2d = None, None, None, None

    # 1. Attempt to find 1D coordinates using standard CF standard_names or common abbreviations
    for lat_name in ['lat', 'latitude', 'rlat', 'y']:
        if lat_name in ds.coords:
            lat_1d = ds[lat_name].values
            break
    for lon_name in ['lon', 'longitude', 'rlon', 'x']:
        if lon_name in ds.coords:
            lon_1d = ds[lon_name].values
            break
            
    # 2. Attempt to find existing 2D auxiliary coordinates
    if 'latitude' in ds: lat_2d = ds['latitude'].values
    if 'longitude' in ds: lon_2d = ds['longitude'].values
    # Fallback for older file versions
    if lat_2d is None and 'lat2d' in ds: lat_2d = ds['lat2d'].values
    if lon_2d is None and 'lon2d' in ds: lon_2d = ds['lon2d'].values

    # 3. Construct 2D meshgrid if missing (Assumes Regular Grid geometry)
    if lat_1d is not None and lon_1d is not None and lat_2d is None:
        # Assuming 'ij' indexing (lat, lon) matches the mcs_id array shape
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing='ij')

    if lat_1d is None or lat_2d is None:
        raise ValueError(f"Could not determine grid coordinates. Found coords: {list(ds.coords)}")

    return {'lat': lat_1d, 'lon': lon_1d, 'lat2d': lat_2d, 'lon2d': lon_2d}


def process_single_timestep(
    tracking_file: str, 
    available_li_files: List[str], 
    config: dict
) -> List[dict]:
    """
    Worker function to extract physical metrics for all tracks in a single timestep.

    Args:
        tracking_file (str): Path to the raw tracking NetCDF file.
        available_li_files (List[str]): List of paths to available Lifted Index files for the year.
        config (dict): Configuration dictionary containing variable names and thresholds.

    Returns:
        List[dict]: A list of dictionaries, one per active track, containing:
            - 'track_id': Unique integer ID of the MCS.
            - 'time': Timestamp of the observation.
            - 'area_km2': Physical area of the system.
            - 'mean_li': Mean Lifted Index within the system mask.
            - 'lat': Center latitude.
            - 'lon': Center longitude.
            Returns an empty list if processing fails or no tracks are present.
    """
    try:
        results = []
        with xr.open_dataset(tracking_file) as ds_track:
            if ds_track.time.size == 0:
                return []
            time_val = pd.to_datetime(ds_track.time.values[0])
            
            # --- 1. Load Active Track Metadata ---
            if 'active_track_id' not in ds_track:
                return []
            
            active_ids = ds_track['active_track_id'].values
            if len(active_ids) == 0:
                return []

            # Retrieve coordinates directly from tracking output
            # These are pre-calculated during the tracking phase and should be valid.
            active_lats = ds_track['active_track_lat'].values
            active_lons = ds_track['active_track_lon'].values
            
            mcs_map = ds_track['mcs_id'].values[0]
            
            # Data Integrity Check
            if len(active_ids) != len(active_lats):
                logger.error(
                    f"Shape mismatch in {tracking_file}: IDs ({len(active_ids)}) vs Lats ({len(active_lats)})"
                )
                return []

            # --- 2. Calculate Precise Grid Area ---
            # Calculate the physical area (km2) for every pixel in the grid based on latitude.
            try:
                grid_dict = get_grid_dict_from_ds(ds_track)
                area_map_km2 = calculate_grid_area_map(grid_dict)
            except Exception as e:
                logger.error(f"Grid Area Calculation Failed for {tracking_file}: {e}")
                return []

            # --- 3. Locate and Load Environmental Data (Lifted Index) ---
            year_str = time_val.strftime("%Y")
            month_str = time_val.strftime("%m")
            day_str = time_val.strftime("%d")
            hour_str = time_val.strftime("%H")
            date_str = f"{year_str}{month_str}{day_str}"
            
            # Efficiently find the matching file in the pre-loaded list
            matching_files = [
                f for f in available_li_files 
                if date_str in os.path.basename(f) and hour_str in os.path.basename(f)
            ]

            li_data_values = None
            if matching_files:
                li_file_path = sorted(matching_files)[0]
                try:
                    # Load data using the project's standard loader.
                    # Note: We do NOT convert units here; we use the raw values (Kelvin/Difference).
                    # We strictly use 'lat'/'lon' as verified by file inspection.
                    _, _, _, _, _, li_da = load_lifted_index_data(
                        li_file_path, 
                        config['liting_index_var_name'], 
                        lat_name="lat", 
                        lon_name="lon",
                        time_index=0 
                    )
                    li_data_values = li_da.values

                except Exception as e:
                    logger.warning(f"Failed to load LI file {li_file_path}: {e}")
                    li_data_values = None

            # --- 4. Extract Metrics for Each Track ---
            for idx, tid in enumerate(active_ids):
                # Create boolean mask for the current system
                mask = (mcs_map == tid)
                area_cells = np.sum(mask)
                
                # Skip "Ghost" Tracks:
                # If a track ID is listed in metadata but has 0 pixels in the map 
                # (e.g., just terminated or formed), it should not be processed.
                if area_cells == 0:
                    continue

                # A. Physical Area
                # Sum the pre-calculated area of all pixels belonging to this system
                if area_map_km2.shape == mask.shape:
                    area_km2 = np.sum(area_map_km2[mask])
                else:
                    # Robust fallback in rare case of shape mismatch
                    area_km2 = area_cells * config.get('grid_cell_area_km2', 121.0)
                
                # B. Instability (Lifted Index)
                mean_li = np.nan
                if li_data_values is not None:
                    if li_data_values.shape == mask.shape:
                        li_vals = li_data_values[mask]
                        # Calculate mean only if valid data exists (avoid RuntimeWarning)
                        if len(li_vals) > 0 and not np.all(np.isnan(li_vals)):
                            mean_li = np.nanmean(li_vals)
                
                # C. Location
                center_lat = active_lats[idx]
                center_lon = active_lons[idx]

                results.append({
                    'track_id': tid,
                    'time': time_val,
                    'area_km2': area_km2,
                    'mean_li': mean_li,
                    'lat': center_lat,
                    'lon': center_lon
                })
        return results

    except Exception as e:
        logger.error(f"Error processing {tracking_file}: {e}")
        return []


def filter_tracks(df_timesteps: pd.DataFrame, config: dict) -> Set[int]:
    """
    Applies physical filtering criteria to the aggregated track histories.

    Filters implemented:
    1.  **Instability**: The track must exist in an unstable environment (Lifted Index threshold).
    2.  **Volatility**: The track's area growth/decay must be physically realistic (removes artifacts).
    3.  **Straightness**: The track must follow a somewhat linear path (removes erratic 'jumping' tracks).

    Args:
        df_timesteps (pd.DataFrame): DataFrame containing metrics for all timesteps of all tracks.
        config (dict): Configuration dictionary with filter thresholds.

    Returns:
        Set[int]: A set of valid Track IDs that passed all filters.
    """
    grouped = df_timesteps.groupby('track_id')
    valid_ids = []
    thresholds = config['postprocessing_filters']
    
    # Statistics counters for logging
    total_tracks = len(grouped)
    rejected_li = 0
    rejected_volatility = 0
    rejected_straightness = 0
    kept_nan_li = 0
    
    for tid, group in grouped:
        group = group.sort_values('time')
        
        # --- Filter 1: Environmental Instability (Lifted Index) ---
        lifetime_mean_li = group['mean_li'].mean()
        
        # Handling Missing Data:
        # If LI data is missing (NaN), we give the track the benefit of the doubt and keep it.
        # We only reject if we have valid data confirming the environment is stable.
        if not np.isnan(lifetime_mean_li):
            if lifetime_mean_li >= thresholds['lifted_index_threshold']:
                rejected_li += 1
                continue 
        else:
            kept_nan_li += 1

        # --- Filter 2: Area Volatility ---
        # Single-timestep tracks cannot assess volatility or movement; usually rejected implicitly.
        if len(group) < 2:
            rejected_volatility += 1
            continue 

        areas = group['area_km2'].values
        prev_area = areas[:-1]
        curr_area = areas[1:]
        
        # Volatility metric: Normalized squared area change
        diff_sq = (curr_area - prev_area)**2
        mean_area_step = (curr_area + prev_area) / 2.0
        
        # Use float division and handle division by zero (e.g., very small systems)
        volatility = np.divide(
            diff_sq, 
            mean_area_step, 
            out=np.zeros_like(diff_sq, dtype=float), 
            where=mean_area_step!=0
        )
        
        max_volatility = np.max(volatility)
        
        if max_volatility > thresholds['max_area_volatility']:
            rejected_volatility += 1
            continue 

        # --- Filter 3: Track Straightness ---
        lats = group['lat'].values
        lons = group['lon'].values
        
        # Robustness: Filter out NaNs or Zeros (invalid coordinates)
        valid_coords = (np.isfinite(lats) & np.isfinite(lons) & (lats != 0) & (lons != 0))
        
        if np.sum(valid_coords) < 2:
            rejected_straightness += 1
            continue

        lats_clean = lats[valid_coords]
        lons_clean = lons[valid_coords]
        
        # Calculate total path length (sum of segments)
        total_dist = 0.0
        for i in range(len(lats_clean)-1):
            dist = great_circle((lats_clean[i], lons_clean[i]), (lats_clean[i+1], lons_clean[i+1])).kilometers
            total_dist += dist
            
        # Calculate net displacement (start to end)
        net_dist = great_circle((lats_clean[0], lons_clean[0]), (lats_clean[-1], lons_clean[-1])).kilometers
        
        # Straightness ratio [0-1]
        straightness = (net_dist / total_dist) if total_dist > 1e-6 else 1.0
        
        if straightness <= thresholds['track_straightness_threshold']:
            rejected_straightness += 1
            continue 
            
        valid_ids.append(tid)

    logger.info(f"--- Track Filtering Statistics ---")
    logger.info(f"Total unique tracks processed: {total_tracks}")
    logger.info(f"  - Rejected by LI Stability: {rejected_li}")
    logger.info(f"  - Rejected by Volatility/Duration: {rejected_volatility}")
    logger.info(f"  - Rejected by Straightness: {rejected_straightness}")
    logger.info(f"  - Kept with Missing Data (NaN): {kept_nan_li}")
    logger.info(f"  - Total Valid MCS Tracks: {len(valid_ids)}")

    return set(valid_ids)


def apply_filter_to_files(raw_files: List[str], valid_ids: Set[int], output_dir: str):
    """
    Applies the valid ID mask to raw tracking files and saves the filtered output.

    Args:
        raw_files (List[str]): List of paths to raw tracking NetCDF files.
        valid_ids (Set[int]): Set of track IDs to retain.
        output_dir (str): Directory where filtered files will be saved.
    """
    logger.info(f"Writing {len(raw_files)} filtered files to {output_dir}...")
    
    for f in raw_files:
        try:
            with xr.open_dataset(f) as ds:
                ds.load()
                
                # 1. Mask Gridded Variables
                # Set pixel values to 0 if the track ID is not in the valid set
                id_vars = ['mcs_id', 'robust_mcs_id', 'mcs_id_merge_split']
                for var in id_vars:
                    if var in ds:
                        data = ds[var].values
                        mask_invalid = ~np.isin(data, list(valid_ids))
                        # Only mask positive IDs (tracks), leaving background (0) as is
                        data[mask_invalid & (data > 0)] = 0
                        ds[var].values = data
                
                # 2. Filter Tabular Variables (Active Tracks)
                # Remove rows from the track metadata if the ID is invalid
                if 'active_track_id' in ds:
                    active_ids = ds['active_track_id'].values
                    valid_indices = np.isin(active_ids, list(valid_ids))
                    ds = ds.isel(tracks=valid_indices)

                # 3. Prepare Output Path
                time_val = pd.to_datetime(ds.time.values[0])
                year_str = time_val.strftime("%Y")
                month_str = time_val.strftime("%m")
                
                out_subdir = os.path.join(output_dir, year_str, month_str)
                os.makedirs(out_subdir, exist_ok=True)
                
                out_name = os.path.basename(f)
                out_path = os.path.join(out_subdir, out_name)
                
                # 4. Update Metadata
                ds.attrs['postprocessing_level'] = 'Filtered (LI, Straightness, Volatility)'
                ds.attrs['history'] += f"; Post-processed on {pd.Timestamp.now().strftime('%Y-%m-%d')}"
                
                # 5. Save with standardized encoding
                save_dataset_to_netcdf(ds, out_path)
                
        except Exception as e:
            logger.error(f"Failed to filter/save {f}: {e}")


def run_postprocessing_year(
    year: int, 
    raw_tracking_dir: str, 
    filtered_output_dir: str, 
    config: dict, 
    n_cores: int
):
    """
    Orchestrates the post-processing workflow for a specific year.

    Workflow:
    1.  Index all raw tracking files and Lifted Index files.
    2.  Extract physical metrics for all tracks in parallel.
    3.  Aggregate metrics and determine valid tracks based on filters.
    4.  Write filtered NetCDF files to the output directory.

    Args:
        year (int): The year to process.
        raw_tracking_dir (str): Directory containing raw tracking output.
        filtered_output_dir (str): Directory to save final filtered files.
        config (dict): Configuration dictionary.
        n_cores (int): Number of CPU cores for parallel processing.
    """
    logger.info(f"--- Starting Post-Processing for Year {year} ---")
    
    # 1. Find Raw Tracking Files
    search_pattern = os.path.join(raw_tracking_dir, str(year), "**", "tracking_*.nc")
    raw_files = sorted(glob.glob(search_pattern, recursive=True))
    if not raw_files:
        logger.warning(f"No raw tracking files found for {year} in {raw_tracking_dir}")
        return

    # 2. Find All Lifted Index Files
    # We index files once to avoid repeated filesystem calls in workers
    li_dir = config['lifted_index_data_directory']
    all_li_files = glob.glob(os.path.join(li_dir, "**", f"*{config['file_suffix']}"), recursive=True)
    if not all_li_files:
         logger.warning("No Lifted Index files found. Metrics will be NaN, but processing will continue.")
         all_li_files = []

    # Filter LI files for the current year to optimize search speed
    li_files_year = [f for f in all_li_files if str(year) in f]
    logger.info(f"Found {len(li_files_year)} LI files for year {year}.")

    # 3. Extract Metrics (Parallel Execution)
    logger.info("Extracting track metrics...")
    all_timestep_rows = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {
            executor.submit(process_single_timestep, f, li_files_year, config): f 
            for f in raw_files
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                all_timestep_rows.extend(result)
            
    if not all_timestep_rows:
        logger.warning("No valid metrics extracted. Skipping filtering.")
        return
        
    df_timesteps = pd.DataFrame(all_timestep_rows)
    
    # 4. Filter Tracks
    logger.info("Applying physical filters...")
    valid_ids = filter_tracks(df_timesteps, config)
    
    # 5. Save Filtered Data
    if valid_ids:
        apply_filter_to_files(raw_files, valid_ids, filtered_output_dir)
    else:
        logger.warning("0 Valid IDs found. Skipping file writing.")
    
    logger.info(f"Post-processing for {year} complete.")