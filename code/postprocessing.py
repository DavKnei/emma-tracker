"""
code/postprocessing.py

Handles the physics-based filtering of MCS tracks.
1. Extracts metrics (Straightness, LI, Volatility) from raw tracking files.
2. Applies thresholds defined in config.
3. Saves new NetCDF files containing only valid MCS tracks.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import logging
from geopy.distance import great_circle
import concurrent.futures

# Import existing helpers to maintain consistency
from input_output import (
    load_precipitation_data, 
    load_lifted_index_data,
    save_tracking_result
)

logger = logging.getLogger(__name__)

def calculate_weighted_center(precip_data, mask, lat2d, lon2d):
    """
    Calculates precipitation-weighted center of mass.
    Logic adapted from development scripts.
    """
    precip_in_mask = precip_data[mask]
    total_weight = np.sum(precip_in_mask)

    if total_weight == 0:
        # Fallback to geometric center
        coords = np.argwhere(mask)
        if len(coords) == 0: return np.nan, np.nan
        # Average indices
        mean_y, mean_x = coords.mean(axis=0).astype(int)
        return lat2d[mean_y, mean_x], lon2d[mean_y, mean_x]

    # Weighted sum
    lats = lat2d[mask]
    lons = lon2d[mask]
    
    center_lat = np.sum(lats * precip_in_mask) / total_weight
    center_lon = np.sum(lons * precip_in_mask) / total_weight
    
    return center_lat, center_lon

def process_single_timestep(
    tracking_file, 
    precip_dir, 
    li_dir, 
    config
):
    """
    Worker function: Opens one tracking file, finds matching source data, 
    and extracts metrics for all tracks present in that frame.
    """
    try:
        results = []
        with xr.open_dataset(tracking_file) as ds_track:
            time_val = pd.to_datetime(ds_track.time.values[0])
            # Load IDs - handling the 3 different ID types if necessary, 
            # usually we filter based on 'mcs_id' (the main track ID)
            mcs_map = ds_track['mcs_id'].values[0] # (lat, lon)
            
            unique_ids = np.unique(mcs_map)
            unique_ids = unique_ids[unique_ids != 0]
            
            if len(unique_ids) == 0:
                return []

            # Construct paths to source files based on time
            # Assumes source structure YYYY/MM/filename
            year_str = time_val.strftime("%Y")
            month_str = time_val.strftime("%m")
            day_str = time_val.strftime("%d")
            hour_str = time_val.strftime("%H")
            
            # Find matching Precip file
            # Pattern: 3B-HHR...V07B_YYYYMMDD_HH0000.nc (Update pattern if needed)
            precip_pattern = os.path.join(
                precip_dir, year_str, month_str, 
                f"*{year_str}{month_str}{day_str}*_{hour_str}*.{config['file_suffix']}"
            )
            precip_files = glob.glob(precip_pattern)
            
            # Find matching LI file
            li_pattern = os.path.join(
                li_dir, year_str, month_str,
                f"*{year_str}{month_str}{day_str}*{hour_str}*.{config['file_suffix']}"
            )
            li_files = glob.glob(li_pattern)

            if not precip_files or not li_files:
                logger.warning(f"Missing source data for {time_val}. Skipping metrics.")
                return []

            # Load Data using input_output functions
            # Note: We pass time_index=0 as these are usually hourly files
            _, lat2d, lon2d, _, _, prec_data = load_precipitation_data(
                precip_files[0], config['precip_var_name'], config['lat_name'], config['lon_name']
            )
            _, _, _, _, _, li_data = load_lifted_index_data(
                li_files[0], config['liting_index_var_name'], config['lat_name'], config['lon_name']
            )

            # Calculate metrics for each ID
            for tid in unique_ids:
                mask = (mcs_map == tid)
                area_cells = np.sum(mask)
                area_km2 = area_cells * config['grid_cell_area_km2']
                
                # Instability
                li_vals = li_data.values[mask]
                mean_li = np.nanmean(li_vals) if len(li_vals) > 0 else np.nan
                
                # Center for Straightness
                # We use precipitation weighted center for accuracy
                c_lat, c_lon = calculate_weighted_center(prec_data.values, mask, lat2d, lon2d)

                results.append({
                    'track_id': tid,
                    'time': time_val,
                    'area_km2': area_km2,
                    'mean_li': mean_li,
                    'lat': c_lat,
                    'lon': c_lon
                })
        
        return results

    except Exception as e:
        logger.error(f"Error processing stats for {tracking_file}: {e}")
        return []

def filter_tracks(df_timesteps, config):
    """
    Aggregates timestep data into track summaries and applies filters.
    Returns: set of valid_track_ids
    """
    # 1. Group by Track
    grouped = df_timesteps.groupby('track_id')
    
    valid_ids = []
    
    thresholds = config['postprocessing_filters']
    
    for tid, group in grouped:
        group = group.sort_values('time')
        
        # --- A. Instability Filter ---
        # Mean LI over the entire lifetime
        lifetime_mean_li = group['mean_li'].mean()
        if lifetime_mean_li >= thresholds['lifted_index_threshold']:
            continue # Reject (Stable environment)

        # --- B. Area Volatility Filter ---
        if len(group) < 2:
            # Single timestep tracks cannot calculate volatility or straightness well
            # Usually rejected, or accepted if strictness is low. 
            # Assuming we reject single-hour blips for "Robust MCS"
            continue 

        areas = group['area_km2'].values
        # (A_next - A_curr)^2 / (mean_A)
        # Using numpy shifts for speed
        prev_area = areas[:-1]
        curr_area = areas[1:]
        
        diff_sq = (curr_area - prev_area)**2
        mean_area_step = (curr_area + prev_area) / 2.0
        
        # Avoid div by zero
        volatility = np.divide(diff_sq, mean_area_step, out=np.zeros_like(diff_sq), where=mean_area_step!=0)
        
        # Pad with 0 for the first timestep to match length
        max_volatility = np.max(volatility)
        
        if max_volatility > thresholds['max_area_volatility']:
            continue # Reject (Unstable growth/decay)

        # --- C. Straightness Filter ---
        lats = group['lat'].values
        lons = group['lon'].values
        
        # Total distance (sum of steps)
        total_dist = 0.0
        for i in range(len(lats)-1):
            dist = great_circle((lats[i], lons[i]), (lats[i+1], lons[i+1])).kilometers
            total_dist += dist
            
        # Net displacement (start to end)
        net_dist = great_circle((lats[0], lons[0]), (lats[-1], lons[-1])).kilometers
        
        straightness = (net_dist / total_dist) if total_dist > 0 else 1.0
        
        if straightness <= thresholds['track_straightness_threshold']:
            continue # Reject (Erratic movement)
            
        # If we reached here, the track is valid
        valid_ids.append(tid)
        
    return set(valid_ids)

def apply_filter_to_files(raw_files, valid_ids, output_dir):
    """
    Reads raw NC files, masks out invalid IDs, and saves to output_dir
    maintaining compression and integer encoding.
    """
    logger.info(f"Writing {len(raw_files)} filtered files to {output_dir}...")
    
    # Define encoding to ensure output is compressed and optimized
    # matches save_tracking_result in input_output.py
    int_encoding = {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": 0, "dtype": "int32"}
    float_encoding = {"zlib": True, "complevel": 4, "dtype": "float32"}

    encoding = {
        "robust_mcs_id": int_encoding,
        "mcs_id": int_encoding,
        "mcs_id_merge_split": int_encoding,
        "latitude": float_encoding,
        "longitude": float_encoding,
        "active_track_id": {"dtype": "int32"},
        "active_track_lat": float_encoding,
        "active_track_lon": float_encoding,
        # Don't forget 'crs' if you added it!
    }
    
    for f in raw_files:
        try:
            with xr.open_dataset(f) as ds:
                # Load data to memory to modify it
                ds.load()
                
                # Arrays to mask
                id_vars = ['mcs_id', 'robust_mcs_id', 'mcs_id_merge_split']
                
                for var in id_vars:
                    if var in ds:
                        data = ds[var].values
                        # Mask: Where ID is NOT in valid_ids, set to 0
                        mask_invalid = ~np.isin(data, list(valid_ids))
                        data[mask_invalid] = 0
                        ds[var].values = data
                
                # Also filter the 'active_track_...' tabular variables
                # so the list of centers matches the grids
                if 'active_track_id' in ds:
                    active_ids = ds['active_track_id'].values
                    # Keep only those in valid_ids
                    valid_indices = np.isin(active_ids, list(valid_ids))
                    
                    # Subset the active track variables
                    ds = ds.isel(tracks=valid_indices)

                # Prepare output path
                time_val = pd.to_datetime(ds.time.values[0])
                year_str = time_val.strftime("%Y")
                month_str = time_val.strftime("%m")
                
                out_subdir = os.path.join(output_dir, year_str, month_str)
                os.makedirs(out_subdir, exist_ok=True)
                
                out_name = os.path.basename(f)
                out_path = os.path.join(out_subdir, out_name)
                
                # Update attributes
                ds.attrs['postprocessing_level'] = 'Filtered (LI, Straightness, Volatility)'
                
                # Save with specific encoding
                ds.to_netcdf(out_path, encoding=encoding)
                
        except Exception as e:
            logger.error(f"Failed to filter/save {f}: {e}")

def run_postprocessing_year(
    year, 
    raw_tracking_dir, 
    filtered_output_dir, 
    config, 
    n_cores
):
    """
    Main orchestrator for a specific year.
    """
    logger.info(f"--- Starting Post-Processing for Year {year} ---")
    
    # 1. Find Raw Tracking Files
    search_pattern = os.path.join(raw_tracking_dir, str(year), "**", "tracking_*.nc")
    raw_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not raw_files:
        logger.warning(f"No raw tracking files found for {year} in {raw_tracking_dir}")
        return

    # 2. Extract Metrics (Parallel)
    logger.info("Extracting track metrics...")
    all_timestep_rows = []
    
    # Pre-calculate source directories to avoid passing full config if possible
    precip_dir = config['precip_data_directory']
    li_dir = config['lifted_index_data_directory'] # Note: Check spelling in your config!
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit tasks
        futures = {
            executor.submit(
                process_single_timestep, f, precip_dir, li_dir, config
            ): f for f in raw_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            rows = future.result()
            all_timestep_rows.extend(rows)
            
    if not all_timestep_rows:
        logger.warning("No valid metrics extracted. Skipping filtering.")
        return
        
    df_timesteps = pd.DataFrame(all_timestep_rows)
    
    # 3. Determine Valid IDs
    logger.info("Applying physical filters...")
    valid_ids = filter_tracks(df_timesteps, config)
    
    logger.info(f"Year {year}: {len(valid_ids)} valid MCS tracks retained.")
    
    # 4. Filter and Save NetCDFs
    apply_filter_to_files(raw_files, valid_ids, filtered_output_dir)
    
    logger.info(f"Post-processing for {year} complete.")