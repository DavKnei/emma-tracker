# main.py

import os
import glob
import gc
import concurrent.futures
import argparse
import yaml
import sys
import logging
import pandas as pd

from .detection_main import detect_mcs_in_file
from .tracking_main import track_mcs
from .input_output import (
    setup_logging,
    handle_exception,
    filter_files_by_date,
    group_files_by_year,
    save_detection_result,
    load_individual_detection_files,
    save_tracking_result,
)
from .postprocessing import run_postprocessing_year

# Set a global exception handler to log uncaught exceptions
sys.excepthook = handle_exception


def process_file(
    precip_file_path,
    precip_data_var,
    lifted_index_file_path,
    lifted_index_data_var,
    lat_name,
    lon_name,
    heavy_precip_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    lifted_index_percentage
):
    """
    Wrapper function to run MCS detection for a single file.
    This is used as the target for parallel processing.
    """
    result = detect_mcs_in_file(
        precip_file_path,
        precip_data_var,
        lifted_index_file_path,
        lifted_index_data_var,
        lat_name,
        lon_name,
        heavy_precip_threshold,
        moderate_precip_threshold,
        min_size_threshold,
        min_nr_plumes,
        lifted_index_percentage
    )
    return result


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MCS Detection and Tracking")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    return parser.parse_args()


def main():
    """
    Main execution script for MCS detection and tracking.

    The workflow is structured as follows:
    1. Load configuration from a YAML file.
    2. Group input data files by year.
    3. Loop through each year for processing:
        a. Run MCS detection for every timestep, saving results to hourly files.
        b. Load the year's detection results back into memory.
        c. Run the tracking algorithm on the full year's data.
        d. Apply the postprocessing filter to the raw tracking data.
        d. Save the final tracking results to hourly files.
    This yearly batch approach ensures scalability for multi-year datasets.
    """
    # --- 1. SETUP AND CONFIGURATION ---
    logger = logging.getLogger(__name__)
    args = parse_arguments()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # General parameters
    precip_data_dir = config["precip_data_directory"]
    file_suffix = config["file_suffix"]
    detection_output_path = config["detection_output_path"]
    raw_tracking_output_dir = config["raw_tracking_output_dir"]  # raw tracking output
    tracking_output_dir = config["filtered_tracking_output_dir"]  # final filtered tracking output
    precip_data_var = config["precip_var_name"]
    lat_name = config["lat_name"]
    lon_name = config["lon_name"]
    data_source = config["data_source"]

    # Read optional date filtering parameters ---
    years_to_process = config.get("years", [])
    months_to_process = config.get("months", [])

    # Detection parameters
    min_size_threshold = config.get("min_size_threshold", 10)
    heavy_precip_threshold = config.get("heavy_precip_threshold", 7)
    moderate_precip_threshold = config.get("moderate_precip_threshold", 1)
    min_nr_plumes = config.get("min_nr_plumes", 1)
    lifted_index_percentage = config.get("lifted_index_percentage_threshold", 0.2)

    # Tracking parameters
    main_lifetime_thresh = config.get("main_lifetime_thresh", 4)
    main_area_thresh = config.get("main_area_thresh", 3000)
    nmaxmerge = config.get("nmaxmerge", 5)

    # Operational parameters
    USE_MULTIPROCESSING = config.get("use_multiprocessing", False)
    NUMBER_OF_CORES = config.get("number_of_cores", 1)
    RUN_DETECTION = config.get("detection", True)
    USE_LIFTED_INDEX = config.get("use_lifted_index", True)
    RUN_TRACKING = config.get("tracking", True)
    RUN_POSTPROCESSING = config.get("postprocessing", True)

    # Create output directories
    os.makedirs(detection_output_path, exist_ok=True)
    os.makedirs(raw_tracking_output_dir, exist_ok=True)

    # Track the file mode for each phase.
    # 'w' = overwrite (start fresh), 'a' = append (add to existing)
    log_modes = {
        "detection": "w",
        "tracking": "w",
        "postprocessing": "w"
    }

    # Initialize logging based on the starting phase
    if RUN_DETECTION:
        setup_logging(detection_output_path, filename="detection.log", mode=log_modes["detection"])
        log_modes["detection"] = "a" 
    elif RUN_TRACKING:
        setup_logging(raw_tracking_output_dir, filename="tracking.log", mode=log_modes["tracking"])
        log_modes["tracking"] = "a"
    elif RUN_POSTPROCESSING:
        os.makedirs(tracking_output_dir, exist_ok=True)
        setup_logging(tracking_output_dir, filename="postprocessing.log", mode=log_modes["postprocessing"])
        log_modes["postprocessing"] = "a"

    # --- 2. FIND, FILTER, AND GROUP INPUT FILES ---
    files_by_year = {}
    li_files_by_year = {}

    if RUN_DETECTION:
        # Find all files recursively
        all_precip_files = sorted(
            glob.glob(
                os.path.join(precip_data_dir, "**", f"*{file_suffix}"), recursive=True
            )
        )
        if not all_precip_files:
            raise FileNotFoundError("Precipitation data directory is empty. Exiting.")
        logger.info(
            f"Found {len(all_precip_files)} total precipitation files in source directory."
        )

        # Apply the date filter
        filtered_precip_files = filter_files_by_date(
            all_precip_files, years_to_process, months_to_process
        )
        logger.info(
            f"After filtering by year/month, {len(filtered_precip_files)} files remain for processing."
        )

        # Now, group the filtered list by year
        files_by_year = group_files_by_year(filtered_precip_files)

        if USE_LIFTED_INDEX:
            lifted_index_data_dir = config["lifted_index_data_directory"]
            lifted_index_data_var = config["liting_index_var_name"]

            all_li_files = sorted(
                glob.glob(
                    os.path.join(lifted_index_data_dir, "**", f"*{file_suffix}"),
                    recursive=True,
                )
            )
            if not all_li_files:
                raise FileNotFoundError("lifted index data directory is empty. Exiting.")
            logger.info(
                f"Found {len(all_li_files)} total lifted index files in source directory."
            )

            # Apply the same filter to the lifted index files
            filtered_li_files = filter_files_by_date(
                all_li_files, years_to_process, months_to_process
            )
            logger.info(
                f"After filtering, {len(filtered_li_files)} lifted index files remain."
            )

            li_files_by_year = group_files_by_year(filtered_li_files)
            
    # Determine the years to iterate over
    if RUN_DETECTION:
        years_to_iterate = sorted(files_by_year.keys())
    else:
        # If detection is skipped, determine years from config or existing output directories
        if years_to_process:
            years_to_iterate = sorted(years_to_process)
        else:
            years_to_iterate = []
            
            # Check detection output for existing year folders
            if os.path.exists(detection_output_path):
                subdirs = [d for d in os.listdir(detection_output_path) if os.path.isdir(os.path.join(detection_output_path, d))]
                for d in subdirs:
                    if d.isdigit():
                        years_to_iterate.append(int(d))
            
            # Check tracking output (useful if detection files were deleted to save space)
            if not years_to_iterate and os.path.exists(raw_tracking_output_dir):
                subdirs = [d for d in os.listdir(raw_tracking_output_dir) if os.path.isdir(os.path.join(raw_tracking_output_dir, d))]
                for d in subdirs:
                    if d.isdigit():
                        years_to_iterate.append(int(d))

            years_to_iterate = sorted(list(set(years_to_iterate)))
            
        if not years_to_iterate:
             logger.warning("Detection is off and no years specified in config or found in output directories. Nothing to do.")

    # --- 3. MAIN YEARLY PROCESSING LOOP ---
    for year in years_to_iterate:
        logger.info(f"--- Starting processing for year: {year} ---")
        
        # Retrieve files for the current year (empty if detection is skipped)
        precip_file_list_year = files_by_year.get(year, [])

        if USE_LIFTED_INDEX:
            li_files_year = li_files_by_year.get(year, [])
            if RUN_DETECTION and len(precip_file_list_year) != len(li_files_year):
                logger.warning(
                    f"Mismatch in file counts for {year}. Precip: {len(precip_file_list_year)}, LI: {len(li_files_year)}. Skipping year."
                )
                continue
        else:
            li_files_year = [None] * len(precip_file_list_year)
            lifted_index_data_var = None

        # --- 3a. DETECTION PHASE ---
        if RUN_DETECTION:
            # Configure logging for detection
            setup_logging(detection_output_path, filename="detection.log", mode=log_modes["detection"])
            log_modes["detection"] = "a" 
            
            logger.info(f"Running detection for {len(precip_file_list_year)} files in {year}...")

            if USE_MULTIPROCESSING:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=NUMBER_OF_CORES
                ) as executor:
                    futures = [
                        executor.submit(
                            process_file,
                            precip_file,
                            precip_data_var,
                            li_file,
                            lifted_index_data_var,
                            lat_name,
                            lon_name,
                            heavy_precip_threshold,
                            moderate_precip_threshold,
                            min_size_threshold,
                            min_nr_plumes,
                            lifted_index_percentage
                        )
                        for precip_file, li_file in zip(
                            precip_file_list_year, li_files_year
                        )
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            detection_result = future.result()
                            save_detection_result(
                                detection_result, detection_output_path, data_source
                            )
                        except Exception as e:
                            logger.error(f"A detection task failed: {e}")
            else:
                for precip_file, li_file in zip(precip_file_list_year, li_files_year):
                    detection_result = detect_mcs_in_file(
                        precip_file,
                        precip_data_var,
                        li_file,
                        lifted_index_data_var,
                        lat_name,
                        lon_name,
                        heavy_precip_threshold,
                        moderate_precip_threshold,
                        min_size_threshold,
                        min_nr_plumes,
                        lifted_index_percentage
                    )
                    save_detection_result(
                        detection_result, detection_output_path, data_source
                    )
            logger.info(f"Detection for year {year} finished.")
            print(f"Detection for year {year} finished.")

        # --- 3b. & 3c. TRACKING PHASE ---
        if RUN_TRACKING:
            # Configure logging for tracking
            setup_logging(raw_tracking_output_dir, filename="tracking.log", mode=log_modes["tracking"])
            log_modes["tracking"] = "a"
            
            logger.info(f"Loading all detection files for year {year}...")

            year_detection_dir = os.path.join(detection_output_path, str(year))
            detection_results, grid_coords = load_individual_detection_files(
                year_detection_dir, USE_LIFTED_INDEX
            )

            # Apply month filter if specified
            if months_to_process and detection_results:
                original_count = len(detection_results)
                detection_results = [
                    res
                    for res in detection_results
                    if pd.to_datetime(res["time"]).month in months_to_process
                ]
                logger.info(
                    f"Filtered detection results for specified months. Kept {len(detection_results)} of {original_count} files."
                )

            if not detection_results:
                logger.warning(
                    f"No detection results found for year {year} (or for the specified months). Skipping tracking."
                )
                continue

            # Tracking Phase
            logger.info(f"Starting tracking for year {year}...")
            (
                robust_mcs_id,
                mcs_id,
                mcs_id_merge_split,
                lifetime_list,
                time_list,
                lat2d,
                lon2d,
                lat,
                lon,
                merging_events,
                splitting_events,
                tracking_centers_list,
            ) = track_mcs(
                detection_results,
                grid_coords,
                main_lifetime_thresh,
                main_area_thresh,
                nmaxmerge,
                use_li_filter=USE_LIFTED_INDEX,
            )
            logger.info(f"Tracking for year {year} finished.")
            print(f"Tracking for year {year} finished.")

            # Saving Phase
            logger.info(f"Saving individual hourly tracking files for year {year}...")
            for i in range(len(time_list)):
                # Package all data for this single timestep into a dictionary
                tracking_data_for_timestep = {
                    "robust_mcs_id": robust_mcs_id[i],
                    "mcs_id": mcs_id[i],
                    "mcs_id_merge_split": mcs_id_merge_split[i],
                    "lifetime": lifetime_list[i],
                    "time": time_list[i],
                    "lat2d": lat2d,
                    "lon2d": lon2d,
                    "lat": lat,
                    "lon": lon,
                    "tracking_centers": tracking_centers_list[i],
                }
                save_tracking_result(
                    tracking_data_for_timestep, raw_tracking_output_dir, data_source, config
                )
            
            del detection_results
            del mcs_id
            del robust_mcs_id
            del lifetime_list
            gc.collect() # Force garbage collection
            logger.info(f"--- Finished tracking for year: {year} ---")
            print(f"--- Finished tracking for year: {year} ---")


        # --- 3e. POST-PROCESSING PHASE ---
        if RUN_POSTPROCESSING:
            # Configure logging for post-processing
            os.makedirs(tracking_output_dir, exist_ok=True)
            setup_logging(tracking_output_dir, filename="postprocessing.log", mode=log_modes["postprocessing"])
            log_modes["postprocessing"] = "a"
            
            logger.info("Logging initialized for POST-PROCESSING phase.")
            try:
                run_postprocessing_year(
                    year,
                    raw_tracking_output_dir,
                    tracking_output_dir,
                    config,
                    NUMBER_OF_CORES
                )
            except Exception as e:
                logger.error(f"Post-processing failed for year {year}: {e}")

        logger.info(f"--- Finished processing for year: {year} ---")

    logger.info("All processing completed successfully.")
    print("All processing completed successfully.")


if __name__ == "__main__":
    main()