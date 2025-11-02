import numpy as np
import logging
from input_output import load_precipitation_data, load_lifting_index_data
from detection_helper_func import (
    smooth_precipitation_field,
    detect_cores_connected,
    morphological_expansion_with_merging,
    unify_checkerboard_simple,
)
from detection_filter_func import (
    filter_mcs_candidates,
    lifting_index_filter,
    compute_cluster_centers_of_mass,
)


def detect_mcs_in_file(
    precip_file_path,
    precip_data_var,
    lifting_index_file_path,
    lifting_index_data_var,
    lat_name,
    lon_name,
    heavy_precip_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    lifting_index_percentage,
    grid_spacing_km,
    time_index=0,
):
    """
    Detect MCSs in a single file.

    Parameters:
    - precip_file_path: Path to the NetCDF file containing precipitation data.
    - precip_data_var: Variable name of detected precipitation variable.
    - lifting_index_file_path: Path to the NetCDF file containing the lifting index data.
    - lifting_index_data_var: Variable name of the lifting index.
    - heavy_precip_threshold: Threshold for heavy precipitation (mm/h).
    - moderate_precip_threshold: Threshold for moderate precipitation (mm/h).
    - min_size_threshold: Minimum size threshold for clusters (number of grid cells).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - lifting_index_percentage: Percentage of Area that needs to fullfil the lifting_index criteria.
    - grid_spacing_km: Approximate grid spacing in kilometers.
    - time_index: Index of the time step to process.

    Returns:
    - detection_result: Dictionary containing detection results.
    """
    logger = logging.getLogger(__name__)
    lifting_index_regions = None
    
    # Load data
    ds, lat2d, lon2d, lat, lon, precipitation = load_precipitation_data(
        precip_file_path, precip_data_var, lat_name, lon_name, time_index
    )


    # Initialize lifting_index_regions as an array of zeros
    # This ensures it always has the correct shape and type for your output format.
    lifting_index_threshold = -2
    lifting_index_regions = np.ones_like(precipitation, dtype=np.int32) * lifting_index_threshold

    # Step 1: Smooth the precipitation field
    precipitation_smooth = smooth_precipitation_field(precipitation)

    # Step 2: Detect heavy precipitation cores with connected component labeling
    core_labels = detect_cores_connected(
        precipitation_smooth,
        core_thresh=heavy_precip_threshold,
        min_cluster_size=4,  # Min number of points in a cluster
    )

    # Step 3: Morphological expansion with merging
    expanded_labels = morphological_expansion_with_merging(
        core_labels,
        precipitation_smooth,
        expand_threshold=moderate_precip_threshold,
        max_iterations=400,
    )

    expanded_labels = unify_checkerboard_simple(
        expanded_labels,
        precipitation_smooth,
        threshold=moderate_precip_threshold,
        max_passes=10,
    )

    # Step 4: Filter MCS candidates based on number of convective plumes, size and lifting index
    mcs_candidate_labels = filter_mcs_candidates(
        expanded_labels,
        core_labels,
        min_size_threshold,
        min_nr_plumes
    )

    # Create final labeled regions for MCS candidates
    final_labeled_regions = np.where(
        np.isin(expanded_labels, mcs_candidate_labels), expanded_labels, 0
    )

    if lifting_index_file_path and lifting_index_file_path.strip(): # Check if path is not None and not just empty spaces
        logger.info("Lifting index file provided. Applying filter...")
        # Load the lifting index data
        ds_li, _, _, _, _, lifting_index = load_lifting_index_data(
            lifting_index_file_path, lifting_index_data_var, lat_name, lon_name, time_index
        )
        # Apply the filter
        lifting_index_regions = lifting_index_filter(
            ds_li[lifting_index_data_var].values,
            final_labeled_regions,
            lifting_index_percentage,
            lifting_index_threshold=lifting_index_threshold,
        )
    else:
        lifting_index_regions = np.zeros_like(final_labeled_regions, dtype=np.uint8)
        logger.info("No lifting index file provided. Skipping filter.")

    # Step 5: Compute cluster centers of mass
    cluster_centers = compute_cluster_centers_of_mass(
        final_labeled_regions, lat2d, lon2d, precipitation
    )

    # Prepare detection result
    detection_result = {
        "final_labeled_regions": final_labeled_regions,
        "lifting_index_regions": lifting_index_regions,
        "lat2d": lat2d,
        "lon2d": lon2d,
        "lat": lat,
        "lon": lon,
        "precipitation": precipitation_smooth,
        "time": ds["time"].values,
        "convective_plumes": core_labels,
        "center_points": cluster_centers,
    }

    logger.info(f"MCS detection completed for {precip_file_path}.")
    return detection_result
