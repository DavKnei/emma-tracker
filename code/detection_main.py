import numpy as np
import logging
from input_output import load_precipitation_data, load_lifted_index_data
from detection_helper_func import (
    smooth_precipitation_field,
    detect_cores_connected,
    morphological_expansion_with_merging,
    unify_checkerboard_simple,
)
from detection_filter_func import (
    filter_mcs_candidates,
    lifted_index_filter,
    compute_cluster_centers_of_mass,
)


def detect_mcs_in_file(
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
    lifted_index_percentage,
    time_index=0,
):
    """
    Detect MCSs in a single file.

    Parameters:
    - precip_file_path: Path to the NetCDF file containing precipitation data.
    - precip_data_var: Variable name of detected precipitation variable.
    - lifted_index_file_path: Path to the NetCDF file containing the lifted index data.
    - lifted_index_data_var: Variable name of the lifted index.
    - heavy_precip_threshold: Threshold for heavy precipitation (mm/h).
    - moderate_precip_threshold: Threshold for moderate precipitation (mm/h).
    - min_size_threshold: Minimum size threshold for clusters (number of grid cells).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - lifted_index_percentage: Percentage of Area that needs to fullfil the lifted_index criteria.
    - time_index: Index of the time step to process.

    Returns:
    - detection_result: Dictionary containing detection results.
    """
    logger = logging.getLogger(__name__)
    lifted_index_regions = None
    
    # Load data
    ds, lat2d, lon2d, lat, lon, precipitation = load_precipitation_data(
        precip_file_path, precip_data_var, lat_name, lon_name, time_index
    )


    # Initialize lifted_index_regions as an array of zeros
    # This ensures it always has the correct shape and type for your output format.
    lifted_index_threshold = -2
    lifted_index_regions = np.ones_like(precipitation, dtype=np.int32) * lifted_index_threshold

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

    # Step 4: Filter MCS candidates based on number of convective plumes, size and lifted index
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

    if lifted_index_file_path and lifted_index_file_path.strip(): # Check if path is not None and not just empty spaces
        logger.info("Lifting index file provided. Applying filter...")
        # Load the lifted index data
        ds_li, _, _, _, _, lifted_index = load_lifted_index_data(
            lifted_index_file_path, lifted_index_data_var, lat_name, lon_name, time_index
        )
        # Apply the filter
        lifted_index_regions = lifted_index_filter(
            ds_li[lifted_index_data_var].values,
            final_labeled_regions,
            lifted_index_percentage,
            lifted_index_threshold=lifted_index_threshold,
        )
    else:
        lifted_index_regions = np.zeros_like(final_labeled_regions, dtype=np.uint8)
        logger.info("No lifted index file provided. Skipping filter.")

    # Step 5: Compute cluster centers of mass
    cluster_centers = compute_cluster_centers_of_mass(
        final_labeled_regions, lat2d, lon2d, precipitation
    )

    # Prepare detection result
    detection_result = {
        "final_labeled_regions": final_labeled_regions,
        "lifted_index_regions": lifted_index_regions,
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
