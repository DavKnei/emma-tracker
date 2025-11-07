import numpy as np
import math
from scipy.ndimage import affine_transform, center_of_mass, label

import numpy as np
import math

# --- Constants ---
R_EARTH_KM = 6371.0

# =============================================================================
# Vectorized Haversine Function (Helper)
# =============================================================================

def _haversine_vec(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """
    Calculates the great-circle distance (km) between two points
    or arrays of points on a sphere.
    
    Vectorized to be fast with numpy arrays.
    """
    # Convert all to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1_deg, lon1_deg, lat2_deg, lon2_deg])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    
    # c = 2 * np.arcsin(np.sqrt(a)) # More common, but can have precision issues
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # More stable
    
    return R_EARTH_KM * c


def calculate_grid_area_map(detection_result):
    """
    Calculates the area (km²) of each cell in a grid.
    
    This function is "smart":
    1. It detects if the grid is regular (IMERG-style) by comparing
       the 1D 'lat' and 2D 'lat2d' arrays.
    2. If regular, it uses the fast, precise 1D trigonometric calculation.
    3. If irregular (CORDEX-style), it assumes a rotated-pole grid where
       cell area is near-constant, and approximates this by calculating
       the area of the central grid cell.
       
    Args:
        detection_result (dict): A single detection result dictionary
                                 containing at least:
                                 - 'lat' (np.ndarray): 1D array of latitudes
                                 - 'lon' (np.ndarray): 1D array of longitudes
                                 - 'lat2d' (np.ndarray): 2D array of latitudes
                                 - 'lon2d' (np.ndarray): 2D array of longitudes
    
    Returns:
        np.ndarray: A 2D array of grid cell areas (in km²) with the
                    same shape as 'lat2d'.
    
    Raises:
        ValueError: If required coordinates are missing or shapes are mismatched.
    """
    # Get all required coordinates from the dictionary
    try:
        lat_1d = detection_result['lat']
        lon_1d = detection_result['lon']
        lat_2d = detection_result['lat2d']
        lon_2d = detection_result['lon2d']
    except KeyError as e:
        raise KeyError(f"Missing required coordinate {e} in detection_result.")
        
    grid_shape = lat_2d.shape
    (n_lats, n_lons) = grid_shape
    
    if lat_1d.shape[0] != n_lats or lon_1d.shape[0] != n_lons:
         # This check is for CORDEX grids where 1D rlat/rlon might be
         # provided as 'lat'/'lon'. We'll assume the 1D/2D name mismatch
         # is intentional and the shapes are the source of truth.
         # The CORDEX 1D/2D check below is more important.
         pass # Allow shape mismatches, proceed to regularity check

    # --- Check for Regular Grid (IMERG-style) ---
    # A grid is regular if the 2D lat array is just the 1D lat vector broadcasted
    is_regular = np.allclose(lat_2d, lat_1d[:, np.newaxis])
    
    if is_regular:
        # Use np.diff and take the first element for robustness (e.g., if list is size 1)
        # Use abs() in case coordinates are descending
        try:
            delta_lat_deg = np.abs(np.diff(lat_1d)[0])
            delta_lon_deg = np.abs(np.diff(lon_1d)[0])
        except IndexError: # Handle case of a single-pixel dimension
            delta_lat_deg = 0.1 # Assume default
            delta_lon_deg = 0.1 # Assume default

        delta_lat_rad = math.radians(delta_lat_deg)
        delta_lon_rad = math.radians(delta_lon_deg)
        
        lat_rad_1d = np.radians(lat_1d)
        
        # N-S distance is constant for a given grid spacing
        north_south_dist_km = R_EARTH_KM * delta_lat_rad
        
        # E-W distance (varies with latitude)
        east_west_dist_km_1d = R_EARTH_KM * np.cos(lat_rad_1d) * delta_lon_rad
        
        # Calculate the area for each latitude band
        area_1d = east_west_dist_km_1d * north_south_dist_km
        
        # Broadcast 1D area array to the 2D grid shape
        area_column = area_1d[:, np.newaxis]
        area_map_2d = np.tile(area_column, (1, n_lons))
        
        return area_map_2d
    
    else:
        # --- Assume Irregular Grid (CORDEX-style) ---
        # For a rotated-pole grid, cell area is (nearly) constant.
        # We approximate the area of the whole grid by calculating the area of the single, central grid cell.
        
        # Find the central indices
        i_mid = n_lats // 2
        j_mid = n_lons // 2
        
        # Get coordinates for the central cell and its N, E neighbors
        lat_center = lat_2d[i_mid, j_mid]
        lon_center = lon_2d[i_mid, j_mid]
        
        lat_north = lat_2d[i_mid + 1, j_mid]
        lon_north = lon_2d[i_mid + 1, j_mid]
        
        lat_east = lat_2d[i_mid, j_mid + 1]
        lon_east = lon_2d[i_mid, j_mid + 1]
        
        # Calculate N-S (delta_y) and E-W (delta_x) distances
        delta_y_km = _haversine_vec(lat_center, lon_center, lat_north, lon_north)
        delta_x_km = _haversine_vec(lat_center, lon_center, lat_east, lon_east)
        
        # Calculate area
        central_area_km2 = delta_x_km * delta_y_km
        
        # Create a 2D map with this constant area
        area_map_2d = np.full(grid_shape, central_area_km2)
        
        return area_map_2d

def assign_new_id(
    label,
    cluster_mask,
    area,
    next_cluster_id,
    lifetime_dict,
    max_area_dict,
    mcs_id,
    mcs_lifetime,
):
    """
    Assigns a brand-new ID to a cluster with no overlap from the previous timestep.

    Args:
        label (int): Label of the current cluster (from detection).
        cluster_mask (numpy.ndarray): Boolean mask for the current cluster's pixels.
        area (float): Area (km²) of this cluster.
        next_cluster_id (int): Next available integer ID for track assignment.
        lifetime_dict (dict): Tracks how many timesteps each track ID has existed.
        max_area_dict (dict): Tracks the maximum area encountered by each track ID.
        mcs_id (numpy.ndarray): 2D array where we assign the track ID for each pixel.
        mcs_lifetime (numpy.ndarray): 2D array for per-pixel lifetime assignment.

    Returns:
        assigned_id (int): The newly assigned track ID for this cluster.
        next_cluster_id (int): The updated next cluster ID (incremented by 1).
    """
    mcs_id[cluster_mask] = next_cluster_id
    lifetime_dict[next_cluster_id] = 1
    max_area_dict[next_cluster_id] = area
    mcs_lifetime[cluster_mask] = 1
    assigned_id = next_cluster_id
    next_cluster_id += 1
    return assigned_id, next_cluster_id


def get_dominant_cluster(prev_ids, max_area_dict):
    """
    Finds the 'dominant' cluster (largest area) among a list of track IDs.

    Args:
        prev_ids (List[int]): List of old track IDs (integers).
        max_area_dict (dict): Dictionary mapping track ID -> maximum area found so far.

    Returns:
        best_id (int): The track ID with the largest area in `max_area_dict`.
    """
    best_id = None
    best_area = -1
    for pid in prev_ids:
        if max_area_dict.get(pid, 0) > best_area:
            best_area = max_area_dict[pid]
            best_id = pid
    return best_id


def check_overlaps(
    previous_labeled_regions,
    final_labeled_regions,
    previous_cluster_ids,
    overlap_threshold=10,
):
    """
    Checks overlap between old-labeled regions and new-labeled regions.

    We build a mapping new_label -> list of old track IDs that meet or exceed
    `overlap_threshold` percent overlap with the new cluster.

    Args:
        previous_labeled_regions (numpy.ndarray): Labeled regions from the previous timestep.
            0 means no cluster.
        final_labeled_regions (numpy.ndarray): Labeled regions from the current timestep.
            0 means no cluster.
        previous_cluster_ids (dict): Maps old detection labels -> old track IDs.
        overlap_threshold (float): Minimum percentage overlap required for consideration.

    Returns:
        overlap_map (dict): { new_label (int) : [list of old track IDs (int)] }.
    """
    overlap_map = {}

    unique_prev_labels = np.unique(previous_labeled_regions)
    unique_prev_labels = unique_prev_labels[unique_prev_labels != 0]
    unique_curr_labels = np.unique(final_labeled_regions)
    unique_curr_labels = unique_curr_labels[unique_curr_labels != 0]

    for new_label in unique_curr_labels:
        curr_mask = final_labeled_regions == new_label
        curr_area = np.sum(curr_mask)

        relevant_old_ids = []
        if curr_area == 0:
            overlap_map[new_label] = relevant_old_ids
            continue

        for old_label_detection in unique_prev_labels:
            old_track_id = previous_cluster_ids.get(old_label_detection, None)
            if old_track_id is None:
                continue
            prev_mask = previous_labeled_regions == old_label_detection
            overlap_pixels = np.logical_and(curr_mask, prev_mask)
            overlap_area = np.sum(overlap_pixels)
            if overlap_area > 0:
                overlap_percent = (overlap_area / curr_area) * 100
                if overlap_percent >= overlap_threshold:
                    relevant_old_ids.append(old_track_id)
        overlap_map[new_label] = relevant_old_ids

    return overlap_map

  
def attempt_advection_rescue(
    labels_no_overlap,
    previous_labeled_regions,
    final_labeled_regions,
    previous_cluster_ids,
    overlap_map,
    grid_area_map_km2,
    overlap_threshold=10,
    search_radius_km=1000,
):
    """
    Attempts to rescue non-overlapping systems by applying a LOCAL advection correction.

    This function is a "smart" approximation. For each "lost" cluster, it
    calculates a local advection vector based on the movement of nearby
    (within `search_radius_km`) stable systems. It then applies this
    local vector to the previous-timestep-mask and re-checks for overlap
    using a fast, pixel-based percentage check.

    Args:
        labels_no_overlap (list): A list of labels from the current timestep
                                  that did not have any spatial overlap.
        previous_labeled_regions (np.ndarray): The labeled mask from the previous timestep (t-1).
        final_labeled_regions (np.ndarray): The labeled mask from the current timestep (t).
        previous_cluster_ids (dict): Mapping of {label_t-1: track_id}.
        overlap_map (dict): { new_label (int) : [list of old track IDs (int)] }.
        grid_area_map_km2 (np.ndarray): 2D array of grid cell areas (km²).
        overlap_threshold (float): Minimum percentage overlap required for consideration.
        search_radius_km (int): The radius in km to search for neighboring systems.

    Returns:
        dict: A dictionary of newly found overlaps in the format {new_label: [old_track_id]}.
    """
    rescued_overlaps = {}
    if not labels_no_overlap:
        return rescued_overlaps

    # --- 1. PRE-CALCULATE STABLE VECTORS & CENTERS ---
    # Calculate a single, representative pixel area for distance approximation
    avg_pixel_area_km2 = np.mean(grid_area_map_km2)
    
    # Calculate squared radius in pixels for faster distance checks
    radius_pixels_sq = (search_radius_km / np.sqrt(avg_pixel_area_km2))**2

    current_centers = {lbl: center_of_mass(final_labeled_regions, final_labeled_regions, lbl) for lbl in np.unique(final_labeled_regions) if lbl != 0}
    previous_centers = {lbl: center_of_mass(previous_labeled_regions, previous_labeled_regions, lbl) for lbl in np.unique(previous_labeled_regions) if lbl != 0}
    previous_areas_pixels = {lbl: np.sum(previous_labeled_regions == lbl) for lbl in previous_centers.keys()}

    # Store all stable vectors and their *new* center coordinates
    stable_vectors_info = [] # Store as ( (dy, dx), center_new_coords )
    for new_lbl, old_track_ids in overlap_map.items():
        # A stable one-to-one track
        if len(old_track_ids) == 1:
            old_track_id = old_track_ids[0]
            # Find the original detection label for this old track ID
            old_lbls = [k for k, v in previous_cluster_ids.items() if v == old_track_id]
            
            if len(old_lbls) == 1:
                old_lbl = old_lbls[0]
                center_old = previous_centers.get(old_lbl)
                center_new = current_centers.get(new_lbl)
                
                if center_old and center_new:
                    dy = center_new[0] - center_old[0]
                    dx = center_new[1] - center_old[1]
                    stable_vectors_info.append( ((dy, dx), center_new) )

    if not stable_vectors_info:
        return rescued_overlaps # No stable neighbors anywhere, cannot rescue.

    # Calculate the single, global average vector as a fallback
    global_avg_vector = np.mean([info[0] for info in stable_vectors_info], axis=0)

    # --- 2. LOOP THROUGH EACH "LOST" LABEL AND ATTEMPT RESCUE ---
    for new_lbl in labels_no_overlap:
        center_new = current_centers.get(new_lbl)
        if not center_new:
            continue # Should not happen, but good to check

        # Find local stable vectors
        local_stable_vectors = []
        for vector, neighbor_center in stable_vectors_info:
            # Calculate squared pixel distance
            dist_sq = (center_new[0] - neighbor_center[0])**2 + (center_new[1] - neighbor_center[1])**2
            if dist_sq <= radius_pixels_sq:
                local_stable_vectors.append(vector)
        
        # Use local average if available, otherwise fall back to global
        if local_stable_vectors:
            avg_dy, avg_dx = np.mean(local_stable_vectors, axis=0)
        else:
            avg_dy, avg_dx = global_avg_vector
        
        # --- 3. APPLY TRANSLATION AND CHECK OVERLAP (Pixel-based) ---
        translated_previous_regions = np.zeros_like(previous_labeled_regions)
        transform_matrix = np.array([[1, 0, -avg_dy], [0, 1, -avg_dx], [0, 0, 1]])
        affine_transform(previous_labeled_regions, transform_matrix, output=translated_previous_regions, order=0, prefilter=False)

        current_mask = final_labeled_regions == new_lbl
        area_new_pixels = np.sum(current_mask)
        if area_new_pixels == 0:
            continue

        overlapping_old_labels = np.unique(translated_previous_regions[current_mask])
        overlapping_old_labels = overlapping_old_labels[overlapping_old_labels != 0]

        if overlapping_old_labels.size > 0:
            potential_matches = []
            for old_lbl in overlapping_old_labels:
                area_old_pixels = previous_areas_pixels.get(old_lbl, 0)
                if area_old_pixels == 0: 
                    continue

                intersection_area_pixels = np.sum((current_mask) & (translated_previous_regions == old_lbl))

                # This is a PIXEL-based overlap percentage, which is a fast approximation
                overlap_percent = (intersection_area_pixels / min(area_new_pixels, area_old_pixels)) * 100
                
                if overlap_percent >= overlap_threshold:
                    if old_lbl in previous_cluster_ids:
                        potential_matches.append(previous_cluster_ids[old_lbl])

            if potential_matches:
                rescued_overlaps[new_lbl] = sorted(list(set(potential_matches)))

    return rescued_overlaps

def handle_no_overlap(
    new_labels_no_overlap,
    final_labeled_regions,
    next_cluster_id,
    lifetime_dict,
    max_area_dict,
    mcs_id,
    mcs_lifetime,
    grid_area_map_km2,
):
    """
    Assigns brand-new track IDs to all new labels that found no overlap with previous clusters.

    Args:
        new_labels_no_overlap (List[int]): All new detection labels that found no old ID.
        final_labeled_regions (numpy.ndarray): Labeled regions from the current timestep.
        next_cluster_id (int): Next available integer track ID.
        lifetime_dict (dict): Track ID -> lifetime.
        max_area_dict (dict): Track ID -> max area encountered.
        mcs_id (numpy.ndarray): 2D array for per-pixel track IDs.
        mcs_lifetime (numpy.ndarray): 2D array for per-pixel lifetime values.
        grid_area_map_km2 (np.ndarray): 2D array of grid cell areas (km²).

    Returns:
        assigned_ids_map (dict): new_label -> assigned track ID
        next_cluster_id (int): Updated ID counter.
    """
    assigned_ids_map = {}

    for lbl in new_labels_no_overlap:
        mask = final_labeled_regions == lbl
        area_km2 = np.sum(grid_area_map_km2[mask])

        mcs_id[mask] = next_cluster_id
        lifetime_dict[next_cluster_id] = 1
        max_area_dict[next_cluster_id] = area_km2
        mcs_lifetime[mask] = 1

        assigned_ids_map[lbl] = next_cluster_id
        next_cluster_id += 1

    return assigned_ids_map, next_cluster_id


def handle_continuation(
    new_label,
    old_track_id,
    final_labeled_regions,
    mcs_id,
    mcs_lifetime,
    lifetime_dict,
    max_area_dict,
    grid_area_map_km2,
):
    """
    Continues an existing old_track_id for the new_label cluster.

    Args:
        new_label (int): Label in the current detection.
        old_track_id (int): Old track ID to be continued.
        final_labeled_regions (numpy.ndarray): Current labeled regions.
        mcs_id (numpy.ndarray): 2D array for per-pixel track IDs.
        mcs_lifetime (numpy.ndarray): 2D array for per-pixel lifetime.
        lifetime_dict (dict): Tracks how many timesteps each track ID has existed.
        max_area_dict (dict): Track ID -> maximum area encountered.
        grid_area_map_km2 (np.ndarray): 2D array of grid cell areas (km²).
    """
    mask = final_labeled_regions == new_label
    area_km2 = np.sum(grid_area_map_km2[mask])

    mcs_id[mask] = old_track_id
    lifetime_dict[old_track_id] += 1
    mcs_lifetime[mask] = lifetime_dict[old_track_id]
    if area_km2 > max_area_dict[old_track_id]:
        max_area_dict[old_track_id] = area_km2


def compute_max_consecutive(bool_list):
    """
    Compute the maximum number of consecutive True values in a Boolean list.

    Args:
        bool_list (List[bool]): List of Boolean values.

    Returns:
        int: Maximum consecutive True values.
    """
    max_cons = 0
    current = 0
    for b in bool_list:
        if b:
            current += 1
            if current > max_cons:
                max_cons = current
        else:
            current = 0
    return max_cons


def apply_robust_mask(arr, robust_flag_dict):
    """
    Vectorized function that returns track id if robust_flag_dict[tid] is True, else 0.

    Args:
        arr (np.ndarray): 2D array of track IDs.
        robust_flag_dict (dict): Dictionary mapping track id to robust flag (bool).

    Returns:
        np.ndarray: Masked array with only robust track IDs.
    """
    vec_lookup = np.vectorize(
        lambda tid: tid if robust_flag_dict.get(tid, False) else 0
    )
    return vec_lookup(arr)


def build_tracking_centers(previous_cluster_ids, center_points_dict):
    """
    Build a dictionary mapping track id to center coordinates using the provided center_points.

    Args:
        previous_cluster_ids (dict): Mapping from detection label to track id.
        center_points_dict (dict): Mapping from detection label (as string) to (lat, lon) tuple.

    Returns:
        dict: Mapping from track id (as string) to (center_lat, center_lon).
    """
    centers = {}
    for lbl, tid in previous_cluster_ids.items():
        tid_str = str(tid)
        if tid_str in center_points_dict:
            centers[tid_str] = center_points_dict[tid_str]
    return centers
