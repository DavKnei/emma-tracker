import numpy as np


def filter_main_mcs(mcs_ids_list, main_mcs_ids):
    """
    Filters tracking results to include only the 'main' MCS IDs.

    Args:
        mcs_ids_list (List[numpy.ndarray]): List of 2D arrays with track IDs.
        main_mcs_ids (List[int]): List of track IDs considered 'main' MCS.

    Returns:
        List[numpy.ndarray]: A new list of arrays, where IDs not in main_mcs_ids are set to 0.
    """
    filtered_mcs_id_list = []
    for mcs_id_array in mcs_ids_list:
        filtered_array = mcs_id_array.copy()
        mask = ~np.isin(filtered_array, main_mcs_ids)
        filtered_array[mask] = 0
        filtered_mcs_id_list.append(filtered_array)
    return filtered_mcs_id_list


def filter_relevant_systems(
    mcs_ids_list, main_mcs_ids, merging_events, splitting_events
):
    """Expands a set of main MCS tracks to include their full 'family tree'.

    This function identifies all systems that are directly or indirectly connected
    to a 'main' MCS through merging or splitting events. It begins with the
    provided list of main MCS track IDs and iteratively searches for all parent
    systems that merged into them and all child systems that split from them.
    This process repeats to find the entire lineage (e.g., grandparents,
    grandchildren), ensuring that the complete history of all relevant convective
    elements is preserved.

    The final output is a series of 2D arrays where any track that is not
    part of this 'family tree' is removed (i.e., set to 0).

    Args:
        mcs_ids_list (List[np.ndarray]): The complete, unfiltered list of 2D
            track ID arrays, one for each timestep, as produced by the initial
            tracking logic.
        main_mcs_ids (List[int]): A list of integer track IDs that have already
            been identified as 'main' MCSs based on area and lifetime criteria.
            This is the starting set of relevant systems.
        merging_events (List[MergingEvent]): A list of all MergingEvent objects
            recorded during the tracking process.
        splitting_events (List[SplittingEvent]): A list of all SplittingEvent
            objects recorded during the tracking process.

    Returns:
        List[np.ndarray]: A new list of 2D arrays containing the filtered
            track IDs. In these arrays, any track ID not part of the 'family
            tree' of a main MCS has been set to 0.

    Usage:
        # After running the main tracking function:
        (
            robust_mcs_id, main_mcs_id, _, lifetime_list, time_list, lat, lon,
            merging_events, splitting_events, _
        ) = track_mcs(...)

        # To get the full family tree based on the list of main MCS IDs:
        main_mcs_ids_list = [np.unique(arr[arr > 0]) for arr in main_mcs_id]
        unique_main_ids = np.unique(np.concatenate(main_mcs_ids_list))

        full_family_tree = filter_relevant_systems(
            mcs_ids_list, # The original, unfiltered list from tracking
            unique_main_ids,
            merging_events,
            splitting_events
        )
    """
    # Start with the main MCS IDs.
    relevant_ids = set(main_mcs_ids)

    # Iteratively add parent/child IDs from merging/splitting events
    # connected to the set of relevant IDs. This ensures the full family tree is found.
    for _ in range(
        10
    ):  # Iterate a few times to catch multi-step parent/child relationships
        new_ids_found = False
        # Add IDs involved in merging events
        for event in merging_events:
            # If the child is relevant, all parents become relevant
            if event.child_id in relevant_ids:
                for pid in event.parent_ids:
                    if pid not in relevant_ids:
                        relevant_ids.add(pid)
                        new_ids_found = True

        # Add IDs involved in splitting events
        for event in splitting_events:
            # If the parent is relevant, all children become relevant
            if event.parent_id in relevant_ids:
                for cid in event.child_ids:
                    if cid not in relevant_ids:
                        relevant_ids.add(cid)
                        new_ids_found = True

        if not new_ids_found:
            break

    # Filter each timestep's array: only keep values in relevant_ids.
    filtered_mcs_ids_list = []
    for mcs_array in mcs_ids_list:
        filtered_array = mcs_array.copy()
        # Any pixel not in relevant_ids is set to 0.
        mask = ~np.isin(filtered_array, list(relevant_ids))
        filtered_array[mask] = 0
        filtered_mcs_ids_list.append(filtered_array)

    return filtered_mcs_ids_list


def apply_li_filter(
    mcs_ids_list, lifted_index_regions_list, time_list, main_lifetime_thresh
):
    """
    Post‑filter MCS tracks by lifted_index_regions (0/1).
    A track passes if any LI==1 occurs within two hours before or at its start,
    or if LI==1 first appears later and the remaining track length ≥ main_lifetime_thresh.

    Returns a new list of 2D arrays (same shape as mcs_ids_list) called main_mcs_id_robust.
    """
    robust_ids = [np.zeros_like(arr, dtype=int) for arr in mcs_ids_list]
    track_ids = np.unique(
        np.concatenate([arr[arr > 0].ravel() for arr in mcs_ids_list])
    )
    track_ids = track_ids[track_ids != 0]

    # Build per-track time indices
    for tid in track_ids:
        # Find all time steps where this track exists
        times_present = [i for i, arr in enumerate(mcs_ids_list) if (arr == tid).any()]

        if not times_present:
            continue
        t0 = times_present[0]
        # Define pre-window (two hours before t0)
        window = list(range(max(0, t0 - 2), t0 + 1))

        # Check for any convective LI in that window
        found = False
        for ti in window:
            mask = mcs_ids_list[ti] == tid
            if mask.any() and (lifted_index_regions_list[ti][mask] == 1).any():
                onset = t0
                found = True
                break

        # If not found, look for first later LI==1
        if not found:
            for ti in times_present:
                mask = mcs_ids_list[ti] == tid
                if mask.any() and (lifted_index_regions_list[ti][mask] == 1).any():
                    onset = ti
                    found = True
                    break

        if not found:
            # No LI==1 anywhere → discard entire track
            continue

        # Ensure remaining lifetime ≥ threshold
        remaining = len([i for i in times_present if i >= onset])
        if remaining < main_lifetime_thresh:
            continue

        # Mark robust IDs from onset onward
        for ti in times_present:
            if ti >= onset:
                robust_ids[ti][mcs_ids_list[ti] == tid] = tid

    return robust_ids
