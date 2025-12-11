import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import datetime
import json
import re
import sys
import logging
from collections import defaultdict


def get_dataset_encoding(ds):
    """
    Centralized encoding logic for all EMMA output files.
    Ensures consistency across detection, tracking, and post-processing.

    Fixes for ncview:
    1. 1D Coordinates (lat, lon, time) are UNCOMPRESSED (zlib=False).
    2. Data variables are COMPRESSED (zlib=True).
    3. Time uses a fixed epoch.
    4. Masks use _FillValue = -1 so 0 is visible as background.
    """
    encoding = {}

    # --- 1. Coordinate Encoding (Uncompressed, No Fill Value) ---
    coord_encoding = {"_FillValue": None, "zlib": False, "dtype": "float32"}

    # Standardize Time Epoch
    time_encoding = {
        "_FillValue": None,
        "zlib": False,
        "dtype": "float64",
        "units": "days since 1950-01-01 00:00:00",
        "calendar": "standard",
    }

    # --- 2. Data Encoding (Compressed) ---
    int_encoding = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "_FillValue": -1,  # Critical: 0 becomes valid background
        "dtype": "int32",
    }
    float_encoding = {"zlib": True, "complevel": 4, "dtype": "float32"}
    byte_encoding = {"dtype": "int8"}

    # Apply Rules based on Variable Existence
    if "time" in ds:
        encoding["time"] = time_encoding

    # 1D Coordinates
    for c in ["lat", "lon", "rlat", "rlon"]:
        if c in ds:
            encoding[c] = coord_encoding

    # 2D Coordinates (Auxiliary) -> Compressed to save space
    for c in ["latitude", "longitude"]:
        if c in ds:
            encoding[c] = float_encoding

    # Rotated Pole Container
    if "rotated_pole" in ds:
        encoding["rotated_pole"] = {"dtype": "int32"}

    # Gridded Data Variables
    grid_vars = [
        "final_labeled_regions",
        "lifted_index_regions",
        "robust_mcs_id",
        "mcs_id",
        "mcs_id_merge_split",
    ]
    for v in grid_vars:
        if v in ds:
            encoding[v] = int_encoding

    # Tabular (Track) Variables
    if "label_id" in ds:
        encoding["label_id"] = {"dtype": "int32"}
    if "label_lat" in ds:
        encoding["label_lat"] = float_encoding
    if "label_lon" in ds:
        encoding["label_lon"] = float_encoding

    if "active_track_id" in ds:
        encoding["active_track_id"] = {"dtype": "int32"}
    if "active_track_lat" in ds:
        encoding["active_track_lat"] = float_encoding
    if "active_track_lon" in ds:
        encoding["active_track_lon"] = float_encoding
    if "active_track_touches_boundary" in ds:
        encoding["active_track_touches_boundary"] = byte_encoding

    return encoding


def save_dataset_to_netcdf(ds, output_path):
    """
    Shared function to save any EMMA Xarray Dataset to NetCDF.
    Applies the standardized encoding and compression rules.
    """
    encoding = get_dataset_encoding(ds)
    ds.to_netcdf(output_path, encoding=encoding)


def _is_regular_grid(lat_1d, lon_1d, lat_2d):
    """
    Check if the grid is a regular lat/lon grid (IMERG/ERA5)
    or a rotated/curvilinear grid (CORDEX).

    Logic: If meshgrid(lon_1d, lat_1d) approximately equals lat_2d, it's regular.
    """
    try:
        # Quick check on shapes
        if lat_2d.shape != (len(lat_1d), len(lon_1d)):
            return False

        # Check values (using a slice to avoid memory overhead on large grids)
        # Check the first column of lat_2d against the 1D lat vector
        if not np.allclose(lat_2d[:, 0], lat_1d, atol=1e-4):
            return False

        return True
    except:
        return False


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log any uncaught exceptions.
    This is assigned to sys.excepthook in main.py.
    """
    # Don't log KeyboardInterrupt (Ctrl+C) as a critical error
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger(__name__)
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def setup_logging(output_dir, filename="mcs_tracking.log", mode="a"):
    """
    Configures logging. Removes old handlers to prevent duplicate messages.
    """
    log_filepath = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)

    # Shut down and remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # Add the new file handler
    file_handler = logging.FileHandler(log_filepath, mode=mode)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Ensure console output is still active
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def group_files_by_year(file_list):
    """
    Groups a list of file paths into a dictionary keyed by year.

    This function uses a regular expression to find a date in YYYYMMDD format
    within the filename, making it robust to different naming conventions.
    """
    files_by_year = defaultdict(list)
    # This regex looks for a sequence of 8 digits (YYYYMMDD)
    date_pattern = re.compile(r"(\d{8})")

    for f in file_list:
        basename = os.path.basename(f)
        match = date_pattern.search(basename)

        if match:
            date_str = match.group(1)

            # Convert the extracted 8-digit string to a datetime object
            year = pd.to_datetime(date_str, format="%Y%m%d").year
            files_by_year[year].append(f)

        else:
            logging.warning(
                f"Could not parse YYYYMMDD date from filename: {basename}. Skipping."
            )

    return files_by_year


def filter_files_by_date(file_list, years=None, months=None):
    """
    Filters a list of file paths based on specified years and/or months.

    The function extracts a YYYYMMDD date from each filename to perform the
    filtering. If `years` or `months` are empty or None, no filtering is
    applied for that criterion.

    Args:
        file_list (list): The initial list of file paths.
        years (list, optional): A list of integer years to include.
        months (list, optional): A list of integer months to include.

    Returns:
        list: A new list containing only the filtered file paths.
    """
    # If both filter lists are empty/None, no filtering is needed.
    if not years and not months:
        return file_list

    filtered_list = []
    date_pattern = re.compile(r"(\d{8})")
    logger = logging.getLogger(__name__)

    for f in file_list:
        basename = os.path.basename(f)
        match = date_pattern.search(basename)

        if match:
            date_str = match.group(1)
            timestamp = pd.to_datetime(date_str, format="%Y%m%d")

            # A file is kept if its year/month is in the respective list,
            # or if that list is empty (which means "accept all").
            year_ok = not years or timestamp.year in years
            month_ok = not months or timestamp.month in months

            if year_ok and month_ok:
                filtered_list.append(f)

    return filtered_list


def convert_precip_units(prec, target_unit="mm/h"):
    """
    Convert the precipitation DataArray to the target unit.

    Recognized unit conversions:
      - 'm', 'meter', 'metre': multiply by 1000 (assumed hourly accumulation)
      - 'kg m-2 s-1': multiply by 3600 (from mm/s to mm/h, given 1 kg/m² = 1 mm water)
      - 'mm', 'mm/h', 'mm/hr': no conversion needed

    Parameters:
    - prec: xarray DataArray of precipitation values.
    - target_unit: Desired unit for the output (default: "mm/h").

    Returns:
    - new_prec: DataArray with converted values and updated units attribute.
    """
    orig_units = prec.attrs.get("units", "").lower()

    if orig_units in ["m", "meter", "metre"]:
        factor = 1000.0
    elif orig_units in ["kg m-2 s-1"]:
        factor = 3600.0
    elif orig_units in ["mm", "mm/h", "mm/hr", "kg m-2"]:
        factor = 1.0
    else:
        print(
            f"Warning: Unrecognized precipitation units '{orig_units}'. No conversion applied."
        )
        factor = 1.0

    new_prec = prec * factor
    new_prec.attrs["units"] = target_unit
    return new_prec


def convert_lifted_index_units(li, target_unit="K"):
    """
    Convert the lifted index DataArray to the target unit.

    Recognized unit conversions:
      - degree Celcius to K

    Parameters:
    - li: xarray DataArray of precipitation values.
    - target_unit: Desired unit for the output (default: "K").

    Returns:
    - new_prec: DataArray with converted values and updated units attribute.
    """
    orig_units = li.attrs.get("units", "")

    if orig_units in ["K", "Kelvin"]:
        constant = 0
    elif orig_units in ["°C", "degree_Celcius"]:
        constant = 0  # Lifted index is a difference measure hence it doesnt matter
    else:
        print(
            f"Warning: Unrecognized lifted_index units '{orig_units}'. No conversion applied."
        )
        constant = 0  # Default to 0 to be safe

    new_li = li + constant
    new_li.attrs["units"] = target_unit
    return new_li


def load_precipitation_data(file_path, data_var, lat_name, lon_name, time_index=0):
    """
    Load the dataset and select the specified time step, scaling the precipitation
    variable to units of mm/h for consistency with the detection threshold.

    Parameters:
    - file_path: Path to the NetCDF file.
    - data_var: Name of the precipitation variable.
    - lat_name: Name of the latitude variable.
    - lon_name: Name of the longitude variable.
    - time_index: Index of the time step to select.

    Returns:
    - ds: xarray Dataset for the selected time.
    - lat2d: 2D array of latitudes.
    - lon2d: 2D array of longitudes.
    - lat: 1D array of latitudes.
    - lon: 1D array of latitudes
    - prec: 2D DataArray of precipitation values (scaled to mm/h).
    """
    ds = xr.open_dataset(file_path)
    ds = ds.isel(time=time_index)  # Select the specified time step
    ds["time"] = ds["time"].values.astype("datetime64[ns]")

    latitude = ds[lat_name].values
    longitude = ds[lon_name].values

    # Ensure lat and lon are 2D arrays.
    if latitude.ndim == 1 and longitude.ndim == 1:
        lon2d, lat2d = np.meshgrid(longitude, latitude)
        lon, lat = longitude, latitude
    else:
        lat2d, lon2d = latitude, longitude
        # Assume a regular grid and extract 1D coordinates. Be carefull with CORDEX
        lat = lat2d[:, 0]
        lon = lon2d[0, :]

    prec = ds[str(data_var)]
    prec_converted = convert_precip_units(prec)
    return ds, lat2d, lon2d, lat, lon, prec_converted


def load_lifted_index_data(file_path, data_var, lat_name, lon_name, time_index=0):
    """
    Load the dataset and select the specified time step, scaling the lifted_index data
    variable to units of K for consistency with the detection threshold.

    Parameters:
    - file_path: Path to the NetCDF file.
    - data_var: Name of the precipitation variable.
    - lat_name: Name of the latitude variable.
    - lon_name: Name of the longitude variable.
    - time_index: Index of the time step to select.

    Returns:
    - ds: xarray Dataset for the selected time.
    - lat2d: 2D array of latitudes.
    - lon2d: 2D array of longitudes.
    - lat: 1D array of latitudes.
    - lon: 1D array of latitudes
    - prec: 2D DataArray of precipitation values (scaled to mm/h).
    """
    ds = xr.open_dataset(file_path)
    ds = ds.isel(time=time_index)  # Select the specified time step
    ds["time"] = ds["time"].values.astype("datetime64[ns]")

    latitude = ds[lat_name].values
    longitude = ds[lon_name].values

    # Ensure lat and lon are 2D arrays.
    if latitude.ndim == 1 and longitude.ndim == 1:
        lon2d, lat2d = np.meshgrid(longitude, latitude)
        lon, lat = longitude, latitude
    else:
        lat2d, lon2d = latitude, longitude
        # Assume a regular grid and extract 1D coordinates. Be carefull with CORDEX
        lat = lat2d[:, 0]
        lon = lon2d[0, :]

    li = ds[str(data_var)]
    # Convert the lifted index data to K using the separate conversion function.
    li_converted = convert_lifted_index_units(li, target_unit="K")

    # Remove all non relevant data variables from dataset
    data_vars_list = [data_var for data_var in ds.data_vars]
    data_vars_list.remove(data_var)
    ds = ds.drop_vars(data_vars_list)

    return ds, lat2d, lon2d, lat, lon, li_converted


def serialize_center_points(center_points):
    """Convert a center_points dict with float32 lat/lon to Python floats so json.dumps() works."""
    casted_dict = {}
    for label_str, (lat_val, lon_val) in center_points.items():
        # Convert float32 -> float
        casted_dict[label_str] = (float(lat_val), float(lon_val))
    return json.dumps(casted_dict)


# emma/input_output.py

# ... existing imports ...


def load_individual_detection_files(year_input_dir, use_li_filter):
    """
    Load a sequence of detection result NetCDF files.
    Optimized: Loads coordinates only once to save memory.

    Returns:
        tuple: (detection_results_list, shared_grid_coords)
    """
    detection_results = []
    shared_coords = None

    file_pattern = os.path.join(year_input_dir, "**", "detection_*.nc")
    filepaths = sorted(glob.glob(file_pattern, recursive=True))

    if not filepaths:
        print(f"Warning: No detection files found matching {file_pattern}")
        return [], None

    for filepath in filepaths:
        try:
            with xr.open_dataset(filepath) as ds:
                time_val = ds["time"].values[0]

                if shared_coords is None:
                    # Robust Dimension Detection
                    if "rlat" in ds.dims:
                        lat_dim, lon_dim = "rlat", "rlon"
                    elif "lat" in ds.dims:
                        lat_dim, lon_dim = "lat", "lon"
                    elif "y" in ds.dims:
                        lat_dim, lon_dim = "y", "x"
                    else:
                        lat_dim, lon_dim = ds.dims[1], ds.dims[2]

                    lat = ds[lat_dim].values
                    lon = ds[lon_dim].values

                    # Coordinate Reconstruction
                    if "latitude" in ds.variables:
                        lat2d = ds["latitude"].values
                        lon2d = ds["longitude"].values
                    elif "lat" in ds.variables and ds["lat"].ndim == 2:
                        lat2d = ds["lat"].values
                        lon2d = ds["lon"].values
                    else:
                        # Recreate 2D mesh for internal processing if missing (Regular Grid)
                        lon2d, lat2d = np.meshgrid(lon, lat)

                    shared_coords = {
                        "lat": lat,
                        "lon": lon,
                        "lat2d": lat2d,
                        "lon2d": lon2d,
                    }

                final_labeled_regions = ds["final_labeled_regions"].values[0]

                # Reconstruct Center Points
                center_points_dict = {}
                if "label_id" in ds:
                    ids = ds["label_id"].values
                    lats = ds["label_lat"].values
                    lons = ds["label_lon"].values
                    for i, label_id in enumerate(ids):
                        lbl_str = str(int(label_id))
                        lbl_lat = float(lats[i])
                        lbl_lon = float(lons[i])
                        if np.isnan(lbl_lat) or np.isnan(lbl_lon):
                            center_points_dict[lbl_str] = None
                        else:
                            center_points_dict[lbl_str] = (lbl_lat, lbl_lon)
                elif "center_points_t0" in ds.attrs:
                    try:
                        import json

                        center_points_dict = json.loads(ds.attrs["center_points_t0"])
                        if isinstance(center_points_dict, str):
                            center_points_dict = json.loads(center_points_dict)
                    except:
                        center_points_dict = {}

                # Create lightweight dictionary WITHOUT redundant coords
                detection_result = {
                    "final_labeled_regions": final_labeled_regions,
                    "time": time_val,
                    "center_points": center_points_dict,
                }

                if use_li_filter:
                    if "lifted_index_regions" in ds:
                        detection_result["lifted_index_regions"] = ds[
                            "lifted_index_regions"
                        ].values[0]
                    else:
                        detection_result["lifted_index_regions"] = np.zeros_like(
                            final_labeled_regions
                        )

                detection_results.append(detection_result)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    detection_results.sort(key=lambda x: x["time"])
    return detection_results, shared_coords


def save_detection_result(detection_result, output_dir, data_source):
    """
    Saves a single timestep's detection results to a compressed, CF-compliant NetCDF file.

    Optimized for CORDEX/Climate Model grids:
    - Uses 'lat'/'lon' for regular grids or 'rlat'/'rlon' for rotated grids.
    - Stores center points as parallel variables.
    - Applies zlib compression.
    """
    time_val = pd.to_datetime(detection_result["time"]).round("s")
    year_str = time_val.strftime("%Y")
    month_str = time_val.strftime("%m")

    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"detection_{time_val.strftime('%Y%m%dT%H')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    # 1. Determine Grid Type
    lat_1d = detection_result["lat"]
    lon_1d = detection_result["lon"]
    lat_2d = detection_result["lat2d"]
    lon_2d = detection_result["lon2d"]

    is_regular = _is_regular_grid(lat_1d, lon_1d, lat_2d)

    if is_regular:
        y_dim, x_dim = "lat", "lon"
    else:
        y_dim, x_dim = "rlat", "rlon"

    # 2. Process Center Points
    center_points = detection_result.get("center_points", {})
    if center_points:
        sorted_labels = sorted(center_points.keys(), key=lambda x: int(x))
        label_ids = np.array([int(lbl) for lbl in sorted_labels], dtype=np.int32)
        label_lats = [
            center_points[lbl][0] if center_points[lbl] else np.nan
            for lbl in sorted_labels
        ]
        label_lons = [
            center_points[lbl][1] if center_points[lbl] else np.nan
            for lbl in sorted_labels
        ]
    else:
        label_ids = np.array([], dtype=np.int32)
        label_lats = []
        label_lons = []

    # 3. Extract Grids
    final_labeled_regions = np.expand_dims(
        detection_result["final_labeled_regions"], axis=0
    )
    lifted_index_regions = np.expand_dims(
        detection_result["lifted_index_regions"], axis=0
    )

    # 4. Create Dataset
    data_vars = {
        "final_labeled_regions": (["time", y_dim, x_dim], final_labeled_regions),
        "lifted_index_regions": (["time", y_dim, x_dim], lifted_index_regions),
        "label_id": (["labels"], label_ids),
        "label_lat": (["labels"], label_lats),
        "label_lon": (["labels"], label_lons),
    }

    # Add CORDEX specific variables only if NOT regular
    if not is_regular:
        data_vars["rotated_pole"] = ([], b"")

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": [time_val],
            y_dim: lat_1d,
            x_dim: lon_1d,
        },
    )

    # Add 2D aux coordinates ONLY if rotated.
    if not is_regular:
        ds["latitude"] = ((y_dim, x_dim), lat_2d)
        ds["longitude"] = ((y_dim, x_dim), lon_2d)

    # Metadata
    ds.attrs = {
        "title": "EMMA-Tracker Detection Output",
        "institution": "Wegener Center for Climate and Global Change, University of Graz",
        "source": data_source,
        "history": f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "references": "Kneidinger et al. (2025)",
        "Conventions": "CF-1.6",
        "project": "EMMA",
    }

    # Variable Attributes
    # IMPORTANT: Do NOT include 'axis' attribute.
    ds[y_dim].attrs = {
        "standard_name": "latitude" if is_regular else "grid_latitude",
        "units": "degrees_north" if is_regular else "degrees",
    }
    ds[x_dim].attrs = {
        "standard_name": "longitude" if is_regular else "grid_longitude",
        "units": "degrees_east" if is_regular else "degrees",
    }
    ds["time"].attrs = {"standard_name": "time"}

    if not is_regular:
        # Rotated grids: Standard_name required for 2D vars
        ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
        ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

        ds["rotated_pole"].attrs = {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 39.25,
            "grid_north_pole_longitude": -162.0,
        }

    for var in ["final_labeled_regions", "lifted_index_regions"]:
        if not is_regular:
            ds[var].attrs["grid_mapping"] = "rotated_pole"
            ds[var].attrs["coordinates"] = "latitude longitude"
        else:
            # Regular: Remove 'coordinates' attribute. ncview infers from dimensions.
            if "coordinates" in ds[var].attrs:
                del ds[var].attrs["coordinates"]
        ds[var].attrs["cell_methods"] = "time: point"

    ds["final_labeled_regions"].attrs.update(
        {"long_name": "Labeled Convective Regions", "units": "1"}
    )
    ds["lifted_index_regions"].attrs.update(
        {"long_name": "Lifted Index Mask", "units": "1"}
    )
    ds["label_id"].attrs.update({"long_name": "Feature Label IDs"})

    # --- USE SHARED SAVER ---
    save_dataset_to_netcdf(ds, output_filepath)


def save_tracking_result(
    tracking_data_for_timestep, output_dir, data_source, config=None
):
    """
    Saves a single timestep's tracking results to a compressed, CF-compliant NetCDF file.

    This function is optimized for Climate Model (CORDEX) grids and GIS compatibility.
    It saves both the 2D segmentation masks (gridded data) and the scalar summary
    statistics of active tracks (tabular data) in a single file.

    The output files are organized into a directory structure: `{output_dir}/YYYY/MM/`.

    Args:
        tracking_data_for_timestep (dict): A dictionary containing all tracking data for one frame:
            - "time" (datetime-like): The timestamp for this frame.
            - "robust_mcs_id" (np.ndarray): 2D array of Track IDs for mature/robust MCS phases.
            - "mcs_id" (np.ndarray): 2D array of Track IDs for the full MCS lifecycle.
            - "mcs_id_merge_split" (np.ndarray): 2D array of Track IDs including merger history.
            - "lat" (np.ndarray): 1D array of y-coordinates (indices or projection y).
            - "lon" (np.ndarray): 1D array of x-coordinates (indices or projection x).
            - "lat2d" (np.ndarray): 2D array of true latitudes (WGS84).
            - "lon2d" (np.ndarray): 2D array of true longitudes (WGS84).
            - "tracking_centers" (dict): A mapping `{track_id: (lat, lon)}` for all active tracks.
        output_dir (str): The root directory where output subfolders will be created.
        data_source (str): A string describing the input source (e.g., "IMERG + ERA5").
        config (dict, optional): The run configuration dictionary. If provided, it is
            serialized into the global attribute 'run_configuration' for provenance.

    Output NetCDF Structure:
        Dimensions:
            time: 1 (unlimited)
            y: Number of latitude indices
            x: Number of longitude indices
            tracks: Number of active tracks in this specific timestep

        Variables:
            robust_mcs_id (time, y, x): Gridded track IDs (compressed).
            mcs_id (time, y, x): Gridded track IDs (compressed).
            mcs_id_merge_split (time, y, x): Gridded track IDs (compressed).
            active_track_id (tracks): List of IDs present in this file.
            active_track_lat (tracks): Center latitude for each ID.
            active_track_lon (tracks): Center longitude for each ID.
            active_track_touches_boundary (tracks): Flag (0/1) if system touches domain edge.

    Returns:
        None
    """
    time_val = pd.to_datetime(tracking_data_for_timestep["time"]).round("s")
    year_str = time_val.strftime("%Y")
    month_str = time_val.strftime("%m")

    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"tracking_{time_val.strftime('%Y%m%dT%H')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    # 1. Determine Grid Type
    lat_1d = tracking_data_for_timestep["lat"]
    lon_1d = tracking_data_for_timestep["lon"]
    lat_2d = tracking_data_for_timestep["lat2d"]

    is_regular = _is_regular_grid(lat_1d, lon_1d, lat_2d)

    if is_regular:
        y_dim, x_dim = "lat", "lon"
    else:
        y_dim, x_dim = "rlat", "rlon"

    # 2. Process Center Points
    centers_dict = tracking_data_for_timestep.get("tracking_centers", {})
    grid = tracking_data_for_timestep["mcs_id"]
    if grid.ndim == 3:
        grid = grid[0]
    ymax, xmax = grid.shape[0] - 1, grid.shape[1] - 1

    active_ids = []
    active_lats = []
    active_lons = []
    active_boundary_flags = []

    if centers_dict:
        sorted_ids = sorted(centers_dict.keys(), key=lambda x: int(x))
        active_ids = np.array([int(tid) for tid in sorted_ids], dtype=np.int32)

        for tid in sorted_ids:
            coords = centers_dict[tid]
            if coords and coords[0] is not None:
                active_lats.append(coords[0])
                active_lons.append(coords[1])
            else:
                active_lats.append(np.nan)
                active_lons.append(np.nan)

            mask = grid == int(tid)
            touches = (
                np.any(mask[0, :])
                or np.any(mask[ymax, :])
                or np.any(mask[:, 0])
                or np.any(mask[:, xmax])
            )
            active_boundary_flags.append(int(touches))

        active_boundary_flags = np.array(active_boundary_flags, dtype=np.int8)
    else:
        active_ids = np.array([], dtype=np.int32)
        active_lats = []
        active_lons = []
        active_boundary_flags = np.array([], dtype=np.int8)

    # 3. Extract Grids
    robust_mcs_id_arr = np.expand_dims(
        tracking_data_for_timestep["robust_mcs_id"], axis=0
    )
    mcs_id_arr = np.expand_dims(tracking_data_for_timestep["mcs_id"], axis=0)
    mcs_id_merge_split_arr = np.expand_dims(
        tracking_data_for_timestep["mcs_id_merge_split"], axis=0
    )

    # 4. Create Dataset
    data_vars = {
        "robust_mcs_id": (["time", y_dim, x_dim], robust_mcs_id_arr),
        "mcs_id": (["time", y_dim, x_dim], mcs_id_arr),
        "mcs_id_merge_split": (["time", y_dim, x_dim], mcs_id_merge_split_arr),
        "active_track_id": (["tracks"], active_ids),
        "active_track_lat": (["tracks"], active_lats),
        "active_track_lon": (["tracks"], active_lons),
        "active_track_touches_boundary": (["tracks"], active_boundary_flags),
    }

    if not is_regular:
        data_vars["rotated_pole"] = ([], b"")

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": [time_val],
            y_dim: lat_1d,
            x_dim: lon_1d,
        },
    )

    # ONLY Add 2D aux coordinates if the grid is rotated
    if not is_regular:
        ds["latitude"] = ((y_dim, x_dim), tracking_data_for_timestep["lat2d"])
        ds["longitude"] = ((y_dim, x_dim), tracking_data_for_timestep["lon2d"])

    # Metadata
    ds.attrs = {
        "title": "EMMA-Tracker Output",
        "institution": "Wegener Center for Climate and Global Change, University of Graz",
        "source": data_source,
        "history": f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Conventions": "CF-1.6",
        "project": "EMMA",
    }
    if not is_regular:
        ds.attrs["CORDEX_domain"] = "EUR-11"

    if config:
        import json

        try:
            ds.attrs["run_configuration"] = json.dumps(config, default=str)
        except:
            pass

    # Coordinate Attributes
    # IMPORTANT: Do NOT include 'axis' attribute.
    ds[y_dim].attrs = {
        "standard_name": "latitude" if is_regular else "grid_latitude",
        "units": "degrees_north" if is_regular else "degrees",
    }
    ds[x_dim].attrs = {
        "standard_name": "longitude" if is_regular else "grid_longitude",
        "units": "degrees_east" if is_regular else "degrees",
    }
    ds["time"].attrs = {"standard_name": "time"}

    if not is_regular:
        # Rotated grid attributes
        ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
        ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

        ds["rotated_pole"].attrs = {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 39.25,
            "grid_north_pole_longitude": -162.0,
        }

    grid_vars = ["robust_mcs_id", "mcs_id", "mcs_id_merge_split"]
    for var in grid_vars:
        if not is_regular:
            ds[var].attrs["grid_mapping"] = "rotated_pole"
            ds[var].attrs["coordinates"] = "latitude longitude"
        else:
            # Regular: Remove 'coordinates' attribute.
            if "coordinates" in ds[var].attrs:
                del ds[var].attrs["coordinates"]
        ds[var].attrs["cell_methods"] = "time: point"

    ds["robust_mcs_id"].attrs.update(
        {"long_name": "Robust Mature MCS Track IDs", "units": "1"}
    )
    ds["mcs_id"].attrs.update({"long_name": "Main MCS Track IDs", "units": "1"})
    ds["mcs_id_merge_split"].attrs.update(
        {"long_name": "Family Tree Track IDs", "units": "1"}
    )
    ds["active_track_id"].attrs = {"long_name": "Active Track IDs"}
    ds["active_track_touches_boundary"].attrs = {
        "long_name": "Boundary Touching Flag",
        "units": "1",
    }

    # --- USE SHARED SAVER ---
    save_dataset_to_netcdf(ds, output_filepath)
