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
        constant = 273.15
    else:
        print(
            f"Warning: Unrecognized lifted_index units '{orig_units}'. No conversion applied."
        )
        factor = 1.0

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


def save_detection_result(detection_result, output_dir, data_source):
    """Saves a single timestep's detection results to a dedicated NetCDF file.

    The output filename is generated from the result's timestamp.

    Args:
        detection_result (dict):
            A dictionary containing the detection results for one timestep.
            Must contain: "final_labeled_regions", "lifted_index_regions", "lat", "lon", "time",
            and optionally "center_points".
        output_dir (str):
            The directory where the output NetCDF file will be saved.
        data_source (str):
            Name of the original data source for the file's metadata.save_sin
    """
    # Extract the timestamp and format it for the filename
    time_val = pd.to_datetime(detection_result["time"]).round("S")

    # Create the year/month subdirectory structure
    year_str = time_val.strftime("%Y")
    month_str = time_val.strftime("%m")
    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"detection_{time_val.strftime('%Y%m%dT%H')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    # Prepare data arrays
    final_labeled_regions = np.expand_dims(
        detection_result["final_labeled_regions"], axis=0
    )
    lifted_index_regions = np.expand_dims(
        detection_result["lifted_index_regions"], axis=0
    )
    lat = detection_result["lat"]
    lon = detection_result["lon"]

    lat2d = detection_result["lat2d"]
    lon2d = detection_result["lon2d"]
    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "final_labeled_regions": (["time", "lat", "lon"], final_labeled_regions),
            "lifted_index_regions": (["time", "lat", "lon"], lifted_index_regions),
        },
        coords={
            "time": [time_val],
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "description": "Detection results of MCSs for a single timestep.",
            "source": data_source,
        },
    )

    # Add lat/lon as DataArray variables
    ds["latitude"] = (("lat", "lon"), lat2d)
    ds["longitude"] = (("lat", "lon"), lon2d)

    # Handle center points if they exist
    center_points = detection_result.get("center_points", {})
    center_points_str = serialize_center_points(center_points)
    center_points_json = json.dumps(center_points_str)
    ds["final_labeled_regions"].attrs["center_points_t0"] = center_points_json

    # Set global attributes
    ds.attrs["title"] = "MCS Tracking Results"
    ds.attrs[
        "institution"
    ] = "Wegener Center for Global and Climate Change / University of Graz"
    ds.attrs["source"] = data_source
    ds.attrs["history"] = f"Created on {datetime.datetime.now()}"
    ds.attrs["references"] = "David Kneidinger <david.kneidinger@uni-graz.at>"

    # Save to NetCDF file
    ds.to_netcdf(output_filepath)


def load_individual_detection_files(year_input_dir, use_li_filter):
    """
    Load a sequence of individual detection result NetCDF files from a directory,
    searching recursively through its subdirectories (e.g., monthly folders).

    Args:
        year_input_dir (str): The base directory for a specific year (e.g., /path/to/output/2020).
        use_li_filter (bool): Flag to determine if lifted_index_regions should be loaded.

    Returns:
        List[dict]: A list of detection_result dictionaries, sorted by time.
                    Returns an empty list if no files are found.
    """
    detection_results = []

    # Create a recursive glob pattern to find files in YYYY/MM/detection/*.nc
    # The "**" wildcard searches through all subdirectories.
    file_pattern = os.path.join(year_input_dir, "**", "detection_*.nc")
    filepaths = sorted(glob.glob(file_pattern, recursive=True))

    if not filepaths:
        # This warning now reflects the pattern being searched
        print(f"Warning: No detection files found matching pattern {file_pattern}")
        return []

    for filepath in filepaths:
        try:
            with xr.open_dataset(filepath) as ds:
                # Reconstruct the detection_result dictionary from the file
                time_val = ds["time"].values[0]
                final_labeled_regions = ds["final_labeled_regions"].values[0]
                lat = ds["lat"].values
                lon = ds["lon"].values
                lat2d = ds["latitude"].values
                lon2d = ds["longitude"].values

                center_points_dict = {}
                if "center_points_t0" in ds["final_labeled_regions"].attrs:
                    center_points_json = ds["final_labeled_regions"].attrs[
                        "center_points_t0"
                    ]
                    try:
                        center_points_intermediate = json.loads(center_points_json)
                        center_points_dict = (
                            json.loads(center_points_intermediate)
                            if isinstance(center_points_intermediate, str)
                            else center_points_intermediate
                        )
                    except json.JSONDecodeError:
                        center_points_dict = {}

                detection_result = {
                    "final_labeled_regions": final_labeled_regions,
                    "time": time_val,
                    "lat2d": lat2d,
                    "lon2d": lon2d,
                    "lat": lat,
                    "lon": lon,
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
                        print(
                            f"Warning: 'lifted_index_regions' not found in {filepath}. Using zeros."
                        )

                detection_results.append(detection_result)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    # Final sort by time is robust, though sorting by filename often suffices
    detection_results.sort(key=lambda x: x["time"])

    return detection_results


def save_tracking_result(tracking_data_for_timestep, output_dir, data_source):
    """Saves a single timestep's tracking results to a NetCDF file
    in a YYYY/MM/tracking/ subdirectory."""

    time_val = pd.to_datetime(tracking_data_for_timestep["time"]).round("S")

    # Create the year/month subdirectory structure
    year_str = time_val.strftime("%Y")
    month_str = time_val.strftime("%m")
    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"tracking_{time_val.strftime('%Y%m%dT%H')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    # Unpack the data for this timestep
    robust_mcs_id_arr = np.expand_dims(
        tracking_data_for_timestep["robust_mcs_id"], axis=0
    )
    mcs_id_arr = np.expand_dims(tracking_data_for_timestep["mcs_id"], axis=0)
    mcs_id_merge_split_arr = np.expand_dims(
        tracking_data_for_timestep["mcs_id_merge_split"], axis=0
    )
    lifetime = np.expand_dims(tracking_data_for_timestep["lifetime"], axis=0)

    lat = tracking_data_for_timestep["lat"]
    lon = tracking_data_for_timestep["lon"]

    lat2d = tracking_data_for_timestep["lat2d"]
    lon2d = tracking_data_for_timestep["lon2d"]

    tracking_centers = tracking_data_for_timestep["tracking_centers"]

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "robust_mcs_id": (["time", "lat", "lon"], robust_mcs_id_arr),
            "mcs_id": (["time", "lat", "lon"], mcs_id_arr),
            "mcs_id_merge_split": (["time", "lat", "lon"], mcs_id_merge_split_arr),
            "lifetime": (["time", "lat", "lon"], lifetime),
        },
        coords={
            "time": [time_val],
            "lat": lat,
            "lon": lon,
        },
    )

    # Attach lat/lon and metadata
    ds["latitude"] = (("lat", "lon"), lat2d)
    ds["longitude"] = (("lat", "lon"), lon2d)

    # Set descriptive attributes for each variable
    ds["robust_mcs_id"].attrs[
        "description"
    ] = "Track IDs of main MCSs, but only for timesteps where the system's area is >= main_area_thresh."
    ds["mcs_id"].attrs[
        "description"
    ] = "Track IDs showing the full lifetime of all identified main MCSs."
    ds["mcs_id_merge_split"].attrs[
        "description"
    ] = "Track IDs for the 'full family tree': main MCSs plus all systems that merged into or split from them."
    ds["lifetime"].attrs[
        "description"
    ] = "Pixel-wise lifetime of all tracked clusters (in timesteps)."

    # Set global attributes
    ds.attrs["title"] = "MCS Tracking Results"
    ds.attrs[
        "institution"
    ] = "Wegener Center for Global and Climate Change / University of Graz"
    ds.attrs["source"] = data_source
    ds.attrs["history"] = f"Created on {datetime.datetime.now()}"
    ds.attrs["references"] = "David Kneidinger <david.kneidinger@uni-graz.at>"

    # Store center points in a JSON attribute
    centers_serialized = serialize_center_points(tracking_centers)
    centers_json = json.dumps(centers_serialized)
    ds["mcs_id"].attrs["center_points"] = centers_json

    # Save the NetCDF file
    ds.to_netcdf(output_filepath)
