 # EMMA-Tracker: European Mesoscale Convective System Model Assessment
 
 **A robust detection and tracking algorithm for Mesoscale Convective Systems (MCS), optimized for climate model evaluation.**
 
 ---
 
 ## Overview
 
 The **EMMA-Tracker** is a Python software designed to identify and track Mesoscale Convective Systems (MCSs) in gridded precipitation data (e.g., IMERG, CPMs, RCMs). 
 
 Unlike standard threshold-based trackers, EMMA employs a **physics-based post-processing filter** to distinguish self-sustaining, propagating MCSs from synoptic-scale frontal precipitation and stationary orographic convection. This makes it particularly suitable for process-based evaluation of regional climate models.
 
 **Key Features:**
 * **Physics-Aware Filtering:** Removes frontal systems and stationary convection using Lifted Index (LI), Track Straightness, and Area Volatility.
 * **Rotated Pole-Ready:** Output files are fully CF-compliant and support Rotated Pole grids (native EURO-CORDEX support).
 * **Robust Tracking:** Handles complex merging and splitting events explicitly.
 * **Reproducible:** Every output file embeds the full run configuration in its metadata.
 
 ## Installation
 
 We recommend using a fresh environment to ensure dependency stability.
 
 ### 1. Create Environment
 ```bash
 # Create a new environment (Python 3.10+ recommended)
 conda create -n emma python=3.11 -y
 conda activate emma
 ```
 
 ### 2. Install Package
 Clone this repository and install it in "editable" mode. This handles all dependencies (xarray, numpy, dask, etc.) automatically via `pyproject.toml`.
 
 ```bash
 git clone [https://github.com/DavKnei/emma-tracker.git](https://github.com/DavKnei/emma-tracker.git)
 cd emma-tracker
 pip install -e .
 ```
 
 ## Quick Start
 
 To run the tracker, you need a configuration YAML file defining your input data paths and thresholds.
 
 1.  **Configure:** Edit `config.yaml` to point to your data (see [Configuration](#-configuration) below).
 2.  **Run:** Use the command-line interface:
 
 ```bash
 emma-tracker --config config.yaml
 ```
 
 For detailed documentation on parameters and preprocessing, please see the [Technical Guide](emma/README.md).
 
 ## Configuration (`config.yaml`)
 
 The behavior of the algorithm is controlled by a single YAML file. Key parameters include:
 
 ```yaml
 # Data Inputs
 precip_data_directory: "/path/to/imerg"
 lifted_index_data_directory: "/path/to/era5"
 
 # Detection Thresholds
 heavy_precip_threshold: 6.82  # mm/h (e.g., 99th percentile)
 min_size_threshold: 10        # Minimum grid cells
 
 # Post-Processing Filters
 run_postprocessing: True
 postprocessing_filters:
   lifted_index_threshold: 0.0          # Reject systems in stable environments (Mean LI > 0)
   track_straightness_threshold: 0.4    # Reject erratic/stationary systems
   max_area_volatility: 120000.0        # Reject unphysical growth (frontal mergers)
 ```
 
 ## Output Data Structure
 
 The output is saved as **CF-compliant NetCDF4** files (compressed), organized by `YYYY/MM/`. 
 
 Each file contains both **Gridded Segmentation Masks** and **Tabular Track Statistics**:
 
 ### Gridded Variables (2D Maps)
 * `robust_mcs_id`: ID mask of MCSs during their mature, "in-phase" stage.
 * `mcs_id`: Full lifecycle mask of Main MCSs.
 * `mcs_id_merge_split`: Complete family tree mask (includes mergers/splits).
 
 ### Tabular Variables (Parallel Arrays)
 For efficient analysis without loading full grids, summary statistics are stored as parallel arrays along the `tracks` dimension:
 * `active_track_id`: The ID of the system.
 * `active_track_lat` / `lon`: The precipitation-weighted center of mass.
 * `active_track_touches_boundary`: Flag (`1`) if the system touches the domain edge (crucial for lifetime statistics).
 
 ## Testing & Validation
 
 This repository includes a regression test suite that generates synthetic convective scenarios (growth, merger, splitting) to verify algorithmic stability.
 
 To run the tests:
 ```bash
 pip install pytest
 pytest
 ```
 
 ## Citation
 
 If you use the EMMA-Tracker in your research, please cite the following paper:
 
 > **Kneidinger, D., Schaffer, A., & Maraun, D. (2025).** *EMMA-Tracker: European Mesoscale Convective System Model Assessment-Tracker.* [Journal Name], [DOI].
 
 ## Improvments and suggestions
 If you encounter any errors or have suggestions for improvment, simply contact me or open um an issue directly in github.

 ## License
 
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
 ---
 **Contact:** David Kneidinger (david.kneidinger@uni-graz.at)  
 *Wegener Center for Climate and Global Change, University of Graz*