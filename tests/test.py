import pytest
import os
import shutil
import yaml
import xarray as xr
import numpy as np
from pathlib import Path
from unittest.mock import patch

from emma.main import main
from tests.create_test_data import create_test_data_scenario


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data" / "input"
REF_DIR = TEST_DIR / "reference"
OUTPUT_DIR = TEST_DIR / "data" / "output"
CONFIG_PATH = TEST_DIR / "config_test.yaml"

@pytest.fixture(scope="session")
def setup_test_environment():
    """
    1. Cleans/Creates test directories.
    2. Generates synthetic input data.
    3. Creates a temporary config file pointing to these paths.
    """
    # Clean up previous runs
    if DATA_DIR.exists(): shutil.rmtree(DATA_DIR)
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    # Create input data
    print(f"\nGenerating synthetic test data in {DATA_DIR}...")
    create_test_data_scenario(str(DATA_DIR))
    
    # Prepare Output Dirs
    (OUTPUT_DIR / "detection").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tracking_raw").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tracking_filtered").mkdir(parents=True, exist_ok=True)

    # Load and Modify Config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Point config to absolute test paths
    config['precip_data_directory'] = str(DATA_DIR)
    config['lifted_index_data_directory'] = str(DATA_DIR)
    config['detection_output_path'] = str(OUTPUT_DIR / "detection")
    config['raw_tracking_output_dir'] = str(OUTPUT_DIR / "tracking_raw")
    config['filtered_tracking_output_dir'] = str(OUTPUT_DIR / "tracking_filtered")
    
    # --- Disable Post-processing for this test ---
    config['run_postprocessing'] = False
    
    # Save temp config
    temp_config_path = TEST_DIR / "temp_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    yield temp_config_path
    

def test_raw_tracking_consistency(setup_test_environment):
    """
    Runs the pipeline and compares RAW tracking output bit-for-bit against reference.
    """
    config_file = setup_test_environment
    
    # Run Main Pipeline
    print("\nRunning EMMA Tracker (Raw Mode)...")
    with patch("sys.argv", ["main.py", "--config", str(config_file)]):
        main()
        
    # --- COMPARISON LOGIC  ---
    
    # 1. Look for generated files in 'tracking_raw'
    generated_files = sorted(list((OUTPUT_DIR / "tracking_raw").rglob("*.nc")))
    
    # 2. Look for reference files in 'reference/tracking_raw'
    reference_files = sorted(list((REF_DIR / "tracking_raw").rglob("*.nc_test")))
    
    assert len(generated_files) > 0, "Pipeline produced no output files in tracking_raw!"
    
    # Ensure reference data exists
    if not reference_files:
        pytest.fail(f"No reference data found in {REF_DIR / 'tracking_raw'}! Please generate reference data first.")

    assert len(generated_files) == len(reference_files), \
        f"File count mismatch! Got {len(generated_files)}, expected {len(reference_files)}"
    
    for gen_path, ref_path in zip(generated_files, reference_files):
        print(f"Comparing {gen_path.name}...")
        ds_gen = xr.open_dataset(gen_path)
        ds_ref = xr.open_dataset(ref_path)
        
        # Compare Data Variables (Floating point tolerance)
        xr.testing.assert_allclose(ds_gen, ds_ref)
        
        # Check if the Boundary Flag exists (sanity check)
        assert "active_track_touches_boundary" in ds_gen, "Boundary flag missing from output!"
        
        # Compare Attributes
        for attr in ds_ref.attrs:
            if attr not in ['history', 'source', 'run_configuration']: 
                assert ds_gen.attrs.get(attr) == ds_ref.attrs[attr], \
                    f"Attribute '{attr}' mismatch in {gen_path.name}"