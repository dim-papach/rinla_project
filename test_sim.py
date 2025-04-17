import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from simulator import FitsProcessor, CosmicConfig, SatelliteConfig
from colorama import Fore

# test_simulator.py

@pytest.fixture
def fits_processor():
    cosmic_cfg = CosmicConfig(fraction=0.01, value=1.0, seed=42)
    satellite_cfg = SatelliteConfig(num_trails=1, trail_width=3, value=2.0)
    return FitsProcessor(cosmic_cfg, satellite_cfg)

@patch("simulator.subprocess.run")
@patch("simulator.os.makedirs")
@patch("simulator.np.save")
@patch("simulator.np.load")
@patch("simulator.os.remove")
def test_process_variants_success(mock_remove, mock_load, mock_save, mock_makedirs, mock_run, fits_processor):
    # Mock successful R script execution and file loading
    mock_run.return_value = None
    mock_load.return_value = np.array([[1, 2], [3, 4]])

    # Input data
    variants = {
        "cosmic": np.array([[0, 1], [1, 0]]),
        "satellite": np.array([[1, 0], [0, 1]])
    }

    # Call the method
    output_dir = "test_output"
    processed = fits_processor.process_variants(variants, output_dir)

    # Assertions
    assert mock_makedirs.call_count == 2  # Ensure directories are created
    assert mock_save.call_count == 2  # Ensure input files are saved
    assert mock_run.call_count == 2  # Ensure R script is called for each variant
    assert mock_load.call_count == 2  # Ensure output files are loaded
    assert mock_remove.call_count == 4  # Ensure temporary files are removed
    assert processed["cosmic"].shape == (2, 2)  # Check processed data
    assert processed["satellite"].shape == (2, 2)

@patch("simulator.subprocess.run")
@patch("simulator.os.makedirs")
@patch("simulator.np.save")
@patch("simulator.np.load")
@patch("simulator.os.remove")
def test_process_variants_failure(mock_remove, mock_load, mock_save, mock_makedirs, mock_run, fits_processor):
    # Mock R script failure
    mock_run.side_effect = Exception("R script failed")

    # Input data
    variants = {
        "cosmic": np.array([[0, 1], [1, 0]]),
        "satellite": np.array([[1, 0], [0, 1]])
    }

    # Call the method
    output_dir = "test_output"
    processed = fits_processor.process_variants(variants, output_dir)

    # Assertions
    assert mock_makedirs.call_count == 2  # Ensure directories are created
    assert mock_save.call_count == 2  # Ensure input files are saved
    assert mock_run.call_count == 2  # Ensure R script is called for each variant
    assert mock_remove.call_count == 4  # Ensure temporary files are removed
    assert processed["cosmic"] is None  # Check failed processing
    assert processed["satellite"] is None