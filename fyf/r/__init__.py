"""
R scripts for the FYF package.

This module provides access to the R scripts used by the FYF package:
- INLA_pipeline.R: The main INLA processing pipeline
- functions.R: R utility functions for INLA processing
- test_functions.R: Tests for the R functions
"""

import os
import tempfile
import subprocess
from pathlib import Path

# Get the path to the R scripts
r_dir = Path(__file__).parent

def check_r_installed():
    """Check if R is installed."""
    try:
        subprocess.run(["Rscript", "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_inla_installed():
    """Check if the INLA package is installed."""
    if not check_r_installed():
        return False
    
    try:
        # Create a temporary script to check for INLA
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            f.write('if(require(INLA)) cat("INLA_INSTALLED") else cat("INLA_NOT_INSTALLED")')
            script_path = f.name
        
        # Run the script
        result = subprocess.run(
            ["Rscript", script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        
        # Clean up
        os.unlink(script_path)
        
        return "INLA_INSTALLED" in result.stdout
    except:
        return False