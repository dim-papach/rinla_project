"""
Utility scripts for the FYF package.

This module provides access to utility scripts used by the FYF package:
- setup_inla.R: Script to install and set up INLA
- run.sh: Shell script for running the pipeline
"""

import os
import subprocess
from pathlib import Path

# Get the path to the scripts
scripts_dir = Path(__file__).parent

def run_setup_inla():
    """Run the setup_inla.R script."""
    setup_script = scripts_dir / "setup_inla.R"
    if not setup_script.exists():
        raise FileNotFoundError(f"Setup script not found: {setup_script}")
    
    try:
        subprocess.run(["Rscript", str(setup_script)], check=True)
        return True
    except subprocess.CalledProcessError:
        return False