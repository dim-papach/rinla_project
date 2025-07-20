"""Path management for FYF package"""

import os
import tempfile
from pathlib import Path

# Get installation directory from environment or use current directory as fallback
INSTALL_DIR = os.environ.get('FYF_INSTALL_DIR', os.getcwd())

# Get data directory from environment or create a default
DATA_DIR = os.environ.get('FYF_DATA_DIR', 
                        os.path.join(INSTALL_DIR, 'share', 'fyf'))

# Get R scripts directory from environment or calculate it
R_SCRIPTS_DIR = os.environ.get('FYF_R_SCRIPTS_DIR')
if R_SCRIPTS_DIR is None:
    # Calculate relative to this file's location
    R_SCRIPTS_DIR = Path(__file__).parent.parent / 'r'
else:
    R_SCRIPTS_DIR = Path(R_SCRIPTS_DIR)

# Temporary directory for variants (user configurable via environment)
VARIANTS_DIR = os.environ.get('FYF_VARIANTS_DIR',
                            os.path.join(tempfile.gettempdir(), 'fyf_variants'))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VARIANTS_DIR, exist_ok=True)

def get_variant_path(variant_name):
    """Get the path to a variant file"""
    return os.path.join(VARIANTS_DIR, f"{variant_name}.npy")

def get_path_file():
    """Get the path to the path.txt file"""
    return os.path.join(VARIANTS_DIR, "path.txt")

def get_inla_script_path():
    """Get the path to the INLA pipeline R script"""
    return os.path.join(R_SCRIPTS_DIR, "INLA_pipeline.R")

def get_output_dir(base_dir, variant_name):
    """Get the output directory for a variant"""
    output_dir = os.path.join(base_dir, variant_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir