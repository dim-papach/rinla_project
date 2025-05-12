"""
FYF (Fill Your FITS) - Process astronomical FITS images using R-INLA

A command-line tool for applying R-INLA to fill missing data in astronomical images,
particularly focusing on cosmic rays and satellite trails.

This package provides:
- Core functionality for processing FITS images
- Command-line interface for easy use
- Visualization tools for analysis and reporting
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fyf")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Default version if not installed via pip

__author__ = "FYF Contributors"
__email__ = "example@example.com"
__license__ = "MIT"

# Import core components for easier access
from fyf.core.config import CosmicConfig, SatelliteConfig, INLAConfig, PlotConfig
from fyf.core.pipeline import SimulationPipeline

# Define what's available at the top level
__all__ = [
    "CosmicConfig",
    "SatelliteConfig",
    "INLAConfig", 
    "PlotConfig",
    "SimulationPipeline",
]

# Check if INLA is installed
from fyf.r import check_inla_installed
_has_inla = check_inla_installed()

if not _has_inla:
    import warnings
    warnings.warn(
        "R-INLA is not installed. Some functionality will be limited. "
        "Run 'Rscript fyf/scripts/setup_inla.R' to install INLA."
    )