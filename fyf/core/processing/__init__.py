"""
Processing functionality for the FYF package.

This subpackage provides tools for processing data in the FYF package:
- FITS image processing via the FitsProcessor class
- INLA wrapper for interacting with R-INLA
- Hash management for tracking processed files
"""

from fyf.core.processing.fits_processor import FitsProcessor
from fyf.core.processing.hash_manager import ArrayHashManager
from fyf.core.processing.methods import (
    ensure_method_available,
    get_supported_methods,
    run_processing_method,
)
from fyf.core.processing.preprocessing import (
    get_supported_preprocessors,
    run_preprocessed_processing,
)

__all__ = [
    "FitsProcessor",
    "ArrayHashManager",
    "ensure_method_available",
    "get_supported_methods",
    "run_processing_method",
    "get_supported_preprocessors",
    "run_preprocessed_processing",
]
