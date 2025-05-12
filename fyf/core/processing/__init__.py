"""
Processing functionality for the FYF package.

This subpackage provides tools for processing data in the FYF package:
- FITS image processing via the FitsProcessor class
- INLA wrapper for interacting with R-INLA
- Hash management for tracking processed files
"""

from fyf.core.processing.fits_processor import FitsProcessor
from fyf.core.processing.hash_manager import ArrayHashManager

__all__ = [
    "FitsProcessor",
    "ArrayHashManager",
]