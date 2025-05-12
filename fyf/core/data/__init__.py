"""
Data handling functionality for the FYF package.

This subpackage provides tools for managing data in the FYF package:
- File I/O via the FileHandler class
- Mask generation via the MaskGenerator class
"""

from fyf.core.data.file_handler import FileHandler
from fyf.core.data.masking import MaskGenerator

__all__ = [
    "FileHandler",
    "MaskGenerator",
]