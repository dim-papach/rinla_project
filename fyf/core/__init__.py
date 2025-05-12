"""
Visualization functionality for the FYF package.

This subpackage provides tools for visualizing data in the FYF package:
- PlotGenerator for creating plots of astronomical images
- Report generation for creating HTML and PDF reports
"""

from fyf.core.visualization.plotting import PlotGenerator, InvalidDataError

__all__ = [
    "PlotGenerator",
    "InvalidDataError",
]