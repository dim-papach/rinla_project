"""
Visualization functionality for the FYF package.

This module provides tools for visualizing data:
- Plotting functionality for generating figures
- Report generation for creating HTML reports
"""

from fyf.core.visualization.plotting import PlotGenerator, InvalidDataError
from fyf.core.visualization.report import ReportGenerator

__all__ = [
    "PlotGenerator",
    "InvalidDataError",
    "ReportGenerator",
]