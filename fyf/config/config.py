"""
Configuration classes for the FYF package.

This module defines configuration dataclasses used throughout the FYF package
for cosmic ray and satellite trail simulation.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

# Default values for configurations
DEFAULT_COSMIC_VALUE: float = np.float32(np.nan)
DEFAULT_SATELLITE_VALUE: float = np.float32(np.nan)
DEFAULT_PLOT_DPI: int = 150
DEFAULT_PERCENTILE_RANGE: Tuple[int, int] = (1, 99)
DEFAULT_COLORMAP: str = "viridis"

@dataclass(frozen=True)
class CosmicConfig:
    """Configuration parameters for cosmic ray simulation
    
    Attributes:
        fraction: Fraction of pixels to affect (0-1)
        value: Replacement value for cosmic ray pixels
        seed: Optional random seed for reproducibility
    """
    fraction: float
    value: float = DEFAULT_COSMIC_VALUE
    seed: Optional[int] = None


@dataclass(frozen=True)
class SatelliteConfig:
    """Configuration parameters for satellite trail simulation
    
    Attributes:
        num_trails: Number of trails to generate
        trail_width: Width of trails in pixels
        min_angle: Minimum trail angle in degrees
        max_angle: Maximum trail angle in degrees
        value: Replacement value for trail pixels
    """
    num_trails: int
    trail_width: int
    min_angle: float = -45.0
    max_angle: float = 45.0
    value: float = DEFAULT_SATELLITE_VALUE


@dataclass(frozen=True)
class PlotConfig:
    """Configuration parameters for visualization
    
    Attributes:
        dpi: Dots per inch for saved plots
        cmap: Colormap for images
        residual_cmap: Colormap for residual plots
        percentile_range: Percentile range for color scaling
        residual_percentile: Percentile range for residual plots
    """
    dpi: int = DEFAULT_PLOT_DPI
    cmap: str = DEFAULT_COLORMAP
    residual_cmap: str = DEFAULT_COLORMAP
    percentile_range: Tuple[int, int] = DEFAULT_PERCENTILE_RANGE
    residual_percentile: Tuple[int, int] = DEFAULT_PERCENTILE_RANGE


@dataclass(frozen=True)
class INLAConfig:
    """Configuration parameters for INLA processing
    
    Attributes:
        shape: Shape parameter for INLA model ('none', 'radius', 'ellipse')
        mesh_cutoff: Cutoff parameter for mesh creation
        tolerance: Convergence tolerance for INLA algorithm
        restart: Number of restarts for INLA algorithm
        scaling: Whether to use log10 scaling
        nonstationary: Whether to use nonstationary model
    """
    shape: str = "none"
    mesh_cutoff: Optional[float] = None
    tolerance: float = 1e-4
    restart: int = 0
    scaling: bool = False
    nonstationary: bool = False