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
    """Extended configuration parameters for INLA processing
    
    This configuration controls how the INLA (Integrated Nested Laplace Approximation)
    algorithm processes astronomical images to fill missing data.
    """
    
    # ========== BASIC PARAMETERS ==========
    
    shape: str = "none"
    """Spatial trend model type:
    - 'none': Constant mean field (fastest, good for uniform backgrounds)
    - 'radius': Radial trend from center (good for stars, circular galaxies)
    - 'ellipse': Elliptical trend (good for elongated objects like edge-on galaxies)
    Default: 'none'
    """
    
    tolerance: float = 1e-4
    """Convergence tolerance for INLA algorithm.
    Smaller values = more accurate but slower convergence.
    Too small may cause numerical issues.
    Typical range: 1e-6 to 1e-3
    Default: 1e-4
    """
    
    restart: int = 0
    """Number of algorithm restarts if convergence fails.
    Higher values increase robustness but computation time.
    Use if getting convergence warnings.
    Typical range: 0 to 5
    Default: 0
    """
    
    scaling: str = "log"
    """Data scaling method.
    - 'none': No transformation
    - 'log': Apply log10 transformation (recommended for astronomical data with large dynamic range)
    Default: 'log'
    """
    
    nonstationary: bool = False
    """Use non-stationary spatial model.
    - False: Assumes spatial correlation is the same everywhere (faster)
    - True: Allows spatial correlation to vary across the image (slower, more flexible)
    Use True for complex astronomical objects with varying structure.
    Default: False
    """
    
    # ========== MESH PARAMETERS ==========
    
    mesh_cutoff: Optional[float] = None
    """Minimum distance between mesh points (in pixels).
    Smaller values = finer mesh = more detail but slower computation.
    If None, automatically calculated as max_edge/mesh_resolution.
    Typical range: 0.1 to 5.0 pixels
    Default: None (auto-calculated)
    """
    
    mesh_resolution: int = 30
    """Controls mesh fineness. Higher = finer mesh = more detail.
    Affects cutoff calculation: cutoff = max_edge / mesh_resolution
    For astronomical images:
    - 10-20: Coarse (fast, less detail)
    - 30-50: Medium (balanced)
    - 50-100: Fine (slow, high detail)
    Default: 30
    """
    
    max_edge_factor: float = 10.0
    """Controls maximum triangle edge length in mesh.
    max_edge = max(image_width, image_height) / max_edge_factor
    Larger factor = finer mesh = more computation time.
    For astronomical images:
    - 5-8: Coarse mesh (large scale structure only)
    - 10-15: Medium mesh (typical use)
    - 20+: Fine mesh (high resolution objects)
    Default: 10.0
    """
    
    outer_edge_factor: float = 1.5
    """Multiplier for outer boundary mesh edge length.
    outer_edge = max_edge * outer_edge_factor
    Controls mesh quality at image boundaries.
    Larger values = coarser boundary mesh = faster computation.
    Typical range: 1.2 to 3.0
    Default: 1.5
    """
    
    offset_inner_factor: float = 0.5
    """Inner boundary extension as fraction of max_edge.
    Controls how far the mesh extends beyond the data points.
    Smaller = mesh closer to data = potential boundary effects.
    Typical range: 0.1 to 1.0
    Default: 0.5
    """
    
    offset_outer_factor: float = 2.0
    """Outer boundary extension as fraction of max_edge.
    Controls the outermost mesh boundary.
    Larger = more stable but slower computation.
    Typical range: 1.0 to 5.0
    Default: 2.0
    """
    
    # ========== SPDE MODEL PARAMETERS ==========
    
    alpha: int = 2
    """Smoothness parameter for the Matérn covariance function.
    - 1: Less smooth, more jagged spatial field (ν = 0)
    - 2: Smoother spatial field (ν = 1)
    For astronomical images, α=2 usually works well.
    Only values 1 and 2 are supported.
    Default: 2
    """
    
    prior_range_prob: float = 0.2
    """Probability that spatial range < prior_range_lower.
    P(range < prior_range_lower) = prior_range_prob
    This is a PC (Penalized Complexity) prior parameter.
    Smaller prob = stronger belief that range is large.
    Typical range: 0.01 to 0.5
    Default: 0.2
    """
    
    prior_range_lower: float = 2.0
    """Lower bound for spatial range prior (in pixels).
    The spatial range controls how far correlations extend.
    For astronomical images:
    - 1-5: Fine structure (star clusters, small features)
    - 5-20: Medium structure (galaxy spiral arms)
    - 20+: Large structure (galaxy halos, nebulae)
    Default: 2.0
    """
    
    prior_sigma_prob: float = 0.2
    """Probability that marginal standard deviation > prior_sigma_upper.
    P(σ > prior_sigma_upper) = prior_sigma_prob
    Controls the prior belief about field variability.
    Smaller prob = stronger belief that field is smooth.
    Typical range: 0.01 to 0.5
    Default: 0.2
    """
    
    prior_sigma_upper: float = 2.0
    """Upper bound for marginal standard deviation prior.
    Controls how much the spatial field can vary.
    For log-scaled astronomical data:
    - 0.5-1.0: Low variability (uniform backgrounds)
    - 1.0-3.0: Medium variability (typical galaxies)
    - 3.0+: High variability (complex nebulae)
    Default: 2.0
    """
    
    # ========== COMPUTATION PARAMETERS ==========
    
    num_threads: int = 6
    """Number of CPU threads for parallel computation.
    Should not exceed your CPU core count.
    Higher values don't always mean faster (diminishing returns).
    Default: 6
    """
    
    openmp_strategy: str = 'huge'
    """OpenMP parallelization strategy for INLA.
    - 'small': For small problems (<1000 mesh points)
    - 'medium': For medium problems (1000-5000 mesh points)  
    - 'large': For large problems (5000-20000 mesh points)
    - 'huge': For very large problems (>20000 mesh points)
    Astronomical images often need 'large' or 'huge'.
    Default: 'huge'
    """
    
    # ========== NON-STATIONARY MODEL PARAMETERS ==========
    
    nbasis: int = 2
    """Number of basis functions for non-stationary model.
    Only used if nonstationary=True.
    Controls flexibility of spatial parameter variation.
    - 2-5: Simple variation patterns
    - 5-10: Complex variation patterns  
    - 10+: Very complex (may overfit)
    Higher values = more computation time.
    Default: 2
    """
    
    spline_degree: int = 3
    """Degree of B-spline basis functions for non-stationary model.
    Only used if nonstationary=True.
    Controls smoothness of parameter variation.
    - 1: Linear variation
    - 2-3: Quadratic or Cubic (Standard)
    - 4+: Highly smooth, computationally intensive
    Default: 3
    """