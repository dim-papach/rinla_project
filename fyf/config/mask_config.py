from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

#region Constants
DEFAULT_COSMIC_VALUE: float = np.float32(np.nan)
DEFAULT_SATELLITE_VALUE: float = np.float32(np.nan)
CMAP: str = 'viridis'
PLOT_DPI: int = 150
PERCENTILE_RANGE: Tuple[int, int] = (1, 99)
#endregion

#region Data Classes
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
#endregion