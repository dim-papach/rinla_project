import numpy as np
from scipy.ndimage import binary_dilation
from typing import Dict, Tuple
from .mask_config import CosmicConfig, SatelliteConfig


class MaskGenerator:
    """Handles generation of cosmic ray and satellite trail masks"""
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        self.cosmic_cfg = cosmic_cfg
        self.satellite_cfg = satellite_cfg
        self.rng = np.random.default_rng(cosmic_cfg.seed)  # Use a single RNG instance

    def generate_all_masks(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate complete set of boolean masks
        
        Args:
            data: Input image data array
            
        Returns:
            Dictionary containing three masks:
            - 'cosmic': Cosmic ray affected pixels
            - 'satellite': Satellite trail pixels
            - 'combined': Union of both mask types
        """
        if data.size == 0:
            return {
                'cosmic': np.zeros_like(data, dtype=bool),
                'satellite': np.zeros_like(data, dtype=bool),
                'combined': np.zeros_like(data, dtype=bool),
            }

        masks = {
            'cosmic': self._generate_cosmic_mask(data),
            'satellite': self._generate_satellite_mask(data.shape),
            'combined': None
        }
        masks['combined'] = masks['cosmic'] | masks['satellite']
        return masks

    def _generate_cosmic_mask(self, data: np.ndarray) -> np.ndarray:
        """Generate random cosmic ray mask"""
        mask = np.zeros_like(data, dtype=bool)
        n_pixels = int(self.cosmic_cfg.fraction * data.size)

        if n_pixels > 0:
            indices = self.rng.choice(data.size, n_pixels, replace=False)
            np.put(mask, indices, True)

        return mask

    def _generate_satellite_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate satellite trail mask with configurable parameters"""
        if self.satellite_cfg.num_trails < 1:
            return np.zeros(shape, dtype=bool)

        mask = np.zeros(shape, dtype=bool)
        height, width = shape

        for _ in range(self.satellite_cfg.num_trails):
            start, angle = self._random_trail_parameters(width, height)
            coords = self._calculate_trail_coordinates(start, angle, max(height, width))
            self._apply_trail(mask, coords)

        if self.satellite_cfg.trail_width > 1:
            mask = binary_dilation(
                mask,
                structure=np.ones((self.satellite_cfg.trail_width, 
                                 self.satellite_cfg.trail_width))
                )
        return mask

    def _random_trail_parameters(self, 
                                width: int, 
                                height: int) -> Tuple[Tuple[int, int], float]:
        """Generate random starting point and angle for trails"""
        edge = self.rng.choice(['top', 'bottom', 'left', 'right'])

        if edge in ['top', 'bottom']:
            x = self.rng.integers(0, width)
            y = 0 if edge == 'top' else height - 1
        else:
            y = self.rng.integers(0, height)
            x = 0 if edge == 'left' else width - 1

        angle = self.rng.uniform(self.satellite_cfg.min_angle,
                                self.satellite_cfg.max_angle)
        return (x, y), np.deg2rad(angle)

    def _calculate_trail_coordinates(self, 
                                   start: Tuple[int, int], 
                                   angle: float, 
                                   length: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate trail coordinates using vectorized operations"""
        x0, y0 = start
        x1 = x0 + length * 1.5 * np.cos(angle)
        y1 = y0 + length * 1.5 * np.sin(angle)

        return (
            np.round(np.linspace(x0, x1, 1000)).astype(int),
            np.round(np.linspace(y0, y1, 1000)).astype(int))
        
    def _apply_trail(self, mask: np.ndarray, coords: Tuple[np.ndarray, np.ndarray]) -> None:
        """Apply valid coordinates to mask in a vectorized manner"""
        x, y = coords
        valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        mask[y[valid], x[valid]] = True
