"""
Mask generation functionality for cosmic rays and satellite trails with custom mask support.

This module provides the MaskGenerator class which creates boolean masks
for simulating cosmic rays and satellite trails in astronomical images,
with support for loading custom masks.
"""

import numpy as np
from scipy.ndimage import binary_dilation
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from astropy.io import fits

from fyf.config import CosmicConfig, SatelliteConfig


class MaskGenerator:
    """Handles generation of cosmic ray and satellite trail masks with custom mask support
    
    Attributes:
        cosmic_cfg: Configuration for cosmic ray generation
        satellite_cfg: Configuration for satellite trail generation
        rng: Random number generator instance
    """
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        """Initialize the mask generator with configuration objects.
        
        Args:
            cosmic_cfg: Configuration for cosmic ray generation
            satellite_cfg: Configuration for satellite trail generation
        """
        self.cosmic_cfg = cosmic_cfg
        self.satellite_cfg = satellite_cfg
        self.rng = np.random.default_rng(cosmic_cfg.seed)  # Use a single RNG instance

    def generate_all_masks(self, data: np.ndarray, 
                         custom_mask_path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
        """Generate complete set of boolean masks
        
        Args:
            data: Input image data array
            custom_mask_path: Optional path to custom mask file
            
        Returns:
            Dictionary containing masks:
            - 'cosmic': Cosmic ray affected pixels
            - 'satellite': Satellite trail pixels
            - 'custom': Custom mask (if provided)
            - 'combined': Union of all mask types
        """
        print("Generating masks...")
        if data.size == 0:
            empty_mask = np.zeros_like(data, dtype=bool)
            return {
                'cosmic': empty_mask,
                'satellite': empty_mask,
                'custom': empty_mask,
                'combined': empty_mask,
            }
        
        masks = {
            'cosmic': self._generate_cosmic_mask(data),
            'satellite': self._generate_satellite_mask(data.shape),
        }
        
        # Only load and add custom mask if a path was provided
        print(f"Loading custom mask from {custom_mask_path}... if provided")
        custom_mask = None
        if custom_mask_path is not None:
            custom_mask = self._load_custom_mask(data.shape, custom_mask_path)
            print(f"Custom mask loaded with shape: {custom_mask.shape}")
            if custom_mask is not None:
                masks['custom'] = custom_mask
                print(f"Custom mask shape: {custom_mask.shape}")
        else:
            print("Custom mask is None, using empty mask instead.")
            
        
        # Create combined mask from all available masks
        combined = masks['cosmic'] | masks['satellite']
        if custom_mask is not None:
            combined = combined | custom_mask
            
        masks['combined'] = combined
        print(f"Combined mask shape: {masks['combined'].shape}")
        print("Mask generation complete.")
        print(f"Cosmic mask shape: {masks['cosmic'].shape}")
        print(f"Satellite mask shape: {masks['satellite'].shape}")
        print(f"Custom mask shape: {masks['custom'].shape if custom_mask is not None else 'None'}")
        print(f"Combined mask shape: {masks['combined'].shape}")
        print("Masks generated successfully.")
        
        return masks        

    def _generate_cosmic_mask(self, data: np.ndarray) -> np.ndarray:
        """Generate random cosmic ray mask
        
        Args:
            data: Input image data array
            
        Returns:
            Boolean mask with cosmic ray affected pixels set to True
        """
        mask = np.zeros_like(data, dtype=bool)
        n_pixels = int(self.cosmic_cfg.fraction * data.size)

        if n_pixels > 0:
            indices = self.rng.choice(data.size, n_pixels, replace=False)
            np.put(mask, indices, True)

        return mask

    def _generate_satellite_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate satellite trail mask with configurable parameters
        
        Args:
            shape: Shape of the image (height, width)
            
        Returns:
            Boolean mask with satellite trail pixels set to True
        """
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
    
    def _load_custom_mask(self, 
                        shape: Tuple[int, int], 
                        mask_path: Optional[Union[str, Path]] = None) -> Optional[np.ndarray]:
        """Load custom mask from file
        
        Args:
            shape: Shape of the image (height, width)
            mask_path: Path to custom mask file
            
        Returns:
            Boolean mask from file or None if no path provided
        """
        if mask_path is None:
            return None
        
        try:
            # Convert path to Path object if it's a string
            mask_path = Path(mask_path) if isinstance(mask_path, str) else mask_path
            
            # Check file extension to determine loading method
            if mask_path.suffix.lower() in ['.fits', '.fit']:
                # Load FITS file
                with fits.open(mask_path) as hdul:
                    mask_data = hdul[0].data
            elif mask_path.suffix.lower() == '.npy':
                # Load NumPy file
                mask_data = np.load(mask_path)
            else:
                # Attempt to load as image using matplotlib
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                mask_data = mpimg.imread(mask_path)
                
                # If it's an RGB image, convert to grayscale
                if len(mask_data.shape) > 2:
                    mask_data = np.mean(mask_data, axis=2)  # Simple grayscale conversion
            
            # Ensure mask is boolean
            mask_data = mask_data > 0
            
            # Resize mask if necessary to match input shape
            if mask_data.shape != shape:
                from scipy.ndimage import zoom
                zoom_factors = (shape[0] / mask_data.shape[0], shape[1] / mask_data.shape[1])
                mask_data = zoom(mask_data.astype(float), zoom_factors, order=0) > 0
            
            return mask_data
            
        except Exception as e:
            print(f"Error loading custom mask: {e}")
            # Return empty mask in case of error
            return np.zeros(shape, dtype=bool)

    def _random_trail_parameters(self, 
                                width: int, 
                                height: int) -> Tuple[Tuple[int, int], float]:
        """Generate random starting point and angle for trails
        
        Args:
            width: Width of the image
            height: Height of the image
            
        Returns:
            Tuple containing:
            - Starting coordinates (x, y)
            - Angle in radians
        """
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
        """Calculate trail coordinates using vectorized operations
        
        Args:
            start: Starting coordinates (x, y)
            angle: Angle in radians
            length: Length of the trail
            
        Returns:
            Tuple containing:
            - Array of x-coordinates
            - Array of y-coordinates
        """
        x0, y0 = start
        x1 = x0 + length * 1.5 * np.cos(angle)
        y1 = y0 + length * 1.5 * np.sin(angle)

        return (
            np.round(np.linspace(x0, x1, 1000)).astype(int),
            np.round(np.linspace(y0, y1, 1000)).astype(int))
        
    def _apply_trail(self, mask: np.ndarray, coords: Tuple[np.ndarray, np.ndarray]) -> None:
        """Apply valid coordinates to mask in a vectorized manner
        
        Args:
            mask: Boolean mask to update
            coords: Tuple of (x, y) coordinate arrays
        """
        x, y = coords
        valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        mask[y[valid], x[valid]] = True