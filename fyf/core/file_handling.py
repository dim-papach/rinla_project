from pathlib import Path
from astropy.io import fits
import numpy as np
from typing import Tuple, Dict


class FileHandler:
    """Manages FITS file I/O and directory structure"""
    
    @staticmethod
    def load_fits(input_path: Path) -> Tuple[np.ndarray, fits.Header]:
        """Load and validate FITS data
        
        Args:
            input_path: Path to input FITS file
            
        Returns:
            Tuple containing:
            - Validated 2D image data as float32 array
            - FITS header from primary HDU
            
        Raises:
            ValueError: If input data is not 2-dimensional
        """
        with fits.open(input_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim != 2:
                raise ValueError("Input data must be 2-dimensional")
            return data, hdul[0].header

    @staticmethod
    def create_output_structure(basename: str) -> Path:
        """Create standardized output directory structure
        
        Args:
            basename: Stem of input filename
            
        Returns:
            Path to created output directory
        """
        output_dir = Path('output') / basename
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def save_outputs(output_dir: Path,
                   variants: Dict[str, np.ndarray],
                   masks: Dict[str, np.ndarray],
                   header: fits.Header) -> None:
        """Save all output files to organized directory structure
        
        Args:
            output_dir: Base output directory
            variants: Dictionary of image variants
            masks: Dictionary of boolean masks
            header: FITS header to preserve metadata
        """
        # Save FITS variants
        data_dir = output_dir / 'data'
        data_dir.mkdir(exist_ok=True)
        for name, data in variants.items():
            fits.writeto(data_dir / f'{name}.fits', data, header, overwrite=True)

        # Save masks
        mask_dir = output_dir / 'masks'
        mask_dir.mkdir(exist_ok=True)
        for name, mask in masks.items():
            fits.writeto(mask_dir / f'{name}_mask.fits',
                       mask.astype(np.uint8), header, overwrite=True)
