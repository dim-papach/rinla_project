#!/usr/bin/env python3
"""
MODULAR COSMIC RAY & SATELLITE TRAIL SIMULATOR

A configurable pipeline for generating realistic artifacts in astronomical images.
Produces FITS files, binary masks, and diagnostic plots in an organized output structure.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import argparse
import glob
import os
import tempfile
import subprocess
import numpy as np
import matplotlib
# Ensure matplotlib uses a non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import binary_dilation

from colorama import Fore, init
init()
colors = [Fore.WHITE, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]


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

class MaskGenerator:
    """Handles generation of cosmic ray and satellite trail masks"""
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        self.cosmic_cfg = cosmic_cfg
        self.satellite_cfg = satellite_cfg
        self.rng = np.random.default_rng(cosmic_cfg.seed)

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
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])

        if edge in ['top', 'bottom']:
            x = np.random.randint(0, width)
            y = 0 if edge == 'top' else height - 1
        else:
            y = np.random.randint(0, height)
            x = 0 if edge == 'left' else width - 1

        angle = np.random.uniform(self.satellite_cfg.min_angle,
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

class FitsProcessor:
    """FitsProcessor is a class designed to handle the creation, processing, saving, 
    and deletion of image variants based on input data and masks. It also integrates 
    with an external R script pipeline for further processing of these variants.

    Methods:
        - __init__(cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
            Initializes the FitsProcessor with configuration objects for cosmic 
            rays and satellite trails.
        - create_variants(data: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            Creates image variants by applying cosmic ray and satellite trail 
            artifacts to the input data based on provided masks.
        - save_masked_variants(data: np.ndarray, masks: Dict[str, np.ndarray]) -> None:
            Saves the masked image variants (cosmic, satellite, combined) to disk 
            in a directory named 'variants'.
        - delete_masked_variants() -> None:
            Deletes the saved masked image variants from the disk.
        - process_variants(variants: Dict[str, np.ndarray], output_dir: str = "INLA_output_NPY") -> Dict[str, np.ndarray]:
            Processes each image variant through an external R script pipeline, 
            saves the results to the specified output directory, and returns the 
            processed data as a dictionary.
    """
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        self.cosmic_cfg = cosmic_cfg
        self.satellite_cfg = satellite_cfg

    def create_variants(self, 
                      data: np.ndarray, 
                      masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create all image variants from base data and masks
        
        Args:
            data: Original image data
            masks: Dictionary of boolean masks
            
        Returns:
            Dictionary containing four image variants:
            - original: Unmodified input data
            - cosmic: Data with cosmic rays applied
            - satellite: Data with satellite trails applied
            - combined: Data with both artifacts applied
        """
        return {
            'original': data,
            'cosmic': np.where(masks['cosmic'], self.cosmic_cfg.value, data),
            'satellite': np.where(masks['satellite'], 
                                 self.satellite_cfg.value, data),
            'combined': np.where(masks['combined'],
                               np.maximum(self.cosmic_cfg.value,
                                        self.satellite_cfg.value),
                               data)
        }

    def save_masked_variants(self,
                            data: np.ndarray,
                            masks: Dict[str, np.ndarray]) -> None:
        """save masked variants to disk"""
        masked_variants = {
            'cosmic': np.where(masks['cosmic'], self.cosmic_cfg.value, data),
            'satellite': np.where(masks['satellite'], 
                                 self.satellite_cfg.value, data),
            'combined': np.where(masks['combined'],
                               np.maximum(self.cosmic_cfg.value,
                                        self.satellite_cfg.value),
                               data)
        }
        # Create directory if it doesn't exist
        os.makedirs('variants', exist_ok=True)
        # Save each masked variant to disk
        for name, variant in masked_variants.items():
            np.save(f'variants/{name}.npy', variant)
            print(f"Saved {name}.npy")
        print("Masked variants saved to disk.")
        
    def delete_masked_variants(self) -> None:
        """Delete masked variants from disk"""
        for name in ['cosmic', 'satellite', 'combined']:
            try:
                os.remove(f'variants/{name}.npy')
                print(f"Deleted {name}.npy")
            except FileNotFoundError:
                print(f"{name}.npy not found, skipping deletion.") 
        print("Masked variants deleted from disk.")
    
    def process_variants(self, 
                variants: Dict[str, np.ndarray],
                output_dir: str = "INLA_output_NPY") -> Dict[str, np.ndarray]:
        """Process each variant through the R targets pipeline.
        
        Args:
            variants: Dictionary of {variant_name: numpy_array} (e.g., {'cosmic': array})
            output_dir: Directory to save processed results
            
        Returns:
            Dictionary of processed arrays with same keys as input.
        """
        processed = {}
        os.makedirs("variants", exist_ok=True)  # Input dir for .npy files
        os.makedirs(output_dir, exist_ok=True)   # Output dir for results

        i=0
        for variant_name, data in variants.items():
            try:
                i += 1
                print(f"{colors[i % len(colors)]}Processing {variant_name}...\n Calling R-INLA{Fore.RESET}")
                # 1. Save input .npy file
                input_path = f"variants/{variant_name}.npy"
                np.save(input_path, data)
                
                # 2. Save the input path to a text file
                path_file = f"variants/path.txt"
                with open(path_file, "w") as f:
                    f.write(input_path)
                
                # 3. Call R script, passing the path text file
                subprocess.run([
                    "Rscript",
                    "INLA_pipeline.R",
                ], check=True)
                
                # 4. Load processed result
                output_path = f"{output_dir}/{variant_name}/out.npy"
                processed[variant_name] = np.load(output_path)
                
                # 5. Clean up temporary input
                os.remove(input_path)
                
                # 6. Delete the path file after it is used
                os.remove(path_file)
                
            except Exception as e:
                print(f"Failed to process {variant_name}:\n {Fore.RED}Error: {str(e)}{Fore.RESET}")
                processed[variant_name] = None  # Mark as failed

        return processed

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

class PlotGenerator:
    """Handles generation of diagnostic visualizations"""
    
    def __init__(self, cmap: str = CMAP, dpi: int = PLOT_DPI):
        self.cmap = cmap
        self.dpi = dpi
        self.percentile_range = PERCENTILE_RANGE

    def generate_all_plots(self,
                        output_dir: Path,
                        variants: Dict[str, np.ndarray],
                        processed: Dict[str, np.ndarray],
                        basename: str) -> None:
        """Generate complete set of visualizations
        
        Args:
            output_dir: Base output directory
            variants: Dictionary of image variants
            processed: Dictionary of processed variants
            basename: Stem of input filename
        """
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        vmin, vmax = self._calculate_intensity_range(variants['original'])
        
        self._save_single_plots(plot_dir, variants, basename, vmin, vmax)
        self._save_comparison_plot(plot_dir, variants, basename, vmin, vmax)
        self._save_final_comparison(plot_dir, variants, processed,basename, vmin, vmax)

    def _calculate_intensity_range(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate intensity range for consistent color scaling"""
        return np.nanpercentile(data, self.percentile_range)

    def _save_single_plots(self,
                         plot_dir: Path,
                         variants: Dict[str, np.ndarray],
                         basename: str,
                         vmin: float,
                         vmax: float) -> None:
        """Save individual plots for each variant"""
        for name, data in variants.items():
            fig = plt.figure(figsize=(8, 6), dpi=self.dpi)
            plt.imshow(data, cmap=self.cmap, origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Intensity (ADU)')
            plt.title(f'{name.capitalize()}\n{basename}')
            plt.tight_layout()
            plt.savefig(plot_dir / f'{name}.png')
            plt.close(fig)

    def _save_comparison_plot(self,
                            plot_dir: Path,
                            variants: Dict[str, np.ndarray],
                            basename: str,
                            vmin: float,
                            vmax: float) -> None:
        """Generate 2x2 comparison plot"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        titles = ['Original', 'Cosmic Rays', 'Satellite Trails', 'Combined']

        for ax, (key, title) in zip(axs.flat, zip(variants.keys(), titles)):
            im = ax.imshow(variants[key], cmap=self.cmap, origin='lower',
                         vmin=vmin, vmax=vmax)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label='Intensity (ADU)')

        fig.suptitle(f"Image Comparison: {basename}")
        plt.tight_layout()
        plt.savefig(plot_dir / 'comparison.png')
        plt.close(fig)

    def _save_final_comparison(self,
                             plot_dir: Path,
                             variants: Dict[str, np.ndarray],
                             processed: Dict[str, np.ndarray],
                             basename: str,
                             vmin: float,
                             vmax: float) -> None:
        """Generate 3x3 comparison plot showing processing stages"""
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        plt.subplots_adjust(right=0.85, wspace=0.4)

        for row, variant in enumerate(['cosmic', 'satellite', 'combined']):
            # Original
            im0 = axs[row, 0].imshow(variants['original'], cmap=self.cmap,
                                   vmin=vmin, vmax=vmax)
            axs[row, 0].set_ylabel(variant.capitalize())
            if row == 0:
                axs[row, 0].set_title("Original")
            fig.colorbar(im0, ax=axs[row, 0], fraction=0.046, pad=0.04)

            # Masked
            im1 = axs[row, 1].imshow(variants[variant], cmap=self.cmap,
                                    vmin=vmin, vmax=vmax)
            if row == 0:
                axs[row, 1].set_title("Masked")
            fig.colorbar(im1, ax=axs[row, 1], fraction=0.046, pad=0.04)

            # Processed (placeholder)
            im2 = axs[row, 2].imshow(processed[variant], cmap=self.cmap,
                                    vmin=vmin, vmax=vmax)
            if row == 0:
                axs[row, 2].set_title("Processed")
            fig.colorbar(im2, ax=axs[row, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(plot_dir / 'final_comparison.png', bbox_inches='tight')
        plt.close(fig)

class SimulationPipeline:
    """Coordinates the complete simulation workflow"""
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        self.mask_generator = MaskGenerator(cosmic_cfg, satellite_cfg)
        self.fits_processor = FitsProcessor(cosmic_cfg, satellite_cfg)
        self.file_handler = FileHandler()
        self.plot_generator = PlotGenerator()

    def process_file(self, input_path: Path) -> None:
        """Execute complete processing pipeline for a single file"""
        try:
            data, header = self.file_handler.load_fits(input_path)
            basename = input_path.stem
            output_dir = self.file_handler.create_output_structure(basename)

            masks = self.mask_generator.generate_all_masks(data)
            variants = self.fits_processor.create_variants(data, masks)
            self.fits_processor.save_masked_variants(data, masks)
            processed = self.fits_processor.process_variants(variants)

            # Add processed variants to the main dictionary
            for key in processed:
                if processed[key] is not None:
                    variants[f"{key}_processed"] = processed[key]

            self.file_handler.save_outputs(output_dir, variants, masks, header)
            self.plot_generator.generate_all_plots(output_dir, variants, processed, basename)

            print(f"✅ Successfully processed {basename}")

        except (ValueError, OSError, RuntimeError) as e:
            print(f"❌ Error processing {input_path.name}: {str(e)}")
            

def main() -> None:
    """Command line interface and processing coordination"""
    parser = argparse.ArgumentParser(
        description='Simulate cosmic rays and satellite trails in FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input parameters
    parser.add_argument(
        'input',
        type=str,
        nargs='+',  # Accept multiple input arguments
        help='Input FITS file(s) (supports wildcards and directories)'
    )

    # Cosmic ray parameters
    parser.add_argument(
        '--cosmic_fraction',
        type=float,
        default=0.01,
        help='Fraction of pixels affected by cosmic rays (0-1)'
    )
    parser.add_argument(
        '--cosmic_value',
        type=float,
        default=DEFAULT_COSMIC_VALUE,
        help='ADU value for cosmic ray pixels'
    )
    parser.add_argument(
        '--cosmic_seed',
        type=int,
        default=None,
        help='Random seed for cosmic ray generation'
    )

    # Satellite parameters
    parser.add_argument(
        '--num_trails',
        type=int,
        default=1,
        help='Number of satellite trails to generate'
    )
    parser.add_argument(
        '--trail_width',
        type=int,
        default=3,
        help='Width of trails in pixels'
    )
    parser.add_argument(
        '--min_angle',
        type=float,
        default=-45.0,
        help='Minimum trajectory angle (degrees from horizontal)'
    )
    parser.add_argument(
        '--max_angle',
        type=float,
        default=45.0,
        help='Maximum trajectory angle (degrees from horizontal)'
    )
    parser.add_argument(
        '--trail_value',
        type=float,
        default=DEFAULT_SATELLITE_VALUE,
        help='ADU value for satellite trail pixels'
    )

    args = parser.parse_args()

    # Create configuration objects
    cosmic_cfg = CosmicConfig(
        fraction=args.cosmic_fraction,
        value=args.cosmic_value,
        seed=args.cosmic_seed
    )

    satellite_cfg = SatelliteConfig(
        num_trails=args.num_trails,
        trail_width=args.trail_width,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        value=args.trail_value
    )

    # Find and process files
    input_files = []
    for pattern in args.input:
        if os.path.isdir(pattern):
            input_files.extend(glob.glob(os.path.join(pattern, '*.fits')))
        else:
            input_files.extend(glob.glob(pattern))

    if not input_files:
        print("❌ No FITS files found matching input pattern")
        return

    processor = SimulationPipeline(cosmic_cfg, satellite_cfg)

    print(f"🚀 Processing {len(input_files)} files with configuration:")
    print(f"  Cosmic rays: {cosmic_cfg.fraction*100:.1f}% @ {cosmic_cfg.value} ADU")
    print(f"  Satellite trails: {satellite_cfg.num_trails} trails @ {satellite_cfg.value} ADU")
    print("─" * 50)

    for file_path in input_files:
        processor.process_file(Path(file_path))

    print("\n🎉 Processing complete! Output structure:")
    print("output/")
    print("└── [basename]/")
    print("    ├── data/")
    print("    ├── masks/")
    print("    └── plots/")


if __name__ == '__main__':
    main()
    