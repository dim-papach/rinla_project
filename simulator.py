#!/usr/bin/env python3
"""
COSMIC RAY & SATELLITE TRAIL SIMULATOR

A configurable pipeline for generating realistic artifacts in astronomical images.
Produces FITS files, binary masks, and diagnostic plots in an organized output structure.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import binary_dilation

# Constants
DEFAULT_COSMIC_VALUE = np.float32(np.nan)
DEFAULT_SATELLITE_VALUE = np.float32(np.nan)
CMAP = 'viridis'
PLOT_DPI = 150
PERCENTILE_RANGE = (1, 99)


@dataclass(frozen=True)
class CosmicConfig:
    """Configuration parameters for cosmic ray simulation"""
    fraction: float
    value: float = DEFAULT_COSMIC_VALUE
    seed: Optional[int] = None


@dataclass(frozen=True)
class SatelliteConfig:
    """Configuration parameters for satellite trail simulation"""
    num_trails: int
    trail_width: int
    min_angle: float = -45.0
    max_angle: float = 45.0
    value: float = DEFAULT_SATELLITE_VALUE


class FitsProcessor:
    """Main processor class for handling FITS files and artifacts"""

    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        self.cosmic_cfg = cosmic_cfg
        self.satellite_cfg = satellite_cfg
        self.rng = np.random.default_rng(cosmic_cfg.seed)

    def process_file(self, input_path: Path) -> None:
        """Enhanced pipeline with processing and final comparison plot"""
        try:
            # 1. Load data (unchanged)
            with fits.open(input_path) as hdul:
                data = self._validate_data(hdul[0].data)
                header = hdul[0].header

            basename = input_path.stem
            output_dir = self._create_output_structure(basename)

            # 2. Generate masks and variants (unchanged)
            masks = self._generate_masks(data)
            variants = self._create_variants(data, masks)

            # 3. NEW: Process variants (placeholder for now)
            processed = self._process_variants(variants)

            # 4. Save outputs (unchanged)
            self._save_outputs(output_dir, variants, masks, header)

            # 5. Generate plots (now includes final comparison)
            self._generate_plots(output_dir, variants, basename)
            self._generate_final_comparison(basename, data, variants, processed)  # NEW

            print(f"✅ Successfully processed {basename}")

        except (ValueError, OSError, RuntimeError) as e:
            print(f"❌ Error processing {input_path.name}: {str(e)}")

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        """Ensure input data is 2D and convert to float"""
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
        return data.astype(np.float32)

    def _generate_masks(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate all required boolean masks"""
        return {
            'cosmic': self._generate_cosmic_mask(data),
            'satellite': self._generate_satellite_mask(data.shape),
            'combined': None  # Will be calculated later
        }

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
            # Vectorized trail generation
            start, angle = self._random_trail_parameters(width, height)
            coords = self._calculate_trail_coordinates(start,
                                                       angle,
                                                       max(height,
                                                           width))
            self._apply_trail(mask, coords)

        if self.satellite_cfg.trail_width > 1:
            mask = binary_dilation(mask,
                                   structure=np.ones((self.satellite_cfg.trail_width,
                                                      self.satellite_cfg.trail_width)))
        return mask

    def _create_variants(self,
                         data: np.ndarray,
                         masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create all image variants from base data and masks"""
        masks['combined'] = masks['cosmic'] | masks['satellite']

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
    def _process_variants(self, variants: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process masked images. Currently returns unmodified data as placeholder."""
        return {
            'original': variants['original'],
            'cosmic': variants['cosmic'],  # Placeholder - no processing yet
            'satellite': variants['satellite'],
            'combined': variants['combined']
        }

    def _save_outputs(self,
                      output_dir: Path,
                      variants: Dict[str, np.ndarray],
                      masks: Dict[str, np.ndarray],
                      header: fits.Header) -> None:
        """Save all output files to organized directory structure"""
        # Save FITS variants
        (output_dir / 'data').mkdir(exist_ok=True)
        for name, data in variants.items():
            fits.writeto(output_dir / 'data' / f'{name}.fits',
                         data, header, overwrite=True)

        # Save masks
        (output_dir / 'masks').mkdir(exist_ok=True)
        for name, mask in masks.items():
            fits.writeto(output_dir / 'masks' / f'{name}_mask.fits',
                         mask.astype(np.uint8), header, overwrite=True)

    def _generate_plots(self,
                        output_dir: Path,
                        variants: Dict[str, np.ndarray],
                        basename: str) -> None:
        """Generate all diagnostic visualizations"""
        (output_dir / 'plots').mkdir(exist_ok=True)
        vmin, vmax = np.nanpercentile(variants['original'], PERCENTILE_RANGE)

        # Individual plots
        for name, data in variants.items():
            self._save_single_plot(data, output_dir / 'plots' / f'{name}.png',
                                   f'{name.capitalize()}\n{basename}',
                                   vmin, vmax)

        # Comparison plot
        self._save_comparison_plot(variants, output_dir / 'plots' / 'comparison.png',
                                   basename, vmin, vmax)

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

        x_coords = np.linspace(x0, x1, 1000)
        y_coords = np.linspace(y0, y1, 1000)
        return np.round(x_coords).astype(int), np.round(y_coords).astype(int)

    def _apply_trail(self,
                     mask: np.ndarray,
                     coords: Tuple[np.ndarray, np.ndarray]) -> None:
        """Apply valid coordinates to mask in a vectorized manner"""
        x, y = coords
        valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        mask[y[valid], x[valid]] = True

    def _save_single_plot(self,
                          data: np.ndarray,
                          path: Path,
                          title: str,
                          vmin: float,
                          vmax: float) -> None:
        """Save individual plot with consistent styling"""
        plt.figure(figsize=(8, 6), dpi=PLOT_DPI)
        plt.imshow(data, cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Intensity (ADU)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _save_comparison_plot(self,
                              variants: Dict[str, np.ndarray],
                              path: Path,
                              basename: str,
                              vmin: float,
                              vmax: float) -> None:
        """Generate 2x2 comparison plot"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=PLOT_DPI)
        titles = ['Original', 'Cosmic Rays', 'Satellite Trails', 'Combined']

        for ax, (key, title) in zip(axs.flat, zip(variants.keys(), titles)):
            im = ax.imshow(variants[key], cmap=CMAP, origin='lower',
                           vmin=vmin, vmax=vmax)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label='Intensity (ADU)')

        fig.suptitle(f"Image Comparison: {basename}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _generate_final_comparison(self, basename: str, original: np.ndarray,
                                variants: Dict[str, np.ndarray], processed: Dict[str, np.ndarray]):
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        plt.subplots_adjust(right=0.85, wspace=0.4)  # Adjust spacing

        vmin, vmax = np.nanpercentile(original, PERCENTILE_RANGE)

        for row, variant in enumerate(['cosmic', 'satellite', 'combined']):
            # Original
            im0 = axs[row, 0].imshow(original, cmap=CMAP, vmin=vmin, vmax=vmax)
            axs[row, 0].set_ylabel(variant.capitalize())
            if row == 0: axs[row, 0].set_title("Original")
            plt.colorbar(im0, ax=axs[row, 0], fraction=0.046, pad=0.04)

            # Masked
            im1 = axs[row, 1].imshow(variants[variant], cmap=CMAP, vmin=vmin, vmax=vmax)
            if row == 0: axs[row, 1].set_title("Masked")
            plt.colorbar(im1, ax=axs[row, 1], fraction=0.046, pad=0.04)

            # Processed
            im2 = axs[row, 2].imshow(processed[variant], cmap=CMAP, vmin=vmin, vmax=vmax)
            if row == 0: axs[row, 2].set_title("Processed")
            plt.colorbar(im2, ax=axs[row, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"output/{basename}/final_comparison.png", bbox_inches='tight')
        plt.close()

    @staticmethod
    def _create_output_structure(basename: str) -> Path:
        """Create standardized output directory structure"""
        output_dir = Path('output') / basename
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


def main():
    """Command line interface and processing coordination"""
    parser = argparse.ArgumentParser(
        description='Simulate cosmic rays and satellite trails in FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input parameters
    parser.add_argument(
        'input',
        type=str,
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
    if os.path.isdir(args.input):
        input_files = glob.glob(os.path.join(args.input, '*.fits'))
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        print("❌ No FITS files found matching input pattern")
        return

    processor = FitsProcessor(cosmic_cfg, satellite_cfg)

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
