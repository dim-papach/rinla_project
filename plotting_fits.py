"""
plotting.py - Visualization components for astronomical data processing
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import logging
import subprocess
import numpy as np
import matplotlib
# Ensure matplotlib uses a non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import binary_dilation

CMAP: str = 'viridis'
RESIDUAL_CMAP: str = 'viridis'
PLOT_DPI: int = 150
PERCENTILE_RANGE: Tuple[int, int] = (1, 99)
RESIDUAL_PERCENTILE: Tuple[int, int] = (1, 99)

class InvalidDataError(Exception):
    """Custom exception for invalid data during plotting"""
    pass

class PlotGenerator:
    """Handles generation of diagnostic visualizations with validation"""
    
    def __init__(self, cmap: str = CMAP, dpi: int = PLOT_DPI, 
                 residual_cmap: str = RESIDUAL_CMAP,
                 percentile_range: Tuple[int, int] = PERCENTILE_RANGE,
                 residual_percentile = RESIDUAL_PERCENTILE) -> None:
        self.cmap = cmap
        self.dpi = dpi
        self.percentile_range = PERCENTILE_RANGE
        self.residual_cmap = residual_cmap
        self.residual_percentile = residual_percentile
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_all_plots(self,
                         output_dir: Path,
                         variants: Dict[str, np.ndarray],
                         processed: Dict[str, np.ndarray],
                         basename: str) -> None:
        """
        Generate complete set of visualizations with validation
        """
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            vmin, vmax = self._calculate_intensity_range(variants['original'])
            
            self._save_single_plots(plot_dir, variants, basename, vmin, vmax)
            self._save_comparison_plot(plot_dir, variants, basename, vmin, vmax)
            self._save_final_comparison(plot_dir, variants, processed, basename, vmin, vmax)
            self._save_residual_plots(plot_dir, variants['original'], processed, basename)
            
        except (ValueError, TypeError) as e:
            raise InvalidDataError(f"Data validation failed: {str(e)}") from e


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
        fig, axs = plt.subplots(3, 4, figsize=(16, 12), dpi=self.dpi)
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
            
            # Residual
            with np.errstate(divide='ignore', invalid='ignore'):
                residual = ((variants['original'] - processed[variant]) / processed[variant]) * 100
                residual[np.isinf(residual)] = np.nan
                vmin_res, vmax_res = np.nanpercentile(residual, self.residual_percentile)
                im3 = axs[row, 3].imshow(residual, cmap=self.residual_cmap,
                                        vmin=vmin_res, vmax=vmax_res)
                if row == 0:
                    axs[row, 3].set_title("Residual")
                fig.colorbar(im3, ax=axs[row, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(plot_dir / 'final_comparison.png', bbox_inches='tight')
        plt.close(fig)

    def _save_residual_plots(self,
                        plot_dir: Path,
                        original: np.ndarray,
                        processed: Dict[str, np.ndarray],
                        basename: str) -> None:
        """Save residual percentage plots for each processed version"""
        for name, proc_data in processed.items():
            with np.errstate(divide='ignore', invalid='ignore'):
                residual = ((original - proc_data) / proc_data) * 100  # Percentage
                # Handle division by zero and invalid values
                residual[np.isinf(residual)] = np.nan
                # Handle residuals that are too large
                residual[np.abs(residual) >= 100] = np.nan
                # Set the color limits based on the specified percentile                
            vmin, vmax = np.nanpercentile(residual, self.residual_percentile)
            
            fig = plt.figure(figsize=(16, 6), dpi=self.dpi)
            
            # Left subplot - Residual image
            ax1 = plt.subplot(1, 2, 1)
            im = ax1.imshow(residual, cmap=self.residual_cmap, 
                        origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax1, label='Residual Percentage (%)')
            ax1.set_title(f'Residual: {name.capitalize()}\n{basename}')
            
            # Right subplot - Histogram
            ax2 = plt.subplot(1, 2, 2)
            # Filter out NaN values for histogram
            residual_hist = residual[~np.isnan(residual)]
            ax2.hist(residual_hist, bins=100, color='blue', alpha=0.7)
            ax2.set_xlabel('Residual Percentage (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Residual Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add vertical lines for mean and percentiles
            mean_residual = np.nanmean(residual)
            ax2.axvline(mean_residual, color='red', linestyle='--', label=f'Mean: {mean_residual:.2f}%')
            ax2.axvline(vmin, color='green', linestyle=':', label=f'Percentile {self.residual_percentile[0]}%: {vmin:.2f}%')
            ax2.axvline(vmax, color='green', linestyle=':', label=f'Percentile {self.residual_percentile[1]}%: {vmax:.2f}%')
            ax2.legend()
            
            # Add stats as suptitle
            min_residual = np.nanmin(residual)
            max_residual = np.nanmax(residual)
            nan_percentage = np.sum(np.isnan(residual)) / residual.size * 100
            plt.suptitle(f'Mean: {mean_residual:.2f}%, Min: {min_residual:.2f}%, Max: {max_residual:.2f}%,' 
                        + r' Higher than $\pm$ 100%: ' + f'{nan_percentage:.2f}%')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(plot_dir / f'residual_{name}.png')
            plt.close(fig)
                
