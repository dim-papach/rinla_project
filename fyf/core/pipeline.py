"""
Main pipeline coordination for the FYF package.

This module provides the SimulationPipeline class which coordinates the complete
workflow for processing astronomical images with R-INLA.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Any, Union

import numpy as np

from fyf.core.config import CosmicConfig, SatelliteConfig, INLAConfig, PlotConfig
from fyf.core.data.file_handler import FileHandler
from fyf.core.data.masking import MaskGenerator
from fyf.core.processing.fits_processor import FitsProcessor
from fyf.core.visualization.plotting import PlotGenerator
from fyf.core.validation import validate_images


class SimulationPipeline:
    """
    Coordinates the complete simulation workflow
    
    This class brings together all components of the FYF package to provide
    a seamless pipeline for processing astronomical images with R-INLA.
    
    Attributes:
        mask_generator: MaskGenerator instance
        fits_processor: FitsProcessor instance
        file_handler: FileHandler instance
        plot_generator: PlotGenerator instance
        inla_config: INLAConfig instance
    """
    
    def __init__(self, 
               cosmic_cfg: CosmicConfig, 
               satellite_cfg: SatelliteConfig,
               inla_cfg: Optional[INLAConfig] = None,
               plot_cfg: Optional[PlotConfig] = None):
        """
        Initialize the SimulationPipeline with configuration objects.
        
        Args:
            cosmic_cfg: Configuration for cosmic ray generation
            satellite_cfg: Configuration for satellite trail generation
            inla_cfg: Configuration for INLA processing (optional)
            plot_cfg: Configuration for plotting (optional)
        """
        self.mask_generator = MaskGenerator(cosmic_cfg, satellite_cfg)
        self.fits_processor = FitsProcessor(cosmic_cfg, satellite_cfg)
        self.file_handler = FileHandler()
        self.plot_generator = PlotGenerator() if plot_cfg is None else PlotGenerator(
            cmap=plot_cfg.cmap,
            dpi=plot_cfg.dpi,
            residual_cmap=plot_cfg.residual_cmap,
            percentile_range=plot_cfg.percentile_range,
            residual_percentile=plot_cfg.residual_percentile
        )
        self.inla_config = inla_cfg or INLAConfig()

    def process_file(self, 
                    input_path: Path, 
                    custom_mask_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Execute complete processing pipeline for a single file
        
        Args:
            input_path: Path to input FITS file
            custom_mask_path: Optional path to custom mask file
            
        Returns:
            Dictionary containing:
            - 'success': Whether processing was successful
            - 'output_dir': Path to output directory
            - 'variants': Dictionary of image variants
            - 'masks': Dictionary of masks
            - 'processed': Dictionary of processed images
            - 'metrics': Dictionary of validation metrics
            - 'process_time': Time taken for processing
            
        Raises:
            ValueError: If input data is invalid
            OSError: If file I/O fails
            RuntimeError: If processing fails
        """
        start_time = time.time()
        result = {
            'success': False,
            'process_time': 0.0,
            'output_dir': None,
            'variants': None,
            'masks': None,
            'processed': None,
            'metrics': None
        }
        
        try:
            # Load FITS data
            data, header = self.file_handler.load_fits(input_path)
            basename = input_path.stem
            output_dir = self.file_handler.create_output_structure(basename)
            result['output_dir'] = output_dir
            
            # Generate masks with optional custom mask
            masks = self.mask_generator.generate_all_masks(data, custom_mask_path)
            result['masks'] = masks
            
            # Create variants
            variants = self.fits_processor.create_variants(data, masks)
            result['variants'] = variants
            
            # Save masked variants for processing
            self.fits_processor.save_masked_variants(data, masks)
            
            # Process with INLA
            if self.inla_config:
                processed = self.fits_processor.process_variants(variants, inla_config=self.inla_config)
            else:
                processed = self.fits_processor.process_variants(variants)
                
            result['processed'] = processed
            
            # Add processed variants to the main dictionary
            for key in processed:
                if processed[key] is not None:
                    variants[f"{key}_processed"] = processed[key]
            
            # Save outputs
            self.file_handler.save_outputs(output_dir, variants, masks, header)
            
            # Generate plots
            if self.plot_generator is not None:
                self.plot_generator.generate_all_plots(output_dir, variants, processed, basename)
            
            # Validate results
            metrics = {}
            for key, proc_data in processed.items():
                if proc_data is not None:
                    metrics[key] = validate_images(variants['original'], proc_data)
            
            result['metrics'] = metrics
            result['success'] = True
            
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error processing {input_path.name}: {str(e)}")
            result['error'] = str(e)
            
        finally:
            # Clean up
            try:
                self.fits_processor.delete_masked_variants()
            except:
                pass
            
            # Record processing time
            result['process_time'] = time.time() - start_time
        
        return result
    
    def batch_process(self, 
                     input_paths: list, 
                     custom_mask_path: Optional[Union[str, Path]] = None
                     ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple files in batch mode
        
        Args:
            input_paths: List of paths to input FITS files
            custom_mask_path: Optional path to custom mask file
            
        Returns:
            Dictionary of {filename: results} for each processed file
        """
        results = {}
        
        for i, input_path in enumerate(input_paths):
            print(f"Processing file {i+1} of {len(input_paths)}: {input_path.name}")
            try:
                result = self.process_file(input_path)
                results[input_path.name] = result
            except Exception as e:
                print(f"Error processing {input_path.name}: {str(e)}")
                results[input_path.name] = {
                    'success': False,
                    'error': str(e),
                    'process_time': 0.0
                }
        
        return results