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
        print("Debug: Initializing SimulationPipeline")
        self.mask_generator = MaskGenerator(cosmic_cfg, satellite_cfg)
        print("Debug: MaskGenerator initialized")
        self.fits_processor = FitsProcessor(cosmic_cfg, satellite_cfg)
        print("Debug: FitsProcessor initialized")
        self.file_handler = FileHandler()
        print("Debug: FileHandler initialized")
        if plot_cfg is None:
            self.plot_generator = PlotGenerator()
            print("Debug: PlotGenerator initialized with default config")
        else:
            self.plot_generator = PlotGenerator(
                cmap=plot_cfg.cmap,
                dpi=plot_cfg.dpi,
                residual_cmap=plot_cfg.residual_cmap,
                percentile_range=plot_cfg.percentile_range,
                residual_percentile=plot_cfg.residual_percentile
            )
            print("Debug: PlotGenerator initialized with custom config")
        self.inla_config = inla_cfg or INLAConfig()
        print("Debug: INLAConfig initialized")

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
        print(f"Debug: Entered process_file with input_path={input_path}, custom_mask_path={custom_mask_path}")

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
            print(f"Debug: Loading FITS file {input_path}")
            data, header = self.file_handler.load_fits(input_path)
            print(f"Debug: FITS loaded, shape: {data.shape}, dtype: {data.dtype}")
            basename = input_path.stem
            output_dir = self.file_handler.create_output_structure(basename)
            print(f"Debug: Output directory created at {output_dir}")
            result['output_dir'] = output_dir
            
            # Generate masks with optional custom mask
            print(f"Debug: Generating masks, custom mask: {custom_mask_path}")
            masks = self.mask_generator.generate_all_masks(data, custom_mask_path)
            print(f"Debug: Masks generated, types: {list(masks.keys())}, shapes: {[m.shape for m in masks.values()]}")
            result['masks'] = masks
            
            # Create variants
            print("Debug: Creating variants")
            variants = self.fits_processor.create_variants(data, masks)
            print(f"Debug: Variants created, types: {list(variants.keys())}, shapes: {[v.shape for v in variants.values()]}")
            result['variants'] = variants
            
            # Save masked variants for processing
            print("Debug: Saving masked variants")
            self.fits_processor.save_masked_variants(data, masks)
            print("Debug: Masked variants saved")
            
            # Process with INLA
            print("Debug: Processing variants with INLA")
            if self.inla_config:
                print("Debug: INLAConfig is set, calling process_variants with inla_config")
                processed = self.fits_processor.process_variants(variants, inla_config=self.inla_config)
            else:
                print("Debug: INLAConfig is not set, calling process_variants without inla_config")
                processed = self.fits_processor.process_variants(variants)
            
            print(f"Debug: Processing complete, results: {list(processed.keys())}")
            for k, v in processed.items():
                print(f"Debug: {k} result type: {type(v)}, none?: {v is None}")
                if v is not None:
                    print(f"Debug: {k} shape: {v.shape}, dtype: {v.dtype}")
        
            result['processed'] = processed
            
            # Add processed variants to the main dictionary
            for key in processed:
                if processed[key] is not None:
                    variants[f"{key}_processed"] = processed[key]
                    print(f"Debug: {key}_processed added to variants, shape: {processed[key].shape}")
                else:
                    print(f"Debug: {key}_processed is None, not added to variants")
            
            # Save outputs
            print(f"Debug: Saving outputs to {output_dir}")
            self.file_handler.save_outputs(output_dir, variants, masks, header)
            print(f"Debug: Outputs saved to {output_dir}")
            
            # Generate plots
            # When generating plots
            if self.plot_generator is not None:
                print(f"Debug: Generating plots for {basename}")
                # Filter out None variants before plotting
                filtered_variants = {k: v for k, v in variants.items() if v is not None}
                filtered_processed = {k: v for k, v in processed.items() if v is not None}
                
                if filtered_variants and filtered_processed:
                    self.plot_generator.generate_all_plots(output_dir, filtered_variants, filtered_processed, basename)
                    print(f"Debug: Plots generated for {basename}")
                else:
                    print(f"Debug: Skipping plot generation (no valid variants/processed data)")
            '''if self.plot_generator is not None:
                print(f"Debug: Generating plots for {basename}")
                self.plot_generator.generate_all_plots(output_dir, variants, processed, basename)
                print(f"Debug: Plots generated for {basename}")'''
            
            # Validate results
            print("Debug: Validating processed images")
            metrics = {}
            for key, proc_data in processed.items():
                if proc_data is not None:
                    print(f"Debug: Validating {key}")
                    metrics[key] = validate_images(variants['original'], proc_data)
                    print(f"Debug: Metrics for {key}: {metrics[key]}")
                else:
                    print(f"Debug: Skipping validation for {key} (None)")
            
            result['metrics'] = metrics
            result['success'] = True
            print(f"Debug: process_file for {input_path} completed successfully")
            
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error processing {input_path.name}: {str(e)}")
            result['error'] = str(e)
        # cleanup    
        finally:
            print("Debug: Cleaning up masked variants")
            try:
                self.fits_processor.delete_masked_variants()
                print("Debug: Masked variants deleted")
            except Exception as cleanup_e:
                print(f"Debug: Exception during cleanup: {cleanup_e}")
            
            result['process_time'] = time.time() - start_time
            print(f"Debug: Processing time for {input_path}: {result['process_time']:.2f} seconds")
        
        print(f"Debug: Returning result for {input_path}")
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
        print(f"Debug: Entered process_file with input_path={input_path}, custom_mask_path={custom_mask_path}")
        results = {}

        for i, input_path in enumerate(input_paths):
            print(f"Debug: Processing file {i+1} of {len(input_paths)}: {input_path.name}")
            try:
                print(f"Debug: Calling process_file for {input_path.name}")
                result = self.process_file(input_path, custom_mask_path=custom_mask_path)
                print(f"Debug: Finished processing {input_path.name}, success: {result.get('success', False)}")
                results[input_path.name] = result
            except Exception as e:
                print(f"Error processing {input_path.name}: {str(e)}")
                results[input_path.name] = {
                    'success': False,
                    'error': str(e),
                    'process_time': 0.0
                }
        
        print("Debug: batch_process completed")
        print(f"Debug: Returning batch results for {len(input_paths)} files")
        return results