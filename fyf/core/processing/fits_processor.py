"""
FITS processing functionality for the FYF package.

This module provides the FitsProcessor class which handles the creation,
processing, saving, and deletion of image variants.
"""

import os
import subprocess
from pathlib import Path
import tempfile
from typing import Dict, Optional, Any

import numpy as np
from colorama import Fore, init
init()
colors = [Fore.WHITE, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

from fyf.config.config import CosmicConfig, SatelliteConfig, INLAConfig
from fyf.core.paths import (
    VARIANTS_DIR, get_variant_path, get_path_file, 
    get_inla_script_path, get_output_dir
)


print("Debug: fits_processor.py module loaded")

class FitsProcessor:
    """
    FitsProcessor handles the creation, processing, saving, and deletion of image variants
    
    This class creates image variants by applying cosmic ray and satellite trail artifacts
    to input data, saves these variants to disk, processes them with R-INLA, and cleans up
    temporary files.
    
    Attributes:
        cosmic_cfg: Configuration for cosmic ray generation
        satellite_cfg: Configuration for satellite trail generation
    """
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        """Initialize the FitsProcessor with configuration objects.
        
        Args:
            cosmic_cfg: Configuration for cosmic ray generation
            satellite_cfg: Configuration for satellite trail generation
        """
        print("Debug: Initializing FitsProcessor")
        self.cosmic_cfg = cosmic_cfg
        self.satellite_cfg = satellite_cfg
        print("Debug: FitsProcessor initialized with configs")

    def create_variants(self, 
                       data: np.ndarray, 
                       masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create all image variants from base data and masks
        
        Args:
            data: Original image data
            masks: Dictionary of boolean masks
            
        Returns:
            Dictionary containing image variants:
            - 'original': Unmodified input data
            - 'cosmic': Data with cosmic rays applied
            - 'satellite': Data with satellite trails applied
            - 'custom': Data with custom mask applied (if provided)
            - 'combined': Data with all artifacts applied
        """
        print("Debug: Entered create_variants")
        if data.size == 0:
            print("Debug: Data is empty, returning empty variants")
            return {
                'original': data,
                'cosmic': None,
                'satellite': None, 
                'custom': None,
                'combined': None,
            }
        
        print("Debug: Creating variants dictionary")
        variants = {'original': data}
        
        # Create each variant only if the corresponding mask has any True values
        for mask_name in ['cosmic', 'satellite', 'custom', 'combined']:
            if mask_name in masks and masks[mask_name] is not None:
                if np.any(masks[mask_name]):
                    variants[mask_name] = np.where(masks[mask_name], np.nan, data)
                    print(f"Debug: {mask_name} mask affects {np.sum(masks[mask_name])} pixels")
                    #print(f"Debug: {mask_name} variant created with shape: {variants[mask_name].shape}")
                else:
                    print(f"Debug: {mask_name} mask is empty, skipping variant")
                    variants[mask_name] = None
            else:
                print(f"Debug: {mask_name} mask not found, skipping variant")
                variants[mask_name] = None
        
        print("Variants created successfully.")
        return variants

    def save_masked_variants(self,
                            data: np.ndarray,
                            masks: Dict[str, np.ndarray],
                            output_dir: str = None) -> None:
        """Save masked variants to disk
        
        Args:
            data: Original image data
            masks: Dictionary of boolean masks
            output_dir: Directory to save variants (default: uses VARIANTS_DIR)
        """
        # Use the configured variants directory if none specified
        if output_dir is None:
            output_dir = VARIANTS_DIR
            
        print(f"Debug: Saving masked variants to {output_dir}")
        
        # Create the masked variants
        masked_variants = {
            'original': data,
            'cosmic': np.where(masks['cosmic'], self.cosmic_cfg.value, data),
            'satellite': np.where(masks['satellite'], self.satellite_cfg.value, data),
            'combined': np.where(masks['combined'],
                            np.maximum(self.cosmic_cfg.value, self.satellite_cfg.value),
                            data)
        }
        
        # Add custom variant if custom mask exists
        if 'custom' in masks and masks['custom'] is not None:
            masked_variants['custom'] = np.where(masks['custom'], np.nan, data)

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save each masked variant to disk
        for name, variant in masked_variants.items():
            # Create absolute path for each variant
            variant_path = os.path.join(output_dir, f"{name}.npy")
            np.save(variant_path, variant)
            print(f"Debug: Saved {name}.npy to {variant_path}")
        
        print("Masked variants saved to disk.")

    def delete_masked_variants(self, output_dir: str = None) -> None:
        """Delete masked variants from disk
        
        Args:
            output_dir: Directory containing variants (default: uses VARIANTS_DIR)
        """
        # Use the configured variants directory if none specified
        if output_dir is None:
            output_dir = VARIANTS_DIR
            
        print(f"Debug: Deleting masked variants from {output_dir}")
        
        variant_names = ['original', 'cosmic', 'satellite', 'combined', 'custom']
        
        for name in variant_names:
            variant_path = os.path.join(output_dir, f"{name}.npy")
            if os.path.exists(variant_path):
                try:
                    os.remove(variant_path)
                    print(f"Debug: Deleted {name}.npy from {output_dir}")
                except Exception as e:
                    print(f"Debug: Error deleting {name}.npy: {e}")
        
        # Also remove path.txt if it exists
        path_file = os.path.join(output_dir, "path.txt")
        if os.path.exists(path_file):
            try:
                os.remove(path_file)
                print(f"Debug: Deleted path.txt from {output_dir}")
            except Exception as e:
                print(f"Debug: Error deleting path.txt: {e}")
        
        print("Masked variants deletion completed.")

    
    def process_variants(self,
                    variants: Dict[str, np.ndarray],
                    inla_config: Optional[INLAConfig] = None,
                    output_dir: str = "INLA_output_NPY") -> Dict[str, np.ndarray]:
        """Process each variant through the R-INLA pipeline
        
        Args:
            variants: Dictionary of {variant_name: numpy_array}
            inla_config: Optional INLA configuration
            output_dir: Directory to save processed results
            
        Returns:
            Dictionary of processed arrays with same keys as input
        """
        # Use absolute paths
        variants_dir = VARIANTS_DIR
        output_dir = os.path.abspath(output_dir)
        
        print(f"Debug: Processing variants. Variants dir: {variants_dir}, Output dir: {output_dir}")
        
        processed = {}
        os.makedirs(variants_dir, exist_ok=True)   # Ensure variants dir exists
        os.makedirs(output_dir, exist_ok=True)     # Ensure output dir exists

        # Get absolute path to the R script
        inla_script_path = get_inla_script_path()
        print(f"Debug: Using R script at: {inla_script_path}")

        i = 0
        for variant_name, data in variants.items():
            # Skip original (often doesn't need processing) and None variants
            if variant_name == 'original' and len(variants) > 1:
                continue
                
            # Skip variants that are None or have no masked pixels (identical to original)
            if data is None:
                processed[variant_name] = None
                continue
                
            # Skip variants that are identical to original
            if variant_name != 'original' and np.array_equal(data, variants['original']):
                processed[variant_name] = None
                continue
            try:
                i += 1
                print(f"Debug: Processing variant {variant_name}")
                

                
                print(f"{colors[i % len(colors)]}Processing {variant_name}...\n Calling R-INLA{Fore.RESET}")
                
                # 1. Save input .npy file with absolute path
                input_path = os.path.join(variants_dir, f"{variant_name}.npy")
                np.save(input_path, data)
                print(f"Debug: Saved input file to {input_path}")
                
                # 2. Save the input path to a text file with absolute path
                path_file = os.path.join(variants_dir, "path.txt")
                with open(path_file, "w") as f:
                    f.write(input_path)
                print(f"Debug: Saved path file to {path_file}")
                
                # 3. Build R script command with INLA config and absolute path
                cmd = ["Rscript", str(inla_script_path)]
                
                # Add INLA configuration parameters if provided
                if inla_config:
                    print("Debug: Adding INLA config to command")
                    if inla_config.shape != "none":
                        cmd.extend(["--shape", inla_config.shape])
                    if inla_config.mesh_cutoff is not None:
                        cmd.extend(["--mesh-cutoff", str(inla_config.mesh_cutoff)])
                    if inla_config.tolerance != 1e-4:
                        cmd.extend(["--tolerance", str(inla_config.tolerance)])
                    if inla_config.restart != 0:
                        cmd.extend(["--restart", str(inla_config.restart)])
                    if inla_config.scaling:
                        cmd.append("--scaling")
                    if inla_config.nonstationary:
                        cmd.append("--nonstationary")
                
                print(f"Debug: R script command: {' '.join(cmd)}")

                # 4. Call R script
                try:
                    print("Debug: Running R script subprocess")
                    # Create a variant-specific output directory
                    variant_output_dir = os.path.join(output_dir, variant_name)
                    os.makedirs(variant_output_dir, exist_ok=True)
                    
                    # Set environment variable to tell the R script where to save output
                    env = os.environ.copy()
                    env["FYF_OUTPUT_DIR"] = variant_output_dir
                    
                    subprocess.run(cmd, check=True, env=env)
                    print(f"Debug: R script executed successfully")
                    
                    # 5. Load processed result
                    output_path = os.path.join(variant_output_dir, "out.npy")
                    print(f"Debug: Checking for output file: {output_path}")
                    
                    if os.path.exists(output_path):
                        processed[variant_name] = np.load(output_path)
                        print(f"Debug: Loaded output from {output_path}")
                    else:
                        print(f"Warning: Output file not found at {output_path}")
                        if os.path.exists(variant_output_dir):
                            print(f"Debug: Output directory contents: {os.listdir(variant_output_dir)}")
                        processed[variant_name] = None
                        
                except subprocess.CalledProcessError as e:
                    print(f"Error running R script: {e}")
                    if hasattr(e, 'output'):
                        print(f"Debug: Command output: {e.output}")
                    processed[variant_name] = None
                
            except Exception as e:
                print(f"Failed to process {variant_name}:\n {Fore.RED}Error: {str(e)}{Fore.RESET}")
                processed[variant_name] = None  # Mark as failed

            finally:
                # 6. Ensure temporary files are removed (only if we created them)
                try:
                    if 'input_path' in locals():
                        if os.path.exists(input_path):
                            print(f"Debug: Cleaning up temporary input file {input_path}")
                            os.remove(input_path)
                    
                    if 'path_file' in locals():
                        if os.path.exists(path_file):
                            print(f"Debug: Cleaning up temporary path file {path_file}")
                            os.remove(path_file)
                except Exception as cleanup_error:
                    print(f"Debug: Error during cleanup: {cleanup_error}")

        print("Debug: Finished processing all variants")
        return processed 