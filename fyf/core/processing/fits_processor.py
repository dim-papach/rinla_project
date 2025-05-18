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

from fyf.core.config import CosmicConfig, SatelliteConfig, INLAConfig


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
            Dictionary containing image variants:
            - 'original': Unmodified input data
            - 'cosmic': Data with cosmic rays applied
            - 'satellite': Data with satellite trails applied
            - 'custom': Data with custom mask applied (if provided)
            - 'combined': Data with all artifacts applied
        """
        variants = {
            'original': data,
            'cosmic': np.where(masks['cosmic'], self.cosmic_cfg.value, data),
            'satellite': np.where(masks['satellite'], self.satellite_cfg.value, data)
        }
        
        # Add custom variant if custom mask exists
        if 'custom' in masks and masks['custom'] is not None:
            variants['custom'] = np.where(masks['custom'], np.nan, data)
        
        # Create combined variant
        variants['combined'] = np.where(masks['combined'], 
                                      np.maximum(self.cosmic_cfg.value, self.satellite_cfg.value),
                                      data)
        
        return variants

    def save_masked_variants(self,
                            data: np.ndarray,
                            masks: Dict[str, np.ndarray],
                            output_dir: str = 'variants') -> None:
        """Save masked variants to disk
        
        Args:
            data: Original image data
            masks: Dictionary of boolean masks
            output_dir: Directory to save variants (default: 'variants')
        """
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
            np.save(f'{output_dir}/{name}.npy', variant)
            print(f"Saved {name}.npy")
        
        print("Masked variants saved to disk.")
        
    def delete_masked_variants(self, output_dir: str = 'variants') -> None:
        """Delete masked variants from disk
        
        Args:
            output_dir: Directory containing variants (default: 'variants')
        """
        variant_names = ['original', 'cosmic', 'satellite', 'combined', 'custom']
        
        for name in variant_names:
            variant_path = f'{output_dir}/{name}.npy'
            if os.path.exists(variant_path):
                try:
                    os.remove(variant_path)
                    print(f"Deleted {name}.npy")
                except FileNotFoundError:
                    print(f"{name}.npy not found, skipping deletion.")
                except Exception as e:
                    print(f"Error deleting {name}.npy: {e}")
        
        print("Masked variants deleted from disk.")
    
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
        processed = {}
        os.makedirs("variants", exist_ok=True)  # Input dir for .npy files
        os.makedirs(output_dir, exist_ok=True)   # Output dir for results

        i = 0
        for variant_name, data in variants.items():
            try:
                i += 1
                print(f"{colors[i % len(colors)]}Processing {variant_name}...\n Calling R-INLA{Fore.RESET}")
                
                # Skip original if it's included (often doesn't need processing)
                if variant_name == 'original' and len(variants) > 1:
                    print(f"Skipping original variant (no processing needed)")
                    continue
                
                # 1. Save input .npy file
                input_path = f"variants/{variant_name}.npy"
                np.save(input_path, data)
                
                # 2. Save the input path to a text file
                path_file = f"variants/path.txt"
                with open(path_file, "w") as f:
                    f.write(input_path)
                
                # 3. Build R script command with INLA config
                cmd = ["Rscript", "fyf/r/INLA_pipeline.R"]
                
                # Add INLA configuration parameters if provided
                if inla_config:
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
                
                # 4. Call R script
                try:
                    subprocess.run(cmd, check=True)
                    
                    # 5. Load processed result
                    output_path = f"{output_dir}/{variant_name}/out.npy"
                    if os.path.exists(output_path):
                        processed[variant_name] = np.load(output_path)
                    else:
                        print(f"Warning: Output file not found at {output_path}")
                        processed[variant_name] = None
                        
                except subprocess.CalledProcessError as e:
                    print(f"Error running R script: {e}")
                    processed[variant_name] = None
                
            except Exception as e:
                print(f"Failed to process {variant_name}:\n {Fore.RED}Error: {str(e)}{Fore.RESET}")
                processed[variant_name] = None  # Mark as failed

            finally:
                # 6. Ensure temporary files are removed
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(path_file):
                    os.remove(path_file)

        return processed