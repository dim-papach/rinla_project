import os
import numpy as np
import subprocess
from typing import Dict
from fyf.core.config import CosmicConfig, SatelliteConfig

from colorama import Fore, init
init()
colors = [Fore.WHITE, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

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
        #os.makedirs(output_dir, exist_ok=True)

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
                
            except Exception as e:
                print(f"Failed to process {variant_name}:\n {Fore.RED}Error: {str(e)}{Fore.RESET}")
                processed[variant_name] = None  # Mark as failed

            finally:
                # 5. Ensure temporary files are removed
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(path_file):
                    os.remove(path_file)

        return processed
