"""
Python wrapper for the R-INLA package.

This module provides a Python interface to the R-INLA package for spatial statistics,
allowing for seamless integration with the rest of the FYF package.
"""

import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Union, Tuple

from fyf.config.config import INLAConfig


class INLAWrapper:
    """Python wrapper for R-INLA
    
    This class provides a Python interface to the R-INLA package, handling the
    interaction between Python and R and managing temporary files.
    
    Attributes:
        r_script_path: Path to the R script containing INLA functions
    """
    
    def __init__(self, r_script_path: Optional[str] = None):
        """Initialize the INLA wrapper.
        
        Args:
            r_script_path: Path to the R script containing INLA functions.
                           If None, defaults to "fyf/r/INLA_pipeline.R".
        """
        if r_script_path is None:
            # Default R script path if not provided
            package_dir = Path(__file__).parent.parent.parent
            self.r_script_path = package_dir / "r" / "INLA_pipeline.R"
        else:
            self.r_script_path = Path(r_script_path)
        
        # Check that the R script exists
        if not self.r_script_path.exists():
            raise FileNotFoundError(f"R script not found: {self.r_script_path}")
    
    def process_data(self, data: np.ndarray, config: INLAConfig,
                   output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Process data using R-INLA.
        
        Args:
            data: Input image data as a 2D numpy array
            config: INLA configuration
            output_dir: Directory to save output files. If None, uses a temporary directory.
            
        Returns:
            Dictionary with processed results:
            - 'out': Reconstructed image
            - 'outsd': Standard deviation of reconstruction
            - Other fields depending on the INLA model
        """
        # Create temporary directory if output_dir not specified
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            var_dir = Path(temp_dir) / "variants"
            out_dir = Path(temp_dir) / "output"
        else:
            var_dir = Path(output_dir) / "variants"
            out_dir = Path(output_dir) / "INLA_output"
        
        var_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save input data as NPY file
            input_path = var_dir / "data.npy"
            np.save(input_path, data)
            
            # Save the input path to a text file
            path_file = var_dir / "path.txt"
            with open(path_file, "w") as f:
                f.write(str(input_path))
            
            # Build command with appropriate arguments
            cmd = ["Rscript", str(self.r_script_path)]
            
            # Add INLA configuration parameters
            if config.shape != "none":
                cmd.extend(["--shape", config.shape])
            if config.mesh_cutoff is not None:
                cmd.extend(["--mesh-cutoff", str(config.mesh_cutoff)])
            if config.tolerance != 1e-4:
                cmd.extend(["--tolerance", str(config.tolerance)])
            if config.restart != 0:
                cmd.extend(["--restart", str(config.restart)])
            if config.scaling:
                cmd.append("--scaling")
            if config.nonstationary:
                cmd.append("--nonstationary")
            
            # Run the R script
            subprocess.run(cmd, check=True)
            
            # Load results
            results = {}
            result_dir = out_dir / "data"
            if result_dir.exists():
                for result_file in result_dir.glob("*.npy"):
                    name = result_file.stem
                    results[name] = np.load(result_file)
            else:
                # Try loading directly from out_dir if no data subdirectory exists
                if (out_dir / "out.npy").exists():
                    results["out"] = np.load(out_dir / "out.npy")
                if (out_dir / "outsd.npy").exists():
                    results["outsd"] = np.load(out_dir / "outsd.npy")
            
            if not results:
                raise RuntimeError("No output files found after INLA processing")
            
            return results
            
        finally:
            # Clean up temporary files if we created them
            if output_dir is None:
                try:
                    input_path.unlink(missing_ok=True)
                    path_file.unlink(missing_ok=True)
                    # We don't remove the temp_dir since it contains the results
                except:
                    pass  # Ignore errors in cleanup
    
    @staticmethod
    def check_inla_installed() -> bool:
        """Check if R-INLA is installed.
        
        Returns:
            True if R-INLA is installed, False otherwise
        """
        try:
            result = subprocess.run(
                ["Rscript", "-e", "if(require(INLA)) cat('INLA_INSTALLED') else cat('INLA_NOT_INSTALLED')"],
                capture_output=True,
                text=True,
                check=False
            )
            return "INLA_INSTALLED" in result.stdout
        except:
            return False
    
    @staticmethod
    def install_inla() -> bool:
        """Install R-INLA.
        
        Returns:
            True if installation was successful, False otherwise
        """
        try:
            subprocess.run(
                ["Rscript", "-e", "install.packages('INLA', repos=c(INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)"],
                check=True
            )
            return INLAWrapper.check_inla_installed()
        except:
            return False