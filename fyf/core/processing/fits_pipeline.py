import os
from pathlib import Path
from astropy.io import fits
import numpy as np
from plotting_fits import PlotGenerator

def process_fits_pipeline(output_dir: Path):
    """
    Pipeline to process .fits files, plot them, and calculate residuals.
    
    Args:
        output_dir (Path): Base directory containing the output folders.
    """
    plot_generator = PlotGenerator()
    
    # Iterate through all subdirectories in the output directory
    for subdir in output_dir.iterdir():
        data_dir = subdir / "data"
             
                    
        # Load the required .fits files
        original_path = data_dir / "original.fits"
        if not original_path.exists():
            print(f"Skipping {subdir}, original.fits not found.")
            continue
        
        # Load original.fits
        with fits.open(original_path) as hdul:
            original_data = hdul[0].data
        
        # Load variants
        variants = {}
        for name in ["original", "cosmic", "satellite", "combined"]:
            variant_path = data_dir / f"{name}.fits"
            if not variant_path.exists():
                print(f"Skipping {subdir}, {variant_path} not found.")
                continue
            with fits.open(variant_path) as hdul:
                variants[name] = hdul[0].data
 
        # Load processed data   
        processed_files = list(data_dir.glob("*_processed.fits"))
        if not processed_files:
            print(f"No processed files found in {data_dir}.")
            continue
        
        processed = {}
        for processed_file in processed_files:
            name = processed_file.stem.replace("_processed", "")
            with fits.open(processed_file) as hdul:
                processed_data = hdul[0].data
                processed[name] = processed_data
                
        # Generate plots
        basename = subdir.name
        try:
            plot_generator.generate_all_plots(subdir, variants, processed, basename)
        except Exception as e:
            print(f"Error generating plots for {basename}: {e}")
            continue
        print(f"Plots generated for {basename} in {data_dir}.")
        
if __name__ == "__main__":
    # Example usage
    output_directory = Path("/home/dp/Documents/GitHub/rinla_project/output")
    process_fits_pipeline(output_directory)