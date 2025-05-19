# script to cut original.fits into a 100x100 pixel image of the middle of the image
# # and save it as a new file test.fits

import numpy as np
from astropy.io import fits
from pathlib import Path
import os

def cut_fits_file(input_file: str, output_file: str, size: int = 100):
    """
    Cut a FITS file to a smaller size and save it as a new file.

    Args:
        input_file (str): Path to the input FITS file.
        output_file (str): Path to save the cut FITS file.
        size (int): Size of the cut image (size x size).
    """
    # Open the original FITS file
    with fits.open(input_file) as hdul:
        # Get the data from the first HDU
        data = hdul[0].data

        # Check if data is None
        if data is None:
            raise ValueError("No data found in the FITS file.")

        # Get the shape of the data
        height, width = data.shape


        # Calculate the center of the image
        center_row = height // 2
        center_col = width // 2
        # Calculate the starting row and column for the cut
        start_row = center_row - size // 2
        start_col = center_col - size // 2
        # Ensure the cut does not go out of bounds
        start_row = max(0, start_row)
        start_col = max(0, start_col)
        # Adjust the size if it exceeds the image dimensions
        size = min(size, height - start_row, width - start_col)
        # Print the cutting area
        print(f"Original image size: {height}x{width}")
        print(f"Cut size: {size}x{size}")
        print(f"Start position: ({start_row}, {start_col})")
        print(f"End position: ({start_row + size}, {start_col + size})")
        # Print the cutting area
        print(f"Cutting data from ({start_row}, {start_col}) to ({start_row + size}, {start_col + size})")
        
        print(f"Cutting data from ({start_row}, {start_col}) to ({start_row + size}, {start_col + size})")
        
        # Cut the data
        cut_data = data[start_row:start_row + size, start_col:start_col + size]

    # Create a new FITS file with the cut data
    hdu = fits.PrimaryHDU(cut_data)
    hdu.writeto(output_file, overwrite=True)
    
if __name__ == "__main__":
    # Example usage
    input_fits = "original.fits"  # Path to the original FITS file
    output_fits = "test.fits"      # Path to save the cut FITS file

    # Check if the input file exists
    if not os.path.exists(input_fits):
        print(f"Input file {input_fits} does not exist.")
    else:
        cut_fits_file(input_fits, output_fits, size=100)
        print(f"Cut FITS file saved as {output_fits}.")
