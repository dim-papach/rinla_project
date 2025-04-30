#!/usr/bin/env python
"""
Remove NaN regions from a FITS file, crop to valid data, and update WCS headers.
Usage:
  fits_remove_nans.py <input_file> [<output_file>] [--overwrite]
"""

import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def remove_nans_and_update_header(input_file, output_file, overwrite=False):
    try:
        with fits.open(input_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()

            if not np.isnan(data).any():
                print(f"No NaNs found in {input_file}. No action taken.")
                return

            # Find bounding box of non-NaN data
            non_nan_mask = ~np.isnan(data)
            rows, cols = np.where(non_nan_mask)
            if len(rows) == 0:
                raise ValueError("All pixels are NaN!")

            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)
            cropped_data = data[y_min:y_max+1, x_min:x_max+1]

            # Update header
            header['NAXIS1'] = cropped_data.shape[1]
            header['NAXIS2'] = cropped_data.shape[0]

            # Adjust WCS if present
            if 'CTYPE1' in header:
                wcs = WCS(header)
                if wcs.is_celestial:
                    header['CRPIX1'] -= x_min
                    header['CRPIX2'] -= y_min

            header.add_history(f"NaN-cropped: original {data.shape} -> new {cropped_data.shape}")

            # Save output
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(
                output_file, overwrite=overwrite
            )
            print(f"Saved cropped FITS to {output_file} (removed NaN borders)")

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop NaN regions from a FITS file and update WCS headers."
    )
    parser.add_argument("input_file", help="Input FITS file path")
    parser.add_argument(
        "output_file", 
        nargs="?", 
        default=None, 
        help="Output FITS file path (default: '<input>_nan_cropped.fits')"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite output file if it exists"
    )
    args = parser.parse_args()

    output_file = args.output_file or args.input_file.replace(
        ".fits", "_nan_cropped.fits"
    )

    remove_nans_and_update_header(
        args.input_file, 
        output_file, 
        overwrite=args.overwrite
    )
