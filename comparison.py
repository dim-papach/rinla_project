import os
import glob
import numpy as np
from astropy.io import fits
from skimage.metrics import structural_similarity as ssim

def compute_ssim(original, processed):
    """
    Compute the structural similarity (SSIM) between the original and processed images.
    - If there is a dimensionality mismatch, crop both images to their smallest common shape.
    - Replace any NaN or infinite values using np.nan_to_num before computing SSIM.
    - Compute data_range for floating point images.
    """
    # Determine the minimal shape along each axis for cropping
    min_shape = tuple(min(o, p) for o, p in zip(original.shape, processed.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    original_cropped = original[slices]
    processed_cropped = processed[slices]

    # Replace NaN and infinite values
    original_cropped = np.nan_to_num(original_cropped, nan=0.0, posinf=0.0, neginf=0.0)
    processed_cropped = np.nan_to_num(processed_cropped, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute the global data range between the two images
    data_min = min(original_cropped.min(), processed_cropped.min())
    data_max = max(original_cropped.max(), processed_cropped.max())
    data_range = data_max - data_min

    # If data_range is near zero for some reason, prevent division issues
    if np.isclose(data_range, 0):
        # If both images are identical, consider them perfectly similar, else dissimilar.
        return 1.0 if np.allclose(original_cropped, processed_cropped) else 0.0

    # Compute and return SSIM with the specified data_range
    return ssim(original_cropped, processed_cropped, data_range=data_range)

def process_directory(base_dir='./output'):
    # Iterate over directories in the base output directory
    for basename in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, basename)
        if os.path.isdir(dir_path):
            original_path = os.path.join(dir_path, 'data/original.fits')
            if not os.path.exists(original_path):
                print(f"Original FITS not found for {basename}")
                continue

            # Load the original FITS file
            try:
                with fits.open(original_path) as hdul:
                    original_data = hdul[0].data.astype(np.float64)
            except Exception as e:
                print(f"Error reading {original_path}: {e}")
                continue

            # Find all processed FITS files in the data subdirectory
            processed_dir = os.path.join(dir_path, 'data')
            processed_files = glob.glob(os.path.join(processed_dir, '*.fits'))
            if not processed_files:
                print(f"No processed FITS files found in {processed_dir}")
                continue

            # Compare each processed FITS file to the original
            for proc_file in processed_files:
                try:
                    with fits.open(proc_file) as hdul_proc:
                        processed_data = hdul_proc[0].data.astype(np.float64)
                except Exception as e:
                    print(f"Error reading {proc_file}: {e}")
                    continue

                try:
                    ssim_value = compute_ssim(original_data, processed_data)
                    print(f"SSIM for {basename}, file {os.path.basename(proc_file)}: {ssim_value:.4f}")
                except Exception as e:
                    print(f"Error computing SSIM for {basename}, file {os.path.basename(proc_file)}: {e}")

if __name__ == '__main__':
    process_directory()
