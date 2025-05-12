"""
Validation utilities for comparing original and processed images.

This module provides metrics and functions for validating the quality of reconstructed images,
including SSIM, MSE, MAE, and residual analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute the structural similarity (SSIM) between the original and processed images.
    
    Args:
        original: Original image data
        processed: Processed image data
        
    Returns:
        SSIM value (higher is better, max is 1.0)
    
    Notes:
        - If there is a dimensionality mismatch, crops both images to their smallest common shape.
        - Replaces any NaN or infinite values with 0.0 before computing SSIM.
        - Computes data_range for floating point images.
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


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute the mean squared error between original and processed images.
    
    Args:
        original: Original image data
        processed: Processed image data
        
    Returns:
        MSE value (lower is better, min is 0.0)
    """
    # Determine the minimal shape along each axis for cropping
    min_shape = tuple(min(o, p) for o, p in zip(original.shape, processed.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    original_cropped = original[slices]
    processed_cropped = processed[slices]

    # Replace NaN and infinite values
    original_cropped = np.nan_to_num(original_cropped, nan=0.0, posinf=0.0, neginf=0.0)
    processed_cropped = np.nan_to_num(processed_cropped, nan=0.0, posinf=0.0, neginf=0.0)
    
    return mean_squared_error(original_cropped, processed_cropped)


def compute_mae(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute the mean absolute error between original and processed images.
    
    Args:
        original: Original image data
        processed: Processed image data
        
    Returns:
        MAE value (lower is better, min is 0.0)
    """
    # Determine the minimal shape along each axis for cropping
    min_shape = tuple(min(o, p) for o, p in zip(original.shape, processed.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    original_cropped = original[slices]
    processed_cropped = processed[slices]

    # Replace NaN and infinite values
    original_cropped = np.nan_to_num(original_cropped, nan=0.0, posinf=0.0, neginf=0.0)
    processed_cropped = np.nan_to_num(processed_cropped, nan=0.0, posinf=0.0, neginf=0.0)
    
    return mean_absolute_error(original_cropped, processed_cropped)


def compute_residual(original: np.ndarray, processed: np.ndarray, 
                   percentage: bool = True) -> np.ndarray:
    """
    Compute the residual between the original and processed images.
    
    Args:
        original: Original image data
        processed: Processed image data
        percentage: Whether to compute residual as percentage (True) or absolute (False)
        
    Returns:
        Residual array of the same shape as the input images
    """
    # Determine the minimal shape along each axis for cropping
    min_shape = tuple(min(o, p) for o, p in zip(original.shape, processed.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    original_cropped = original[slices]
    processed_cropped = processed[slices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        if percentage:
            # Compute residual as percentage
            residual = (original_cropped - processed_cropped) / processed_cropped * 100
        else:
            # Compute absolute residual
            residual = original_cropped - processed_cropped
            
        # Handle NaN and infinite values
        residual[np.isinf(residual)] = np.nan
    
    return residual


def analyze_residual(residual: np.ndarray) -> Dict[str, float]:
    """
    Analyze the residual array and compute summary statistics.
    
    Args:
        residual: Residual array
        
    Returns:
        Dictionary containing:
        - mean: Mean residual
        - median: Median residual
        - min: Minimum residual
        - max: Maximum residual
        - std: Standard deviation of residual
        - nan_percentage: Percentage of NaN values
    """
    # Compute summary statistics
    return {
        'mean': float(np.nanmean(residual)),
        'median': float(np.nanmedian(residual)),
        'min': float(np.nanmin(residual)),
        'max': float(np.nanmax(residual)),
        'std': float(np.nanstd(residual)),
        'nan_percentage': float(np.sum(np.isnan(residual)) / residual.size * 100)
    }


def validate_images(original: np.ndarray, processed: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Perform comprehensive validation between original and processed images.
    
    Args:
        original: Original image data
        processed: Processed image data
        
    Returns:
        Dictionary containing:
        - ssim: SSIM value
        - mse: MSE value
        - mae: MAE value
        - residual_stats: Dictionary with residual statistics
    """
    # Compute metrics
    ssim_value = compute_ssim(original, processed)
    mse_value = compute_mse(original, processed)
    mae_value = compute_mae(original, processed)
    
    # Compute and analyze residual
    residual = compute_residual(original, processed, percentage=True)
    residual_stats = analyze_residual(residual)
    
    return {
        'ssim': ssim_value,
        'mse': mse_value,
        'mae': mae_value,
        'residual_stats': residual_stats
    }