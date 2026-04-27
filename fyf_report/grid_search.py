import os
import subprocess
import re
import logging
import sys
import csv
import shutil
from pathlib import Path

import itertools
from datetime import datetime
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric

# Parameters to grid search
param_grid = {
    'shape': ['none'],
    'mesh-resolution': [20, 30, 40],
    'scaling': ['log'],
    'max-edge-factor': [8, 10, 12],
    'outer-edge-factor': [1.2, 1.5, 2.0],
    'offset-inner-factor': [0.1, 0.5, 0.8],
    'offset-outer-factor': [1.0, 2.0, 3.0],
    'prior-range-prob': [0.1, 0.2],
    'prior-range-lower': [1.0, 2.0, 5.0],
    'prior-sigma-prob': [0.1, 0.2],
    'prior-sigma-upper': [1.0, 2.0, 3.0]
}

def setup_logging():
    """Setup dual logging to console and file."""
    log_file = Path("grid_search.log")
    # Ensure log file exists
    log_file.touch(exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def run_docker_fyf(args):
    """Run command inside Docker container."""
    cwd = os.getcwd()
    # Add verbose flag if it's an fyf command
    if args and args[0] == "fyf":
        args.insert(1, "-v")
        
    cmd = [
        "podman", "run", "--rm",
        "-v", f"{cwd}:/data:z",
        "localhost/fyf"
    ] + args
    
    logger.debug(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        logger.info(f"STDOUT ({' '.join(args[:3])}...):")
        for line in result.stdout.splitlines():
            logger.info(f"  [DOCKER OUT] {line}")
    if result.stderr:
        logger.info(f"STDERR ({' '.join(args[:3])}...):")
        for line in result.stderr.splitlines():
            logger.info(f"  [DOCKER ERR] {line}")
            
    return result

def get_ssim(output):
    """Extract SSIM from validate output."""
    # Look for SSIM: 0.XXXXX in the output
    match = re.search(r"SSIM:\s*([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None

def compute_extra_metrics(original_data, processed_data):
    """Compute additional metrics and residual array."""
    # Ensure same shape by cropping
    min_shape = tuple(min(o, p) for o, p in zip(original_data.shape, processed_data.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    orig = np.nan_to_num(original_data[slices], nan=0.0)
    proc = np.nan_to_num(processed_data[slices], nan=0.0)
    
    # MSE
    mse = np.mean((orig - proc)**2)
    
    # Residual %: (orig - proc) / orig * 100
    with np.errstate(divide='ignore', invalid='ignore'):
        res = (orig - proc) / orig * 100
        res[np.isinf(res)] = np.nan
        # Handle residuals that are too large (e.g. division by near-zero)
        res[res >= 150] = 150
        res[res <= -150] = -150
        
    mean_res = np.nanmean(res)
    var_res = np.nanvar(res)
    
    return mse, mean_res, var_res, res

def save_best_plots(slice_id, residual, mask, output_dir):
    """Generate and save residual map and histogram with masked/non-masked separation."""
    sanitized_id = slice_id.replace(".fits", "").replace("[", "_").replace("]", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Residual Map
    plt.figure(figsize=(10, 8))
    vmin, vmax = np.nanpercentile(residual, [1, 99])
    im = plt.imshow(residual, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Residual Percentage (%)')
    plt.title(f'Residual Map: {slice_id}')
    plt.savefig(output_dir / f"{sanitized_id}_residual_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Residual Histogram
    plt.figure(figsize=(12, 10))
    
    # Prepare data subsets
    res_all = residual[~np.isnan(residual)].flatten()
    
    if mask is not None:
        # Ensure mask is boolean and same shape as residual
        mask_bool = mask.astype(bool)
        if mask_bool.shape != residual.shape:
            # Crop mask if needed (should match residual cropping logic)
            min_shape = tuple(min(r, m) for r, m in zip(residual.shape, mask_bool.shape))
            mask_bool = mask_bool[:min_shape[0], :min_shape[1]]
            
        res_masked = residual[mask_bool & ~np.isnan(residual)].flatten()
        res_non_masked = residual[~mask_bool & ~np.isnan(residual)].flatten()
    else:
        res_masked = np.array([])
        res_non_masked = res_all

    def get_stats_str(data, label):
        if len(data) == 0:
            return f"{label}: N/A"
        return (f"{label}\n"
                f"Mean: {np.mean(data):.4f}%\n"
                f"Med: {np.median(data):.4f}%\n"
                f"Var: {np.var(data):.4f}")

    # Plot all pixels
    plt.hist(res_all, bins=100, color='blue', alpha=0.3, label=get_stats_str(res_all, "All Pixels"))
    
    # Plot masked pixels
    if len(res_masked) > 0:
        plt.hist(res_masked, bins=100, color='grey', alpha=0.5, label=get_stats_str(res_masked, "Masked Pixels"))
        
    # Plot non-masked pixels
    if len(res_non_masked) > 0:
        plt.hist(res_non_masked, bins=100, color='green', alpha=0.4, label=get_stats_str(res_non_masked, "Non-masked Pixels"))

    plt.xlabel('Residual Percentage (%)')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution: {slice_id}')
    
    # Place legend with stats
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{sanitized_id}_residual_hist.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Ensure results file and output directory exist
    Path("best_model.txt").touch(exist_ok=True)
    Path("report_output").mkdir(parents=True, exist_ok=True)
    
    history_file = Path("history.csv")
    headers = ['image'] + list(param_grid.keys()) + ['ssim', 'mse', 'mean_residual', 'sigma']
    if not history_file.exists():
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    # Ensure we are in fyf_report
    if not os.path.exists("data"):
        logger.error("'data' directory not found. Please run from 'fyf_report' directory.")
        return

    data_dir = Path("data")
    fits_files = list(data_dir.glob("*.fits")) + list(data_dir.glob("*.fit"))
    
    if not fits_files:
        logger.error("No FITS files found in data directory.")
        return

    report_output = Path("report_output")
    report_output.mkdir(exist_ok=True)

    best_results = {}

    for fits_file in fits_files:
        logger.info(f"Processing {fits_file.name}...")
        
        # Determine number of slices
        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                if data is None:
                    logger.error(f"    No data in {fits_file.name}")
                    continue
                n_slices = data.shape[0] if data.ndim == 3 else 1
                is_3d = data.ndim == 3
        except Exception as e:
            logger.error(f"    Error opening {fits_file.name}: {e}")
            continue

        for i in range(n_slices):
            slice_id = f"{fits_file.name}[{i}]"
            logger.info(f"  Grid search for {slice_id}...")
            
            # Extract slice to temporary FITS
            temp_slice = Path(f"temp_slice_{i}.fits")
            try:
                with fits.open(fits_file) as hdul:
                    d = hdul[0].data if hdul[0].data is not None else hdul[1].data
                    s_data = d[i] if is_3d else d
                    fits.writeto(temp_slice, s_data, overwrite=True)
            except Exception as e:
                logger.error(f"    Error creating temp slice: {e}")
                continue
            
            # 1. Simulate artifacts for this slice
            sim_out_dir = Path(f"temp_sim_{i}")
            docker_temp_slice = f"/data/{temp_slice}"
            docker_sim_out = f"/data/{sim_out_dir}"
            
            # fyf -v simulate ...
            sim_res = run_docker_fyf(["fyf", "simulate", docker_temp_slice, "-c", "0.1", "-t", "2", "-o", docker_sim_out])
            if sim_res.returncode != 0:
                logger.error(f"    Simulation failed for {slice_id}")
                if temp_slice.exists(): os.remove(temp_slice)
                continue

            docker_combined = f"{docker_sim_out}/{temp_slice.stem}/data/combined.fits"
            local_combined = sim_out_dir / temp_slice.stem / "data" / "combined.fits"
            local_mask_path = sim_out_dir / temp_slice.stem / "masks" / "combined_mask.fits"
            
            if not local_combined.exists():
                logger.warning(f"    Combined file not found at {local_combined}")
                if temp_slice.exists(): os.remove(temp_slice)
                run_docker_fyf(["rm", "-rf", f"/data/{sim_out_dir}"])
                continue
            
            # Load the mask
            best_mask = None
            if local_mask_path.exists():
                try:
                    with fits.open(local_mask_path) as h_mask:
                        best_mask = h_mask[0].data
                except Exception as e:
                    logger.error(f"    Error loading mask: {e}")

            best_ssim = -1.0
            best_params = None
            best_metrics = {}
            best_residual = None
            
            # Grid search
            keys = param_grid.keys()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
            
            for combo in combinations:
                logger.info(f"    Testing combo: {combo}")
                proc_out_dir = Path(f"temp_proc_{i}")
                docker_proc_out = f"/data/{proc_out_dir}"
                
                # fyf -v process ...
                proc_args = ["fyf", "process", docker_combined, "--method", "inla", "-o", docker_proc_out]
                for k, v in combo.items():
                    if k == 'scaling':
                        proc_args.extend(["--scaling", v])
                    else:
                        proc_args.extend([f"--{k}", str(v)])
                
                proc_res = run_docker_fyf(proc_args)
                
                local_processed = proc_out_dir / "combined_inla" / "data" / "processed.fits"
                docker_processed = f"{docker_proc_out}/combined_inla/data/processed.fits"
                
                if local_processed.exists():
                    # 3. Validate
                    # fyf -v validate ...
                    val_res = run_docker_fyf(["fyf", "validate", docker_temp_slice, docker_processed, "--metrics", "ssim"])
                    ssim_val = get_ssim(val_res.stdout)
                    
                    if ssim_val is not None:
                        logger.info(f"      SSIM: {ssim_val:.4f}")
                        # Compute metrics for history and best model tracking
                        try:
                            with fits.open(temp_slice) as h_orig, fits.open(local_processed) as h_proc:
                                mse, mean_res, var_res, res = compute_extra_metrics(h_orig[0].data, h_proc[0].data)
                                
                                # Append to history.csv
                                with open(history_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    row = [slice_id] + [combo.get(k, "none") for k in param_grid.keys()] + [ssim_val, mse, mean_res, var_res]
                                    writer.writerow(row)

                                if ssim_val > best_ssim:
                                    best_ssim = ssim_val
                                    best_params = combo
                                    best_metrics = {
                                        'ssim': ssim_val,
                                        'mse': mse,
                                        'mean_residual': mean_res,
                                        'sigma2': var_res
                                    }
                                    best_residual = res
                                    # Save the best FITS file
                                    sanitized_id = slice_id.replace(".fits", "").replace("[", "_").replace("]", "")
                                    shutil.copy2(local_processed, report_output / f"{sanitized_id}_best_processed.fits")
                        except Exception as e:
                            logger.error(f"      Error calculating metrics: {e}")
                    else:
                        logger.warning(f"      Failed to parse SSIM from output")
                else:
                    logger.warning(f"      Processing failed: {local_processed} not found")
                
                # Cleanup proc dir using docker to avoid permission issues
                run_docker_fyf(["rm", "-rf", f"/data/{proc_out_dir}"])

            if best_params:
                best_results[slice_id] = {'params': best_params, 'metrics': best_metrics}
                logger.info(f"  ==> Best for {slice_id}: SSIM {best_ssim:.4f} with {best_params}")
                if best_residual is not None:
                    save_best_plots(slice_id, best_residual, best_mask, report_output)
            else:
                logger.error(f"  ==> No successful model found for {slice_id}")
            
            # Cleanup temp files for this slice
            if temp_slice.exists():
                os.remove(temp_slice)
            run_docker_fyf(["rm", "-rf", f"/data/{sim_out_dir}"])

    # Save best results
    if best_results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("best_model.txt", "a") as f:
            f.write(f"\n--- Grid Search Run: {timestamp} ---\n")
            f.write(f"Grid: {param_grid}\n")
            for slice_id, data in best_results.items():
                param_str = ", ".join([f"{k}={v}" for k, v in data['params'].items()])
                metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in data['metrics'].items()])
                f.write(f"{slice_id}: {{parameters: {param_str}, {metrics_str}}}\n")
        logger.info("Grid search complete. Results appended to best_model.txt and report_output/")
    else:
        logger.info("Grid search complete. No results found.")

if __name__ == "__main__":
    main()
