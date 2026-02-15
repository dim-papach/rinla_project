#!/usr/bin/env python3
"""
FYF (Fill Your FITS) - CLI Tool

A command-line tool for processing astronomical FITS images using R-INLA.
"""

import sys
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional
import click
import numpy as np
from astropy.io import fits
from colorama import Fore, Style, init
init(autoreset=True)

# Import existing FYF components
try:
    from fyf.config import CosmicConfig, SatelliteConfig, INLAConfig, PlotConfig
    from fyf.core.data.masking import MaskGenerator
    from fyf.core.processing.fits_processor import FitsProcessor
    from fyf.core.processing.methods import (
        ensure_method_available,
        get_supported_methods,
    )
    from fyf.core.processing.preprocessing import (
        get_supported_preprocessors,
        run_preprocessed_processing,
    )
    from fyf.core.data.file_handler import FileHandler
    from fyf.core.validation import validate_images
    from fyf.visualization.plotting import PlotGenerator
    from fyf.visualization.report import ReportGenerator
    from fyf.config.config_manager import ConfigManager
except ImportError as e:
    click.echo(f"Error importing FYF modules: {e}", err=True)
    sys.exit(1)

# Version and constants
VERSION = "0.1.0"
DEFAULT_COSMIC_VALUE = np.float32(np.nan)
DEFAULT_SATELLITE_VALUE = np.float32(np.nan)

# Color utilities
class Colors:
    ERROR = Fore.RED
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    INFO = Fore.CYAN
    BOLD = Style.BRIGHT

def echo_colored(msg: str, color: str = Colors.INFO):
    """Print colored message"""
    click.echo(f"{color}{msg}{Style.RESET_ALL}")

def echo_banner(title: str):
    """Print a banner"""
    echo_colored("=" * 50, Colors.INFO)
    echo_colored(f" {title}", f"{Colors.BOLD}{Colors.INFO}")
    echo_colored("=" * 50, Colors.INFO)

# File validation callback
def validate_fits_files(ctx, param, value):
    """Validate and expand FITS file patterns"""
    if not value:
        return []
    
    import glob
    files = []

    def _iter_candidates(raw_pattern: str):
        raw_path = Path(raw_pattern)
        if raw_path.is_absolute():
            return [raw_path]
        return [raw_path, Path('/data') / raw_path, Path('/app') / raw_path]

    for pattern in value:
        found_for_pattern = False
        for candidate_path in _iter_candidates(pattern):
            candidate = str(candidate_path)

            if candidate_path.is_dir():
                dir_matches = sorted(candidate_path.glob("*.fits")) + sorted(candidate_path.glob("*.fit"))
                if dir_matches:
                    files.extend(dir_matches)
                    found_for_pattern = True
                    break

            if glob.has_magic(candidate):
                matches = sorted(
                    Path(m)
                    for m in glob.glob(candidate)
                    if m.endswith(('.fits', '.fit'))
                )
                if matches:
                    files.extend(matches)
                    found_for_pattern = True
                    break

            if candidate_path.suffix.lower() in {'.fits', '.fit'} and candidate_path.exists():
                files.append(candidate_path)
                found_for_pattern = True
                break

        if not found_for_pattern and pattern.lower().endswith(('.fits', '.fit')):
            click.echo(f"Warning: Skipping invalid file: {pattern}", err=True)
    
    # Filter existing files and deduplicate by underlying inode when possible.
    existing = []
    seen_keys = set()
    for f in files:
        if not f.exists():
            continue
        try:
            st = f.stat()
            key = (st.st_dev, st.st_ino, st.st_size)
        except OSError:
            key = ("path", str(f.resolve()))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        existing.append(f)

    if not existing and value:
        raise click.BadParameter("No valid FITS files found")
    
    return sorted(existing)

# Main CLI group
@click.group()
@click.version_option(version=VERSION)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """FYF - Fill Your FITS: Process astronomical images"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        echo_banner(f"FYF v{VERSION}")

# Config command group
@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command('generate')
@click.argument('output', type=click.Path(), default='fyf-config.json')
@click.option('--force', is_flag=True, help='Overwrite existing file')
def config_generate(output, force):
    """Generate a template configuration file"""
    echo_banner("Config Generation")
    
    output_path = Path(output)
    
    if output_path.exists() and not force:
        echo_colored(f"File {output_path} already exists. Use --force to overwrite.", Colors.WARNING)
        return
    
    ConfigManager.generate_template(output_path)
    echo_colored(f"Configuration template generated: {output_path}", Colors.SUCCESS)

@config.command('validate')
@click.argument('config_file', type=click.Path(exists=True))
def config_validate(config_file):
    """Validate a configuration file"""
    echo_banner("Config Validation")
    
    try:
        config_data = ConfigManager.load_config(Path(config_file))
        valid = ConfigManager.validate_config(config_data)
        
        if valid:
            echo_colored("Configuration is valid", Colors.SUCCESS)
        else:
            echo_colored("Configuration has issues", Colors.WARNING)
    except Exception as e:
        echo_colored(f"Error validating config: {e}", Colors.ERROR)

# Updated simulate command with config support
@cli.command()
@click.argument('files', nargs=-1, required=True, callback=validate_fits_files)
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--cosmic-fraction', '-c', type=float, help='Cosmic ray fraction (0-1)')
@click.option('--trails', '-t', type=int, help='Number of satellite trails')
@click.option('--output-dir', '-o', type=Path, help='Output directory')
@click.option('--report', '-r', is_flag=True, help='Generate HTML report')
@click.option('--custom-mask', type=click.Path(exists=True), help='Path to custom mask file')
@click.pass_context
def simulate(ctx, files, config, cosmic_fraction, trails, output_dir, report, custom_mask):
    """Simulate cosmic rays and satellite trails on FITS images"""
    echo_banner("FYF Simulation")
    
    # Load config if provided
    config_data = {}
    if config:
        config_data = ConfigManager.load_config(Path(config))
    
    # Merge CLI args with config
    cli_args = {
        'cosmic_fraction': cosmic_fraction,
        'trails': trails,
        'output_dir': str(output_dir) if output_dir else None
    }
    
    simulate_config = ConfigManager.merge_with_cli_args(config_data, 'simulate', cli_args)
    
    # Create configurations using merged values
    cosmic_cfg = CosmicConfig(
        fraction=simulate_config.get('cosmic_fraction', 0.01)
    )
    satellite_cfg = SatelliteConfig(
        num_trails=simulate_config.get('trails', 1),
        trail_width=simulate_config.get('trail_width', 3)
    )
    
    # Set output directory
    output_dir = Path(simulate_config.get('output_dir', './output'))
    
    # Initialize components
    file_handler = FileHandler()
    mask_generator = MaskGenerator(cosmic_cfg, satellite_cfg)
    fits_processor = FitsProcessor(cosmic_cfg, satellite_cfg)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    echo_colored(f"Cosmic rays: {cosmic_cfg.fraction*100:.1f}%", Colors.INFO)
    echo_colored(f"Satellite trails: {satellite_cfg.num_trails}", Colors.INFO)
    echo_colored(f"Output directory: {output_dir}", Colors.INFO)
    
    # Process files
    results = {}
    with click.progressbar(files, label='Processing') as bar:
        for file_path in bar:
            try:
                start_time = time.time()
                
                # Load FITS data
                data, header = file_handler.load_fits(file_path)
                basename = file_path.stem
                file_output_dir = output_dir / basename
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate masks with optional custom mask
                masks = mask_generator.generate_all_masks(data, custom_mask)
                
                # Create variants
                variants = fits_processor.create_variants(data, masks)
                
                # Save outputs
                file_handler.save_outputs(file_output_dir, variants, masks, header)
                
                process_time = time.time() - start_time
                result = {
                    'success': True,
                    'output_dir': str(file_output_dir),
                    'process_time': process_time
                }
                
                results[file_path.name] = result
                echo_colored(f"✓ {file_path.name}", Colors.SUCCESS)
                
            except Exception as e:
                echo_colored(f"✗ {file_path.name}: {e}", Colors.ERROR)
                results[file_path.name] = {'success': False, 'error': str(e)}
    
    # Summary
    success_count = sum(1 for r in results.values() if r.get('success', False))
    echo_colored(f"\nProcessed: {success_count}/{len(files)} files", Colors.INFO)
    
    # Generate report if requested
    if report and results:
        try:
            report_gen = ReportGenerator(output_dir)
            report_path = report_gen.generate_summary_report("FYF Simulation", results)
            echo_colored(f"Report: {report_path}", Colors.SUCCESS)
        except Exception as e:
            echo_colored(f"Report error: {e}", Colors.ERROR)


def _build_process_config(
    config_path: Optional[str],
    cli_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Load process configuration and merge explicit CLI overrides."""
    config_data: Dict[str, Any] = {}
    if config_path:
        config_data = ConfigManager.load_config(Path(config_path))
    return ConfigManager.merge_with_cli_args(config_data, 'process', cli_args)


def _normalize_processing_method(raw_method: Optional[str]) -> Optional[str]:
    """Normalize optional processing method name."""
    if not raw_method:
        return None
    return str(raw_method).lower().strip()


def _build_inla_config(process_config: Dict[str, Any]) -> INLAConfig:
    """Create INLA configuration from merged process config."""
    return INLAConfig(
        shape=process_config.get('shape', 'none'),
        scaling=process_config.get('scaling', 'log'),
        nonstationary=process_config.get('nonstationary', False),
        mesh_cutoff=process_config.get('mesh_cutoff', None),
        mesh_resolution=process_config.get('mesh_resolution', 30),
        max_edge_factor=process_config.get('max_edge_factor', 10.0),
        outer_edge_factor=process_config.get('outer_edge_factor', 1.5),
        offset_inner_factor=process_config.get('offset_inner_factor', 0.5),
        offset_outer_factor=process_config.get('offset_outer_factor', 2.0),
        alpha=process_config.get('alpha', 2),
        prior_range_prob=process_config.get('prior_range_prob', 0.2),
        prior_range_lower=process_config.get('prior_range_lower', 2.0),
        prior_sigma_prob=process_config.get('prior_sigma_prob', 0.2),
        prior_sigma_upper=process_config.get('prior_sigma_upper', 2.0),
        num_threads=process_config.get('num_threads', 6),
        openmp_strategy=process_config.get('openmp_strategy', 'huge'),
        nbasis=process_config.get('nbasis', 2),
        spline_degree=process_config.get('spline_degree', 10),
        tolerance=process_config.get('tolerance', 1e-4),
        restart=process_config.get('restart', 0)
    )


def _resolve_base_output_dir(process_config: Dict[str, Any]) -> Optional[Path]:
    """Create and return configured base output directory when present."""
    output_dir_value = process_config.get('output_dir')
    if not output_dir_value:
        return None
    base_output_dir = Path(output_dir_value)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir


def _echo_process_settings(
    processing_method: Optional[str],
    preprocessing_method: str,
    inla_cfg: INLAConfig,
    final_base_output_dir: Optional[Path],
) -> None:
    """Print merged process settings for observability."""
    echo_colored(
        f"Method: {processing_method if processing_method else 'preprocess-only (no processing method)'}",
        Colors.INFO,
    )
    echo_colored(f"3D preprocess: {preprocessing_method}", Colors.INFO)
    echo_colored(f"INLA shape: {inla_cfg.shape}", Colors.INFO)
    echo_colored(f"Scaling: {'Enabled' if inla_cfg.scaling else 'Disabled'}", Colors.INFO)
    if final_base_output_dir:
        echo_colored(f"Output directory: {final_base_output_dir}", Colors.INFO)
    else:
        echo_colored("Output directory: Each file's original directory", Colors.INFO)
    echo_colored(
        "Per-file folder pattern: {original-name}_{method-or-preprocess}",
        Colors.INFO,
    )


def _resolve_file_output_dir(
    file_path: Path,
    processing_method: Optional[str],
    preprocessing_method: str,
    final_base_output_dir: Optional[Path],
) -> Path:
    """Resolve output directory for a single input file."""
    basename = file_path.stem
    output_folder_name = (
        f"{basename}_{processing_method}"
        if processing_method
        else f"{basename}_{preprocessing_method}"
    )
    if final_base_output_dir:
        file_output_dir = final_base_output_dir / output_folder_name
    else:
        file_output_dir = file_path.parent / output_folder_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    return file_output_dir


def _process_single_file(
    file_path: Path,
    processing_method: Optional[str],
    preprocessing_method: str,
    inla_cfg: INLAConfig,
    final_base_output_dir: Optional[Path],
    file_handler: FileHandler,
) -> str:
    """Process one FITS file and return success message."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    if data.ndim not in (2, 3):
        raise ValueError(f"Input data must be 2D or 3D, got {data.ndim}D.")

    current_file_output_dir = _resolve_file_output_dir(
        file_path=file_path,
        processing_method=processing_method,
        preprocessing_method=preprocessing_method,
        final_base_output_dir=final_base_output_dir,
    )

    if processing_method is None and data.ndim != 3:
        raise ValueError(
            "No processing method selected. Preprocess-only mode is supported for 3D inputs only."
        )

    method_result = run_preprocessed_processing(
        preprocess=preprocessing_method,
        method=processing_method,
        data=data,
        output_dir=current_file_output_dir,
        inla_config=inla_cfg,
        header=header,
    )

    # Keep a copy of the input FITS alongside original variant NPY output.
    original_variant_dir = current_file_output_dir / "original"
    original_variant_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, original_variant_dir / file_path.name)

    restored = method_result.get('restored')
    uncertainty = method_result.get('uncertainty')

    if restored is None:
        return "Preprocessing complete (no processing method selected)"

    results_to_save = {'processed': restored}
    if uncertainty is not None:
        results_to_save['uncertainty'] = uncertainty
    file_handler.save_outputs(current_file_output_dir, results_to_save, {}, header)
    return "Success"


@cli.command()
@click.argument('files', nargs=-1, required=True, callback=validate_fits_files)
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option(
    '--method',
    type=click.Choice(get_supported_methods(), case_sensitive=False),
    default=None,
    help='Processing backend: inla, mcmc, or convolution. Omit to run preprocess-only for 3D FITS.',
)
@click.option(
    '--preprocess',
    type=click.Choice(get_supported_preprocessors(), case_sensitive=False),
    default='split2d',
    show_default=True,
    help='3D preprocessing strategy: split2d, pca, or svd',
)
@click.option('--shape', type=click.Choice(['none', 'radius', 'ellipse']), help='Shape parameter')
@click.option('--scaling', '-s', type=click.Choice(['log', 'none']), help='Enable log10 scaling')
@click.option('--nonstationary', is_flag=True, help='Enable non-stationary model')
@click.option('--output-dir', '-o', type=Path, help='Output directory')
@click.option('--mesh-cutoff', type=float, help='Minimum distance between mesh points')
@click.option('--mesh-resolution', type=int, help='Mesh resolution factor')
@click.option('--max-edge-factor', type=float, help='Max edge factor for mesh')
@click.option('--outer-edge-factor', type=float, help='Outer edge factor for mesh')
@click.option('--offset-inner-factor', type=float, help='Inner offset factor for mesh')
@click.option('--offset-outer-factor', type=float, help='Outer offset factor for mesh')
@click.option('--alpha', type=int, help='SPDE smoothness parameter (1 or 2)')
@click.option('--prior-range-prob', type=float, help='Prior probability for range parameter')
@click.option('--prior-range-lower', type=float, help='Lower bound for range prior')
@click.option('--prior-sigma-prob', type=float, help='Prior probability for sigma parameter')
@click.option('--prior-sigma-upper', type=float, help='Upper bound for sigma prior')
@click.option('--num-threads', type=int, help='Number of CPU threads')
@click.option('--openmp-strategy', type=click.Choice(['small', 'medium', 'large', 'huge']), help='OpenMP strategy')
@click.option('--nbasis', type=int, help='Number of basis functions for non-stationary model')
@click.option('--spline-degree', type=int, help='Degree of B-spline basis functions')
@click.option('--tolerance', type=float, help='INLA convergence tolerance')
@click.option('--restart', type=int, help='Number of INLA restarts')
@click.pass_context
def process(ctx, files, config, method, preprocess, shape, scaling, nonstationary, output_dir,
            mesh_cutoff, mesh_resolution, max_edge_factor, outer_edge_factor,
            offset_inner_factor, offset_outer_factor, alpha, prior_range_prob,
            prior_range_lower, prior_sigma_prob, prior_sigma_upper, num_threads,
            openmp_strategy, nbasis, spline_degree, tolerance, restart):
    """Process FITS images to fill missing data"""
    echo_banner("FYF Processing")

    validated_files = files
    
    echo_colored(f"Processing {len(validated_files)} files", Colors.INFO)
    
    # Merge CLI args with config
    cli_args = {
        'method': method,
        'preprocess': preprocess,
        'shape': shape,
        'scaling': scaling,
        'nonstationary': nonstationary,
        'mesh_cutoff': mesh_cutoff,
        'mesh_resolution': mesh_resolution,
        'max_edge_factor': max_edge_factor,
        'outer_edge_factor': outer_edge_factor,
        'offset_inner_factor': offset_inner_factor,
        'offset_outer_factor': offset_outer_factor,
        'alpha': alpha,
        'prior_range_prob': prior_range_prob,
        'prior_range_lower': prior_range_lower,
        'prior_sigma_prob': prior_sigma_prob,
        'prior_sigma_upper': prior_sigma_upper,
        'num_threads': num_threads,
        'openmp_strategy': openmp_strategy,
        'nbasis': nbasis,
        'spline_degree': spline_degree,
        'tolerance': tolerance,
        'restart': restart,
        'output_dir': str(output_dir) if output_dir else None
    }
    process_config = _build_process_config(config, cli_args)
    processing_method = _normalize_processing_method(process_config.get('method', None))
    preprocessing_method = str(process_config.get('preprocess', 'split2d')).lower()

    # Validate selected method early (non-INLA placeholders fail fast).
    if processing_method:
        try:
            ensure_method_available(processing_method)
        except (ValueError, NotImplementedError) as e:
            echo_colored(f"Error: {e}", Colors.ERROR)
            return

    # Check R-INLA availability only when INLA backend is selected.
    if processing_method == 'inla':
        try:
            from fyf.r import check_inla_installed
            if not check_inla_installed():
                echo_colored("Warning: R-INLA not detected", Colors.WARNING)
        except ImportError:
            pass
    
    inla_cfg = _build_inla_config(process_config)
    final_base_output_dir = _resolve_base_output_dir(process_config)
    _echo_process_settings(
        processing_method=processing_method,
        preprocessing_method=preprocessing_method,
        inla_cfg=inla_cfg,
        final_base_output_dir=final_base_output_dir,
    )
    
    # Initialize file handler
    file_handler = FileHandler()

    # Process files using selected backend
    with click.progressbar(validated_files, label='Processing') as bar:
        for file_path in bar:
            try:
                status = _process_single_file(
                    file_path=file_path,
                    processing_method=processing_method,
                    preprocessing_method=preprocessing_method,
                    inla_cfg=inla_cfg,
                    final_base_output_dir=final_base_output_dir,
                    file_handler=file_handler,
                )
                echo_colored(f"✓ {file_path.name}: {status}", Colors.SUCCESS)
            except Exception as e:
                echo_colored(f"✗ {file_path.name}: {e}", Colors.ERROR)
                
# Validate command  
@cli.command()
@click.argument('original', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--output-dir', '-o', type=Path, help='Output directory')
@click.option('--plot', is_flag=True, help='Generate validation plots')
@click.option('--metrics', multiple=True, type=click.Choice(['ssim', 'mse', 'mae']), help='Metrics to compute')
@click.pass_context
def validate(ctx, original, processed, config, output_dir, plot, metrics):
    """Validate processing results by comparing original and processed images"""
    echo_banner("FYF Validation")
    
    # Load config if provided
    config_data = {}
    if config:
        config_data = ConfigManager.load_config(Path(config))
    
    # Merge CLI args with config
    cli_args = {
        'output_dir': str(output_dir) if output_dir else None,
        'generate_plots': plot,
        'metrics': list(metrics) if metrics else None
    }
    
    validate_config = ConfigManager.merge_with_cli_args(config_data, 'validate', cli_args)
    
    # Set defaults
    output_dir = Path(validate_config.get('output_dir', './validation'))
    generate_plots = validate_config.get('generate_plots', False)
    selected_metrics = validate_config.get('metrics', ['ssim', 'mse', 'mae'])
    
    # Load FITS files
    try:
        from astropy.io import fits
        
        with fits.open(original) as hdul:
            original_data = hdul[0].data.astype(np.float64)
        with fits.open(processed) as hdul:
            processed_data = hdul[0].data.astype(np.float64)
            
        echo_colored("Files loaded successfully", Colors.SUCCESS)
    except Exception as e:
        echo_colored(f"Error loading files: {e}", Colors.ERROR)
        return
    
    # Run validation using existing function
    try:
        metrics_results = validate_images(original_data, processed_data)
        
        # Display results
        echo_colored("\nValidation Results:", Colors.INFO)
        for metric in selected_metrics:
            if metric in metrics_results:
                value = metrics_results[metric]
                echo_colored(f"  {metric.upper()}: {value:.4f}", Colors.INFO)
        
        residual_stats = metrics_results.get('residual_stats', {})
        echo_colored("\nResidual Statistics:", Colors.INFO)
        echo_colored(f"  Mean:   {residual_stats.get('mean', 0):.2f}%", Colors.INFO)
        echo_colored(f"  StdDev: {residual_stats.get('std', 0):.2f}%", Colors.INFO)
        echo_colored(f"  NaN:    {residual_stats.get('nan_percentage', 0):.2f}%", Colors.INFO)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / "validation_results.txt"
        
        with open(result_file, 'w') as f:
            f.write("FYF Validation Results\n")
            f.write("=====================\n\n")
            f.write(f"Original:  {original}\n")
            f.write(f"Processed: {processed}\n\n")
            for metric in selected_metrics:
                if metric in metrics_results:
                    f.write(f"{metric.upper()}: {metrics_results[metric]:.6f}\n")
            f.write("\nResidual Statistics:\n")
            for key, value in residual_stats.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        echo_colored(f"\nResults saved: {result_file}", Colors.SUCCESS)
        
        # Generate plots if requested
        if generate_plots:
            from fyf.visualization.plotting import PlotGenerator
            
            plot_gen = PlotGenerator()
            plot_dir = output_dir / "plots" 
            plot_dir.mkdir(exist_ok=True)
            
            # Use existing plotting functionality
            variants = {'original': original_data}
            processed_dict = {'processed': processed_data}
            basename = Path(original).stem
            
            plot_gen.generate_all_plots(plot_dir, variants, processed_dict, basename)
            echo_colored(f"Plots saved to: {plot_dir}", Colors.SUCCESS)
            
    except Exception as e:
        echo_colored(f"Validation error: {e}", Colors.ERROR)

# Updated plot command with config support
@cli.command()
@click.argument('original', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--plot-type', type=click.Choice(['comparison', 'residual', 'all']), help='Type of plot')
@click.option('--output-dir', '-o', type=Path, help='Output directory')
@click.option('--dpi', type=int, help='DPI for plots')
@click.option('--cmap', type=str, help='Colormap for images')
@click.option('--residual-cmap', type=str, help='Colormap for residual plots')
@click.pass_context
def plot(ctx, original, processed, config, plot_type, output_dir, dpi, cmap, residual_cmap):
    """Generate plots from processed data"""
    echo_banner("FYF Plot Generation")
    
    # Load config if provided
    config_data = {}
    if config:
        config_data = ConfigManager.load_config(Path(config))
    
    # Merge CLI args with config
    cli_args = {
        'plot_type': plot_type,
        'output_dir': str(output_dir) if output_dir else None,
        'dpi': dpi,
        'cmap': cmap,
        'residual_cmap': residual_cmap
    }
    
    plot_config = ConfigManager.merge_with_cli_args(config_data, 'plot', cli_args)
    
    # Set values from config
    plot_type = plot_config.get('plot_type', 'all')
    output_dir = Path(plot_config.get('output_dir', './plots'))
    dpi = plot_config.get('dpi', 150)
    cmap = plot_config.get('cmap', 'viridis')
    residual_cmap = plot_config.get('residual_cmap', 'viridis')
    
    # Load FITS files
    try:
        from astropy.io import fits
        
        with fits.open(original) as hdul:
            original_data = hdul[0].data.astype(np.float64)
        with fits.open(processed) as hdul:
            processed_data = hdul[0].data.astype(np.float64)
    except Exception as e:
        echo_colored(f"Error loading files: {e}", Colors.ERROR)
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    echo_colored(f"Plot type: {plot_type}", Colors.INFO)
    echo_colored(f"DPI: {dpi}", Colors.INFO)
    echo_colored(f"Colormap: {cmap}", Colors.INFO)
    echo_colored(f"Output directory: {output_dir}", Colors.INFO)
    
    try:
        # Create plot configuration
        plot_cfg = PlotConfig(
            dpi=dpi,
            cmap=cmap,
            residual_cmap=residual_cmap,
            percentile_range=(plot_config.get('percentile_min', 1), plot_config.get('percentile_max', 99)),
            residual_percentile=(plot_config.get('residual_percentile_min', 1), plot_config.get('residual_percentile_max', 99))
        )
        
        # Use existing PlotGenerator
        from fyf.visualization.plotting import PlotGenerator
        plot_gen = PlotGenerator(
            cmap=plot_cfg.cmap,
            dpi=plot_cfg.dpi,
            residual_cmap=plot_cfg.residual_cmap,
            percentile_range=plot_cfg.percentile_range,
            residual_percentile=plot_cfg.residual_percentile
        )
        
        # Prepare data
        variants = {'original': original_data}
        processed_dict = {'processed': processed_data}
        basename = Path(original).stem
        
        # Generate requested plots
        if plot_type in ['comparison', 'all']:
            # Generate comparison plot using matplotlib directly for simplicity
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
            
            # Calculate common color scale
            vmin, vmax = np.nanpercentile(
                np.concatenate([original_data.flatten(), processed_data.flatten()]),
                [1, 99]
            )
            
            # Original image
            im0 = axes[0].imshow(original_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0].set_title("Original")
            plt.colorbar(im0, ax=axes[0])
            
            # Processed image
            im1 = axes[1].imshow(processed_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title("Processed")
            plt.colorbar(im1, ax=axes[1])
            
            plt.suptitle(f"Comparison: {basename}")
            plt.tight_layout()
            
            plot_path = output_dir / f"comparison_{basename}.png"
            plt.savefig(plot_path, dpi=dpi)
            plt.close()
            
            echo_colored(f"Comparison plot saved: {plot_path}", Colors.SUCCESS)
        
        if plot_type in ['residual', 'all']:
            # Generate residual plots using existing functionality
            plot_gen._save_residual_plots(output_dir, original_data, processed_dict, basename)
            echo_colored(f"Residual plots saved to: {output_dir}", Colors.SUCCESS)
            
    except Exception as e:
        echo_colored(f"Error generating plots: {e}", Colors.ERROR)

# Version command
@cli.command()
def version():
    """Show version information"""
    echo_colored(f"FYF v{VERSION}", Colors.INFO)
    
    # Try to show R/INLA status
    try:
        from fyf.r import check_r_installed, check_inla_installed
        r_status = "✓" if check_r_installed() else "✗"
        inla_status = "✓" if check_inla_installed() else "✗"
        echo_colored(f"R installed: {r_status}", Colors.INFO)
        echo_colored(f"R-INLA installed: {inla_status}", Colors.INFO)
    except ImportError:
        echo_colored("R status: Unable to check", Colors.WARNING)

# Examples command
@cli.command()
def examples():
    """Show usage examples"""
    echo_banner("FYF Usage Examples")
    
    examples = [
        "Basic simulation:",
        "  fyf simulate image.fits --cosmic-fraction 0.02 --trails 1",
        "",
        "Process with INLA:",
        "  fyf process masked_image.fits --shape radius --scaling",
        "",
        "Manual workflow (replaces pipeline):",
        "  # Step 1: Simulate artifacts",
        "  fyf simulate data.fits -c 0.01 -t 2 -o ./artifacts/",
        "  # Step 2: Process with INLA",
        "  fyf process ./artifacts/data/combined.fits --shape ellipse -o ./processed/",
        "  # Step 3: Validate results",
        "  fyf validate data.fits ./processed/data/original_processed.fits --plot",
        "  # Step 4: Generate detailed plots",
        "  fyf plot data.fits ./processed/data/original_processed.fits --plot-type all",
        "",
        "Validate results:",
        "  fyf validate original.fits processed.fits --plot",
        "",
        "Generate plots:",
        "  fyf plot original.fits processed.fits --plot-type residual"
    ]
    
    for line in examples:
        if line.startswith("  fyf"):
            echo_colored(line, f"{Colors.BOLD}{Fore.WHITE}")
        elif line.startswith("  #"):
            echo_colored(line, Colors.WARNING)
        else:
            echo_colored(line, Colors.INFO)

# Help command
@cli.command()
@click.argument('command', required=False)
@click.pass_context
def help(ctx, command):
    """Show help for a specific command"""
    if command:
        # Show help for specific command
        cmd = cli.get_command(ctx, command)
        if cmd:
            echo_colored(cmd.get_help(ctx), Colors.INFO)
        else:
            echo_colored(f"Unknown command: {command}", Colors.ERROR)
    else:
        # Show general help
        echo_colored(cli.get_help(ctx), Colors.INFO)

# Main entry point
def main():
    """Main CLI entry point"""
    try:
        cli()
    except Exception as e:
        echo_colored(f"Error: {e}", Colors.ERROR)
        sys.exit(1)

if __name__ == '__main__':
    main()
