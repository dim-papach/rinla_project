#!/usr/bin/env python3
"""
FYF (Fill Your FITS) - CLI Tool

A command-line tool for processing astronomical FITS images using R-INLA.
"""

import sys
from pathlib import Path
import click
import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)

# Import existing FYF components
try:
    from fyf.core.config import CosmicConfig, SatelliteConfig, INLAConfig, PlotConfig
    from fyf.core.pipeline import SimulationPipeline
    from fyf.core.validation import validate_images
    from fyf.core.visualization.report import ReportGenerator
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
    for pattern in value:
        if Path(pattern).is_dir():
            # Directory: find all FITS files
            files.extend(Path(pattern).glob("*.fits"))
            files.extend(Path(pattern).glob("*.fit"))
        else:
            # Pattern: expand wildcards
            matches = glob.glob(pattern)
            if matches:
                files.extend([Path(f) for f in matches if f.endswith(('.fits', '.fit'))])
            elif pattern.endswith(('.fits', '.fit')):
                files.append(Path(pattern))
    
    # Filter existing files
    existing = [f for f in files if f.exists()]
    if not existing and value:
        raise click.BadParameter("No valid FITS files found")
    
    return sorted(set(existing))

# Main CLI group
@click.group()
@click.version_option(version=VERSION)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """FYF - Fill Your FITS: Process astronomical images using R-INLA"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        echo_banner(f"FYF v{VERSION}")

# Simulate command
@cli.command()
@click.argument('files', nargs=-1, required=True, callback=validate_fits_files)
@click.option('--cosmic-fraction', '-c', type=float, default=0.01, help='Cosmic ray fraction (0-1)')
@click.option('--trails', '-t', type=int, default=1, help='Number of satellite trails')
@click.option('--output-dir', '-o', type=Path, default=Path('./output'), help='Output directory')
@click.option('--report', '-r', is_flag=True, help='Generate HTML report')
@click.pass_context
def simulate(ctx, files, cosmic_fraction, trails, output_dir, report):
    """Simulate cosmic rays and satellite trails on FITS images"""
    echo_banner("FYF Simulation")
    
    # Create configurations using existing classes
    cosmic_cfg = CosmicConfig(fraction=cosmic_fraction)
    satellite_cfg = SatelliteConfig(num_trails=trails, trail_width=3)
    
    # Use existing pipeline
    pipeline = SimulationPipeline(cosmic_cfg, satellite_cfg, inla_cfg = None, plot_cfg=None)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    results = {}
    with click.progressbar(files, label='Processing') as bar:
        for file_path in bar:
            try:
                result = pipeline.process_file(file_path)
                results[file_path.name] = result
                status = "✓" if result['success'] else "✗"
                color = Colors.SUCCESS if result['success'] else Colors.ERROR
                echo_colored(f"{status} {file_path.name}", color)
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

# Process command
@cli.command()
@click.argument('files', nargs=-1, required=True, callback=validate_fits_files)
@click.option('--shape', type=click.Choice(['none', 'radius', 'ellipse']), default='none')
@click.option('--scaling', '-s', is_flag=True, help='Enable log10 scaling')
@click.option('--output-dir', '-o', type=Path, default=Path('./processed'), help='Output directory')
@click.pass_context
def process(ctx, files, shape, scaling, output_dir):
    """Process FITS images using R-INLA"""
    echo_banner("FYF Processing")
    
    # Check if R-INLA is available
    try:
        from fyf.r import check_inla_installed
        if not check_inla_installed():
            echo_colored("Warning: R-INLA not detected", Colors.WARNING)
    except ImportError:
        pass
    
    # Create INLA configuration
    inla_cfg = INLAConfig(shape=shape, scaling=scaling)
    plot_cfg = PlotConfig()
    
    # Empty configs for simulation (we're just processing)
    cosmic_cfg = CosmicConfig(fraction=0.0)
    satellite_cfg = SatelliteConfig(num_trails=0, trail_width=1)
    
    pipeline = SimulationPipeline(cosmic_cfg, satellite_cfg, inla_cfg = None, plot_cfg = None)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    with click.progressbar(files, label='Processing') as bar:
        for file_path in bar:
            try:
                result = pipeline.process_file(file_path)
                status = "✓" if result['success'] else "✗"
                color = Colors.SUCCESS if result['success'] else Colors.ERROR
                echo_colored(f"{status} {file_path.name}", color)
            except Exception as e:
                echo_colored(f"✗ {file_path.name}: {e}", Colors.ERROR)

# Pipeline command
@cli.command()
@click.argument('files', nargs=-1, required=True, callback=validate_fits_files)
@click.option('--cosmic-fraction', '-c', type=float, default=0.01)
@click.option('--trails', '-t', type=int, default=1)
@click.option('--shape', type=click.Choice(['none', 'radius', 'ellipse']), default='none')
@click.option('--scaling', '-s', is_flag=True)
@click.option('--output-dir', '-o', type=Path, default=Path('./output'))
@click.option('--report', '-r', is_flag=True)
@click.option('--workers', '-w', type=int, default=1, help='Parallel workers (0=auto)')
@click.pass_context
def pipeline(ctx, files, cosmic_fraction, trails, shape, scaling, output_dir, report, workers):
    """Run complete pipeline: simulate artifacts and process with INLA"""
    echo_banner("FYF Complete Pipeline")
    
    # Create all configurations
    cosmic_cfg = CosmicConfig(fraction=cosmic_fraction)
    satellite_cfg = SatelliteConfig(num_trails=trails, trail_width=3)
    inla_cfg = INLAConfig(shape=shape, scaling=scaling)
    plot_cfg = PlotConfig()
    
    pipeline = SimulationPipeline(cosmic_cfg, satellite_cfg, inla_cfg=None, plot_cfg = None)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    echo_colored(f"Cosmic rays: {cosmic_fraction*100:.1f}%", Colors.INFO)
    echo_colored(f"Satellite trails: {trails}", Colors.INFO)
    echo_colored(f"INLA shape: {shape}", Colors.INFO)
    
    # Process files
    results = {}
    if workers > 1:
        # Parallel processing
        import concurrent.futures
        import multiprocessing
        
        if workers == 0:
            workers = multiprocessing.cpu_count()
        
        echo_colored(f"Using {workers} workers", Colors.INFO)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(pipeline.process_file, file_path): file_path
                for file_path in files
            }
            
            # Collect results
            with click.progressbar(length=len(files), label='Processing') as bar:
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results[file_path.name] = result
                        status = "✓" if result['success'] else "✗"
                        color = Colors.SUCCESS if result['success'] else Colors.ERROR
                        echo_colored(f"{status} {file_path.name}", color)
                    except Exception as e:
                        echo_colored(f"✗ {file_path.name}: {e}", Colors.ERROR)
                        results[file_path.name] = {'success': False, 'error': str(e)}
                    bar.update(1)
    else:
        # Sequential processing
        with click.progressbar(files, label='Processing') as bar:
            for file_path in bar:
                try:
                    result = pipeline.process_file(file_path)
                    results[file_path.name] = result
                    status = "✓" if result['success'] else "✗"
                    color = Colors.SUCCESS if result['success'] else Colors.ERROR
                    echo_colored(f"{status} {file_path.name}", color)
                except Exception as e:
                    echo_colored(f"✗ {file_path.name}: {e}", Colors.ERROR)
                    results[file_path.name] = {'success': False, 'error': str(e)}
    
    # Summary
    success_count = sum(1 for r in results.values() if r.get('success', False))
    echo_colored(f"\nCompleted: {success_count}/{len(files)} files", Colors.INFO)
    
    # Generate report if requested
    if report and results:
        try:
            report_gen = ReportGenerator(output_dir)
            report_path = report_gen.generate_summary_report("FYF Pipeline", results)
            echo_colored(f"Report: {report_path}", Colors.SUCCESS)
        except Exception as e:
            echo_colored(f"Report error: {e}", Colors.ERROR)

# Validate command  
@cli.command()
@click.argument('original', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=Path, default=Path('./validation'))
@click.option('--plot', is_flag=True, help='Generate validation plots')
@click.pass_context
def validate(ctx, original, processed, output_dir, plot):
    """Validate processing results by comparing original and processed images"""
    echo_banner("FYF Validation")
    
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
        metrics = validate_images(original_data, processed_data)
        
        # Display results
        echo_colored("\nValidation Results:", Colors.INFO)
        echo_colored(f"  SSIM: {metrics['ssim']:.4f}", Colors.INFO)
        echo_colored(f"  MSE:  {metrics['mse']:.4f}", Colors.INFO)
        echo_colored(f"  MAE:  {metrics['mae']:.4f}", Colors.INFO)
        
        residual_stats = metrics['residual_stats']
        echo_colored("\nResidual Statistics:", Colors.INFO)
        echo_colored(f"  Mean:   {residual_stats['mean']:.2f}%", Colors.INFO)
        echo_colored(f"  StdDev: {residual_stats['std']:.2f}%", Colors.INFO)
        echo_colored(f"  NaN:    {residual_stats['nan_percentage']:.2f}%", Colors.INFO)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / "validation_results.txt"
        
        with open(result_file, 'w') as f:
            f.write("FYF Validation Results\n")
            f.write("=====================\n\n")
            f.write(f"Original:  {original}\n")
            f.write(f"Processed: {processed}\n\n")
            f.write(f"SSIM: {metrics['ssim']:.6f}\n")
            f.write(f"MSE:  {metrics['mse']:.6f}\n")
            f.write(f"MAE:  {metrics['mae']:.6f}\n\n")
            f.write("Residual Statistics:\n")
            for key, value in residual_stats.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        echo_colored(f"\nResults saved: {result_file}", Colors.SUCCESS)
        
        # Generate plots if requested
        if plot:
            from fyf.core.visualization.plotting import PlotGenerator
            
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

# Plot command
@cli.command()
@click.argument('original', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
@click.option('--plot-type', type=click.Choice(['comparison', 'residual', 'all']), default='all')
@click.option('--output-dir', '-o', type=Path, default=Path('./plots'))
@click.option('--dpi', type=int, default=150, help='DPI for plots')
@click.option('--cmap', type=str, default='viridis', help='Colormap for images')
@click.pass_context
def plot(ctx, original, processed, plot_type, output_dir, dpi, cmap):
    """Generate plots from processed data"""
    echo_banner("FYF Plot Generation")
    
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
    
    try:
        # Create plot configuration
        plot_cfg = PlotConfig(dpi=dpi, cmap=cmap)
        
        # Use existing PlotGenerator
        from fyf.core.visualization.plotting import PlotGenerator
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
        "Complete pipeline:",
        "  fyf pipeline data/*.fits -c 0.01 -t 2 --shape ellipse",
        "",
        "Parallel processing:",
        "  fyf pipeline *.fits --workers 4 --report",
        "",
        "Validate results:",
        "  fyf validate original.fits processed.fits --plot",
        "",
        "Generate plots:",
        "  fyf plot original.fits processed.fits --plot-type residual",
        "",
        "Batch processing with reports:",
        "  fyf pipeline data/ --output-dir results --report"
    ]
    
    for line in examples:
        if line.startswith("  fyf"):
            echo_colored(line, f"{Colors.BOLD}{Fore.WHITE}")
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
    cli()

if __name__ == '__main__':
    main()