#!/usr/bin/env python3
"""
Batch processing script for FYF

This script provides advanced batch processing capabilities for the FYF package,
including parallel processing, custom configurations, and report generation.
"""

import argparse
import concurrent.futures
import glob
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fyf.config.config import CosmicConfig, SatelliteConfig, INLAConfig, PlotConfig
from fyf.pipeline import SimulationPipeline
from fyf.visualization.report import ReportGenerator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch process FITS files with FYF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        'input_patterns',
        nargs='+',
        help='Input FITS file patterns (supports wildcards and directories)'
    )
    
    # Cosmic ray configuration
    cosmic_group = parser.add_argument_group('Cosmic Ray Options')
    cosmic_group.add_argument(
        '--cosmic-fraction', '-c',
        type=float,
        default=0.01,
        help='Fraction of pixels affected by cosmic rays (0-1)'
    )
    cosmic_group.add_argument(
        '--cosmic-value',
        type=float,
        default=float('nan'),
        help='Value to use for cosmic ray pixels'
    )
    cosmic_group.add_argument(
        '--cosmic-seed',
        type=int,
        default=None,
        help='Random seed for cosmic ray generation'
    )
    
    # Satellite trail configuration
    satellite_group = parser.add_argument_group('Satellite Trail Options')
    satellite_group.add_argument(
        '--trails', '-t',
        type=int,
        default=1,
        help='Number of satellite trails to generate'
    )
    satellite_group.add_argument(
        '--trail-width',
        type=int,
        default=3,
        help='Width of satellite trails in pixels'
    )
    satellite_group.add_argument(
        '--trail-value',
        type=float,
        default=float('nan'),
        help='Value to use for satellite trail pixels'
    )
    
    # INLA configuration
    inla_group = parser.add_argument_group('INLA Options')
    inla_group.add_argument(
        '--shape', '-s',
        type=str,
        default='none',
        choices=['none', 'radius', 'ellipse'],
        help='Shape parameter for INLA model'
    )
    inla_group.add_argument(
        '--mesh-cutoff',
        type=float,
        default=None,
        help='Mesh cutoff for INLA'
    )
    inla_group.add_argument(
        '--tolerance',
        type=float,
        default=1e-4,
        help='Tolerance for INLA algorithm'
    )
    inla_group.add_argument(
        '--restart',
        type=int,
        default=0,
        help='Number of restarts for INLA algorithm'
    )
    inla_group.add_argument(
        '--scaling',
        action='store_true',
        help='Enable log10 scaling of values'
    )
    inla_group.add_argument(
        '--nonstationary',
        action='store_true',
        help='Use nonstationary model'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Base output directory'
    )
    output_group.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate HTML report'
    )
    output_group.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generation of plots'
    )
    
    # Processing configuration
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of worker processes (use 0 for number of CPU cores)'
    )
    proc_group.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout per file in seconds'
    )
    proc_group.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that have already been processed'
    )
    proc_group.add_argument(
        '--config-file',
        type=str,
        help='JSON configuration file'
    )
    
    return parser.parse_args()

def resolve_input_patterns(patterns: List[str]) -> List[Path]:
    """
    Resolve input patterns to file paths
    
    Args:
        patterns: List of input patterns (files, directories, wildcards)
        
    Returns:
        List of resolved file paths
    """
    resolved_paths = []
    
    for pattern in patterns:
        # Check if it's a directory
        if os.path.isdir(pattern):
            # Find all .fits files in the directory
            for ext in ['.fits', '.fit', '.FITS', '.FIT']:
                resolved_paths.extend(Path(pattern).glob(f'*{ext}'))
        else:
            # Expand wildcards
            for path in glob.glob(pattern):
                if path.endswith(('.fits', '.fit', '.FITS', '.FIT')):
                    resolved_paths.append(Path(path))
    
    # Remove duplicates and sort
    return sorted(set(resolved_paths))

def load_config_file(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Dictionary with configuration values
    """
    import json
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def process_file(
    file_path: Path,
    cosmic_cfg: CosmicConfig,
    satellite_cfg: SatelliteConfig,
    inla_cfg: INLAConfig,
    plot_cfg: Optional[PlotConfig] = None,
    output_dir: str = './output',
) -> Dict[str, Any]:
    """
    Process a single file
    
    Args:
        file_path: Path to the input file
        cosmic_cfg: Cosmic ray configuration
        satellite_cfg: Satellite trail configuration
        inla_cfg: INLA configuration
        plot_cfg: Plot configuration
        output_dir: Output directory
        
    Returns:
        Dictionary with processing results
    """
    # Create base output directory
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize pipeline
    pipeline = SimulationPipeline(
        cosmic_cfg, satellite_cfg, inla_cfg, plot_cfg
    )
    
    # Process the file
    try:
        result = pipeline.process_file(file_path)
        result['file_path'] = str(file_path)
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            'file_path': str(file_path),
            'success': False,
            'error': str(e),
            'process_time': 0.0
        }

def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration from file if specified
    config = {}
    if args.config_file:
        config = load_config_file(args.config_file)
    
    # Resolve input patterns
    input_files = resolve_input_patterns(args.input_patterns)
    
    if not input_files:
        print("No input files found.")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input files.")
    
    # Create configurations
    cosmic_cfg = CosmicConfig(
        fraction=config.get('cosmic_fraction', args.cosmic_fraction),
        value=config.get('cosmic_value', args.cosmic_value),
        seed=config.get('cosmic_seed', args.cosmic_seed)
    )
    
    satellite_cfg = SatelliteConfig(
        num_trails=config.get('trails', args.trails),
        trail_width=config.get('trail_width', args.trail_width),
        min_angle=config.get('min_angle', -45.0),
        max_angle=config.get('max_angle', 45.0),
        value=config.get('trail_value', args.trail_value)
    )
    
    inla_cfg = INLAConfig(
        shape=config.get('shape', args.shape),
        mesh_cutoff=config.get('mesh_cutoff', args.mesh_cutoff),
        tolerance=config.get('tolerance', args.tolerance),
        restart=config.get('restart', args.restart),
        scaling=config.get('scaling', args.scaling),
        nonstationary=config.get('nonstationary', args.nonstationary)
    )
    
    plot_cfg = None
    if not args.skip_plots:
        plot_cfg = PlotConfig()
    
    # Create output directory
    output_dir = Path(config.get('output_dir', args.output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine number of workers
    workers = args.workers
    if workers <= 0:
        import multiprocessing
        workers = multiprocessing.cpu_count()
    
    # Skip existing files if requested
    if args.skip_existing:
        filtered_files = []
        for file_path in input_files:
            output_file = output_dir / file_path.stem / 'data' / 'original_processed.fits'
            if not output_file.exists():
                filtered_files.append(file_path)
            else:
                print(f"Skipping existing file: {file_path}")
        
        input_files = filtered_files
    
    # Display configuration
    print(f"Configuration:")
    print(f"  - Cosmic rays: {cosmic_cfg.fraction*100:.1f}% @ {cosmic_cfg.value} ADU")
    print(f"  - Satellite trails: {satellite_cfg.num_trails} trails @ {satellite_cfg.value} ADU")
    print(f"  - INLA shape: {inla_cfg.shape}")
    print(f"  - Workers: {workers}")
    print(f"  - Output directory: {output_dir}")
    
    # Process files
    start_time = time.time()
    results = {}
    
    if workers > 1 and len(input_files) > 1:
        print(f"Processing {len(input_files)} files with {workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(
                    process_file,
                    file_path,
                    cosmic_cfg,
                    satellite_cfg,
                    inla_cfg,
                    plot_cfg,
                    str(output_dir)
                ): file_path
                for file_path in input_files
            }
            
            # Process as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=args.timeout)
                    results[file_path.name] = result
                    
                    status = "✅" if result.get('success', False) else "❌"
                    print(f"[{i}/{len(input_files)}] {status} {file_path.name}")
                    
                except concurrent.futures.TimeoutError:
                    print(f"[{i}/{len(input_files)}] ⏱️ Timeout: {file_path.name}")
                    results[file_path.name] = {
                        'file_path': str(file_path),
                        'success': False,
                        'error': 'Timeout',
                        'process_time': args.timeout
                    }
                except Exception as e:
                    print(f"[{i}/{len(input_files)}] ❌ Error: {file_path.name} - {e}")
                    results[file_path.name] = {
                        'file_path': str(file_path),
                        'success': False,
                        'error': str(e),
                        'process_time': 0.0
                    }
    else:
        print(f"Processing {len(input_files)} files sequentially...")
        for i, file_path in enumerate(input_files, 1):
            print(f"[{i}/{len(input_files)}] Processing: {file_path.name}")
            result = process_file(
                file_path,
                cosmic_cfg,
                satellite_cfg,
                inla_cfg,
                plot_cfg,
                str(output_dir)
            )
            results[file_path.name] = result
            
            status = "✅" if result.get('success', False) else "❌"
            print(f"[{i}/{len(input_files)}] {status} {file_path.name}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print summary
    success_count = sum(1 for r in results.values() if r.get('success', False))
    print(f"\nProcessing complete!")
    print(f"  - Total files: {len(results)}")
    print(f"  - Successfully processed: {success_count}")
    print(f"  - Failed: {len(results) - success_count}")
    print(f"  - Total time: {total_time:.2f} seconds")
    
    # Generate report if requested
    if args.report:
        try:
            report_gen = ReportGenerator(output_dir)
            report_path = report_gen.generate_summary_report(
                title="FYF Batch Processing Results",
                results=results
            )
            print(f"Report generated: {report_path}")
        except Exception as e:
            print(f"Error generating report: {e}")
    
    # Return success if all files processed successfully
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())