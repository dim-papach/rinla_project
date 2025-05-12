import argparse
import glob
import os
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
from colorama import Fore, init
from .mask_config import CosmicConfig, SatelliteConfig
from .pipeline import SimulationPipeline

#region Constants
DEFAULT_COSMIC_VALUE: float = np.float32(np.nan)
DEFAULT_SATELLITE_VALUE: float = np.float32(np.nan)
CMAP: str = 'viridis'
PLOT_DPI: int = 150
PERCENTILE_RANGE: Tuple[int, int] = (1, 99)
#endregion

def main() -> None:
    """Command line interface and processing coordination"""
    parser = argparse.ArgumentParser(
        description='Simulate cosmic rays and satellite trails in FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input parameters
    parser.add_argument(
        'input',
        type=str,
        nargs='+',  # Accept multiple input arguments
        help='Input FITS file(s) (supports wildcards and directories)'
    )

    # Cosmic ray parameters
    parser.add_argument(
        '--cosmic_fraction',
        type=float,
        default=0.01,
        help='Fraction of pixels affected by cosmic rays (0-1)'
    )
    parser.add_argument(
        '--cosmic_value',
        type=float,
        default=DEFAULT_COSMIC_VALUE,
        help='ADU value for cosmic ray pixels'
    )
    parser.add_argument(
        '--cosmic_seed',
        type=int,
        default=None,
        help='Random seed for cosmic ray generation'
    )

    # Satellite parameters
    parser.add_argument(
        '--num_trails',
        type=int,
        default=1,
        help='Number of satellite trails to generate'
    )
    parser.add_argument(
        '--trail_width',
        type=int,
        default=3,
        help='Width of trails in pixels'
    )
    parser.add_argument(
        '--min_angle',
        type=float,
        default=-45.0,
        help='Minimum trajectory angle (degrees from horizontal)'
    )
    parser.add_argument(
        '--max_angle',
        type=float,
        default=45.0,
        help='Maximum trajectory angle (degrees from horizontal)'
    )
    parser.add_argument(
        '--trail_value',
        type=float,
        default=DEFAULT_SATELLITE_VALUE,
        help='ADU value for satellite trail pixels'
    )

    args = parser.parse_args()

    # Create configuration objects
    cosmic_cfg = CosmicConfig(
        fraction=args.cosmic_fraction,
        value=args.cosmic_value,
        seed=args.cosmic_seed
    )

    satellite_cfg = SatelliteConfig(
        num_trails=args.num_trails,
        trail_width=args.trail_width,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        value=args.trail_value
    )

    # Find and process files
    input_files = []
    for pattern in args.input:
        if os.path.isdir(pattern):
            input_files.extend(glob.glob(os.path.join(pattern, '*.fits')))
        else:
            input_files.extend(glob.glob(pattern))

    if not input_files:
        print("❌ No FITS files found matching input pattern")
        return

    processor = SimulationPipeline(cosmic_cfg, satellite_cfg)

    print(f"🚀 Processing {len(input_files)} files with configuration:")
    print(f"  Cosmic rays: {cosmic_cfg.fraction*100:.1f}% @ {cosmic_cfg.value} ADU")
    print(f"  Satellite trails: {satellite_cfg.num_trails} trails @ {satellite_cfg.value} ADU")
    print("─" * 50)

    for file_path in input_files:
        processor.process_file(Path(file_path))
        print("\n🎉 Processing complete! Output structure:")
        print("output/")
        print("└── {input_files}/")
        print("    ├── data/")
        print("    ├── masks/")
        print("    └── plots/")
