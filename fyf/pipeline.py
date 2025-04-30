from pathlib import Path
from typing import Dict
import numpy as np
from .mask_generator import MaskGenerator
from .fits_processing import FitsProcessor
from .file_handling import FileHandler
from .plotting_fits import PlotGenerator
from .mask_config import CosmicConfig, SatelliteConfig

class SimulationPipeline:
    """Coordinates the complete simulation workflow"""
    
    def __init__(self, cosmic_cfg: CosmicConfig, satellite_cfg: SatelliteConfig):
        self.mask_generator = MaskGenerator(cosmic_cfg, satellite_cfg)
        self.fits_processor = FitsProcessor(cosmic_cfg, satellite_cfg)
        self.file_handler = FileHandler()
        self.plot_generator = PlotGenerator()

    def process_file(self, input_path: Path) -> None:
        """Execute complete processing pipeline for a single file"""
        try:
            data, header = self.file_handler.load_fits(input_path)
            basename = input_path.stem
            output_dir = self.file_handler.create_output_structure(basename)

            masks = self.mask_generator.generate_all_masks(data)
            variants = self.fits_processor.create_variants(data, masks)
            self.fits_processor.save_masked_variants(data, masks)
            processed = self.fits_processor.process_variants(variants)

            # Add processed variants to the main dictionary
            for key in processed:
                if processed[key] is not None:
                    variants[f"{key}_processed"] = processed[key]

            self.file_handler.save_outputs(output_dir, variants, masks, header)
            self.plot_generator.generate_all_plots(output_dir, variants, processed, basename)

            print(f"✅ Successfully processed {basename}")

        except (ValueError, OSError, RuntimeError) as e:
            print(f"❌ Error processing {input_path.name}: {str(e)}")
  