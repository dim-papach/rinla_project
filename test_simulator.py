import unittest
import numpy as np
from pathlib import Path
from simulator import CosmicConfig, SatelliteConfig, MaskGenerator, FileHandler, FitsProcessor

class TestSimulator(unittest.TestCase):
    def setUp(self):
        """Set up configurations and test data."""
        self.cosmic_cfg = CosmicConfig(fraction=0.05, value=100.0, seed=42)
        self.satellite_cfg = SatelliteConfig(num_trails=2, trail_width=3, min_angle=-30, max_angle=30, value=200.0)
        self.mask_generator = MaskGenerator(self.cosmic_cfg, self.satellite_cfg)
        self.data = np.zeros((100, 100), dtype=np.float32)  # Test data array

    def test_generate_cosmic_mask(self):
        """Test generation of cosmic ray mask."""
        masks = self.mask_generator.generate_all_masks(self.data)
        self.assertEqual(masks['cosmic'].shape, self.data.shape)
        self.assertTrue(np.any(masks['cosmic']))  # Ensure some pixels are affected

    def test_generate_satellite_mask(self):
        """Test generation of satellite trail mask."""
        masks = self.mask_generator.generate_all_masks(self.data)
        self.assertEqual(masks['satellite'].shape, self.data.shape)
        self.assertTrue(np.any(masks['satellite']))  # Ensure some pixels are affected

    def test_combined_mask(self):
        """Test combined mask generation."""
        masks = self.mask_generator.generate_all_masks(self.data)
        combined_mask = masks['combined']
        self.assertTrue(np.array_equal(combined_mask, masks['cosmic'] | masks['satellite']))

    def test_load_fits(self):
        """Test loading of FITS file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_fits_path = Path(temp_dir) / "test.fits"
            fits_data = np.ones((50, 50), dtype=np.float32)
            fits_header = fits.Header()
            fits.writeto(test_fits_path, fits_data, fits_header, overwrite=True)

            loaded_data, loaded_header = FileHandler.load_fits(test_fits_path)
            self.assertTrue(np.array_equal(loaded_data, fits_data))
            self.assertEqual(loaded_header, fits_header)

    def test_create_variants(self):
        """Test creation of image variants."""
        masks = self.mask_generator.generate_all_masks(self.data)
        fits_processor = FitsProcessor(self.cosmic_cfg, self.satellite_cfg)
        variants = fits_processor.create_variants(self.data, masks)

        self.assertIn('original', variants)
        self.assertIn('cosmic', variants)
        self.assertIn('satellite', variants)
        self.assertIn('combined', variants)
        self.assertTrue(np.array_equal(variants['original'], self.data))

if __name__ == '__main__':
    unittest.main()