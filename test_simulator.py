import unittest
import numpy as np
from pathlib import Path
import tempfile  # Import tempfile for temporary directory creation
from simulator import CosmicConfig, SatelliteConfig, MaskGenerator, FileHandler, FitsProcessor
from astropy.io import fits

class TestSimulator(unittest.TestCase):
    def setUp(self):
        """Set up configurations and test data for the simulator."""
        self.cosmic_cfg = CosmicConfig(fraction=0.05, value=100.0, seed=42)
        self.satellite_cfg = SatelliteConfig(num_trails=2, trail_width=3, min_angle=-30, max_angle=30, value=200.0)
        self.mask_generator = MaskGenerator(self.cosmic_cfg, self.satellite_cfg)
        self.data = np.zeros((100, 100), dtype=np.float32)  # Test data array

    def test_generate_cosmic_mask(self):
        """
        Test the generation of a cosmic ray mask.

        Validates:
        - The shape of the generated mask matches the input data.
        - The mask contains some affected pixels (i.e., not all zeros).
        """
        masks = self.mask_generator.generate_all_masks(self.data)
        self.assertEqual(masks['cosmic'].shape, self.data.shape)
        self.assertTrue(np.any(masks['cosmic']))  # Ensure some pixels are affected

    def test_generate_satellite_mask(self):
        """
        Test the generation of a satellite trail mask.

        Validates:
        - The shape of the generated mask matches the input data.
        - The mask contains some affected pixels (i.e., not all zeros).
        """
        masks = self.mask_generator.generate_all_masks(self.data)
        self.assertEqual(masks['satellite'].shape, self.data.shape)
        self.assertTrue(np.any(masks['satellite']))  # Ensure some pixels are affected

    def test_combined_mask(self):
        """
        Test the generation of a combined mask (cosmic rays + satellite trails).

        Validates:
        - The combined mask is the logical OR of the cosmic and satellite masks.
        """
        masks = self.mask_generator.generate_all_masks(self.data)
        combined_mask = masks['combined']
        self.assertTrue(np.array_equal(combined_mask, masks['cosmic'] | masks['satellite']))

    def test_load_fits(self):
        """
        Test the loading of a FITS file.

        Validates:
        - The loaded data matches the original data written to the FITS file.
        - The loaded header matches the original header written to the FITS file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            test_fits_path = Path(temp_dir) / "test.fits"
            fits_data = np.ones((50, 50), dtype=np.float32)
            fits_header = fits.Header()
            fits_header['TESTKEY'] = 'TESTVALUE'  # Add a test key-value pair
            fits.writeto(test_fits_path, fits_data, fits_header, overwrite=True)

            loaded_data, loaded_header = FileHandler.load_fits(test_fits_path)
            self.assertTrue(np.array_equal(loaded_data, fits_data))

            # Compare specific header keys
            for key in fits_header.keys():
                self.assertEqual(loaded_header.get(key), fits_header.get(key))

    def test_create_variants(self):
        """
        Test the creation of image variants (original, cosmic, satellite, combined).

        Validates:
        - All expected variants ('original', 'cosmic', 'satellite', 'combined') are created.
        - The 'original' variant matches the input data.
        """
        masks = self.mask_generator.generate_all_masks(self.data)
        fits_processor = FitsProcessor(self.cosmic_cfg, self.satellite_cfg)
        variants = fits_processor.create_variants(self.data, masks)

        self.assertIn('original', variants)
        self.assertIn('cosmic', variants)
        self.assertIn('satellite', variants)
        self.assertIn('combined', variants)
        self.assertTrue(np.array_equal(variants['original'], self.data))
    
    def test_large_data_handling(self):
        """
        Test handling of large input data.

        Validates:
        - The system processes large data arrays without errors.
        """
        large_data = np.zeros((10000, 10000), dtype=np.float32)
        masks = self.mask_generator.generate_all_masks(large_data)
        self.assertEqual(masks['cosmic'].shape, large_data.shape)
        self.assertEqual(masks['satellite'].shape, large_data.shape)
        self.assertEqual(masks['combined'].shape, large_data.shape)

    def test_missing_keys_in_variants(self):
        """
        Test that all expected keys are present in the generated variants.
    
        Validates:
        - The keys 'original', 'cosmic', 'satellite', and 'combined' are present.
        """
        masks = self.mask_generator.generate_all_masks(self.data)
        fits_processor = FitsProcessor(self.cosmic_cfg, self.satellite_cfg)
        variants = fits_processor.create_variants(self.data, masks)
    
        expected_keys = {'original', 'cosmic', 'satellite', 'combined'}
        self.assertEqual(set(variants.keys()), expected_keys)

    def test_random_seed_consistency(self):
        """
        Test that using the same random seed produces consistent results.
    
        Validates:
        - The generated masks are identical when the same seed is used.
        """
        cfg1 = CosmicConfig(fraction=0.05, value=100.0, seed=42)
        cfg2 = CosmicConfig(fraction=0.05, value=100.0, seed=42)
        mask_gen1 = MaskGenerator(cfg1, self.satellite_cfg)
        mask_gen2 = MaskGenerator(cfg2, self.satellite_cfg)
    
        masks1 = mask_gen1.generate_all_masks(self.data)
        masks2 = mask_gen2.generate_all_masks(self.data)
    
        self.assertTrue(np.array_equal(masks1['cosmic'], masks2['cosmic']))
        self.assertTrue(np.array_equal(masks1['satellite'], masks2['satellite']))
        
    def test_empty_data(self):
        """
        Test handling of empty input data.
    
        Validates:
        - The mask generator returns empty masks for empty input data.
        - No exceptions are raised during processing.
        """
        empty_data = np.zeros((0, 0), dtype=np.float32)
        masks = self.mask_generator.generate_all_masks(empty_data)
        self.assertEqual(masks['cosmic'].shape, empty_data.shape)
        self.assertEqual(masks['satellite'].shape, empty_data.shape)
        self.assertEqual(masks['combined'].shape, empty_data.shape)
    
    def test_invalid_fits_file(self):
        """
        Test loading of an invalid FITS file.
    
        Validates:
        - An appropriate exception is raised when loading a corrupted or invalid FITS file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_fits_path = Path(temp_dir) / "invalid.fits"
            with open(invalid_fits_path, "w") as f:
                f.write("This is not a valid FITS file.")
    
            with self.assertRaises(Exception):
                FileHandler.load_fits(invalid_fits_path)
    
if __name__ == '__main__':
    unittest.main()