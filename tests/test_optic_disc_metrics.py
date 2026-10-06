import unittest

import numpy as np

from src.metrics.optic_disc import calculate_vertical_cdr


class VerticalCDRTests(unittest.TestCase):
    def test_nested_rectangles_have_known_vertical_cdr(self):
        disc = np.zeros((12, 10), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[2:10, 1:9] = 1
        cup[4:8, 3:7] = 1

        result = calculate_vertical_cdr(disc, cup)

        self.assertEqual(result.cdr, 0.5)
        self.assertTrue(result.segmentation_valid)
        self.assertEqual(result.status, "ok")

    def test_empty_disc_is_invalid(self):
        disc = np.zeros((12, 10), dtype=np.uint8)
        cup = np.zeros_like(disc)
        cup[4:8, 3:7] = 1

        result = calculate_vertical_cdr(disc, cup)

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "empty_disc")

    def test_empty_cup_is_invalid(self):
        disc = np.zeros((12, 10), dtype=np.uint8)
        disc[2:10, 1:9] = 1

        result = calculate_vertical_cdr(disc, np.zeros_like(disc))

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "empty_cup")

    def test_cup_outside_disc_is_invalid(self):
        disc = np.zeros((12, 10), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[2:10, 1:9] = 1
        cup[4:8, 0:8] = 1

        result = calculate_vertical_cdr(disc, cup)

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "cup_outside_disc")

    def test_cup_larger_than_disc_is_invalid(self):
        disc = np.zeros((12, 12), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[2:10, 2:10] = 1
        cup[1:11, 1:11] = 1

        result = calculate_vertical_cdr(disc, cup)

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "cup_outside_disc")

    def test_isolated_single_pixel_components_do_not_change_cdr(self):
        disc = np.zeros((14, 12), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[2:10, 1:9] = 1
        cup[4:8, 3:7] = 1
        disc[12, 11] = 1
        cup[0, 0] = 1

        result = calculate_vertical_cdr(disc, cup)

        self.assertEqual(result.cdr, 0.5)
        self.assertTrue(result.segmentation_valid)
        self.assertEqual(result.status, "ok")

    def test_single_pixel_only_mask_is_invalid(self):
        disc = np.zeros((12, 10), dtype=np.uint8)
        cup = np.zeros_like(disc)
        disc[4, 4] = 1
        cup[4, 4] = 1

        result = calculate_vertical_cdr(disc, cup)

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "single_pixel_mask")

    def test_mismatched_shapes_are_invalid(self):
        result = calculate_vertical_cdr(np.zeros((5, 5)), np.zeros((6, 5)))

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "shape_mismatch")

    def test_non_2d_masks_are_invalid(self):
        result = calculate_vertical_cdr(np.zeros((5, 5, 3)), np.zeros((5, 5)))

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "invalid_dimensions")

    def test_nonbinary_or_nonfinite_values_are_invalid(self):
        disc = np.zeros((12, 10), dtype=np.float32)
        cup = np.zeros_like(disc)
        disc[2:10, 1:9] = 1
        cup[4:8, 3:7] = 1
        disc[2, 1] = np.nan

        result = calculate_vertical_cdr(disc, cup)

        self.assertFalse(result.segmentation_valid)
        self.assertTrue(np.isnan(result.cdr))
        self.assertEqual(result.status, "invalid_mask_values")


if __name__ == "__main__":
    unittest.main()
