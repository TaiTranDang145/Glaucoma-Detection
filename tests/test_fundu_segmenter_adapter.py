import unittest

import numpy as np
import torch

from src.models.fundu_segmenter_adapter import (
    FunduSegmenterAdapter,
    decode_class_map,
    preprocess_image,
)


class FixedLogitModel(torch.nn.Module):
    def forward(self, image):
        logits = torch.zeros((image.shape[0], 3, 4, 4), device=image.device)
        logits[:, 1, 1:3, 1:3] = 2.0
        logits[:, 2, 2, 2] = 4.0
        return logits


class FunduSegmenterAdapterTests(unittest.TestCase):
    def test_decode_uses_classes_one_and_two_for_disc_and_two_for_cup(self):
        labels = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.uint8)

        disc, cup = decode_class_map(labels)

        np.testing.assert_array_equal(disc, labels != 0)
        np.testing.assert_array_equal(cup, labels == 2)

    def test_decode_rejects_non_2d_and_unknown_classes(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            decode_class_map(np.zeros((2, 2, 1), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "class IDs"):
            decode_class_map(np.array([[3]], dtype=np.uint8))

    def test_preprocessing_matches_official_256_imagenet_transform(self):
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        image[..., 0] = 255

        tensor = preprocess_image(image)

        self.assertEqual(tuple(tensor.shape), (1, 3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 100, 100]), (1.0 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(float(tensor[0, 1, 100, 100]), (0.0 - 0.456) / 0.224, places=5)

    def test_predict_masks_upsamples_to_input_and_returns_nested_binary_masks(self):
        adapter = FunduSegmenterAdapter(FixedLogitModel(), device="cpu")
        image = np.zeros((20, 30, 3), dtype=np.uint8)

        disc, cup = adapter.predict_masks(image)

        self.assertEqual(disc.shape, image.shape[:2])
        self.assertEqual(cup.shape, image.shape[:2])
        self.assertTrue(np.isin(disc, (0, 1)).all())
        self.assertTrue(np.isin(cup, (0, 1)).all())
        self.assertTrue(np.all(cup <= disc))
        self.assertGreater(int(disc.sum()), 0)
        self.assertGreater(int(cup.sum()), 0)


if __name__ == "__main__":
    unittest.main()
