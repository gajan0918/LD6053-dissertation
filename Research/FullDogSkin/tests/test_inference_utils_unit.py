import unittest

import torch
from PIL import Image

import path_setup  # noqa: F401
from inference_utils import build_prediction_tensor_batch, build_tta_images, normalize_tta_mode


class InferenceUtilsUnitTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGBA", (80, 60), color=(100, 80, 60, 255))

    def test_normalize_tta_mode_accepts_defaults_and_case_variants(self):
        self.assertEqual(normalize_tta_mode(None), "none")
        self.assertEqual(normalize_tta_mode(" HFLIP "), "hflip")
        self.assertEqual(normalize_tta_mode("ten_crop"), "ten_crop")

    def test_normalize_tta_mode_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported TTA mode"):
            normalize_tta_mode("diagonal")

    def test_build_tta_images_returns_expected_counts_and_sizes(self):
        expected_counts = {
            "none": 1,
            "hflip": 2,
            "five_crop": 5,
            "ten_crop": 10,
        }

        for mode, expected_count in expected_counts.items():
            with self.subTest(mode=mode):
                images = build_tta_images(self.image, image_size=32, tta_mode=mode)
                self.assertEqual(len(images), expected_count)
                self.assertTrue(all(image.size == (32, 32) for image in images))
                self.assertTrue(all(image.mode == "RGB" for image in images))

    def test_build_prediction_tensor_batch_returns_normalized_batch(self):
        batch = build_prediction_tensor_batch(self.image, image_size=32, tta_mode="hflip")

        self.assertEqual(tuple(batch.shape), (2, 3, 32, 32))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch).all())


if __name__ == "__main__":
    unittest.main()
