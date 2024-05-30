import unittest
from unittest.mock import patch, Mock
import numpy as np
import cv2
from artgen import average_color, create_hex_mask, best_match_colors

class TestImageProcessing(unittest.TestCase):
    
    def test_average_color(self):
        # Mock an image and a mask
        image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
        mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        expected_average = np.array([255, 0, 0])
        np.testing.assert_array_equal(average_color(image, mask), expected_average)

    def test_create_hex_mask(self):
        center_x, center_y, radius, shape = 100, 100, 50, (200, 200)
        mask = create_hex_mask(center_x, center_y, radius, shape)
        self.assertEqual(mask.shape, (200, 200))  # Basic check for output mask dimensions

    def test_best_match_colors(self):
        lab_color = np.array([60, 10, 10])
        palette_lab = np.array([
            [60 + i, 10 + (i % 5), 10 + (i % 7)] for i in range(32)
        ])
        indices = best_match_colors(lab_color, palette_lab)
        
        # Calculate the expected indices manually
        distances = np.linalg.norm(palette_lab - lab_color, axis=1)
        expected_indices = np.argpartition(distances, 3)[:3]

        np.testing.assert_array_equal(indices, expected_indices)

# Add more tests to cover other functions and edge cases

if __name__ == '__main__':
    unittest.main()
