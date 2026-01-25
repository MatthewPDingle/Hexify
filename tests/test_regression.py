"""Regression tests for Hexify image processing.

These tests compare output from the current code against pre-generated
reference outputs to ensure that refactoring does not change behavior.
"""
import os
import sys
import pytest
import numpy as np
import cv2
from typing import Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexagon_processor import HexagonProcessor

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
REFERENCE_DIR = os.path.join(TESTS_DIR, 'reference_outputs')


# =============================================================================
# Helper Functions
# =============================================================================

def compare_images(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """
    Compare two images and return similarity metrics.

    Args:
        img1: First image as numpy array (H, W, C) or (H, W)
        img2: Second image as numpy array (H, W, C) or (H, W)

    Returns:
        Dictionary containing:
        - 'identical': bool, True if images are pixel-perfect identical
        - 'shape_match': bool, True if shapes are identical
        - 'max_diff': float, maximum absolute difference between any pixels
        - 'mean_diff': float, mean absolute difference across all pixels
        - 'rmse': float, root mean square error
        - 'psnr': float, peak signal-to-noise ratio (higher is better, inf if identical)
        - 'percent_different': float, percentage of pixels that differ
    """
    result = {
        'identical': False,
        'shape_match': img1.shape == img2.shape,
        'max_diff': float('inf'),
        'mean_diff': float('inf'),
        'rmse': float('inf'),
        'psnr': 0.0,
        'percent_different': 100.0
    }

    if not result['shape_match']:
        return result

    # Convert to float for calculations
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)

    # Calculate difference
    diff = np.abs(img1_float - img2_float)

    result['max_diff'] = float(np.max(diff))
    result['mean_diff'] = float(np.mean(diff))
    result['identical'] = result['max_diff'] == 0.0

    # RMSE
    mse = np.mean((img1_float - img2_float) ** 2)
    result['rmse'] = float(np.sqrt(mse))

    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        result['psnr'] = float('inf')
    else:
        max_pixel = 255.0
        result['psnr'] = float(20 * np.log10(max_pixel / np.sqrt(mse)))

    # Percentage of different pixels
    if len(img1.shape) == 3:
        # For color images, a pixel is different if any channel differs
        pixel_diff = np.any(diff > 0, axis=2)
    else:
        pixel_diff = diff > 0
    result['percent_different'] = float(100.0 * np.sum(pixel_diff) / pixel_diff.size)

    return result


def assert_images_equal(
    img1: np.ndarray,
    img2: np.ndarray,
    tolerance: int = 0,
    msg: str = ""
) -> None:
    """
    Assert that two images are equal within a tolerance.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
        tolerance: Maximum allowed difference per pixel (0 for exact match)
        msg: Optional message to include in assertion error

    Raises:
        AssertionError: If images differ beyond tolerance
    """
    metrics = compare_images(img1, img2)

    if not metrics['shape_match']:
        raise AssertionError(
            f"{msg}Image shapes do not match: {img1.shape} vs {img2.shape}"
        )

    if tolerance == 0:
        if not metrics['identical']:
            raise AssertionError(
                f"{msg}Images are not identical. "
                f"Max diff: {metrics['max_diff']}, "
                f"Mean diff: {metrics['mean_diff']:.4f}, "
                f"Pixels different: {metrics['percent_different']:.2f}%"
            )
    else:
        if metrics['max_diff'] > tolerance:
            raise AssertionError(
                f"{msg}Images differ beyond tolerance ({tolerance}). "
                f"Max diff: {metrics['max_diff']}, "
                f"Mean diff: {metrics['mean_diff']:.4f}, "
                f"Pixels different: {metrics['percent_different']:.2f}%"
            )


def load_reference_output(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a reference output image and its palette.

    Args:
        name: Base name of the reference output (e.g., 'ref_gradient_64x64_c16')

    Returns:
        Tuple of (image, palette) as numpy arrays
    """
    img_path = os.path.join(REFERENCE_DIR, f"{name}.png")
    palette_path = os.path.join(REFERENCE_DIR, f"{name}_palette.npy")

    img = cv2.imread(img_path)
    if img is None:
        pytest.skip(f"Reference image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    palette = None
    if os.path.exists(palette_path):
        palette = np.load(palette_path)

    return img, palette


def load_fixture(name: str) -> np.ndarray:
    """
    Load a fixture image.

    Args:
        name: Name of the fixture file (e.g., 'gradient_64x64.png')

    Returns:
        Image as numpy array in RGB format
    """
    path = os.path.join(FIXTURES_DIR, name)
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_image(input_image: np.ndarray, num_colors: int = 16) -> np.ndarray:
    """
    Process an image using HexagonProcessor with deterministic settings.

    Args:
        input_image: Input image as numpy array in RGB format
        num_colors: Number of palette colors

    Returns:
        Processed output image as numpy array
    """
    processor = HexagonProcessor(
        num_palette_colors=num_colors,
        num_processes=1,  # Single process for determinism
        hexagons_dir=None,
        chunk_size=32,
        save_hexagons=False
    )
    return processor.process_image(input_image)


# =============================================================================
# Pixel-Perfect Regression Tests
# =============================================================================

class TestPixelPerfectRegression:
    """Test suite for exact pixel-by-pixel comparison with reference outputs."""

    def test_gradient_64x64_c16_exact(self):
        """Test gradient 64x64 with 16 colors produces exact reference output."""
        input_img = load_fixture('gradient_64x64.png')
        ref_img, ref_palette = load_reference_output('ref_gradient_64x64_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=0,
            msg="Gradient 64x64 c16 regression: "
        )

    def test_color_blocks_64x64_c16_exact(self):
        """Test color blocks 64x64 with 16 colors produces exact reference output."""
        input_img = load_fixture('color_blocks_64x64.png')
        ref_img, ref_palette = load_reference_output('ref_color_blocks_64x64_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=0,
            msg="Color blocks 64x64 c16 regression: "
        )

    def test_gradient_32x48_c16_exact(self):
        """Test non-square gradient 32x48 with 16 colors produces exact reference output."""
        input_img = load_fixture('gradient_32x48.png')
        ref_img, ref_palette = load_reference_output('ref_gradient_32x48_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=0,
            msg="Gradient 32x48 c16 regression: "
        )

    def test_gradient_64x64_c8_exact(self):
        """Test gradient 64x64 with 8 colors produces exact reference output."""
        input_img = load_fixture('gradient_64x64.png')
        ref_img, ref_palette = load_reference_output('ref_gradient_64x64_c8')

        output = process_image(input_img, num_colors=8)

        assert_images_equal(
            output, ref_img, tolerance=0,
            msg="Gradient 64x64 c8 regression: "
        )

    def test_fire_64x64_c16_exact(self):
        """Test fire image 64x64 with 16 colors produces exact reference output."""
        fire_path = os.path.join(FIXTURES_DIR, 'fire_64x64.png')
        if not os.path.exists(fire_path):
            pytest.skip("Fire fixture not available")

        input_img = load_fixture('fire_64x64.png')
        ref_img, ref_palette = load_reference_output('ref_fire_64x64_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=0,
            msg="Fire 64x64 c16 regression: "
        )


# =============================================================================
# Tolerance-Based Regression Tests
# =============================================================================

class TestToleranceRegression:
    """Test suite with tolerance for floating-point variations."""

    @pytest.mark.parametrize("tolerance", [1, 2, 5])
    def test_gradient_64x64_c16_with_tolerance(self, tolerance):
        """Test gradient with various tolerances for floating-point variations."""
        input_img = load_fixture('gradient_64x64.png')
        ref_img, _ = load_reference_output('ref_gradient_64x64_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=tolerance,
            msg=f"Gradient 64x64 c16 with tolerance {tolerance}: "
        )

    @pytest.mark.parametrize("tolerance", [1, 2, 5])
    def test_color_blocks_64x64_c16_with_tolerance(self, tolerance):
        """Test color blocks with various tolerances for floating-point variations."""
        input_img = load_fixture('color_blocks_64x64.png')
        ref_img, _ = load_reference_output('ref_color_blocks_64x64_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=tolerance,
            msg=f"Color blocks 64x64 c16 with tolerance {tolerance}: "
        )

    @pytest.mark.parametrize("tolerance", [1, 2, 5])
    def test_gradient_32x48_c16_with_tolerance(self, tolerance):
        """Test non-square gradient with various tolerances."""
        input_img = load_fixture('gradient_32x48.png')
        ref_img, _ = load_reference_output('ref_gradient_32x48_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=tolerance,
            msg=f"Gradient 32x48 c16 with tolerance {tolerance}: "
        )

    @pytest.mark.parametrize("tolerance", [1, 2, 5])
    def test_gradient_64x64_c8_with_tolerance(self, tolerance):
        """Test 8-color gradient with various tolerances."""
        input_img = load_fixture('gradient_64x64.png')
        ref_img, _ = load_reference_output('ref_gradient_64x64_c8')

        output = process_image(input_img, num_colors=8)

        assert_images_equal(
            output, ref_img, tolerance=tolerance,
            msg=f"Gradient 64x64 c8 with tolerance {tolerance}: "
        )

    @pytest.mark.parametrize("tolerance", [1, 2, 5])
    def test_fire_64x64_c16_with_tolerance(self, tolerance):
        """Test fire image with various tolerances."""
        fire_path = os.path.join(FIXTURES_DIR, 'fire_64x64.png')
        if not os.path.exists(fire_path):
            pytest.skip("Fire fixture not available")

        input_img = load_fixture('fire_64x64.png')
        ref_img, _ = load_reference_output('ref_fire_64x64_c16')

        output = process_image(input_img, num_colors=16)

        assert_images_equal(
            output, ref_img, tolerance=tolerance,
            msg=f"Fire 64x64 c16 with tolerance {tolerance}: "
        )


# =============================================================================
# Image Comparison Metrics Tests
# =============================================================================

class TestComparisonMetrics:
    """Tests for the compare_images helper function itself."""

    def test_identical_images(self):
        """Verify metrics for identical images."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        metrics = compare_images(img, img.copy())

        assert metrics['identical'] is True
        assert metrics['shape_match'] is True
        assert metrics['max_diff'] == 0.0
        assert metrics['mean_diff'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['psnr'] == float('inf')
        assert metrics['percent_different'] == 0.0

    def test_different_shapes(self):
        """Verify handling of different shaped images."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 50, 3), dtype=np.uint8)
        metrics = compare_images(img1, img2)

        assert metrics['identical'] is False
        assert metrics['shape_match'] is False

    def test_known_difference(self):
        """Verify metrics for images with known difference."""
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2[0, 0] = [10, 10, 10]  # Single pixel difference

        metrics = compare_images(img1, img2)

        assert metrics['identical'] is False
        assert metrics['shape_match'] is True
        assert metrics['max_diff'] == 10.0
        assert metrics['percent_different'] == 1.0  # 1 of 100 pixels

    def test_grayscale_images(self):
        """Verify metrics work for grayscale images."""
        img1 = np.zeros((50, 50), dtype=np.uint8)
        img2 = np.ones((50, 50), dtype=np.uint8) * 128

        metrics = compare_images(img1, img2)

        assert metrics['identical'] is False
        assert metrics['shape_match'] is True
        assert metrics['max_diff'] == 128.0
        assert metrics['mean_diff'] == 128.0
        assert metrics['percent_different'] == 100.0


class TestAssertImagesEqual:
    """Tests for the assert_images_equal helper function."""

    def test_identical_passes(self):
        """Identical images should pass assertion."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        assert_images_equal(img, img.copy())  # Should not raise

    def test_different_fails(self):
        """Different images should fail assertion with tolerance=0."""
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.ones((50, 50, 3), dtype=np.uint8)

        with pytest.raises(AssertionError):
            assert_images_equal(img1, img2, tolerance=0)

    def test_within_tolerance_passes(self):
        """Images within tolerance should pass."""
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.ones((50, 50, 3), dtype=np.uint8) * 5

        assert_images_equal(img1, img2, tolerance=5)  # Should not raise

    def test_beyond_tolerance_fails(self):
        """Images beyond tolerance should fail."""
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.ones((50, 50, 3), dtype=np.uint8) * 10

        with pytest.raises(AssertionError):
            assert_images_equal(img1, img2, tolerance=5)

    def test_shape_mismatch_fails(self):
        """Different shaped images should fail."""
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.zeros((50, 100, 3), dtype=np.uint8)

        with pytest.raises(AssertionError) as excinfo:
            assert_images_equal(img1, img2)
        assert "shapes do not match" in str(excinfo.value)


# =============================================================================
# Palette Consistency Tests
# =============================================================================

class TestPaletteConsistency:
    """Tests to verify palette generation is deterministic."""

    def test_palette_reproducibility(self):
        """Verify that the same input produces the same palette."""
        input_img = load_fixture('gradient_64x64.png')
        _, ref_palette = load_reference_output('ref_gradient_64x64_c16')

        if ref_palette is None:
            pytest.skip("Reference palette not available")

        processor = HexagonProcessor(
            num_palette_colors=16,
            num_processes=1,
            hexagons_dir=None,
            save_hexagons=False
        )
        processor.generate_palette(input_img)

        np.testing.assert_array_equal(
            processor.palette, ref_palette,
            err_msg="Palette should be identical to reference"
        )

    def test_palette_color_count(self):
        """Verify palette has the correct number of colors."""
        input_img = load_fixture('gradient_64x64.png')

        for num_colors in [8, 16, 32]:
            processor = HexagonProcessor(
                num_palette_colors=num_colors,
                num_processes=1,
                hexagons_dir=None,
                save_hexagons=False
            )
            processor.generate_palette(input_img)

            assert processor.palette.shape == (num_colors, 3), \
                f"Expected palette shape ({num_colors}, 3), got {processor.palette.shape}"


# =============================================================================
# Output Dimension Tests
# =============================================================================

class TestOutputDimensions:
    """Tests to verify output dimensions are correct."""

    @pytest.mark.parametrize("input_shape,expected_multiplier", [
        ((64, 64), 16),
        ((32, 48), 16),
        ((100, 100), 16),
    ])
    def test_output_size_multiplier(self, input_shape, expected_multiplier):
        """Verify output is 16x the input dimensions."""
        input_img = np.random.randint(0, 256, (*input_shape, 3), dtype=np.uint8)

        processor = HexagonProcessor(
            num_palette_colors=16,
            num_processes=1,
            hexagons_dir=None,
            save_hexagons=False
        )
        output = processor.process_image(input_img)

        expected_height = input_shape[0] * expected_multiplier
        expected_width = input_shape[1] * expected_multiplier

        assert output.shape[0] == expected_height, \
            f"Expected height {expected_height}, got {output.shape[0]}"
        assert output.shape[1] == expected_width, \
            f"Expected width {expected_width}, got {output.shape[1]}"
        assert output.shape[2] == 3, \
            f"Expected 3 channels, got {output.shape[2]}"
