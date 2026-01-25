"""
Tests for custom exceptions and input validation.
"""

import pytest
import numpy as np

from v2 import HexagonProcessor
from v2.exceptions import (
    HexifyError,
    InvalidImageError,
    PaletteError,
    PaletteNotGeneratedError,
    InsufficientColorsError,
    GridError,
    GridNotSetupError,
)


class TestExceptionHierarchy:
    """Test that exceptions have the correct hierarchy."""

    def test_invalid_image_error_is_hexify_error(self):
        assert issubclass(InvalidImageError, HexifyError)

    def test_palette_error_is_hexify_error(self):
        assert issubclass(PaletteError, HexifyError)

    def test_palette_not_generated_is_palette_error(self):
        assert issubclass(PaletteNotGeneratedError, PaletteError)

    def test_insufficient_colors_is_palette_error(self):
        assert issubclass(InsufficientColorsError, PaletteError)

    def test_grid_error_is_hexify_error(self):
        assert issubclass(GridError, HexifyError)

    def test_grid_not_setup_is_grid_error(self):
        assert issubclass(GridNotSetupError, GridError)


class TestInvalidImageErrorAttributes:
    """Test InvalidImageError stores shape information."""

    def test_stores_shape(self):
        err = InvalidImageError("test message", shape=(100, 100, 3))
        assert err.shape == (100, 100, 3)

    def test_shape_can_be_none(self):
        err = InvalidImageError("test message", shape=None)
        assert err.shape is None


class TestInsufficientColorsErrorAttributes:
    """Test InsufficientColorsError stores count information."""

    def test_stores_required_and_available(self):
        err = InsufficientColorsError(required=16, available=4)
        assert err.required == 16
        assert err.available == 4
        assert "16" in str(err)
        assert "4" in str(err)


class TestInputValidation:
    """Test that process_image validates input correctly."""

    def test_rejects_non_array(self):
        processor = HexagonProcessor(num_palette_colors=4)
        with pytest.raises(InvalidImageError) as excinfo:
            processor.process_image("not an array")
        assert "numpy array" in str(excinfo.value)

    def test_rejects_2d_array(self):
        processor = HexagonProcessor(num_palette_colors=4)
        image = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(InvalidImageError) as excinfo:
            processor.process_image(image)
        assert "3D array" in str(excinfo.value)

    def test_rejects_4d_array(self):
        processor = HexagonProcessor(num_palette_colors=4)
        image = np.zeros((1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError) as excinfo:
            processor.process_image(image)
        assert "3D array" in str(excinfo.value)

    def test_rejects_4_channels(self):
        processor = HexagonProcessor(num_palette_colors=4)
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(InvalidImageError) as excinfo:
            processor.process_image(image)
        assert "3 channels" in str(excinfo.value)

    def test_rejects_1_channel(self):
        processor = HexagonProcessor(num_palette_colors=4)
        image = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(InvalidImageError) as excinfo:
            processor.process_image(image)
        assert "3 channels" in str(excinfo.value)

    def test_accepts_valid_rgb_image(self):
        processor = HexagonProcessor(num_palette_colors=4)
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        # Should not raise
        result = processor.process_image(image)
        assert result is not None
        assert result.ndim == 3


class TestCatchingByBaseClass:
    """Test that exceptions can be caught by base class."""

    def test_catch_invalid_image_as_hexify_error(self):
        processor = HexagonProcessor(num_palette_colors=4)
        with pytest.raises(HexifyError):
            processor.process_image("not an array")

    def test_catch_palette_not_generated_as_palette_error(self):
        with pytest.raises(PaletteError):
            raise PaletteNotGeneratedError()

    def test_catch_grid_not_setup_as_grid_error(self):
        with pytest.raises(GridError):
            raise GridNotSetupError()
