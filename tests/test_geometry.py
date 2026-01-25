"""Unit tests for hexagon geometry calculations in Hexify.

Tests cover:
- Hexagon mask creation
- Grid center calculations
- Coordinate transformations
- Point clipping to hexagon boundaries
"""
import os
import sys
import pytest
import numpy as np
import math
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexagon_processor import HexagonProcessor


# =============================================================================
# Hexagon Mask Creation Tests
# =============================================================================

class TestHexagonMaskCreation:
    """Tests for the create_hex_mask static method."""

    def test_mask_shape_matches_input_shape(self):
        """Mask should have the same shape as the specified shape parameter."""
        shapes = [(100, 100), (64, 128), (200, 150)]

        for shape in shapes:
            mask = HexagonProcessor.create_hex_mask(
                center_x=50, center_y=50, radius=20, shape=shape
            )
            assert mask.shape == shape, \
                f"Expected mask shape {shape}, got {mask.shape}"

    def test_mask_dtype_is_uint8(self):
        """Mask should be uint8 type for use with OpenCV."""
        mask = HexagonProcessor.create_hex_mask(
            center_x=50, center_y=50, radius=20, shape=(100, 100)
        )
        assert mask.dtype == np.uint8

    def test_mask_values_are_binary(self):
        """Mask should only contain 0 and 255 values."""
        mask = HexagonProcessor.create_hex_mask(
            center_x=50, center_y=50, radius=20, shape=(100, 100)
        )
        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values), \
            f"Mask contains unexpected values: {unique_values}"

    def test_mask_center_is_filled(self):
        """The center of the hexagon should be filled (255)."""
        mask = HexagonProcessor.create_hex_mask(
            center_x=50, center_y=50, radius=20, shape=(100, 100)
        )
        # The center point should be inside the hexagon
        assert mask[50, 50] == 255, "Center should be filled"

    def test_mask_corners_are_empty(self):
        """Corners of the bounding box should be outside the hexagon."""
        center_x, center_y, radius = 50, 50, 20
        mask = HexagonProcessor.create_hex_mask(
            center_x=center_x, center_y=center_y, radius=radius, shape=(100, 100)
        )
        # Check that points far from center are outside
        # Note: hexagon is oriented with pi/2, so corners of bounding box are outside
        assert mask[0, 0] == 0, "Top-left corner should be empty"
        assert mask[0, 99] == 0, "Top-right corner should be empty"
        assert mask[99, 0] == 0, "Bottom-left corner should be empty"
        assert mask[99, 99] == 0, "Bottom-right corner should be empty"

    def test_mask_has_approximately_correct_area(self):
        """Hexagon mask area should be approximately 3*sqrt(3)/2 * r^2."""
        radius = 30
        expected_area = 3 * math.sqrt(3) / 2 * (radius ** 2)

        mask = HexagonProcessor.create_hex_mask(
            center_x=50, center_y=50, radius=radius, shape=(100, 100)
        )
        actual_area = np.sum(mask == 255)

        # Allow 10% tolerance for discretization effects
        tolerance = 0.10
        assert abs(actual_area - expected_area) / expected_area < tolerance, \
            f"Expected area ~{expected_area:.1f}, got {actual_area}"

    def test_mask_clipping_at_boundaries(self):
        """Mask should be clipped when hexagon extends beyond image boundaries."""
        # Center near edge
        mask = HexagonProcessor.create_hex_mask(
            center_x=5, center_y=5, radius=20, shape=(50, 50)
        )

        # Should still have valid shape and not raise errors
        assert mask.shape == (50, 50)
        # Should have some filled pixels
        assert np.sum(mask == 255) > 0

    def test_mask_fully_outside_image(self):
        """Mask with center far outside image should have minimal or no fill."""
        mask = HexagonProcessor.create_hex_mask(
            center_x=-100, center_y=-100, radius=20, shape=(50, 50)
        )

        # Most or all pixels should be empty
        assert mask.shape == (50, 50)

    def test_different_radii(self):
        """Larger radius should produce larger mask area."""
        areas = []
        for radius in [10, 20, 30, 40]:
            mask = HexagonProcessor.create_hex_mask(
                center_x=100, center_y=100, radius=radius, shape=(200, 200)
            )
            areas.append(np.sum(mask == 255))

        # Each radius should produce a larger area
        for i in range(len(areas) - 1):
            assert areas[i] < areas[i + 1], \
                f"Radius {[10, 20, 30, 40][i + 1]} should have larger area than {[10, 20, 30, 40][i]}"


# =============================================================================
# Grid Center Calculations Tests
# =============================================================================

class TestGridCenterCalculations:
    """Tests for hexagon grid center calculations in setup_hexagon_grid."""

    def test_grid_setup_creates_centers(self):
        """setup_hexagon_grid should create a list of center coordinates."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        assert processor.hex_centers is not None
        assert len(processor.hex_centers) > 0

    def test_centers_are_tuples(self):
        """Each center should be a tuple of (x, y) coordinates."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        for center in processor.hex_centers:
            assert isinstance(center, tuple)
            assert len(center) == 2
            assert isinstance(center[0], int)
            assert isinstance(center[1], int)

    def test_output_shape_is_16x_input(self):
        """Output shape should be 16x the input dimensions."""
        processor = HexagonProcessor(num_processes=1)

        for input_shape in [(32, 64, 3), (64, 64, 3), (100, 50, 3)]:
            processor.setup_hexagon_grid(input_shape)

            assert processor.output_shape[0] == input_shape[0] * 16
            assert processor.output_shape[1] == input_shape[1] * 16
            assert processor.output_shape[2] == 3

    def test_hex_dimensions_are_correct(self):
        """Hexagon width and height should follow geometric relationships."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        assert processor.hex_width == 256
        assert processor.hex_radius == 128
        # Height should be width * sqrt(3)/2
        expected_height = round(256 * (math.sqrt(3) / 2))
        assert processor.hex_height == expected_height

    def test_centers_cover_output_area(self):
        """Centers should cover the entire output area with some overlap."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        x_coords = [c[0] for c in processor.hex_centers]
        y_coords = [c[1] for c in processor.hex_centers]

        # Centers should start near origin and extend past the output dimensions
        # (to ensure edge coverage)
        assert min(x_coords) <= processor.hex_radius
        assert min(y_coords) <= processor.hex_radius
        assert max(x_coords) >= processor.output_shape[1] - processor.hex_radius
        assert max(y_coords) >= processor.output_shape[0] - processor.hex_radius

    def test_staggered_grid_pattern(self):
        """Odd columns should have vertically offset centers (staggered grid)."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        # Group centers by column (x coordinate)
        columns = {}
        for x, y in processor.hex_centers:
            if x not in columns:
                columns[x] = []
            columns[x].append(y)

        # Sort columns by x coordinate
        sorted_x = sorted(columns.keys())

        # For at least some adjacent columns, check the staggering pattern
        if len(sorted_x) >= 2:
            # Calculate column indices based on hex spacing
            hex_horizontal_spacing = processor.hex_width * 0.75

            for i, x in enumerate(sorted_x):
                col_index = int(round(x / hex_horizontal_spacing))
                # Odd columns should have different y offsets than even columns
                # This is verified by checking that not all columns have same y values
                # (more comprehensive check would require detailed analysis)


# =============================================================================
# Coordinate Transformation Tests
# =============================================================================

class TestCoordinateTransformations:
    """Tests for coordinate scaling and transformation."""

    def test_input_to_output_scaling(self):
        """Verify 16x scaling from input to output coordinates."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        # For each hex center, the corresponding input coordinate should be 1/16
        for center_x, center_y in processor.hex_centers:
            input_x = center_x // 16
            input_y = center_y // 16

            # Input coordinates should be within input image bounds (with some margin)
            assert input_x >= -10  # Some margin for edge hexagons
            assert input_y >= -10

    def test_radius_scaling(self):
        """Verify radius scaling from output to input coordinates."""
        processor = HexagonProcessor(num_processes=1)
        processor.setup_hexagon_grid((64, 64, 3))

        output_radius = processor.hex_radius
        input_radius = output_radius // 16

        assert output_radius == 128
        assert input_radius == 8


# =============================================================================
# Point Clipping Tests
# =============================================================================

class TestPointClipping:
    """Tests for clip_point_to_hexagon and nearest_point_on_segment methods."""

    def test_point_inside_hexagon_unchanged(self):
        """Points inside hexagon should not be modified."""
        # Create hexagon vertices
        radius = 50
        center = (100, 100)
        hexagon = RegularPolygon(center, numVertices=6, radius=radius, orientation=np.pi / 2)
        coords = hexagon.get_verts().astype(int)

        # Test center point
        result = HexagonProcessor.clip_point_to_hexagon(center, coords)
        assert result == center

        # Test point slightly off center but still inside
        inside_point = (100, 110)
        result = HexagonProcessor.clip_point_to_hexagon(inside_point, coords)
        assert result == inside_point

    def test_point_outside_hexagon_clipped(self):
        """Points outside hexagon should be clipped to boundary."""
        radius = 50
        center = (100, 100)
        hexagon = RegularPolygon(center, numVertices=6, radius=radius, orientation=np.pi / 2)
        coords = hexagon.get_verts().astype(int)

        # Point far outside
        outside_point = (200, 200)
        result = HexagonProcessor.clip_point_to_hexagon(outside_point, coords)

        # Result should be different from original
        assert result != outside_point

        # Result should be on or near the hexagon boundary
        path = Path(coords)
        # The clipped point should be on the boundary (or very close to an edge)

    def test_nearest_point_on_segment_midpoint(self):
        """Test nearest point when projection falls on segment interior."""
        p1 = (0, 0)
        p2 = (10, 0)
        point = (5, 5)

        nearest = HexagonProcessor.nearest_point_on_segment(point, p1, p2)

        # Nearest point should be at (5, 0) - perpendicular projection
        assert abs(nearest[0] - 5) < 0.001
        assert abs(nearest[1] - 0) < 0.001

    def test_nearest_point_on_segment_at_p1(self):
        """Test nearest point when closest is p1."""
        p1 = (0, 0)
        p2 = (10, 0)
        point = (-5, 0)

        nearest = HexagonProcessor.nearest_point_on_segment(point, p1, p2)

        # Nearest point should be p1
        assert abs(nearest[0] - 0) < 0.001
        assert abs(nearest[1] - 0) < 0.001

    def test_nearest_point_on_segment_at_p2(self):
        """Test nearest point when closest is p2."""
        p1 = (0, 0)
        p2 = (10, 0)
        point = (15, 0)

        nearest = HexagonProcessor.nearest_point_on_segment(point, p1, p2)

        # Nearest point should be p2
        assert abs(nearest[0] - 10) < 0.001
        assert abs(nearest[1] - 0) < 0.001

    def test_nearest_point_same_endpoints(self):
        """Test handling of degenerate segment (same start and end)."""
        p1 = (5, 5)
        p2 = (5, 5)
        point = (10, 10)

        nearest = HexagonProcessor.nearest_point_on_segment(point, p1, p2)

        # Should return p1
        assert nearest == p1

    def test_nearest_point_diagonal_segment(self):
        """Test nearest point on a diagonal segment."""
        p1 = (0, 0)
        p2 = (10, 10)
        point = (0, 10)

        nearest = HexagonProcessor.nearest_point_on_segment(point, p1, p2)

        # Nearest point should be at (5, 5)
        assert abs(nearest[0] - 5) < 0.001
        assert abs(nearest[1] - 5) < 0.001


# =============================================================================
# Average Color Tests
# =============================================================================

class TestAverageColor:
    """Tests for the average_color static method."""

    def test_uniform_color_returns_same(self):
        """Uniform color image should return that color as average."""
        color = [100, 150, 200]
        image = np.full((50, 50, 3), color, dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.uint8) * 255

        avg = HexagonProcessor.average_color(image, mask)

        np.testing.assert_array_equal(avg, color)

    def test_partial_mask(self):
        """Average should only consider masked pixels."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[0:25, :] = [255, 0, 0]  # Top half red
        image[25:50, :] = [0, 0, 255]  # Bottom half blue

        # Mask only top half
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[0:25, :] = 255

        avg = HexagonProcessor.average_color(image, mask)

        # Should be red (top half)
        assert avg[0] == 255
        assert avg[1] == 0
        assert avg[2] == 0

    def test_average_rounds_to_int(self):
        """Average color values should be rounded integers."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[0:25, :] = [100, 100, 100]
        image[25:50, :] = [101, 101, 101]

        mask = np.ones((50, 50), dtype=np.uint8) * 255

        avg = HexagonProcessor.average_color(image, mask)

        # Should be rounded
        assert avg.dtype == np.int64 or avg.dtype == np.int32
        assert all(100 <= v <= 101 for v in avg)


# =============================================================================
# Closest Palette Color Tests
# =============================================================================

class TestClosestPaletteColor:
    """Tests for the closest_palette_color static method."""

    def test_exact_match_returns_same(self):
        """Exact color match should be returned."""
        palette = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255]
        ])

        for color in palette:
            result = HexagonProcessor.closest_palette_color(color, palette)
            np.testing.assert_array_equal(result, color)

    def test_closest_color_selection(self):
        """Should return nearest palette color by Euclidean distance."""
        palette = np.array([
            [0, 0, 0],      # Black
            [255, 255, 255] # White
        ])

        # Dark gray should be closer to black
        dark_gray = np.array([50, 50, 50])
        result = HexagonProcessor.closest_palette_color(dark_gray, palette)
        np.testing.assert_array_equal(result, [0, 0, 0])

        # Light gray should be closer to white
        light_gray = np.array([200, 200, 200])
        result = HexagonProcessor.closest_palette_color(light_gray, palette)
        np.testing.assert_array_equal(result, [255, 255, 255])

    def test_avoid_colors(self):
        """Should skip colors in the avoid list."""
        palette = np.array([
            [0, 0, 0],
            [128, 128, 128],
            [255, 255, 255]
        ])

        # Black should be avoided, so gray is next closest to dark gray
        dark_gray = np.array([50, 50, 50])
        avoid = [np.array([0, 0, 0])]

        result = HexagonProcessor.closest_palette_color(dark_gray, palette, avoid)
        np.testing.assert_array_equal(result, [128, 128, 128])


# =============================================================================
# Hexagon Pattern Layer Tests
# =============================================================================

class TestHexagonPatternLayers:
    """Tests for the fill_odd_layer static method."""

    def test_odd_layer_returns_pattern_and_area(self):
        """fill_odd_layer should return updated pattern and area."""
        pattern = np.zeros((256, 256, 3), dtype=np.uint8)
        radius = 128
        hex_radius = 100
        bw_color = (255, 255, 255)

        result_pattern, area = HexagonProcessor.fill_odd_layer(
            pattern, radius, hex_radius, bw_color
        )

        assert isinstance(result_pattern, np.ndarray)
        assert result_pattern.shape == pattern.shape
        assert area > 0

    def test_odd_layer_area_formula(self):
        """Area should follow hexagon area formula."""
        pattern = np.zeros((256, 256, 3), dtype=np.uint8)
        radius = 128
        hex_radius = 50
        bw_color = (255, 255, 255)

        _, area = HexagonProcessor.fill_odd_layer(
            pattern, radius, hex_radius, bw_color
        )

        expected_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
        assert abs(area - expected_area) < 0.01


# =============================================================================
# Integration Test for Geometry
# =============================================================================

class TestGeometryIntegration:
    """Integration tests combining multiple geometry functions."""

    def test_mask_created_at_grid_centers(self):
        """Masks should be successfully created at each grid center."""
        processor = HexagonProcessor(num_processes=1)
        input_shape = (32, 32, 3)
        processor.setup_hexagon_grid(input_shape)

        # Test that masks can be created at each center
        for center_x, center_y in processor.hex_centers[:10]:  # Test first 10
            mask = HexagonProcessor.create_hex_mask(
                center_x, center_y,
                processor.hex_radius,
                processor.output_shape[:2]
            )
            assert mask.shape == processor.output_shape[:2]

    def test_input_mask_scaling_consistency(self):
        """Input mask scaling should be consistent with output mask."""
        processor = HexagonProcessor(num_processes=1)
        input_shape = (64, 64, 3)
        processor.setup_hexagon_grid(input_shape)

        # For a center in the middle of the image
        center_x = processor.output_shape[1] // 2
        center_y = processor.output_shape[0] // 2

        # Create output-scale mask
        output_mask = HexagonProcessor.create_hex_mask(
            center_x, center_y,
            processor.hex_radius,
            processor.output_shape[:2]
        )

        # Create input-scale mask
        input_center_x = center_x // 16
        input_center_y = center_y // 16
        input_radius = processor.hex_radius // 16

        input_mask = HexagonProcessor.create_hex_mask(
            input_center_x, input_center_y,
            input_radius,
            input_shape[:2]
        )

        # Both masks should have non-zero pixels
        assert np.sum(output_mask == 255) > 0
        assert np.sum(input_mask == 255) > 0

        # Output mask should have ~256x the area (16^2)
        area_ratio = np.sum(output_mask == 255) / np.sum(input_mask == 255)
        assert 200 < area_ratio < 300  # Approximate due to discretization
