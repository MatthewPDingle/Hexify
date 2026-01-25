"""
Geometric utilities for hexagon grid and mask operations.

This module handles all geometric calculations:
- Setting up the hexagon grid layout
- Creating hexagonal masks for sampling and rendering
- Point-in-polygon tests and clipping operations
- Coordinate transformations between input and output spaces
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path

from .config import (
    HEX_HEIGHT,
    HEX_HORIZONTAL_SPACING,
    HEX_NUM_VERTICES,
    HEX_ORIENTATION,
    HEX_RADIUS,
    HEX_SCALE_FACTOR,
    HEX_VERTICAL_SPACING,
    HEX_WIDTH,
)

# Module-level logger
logger = logging.getLogger(__name__)


class HexagonGrid:
    """
    Manages the hexagon grid layout for the output image.

    The grid uses an offset coordinate system where:
    - Hexagons are arranged in columns
    - Odd columns are offset vertically by half a hexagon height
    - This creates the characteristic honeycomb tessellation

    The output image is HEX_SCALE_FACTOR times larger than the input,
    allowing detailed hexagon patterns to be rendered.
    """

    def __init__(self):
        """Initialize the hexagon grid."""
        self.hex_centers = None
        self.output_shape = None
        self.hex_width = HEX_WIDTH
        self.hex_height = HEX_HEIGHT
        self.hex_radius = HEX_RADIUS

    def setup(self, input_shape: tuple) -> None:
        """
        Set up the hexagon grid based on input image dimensions.

        Calculates the output image size and generates all hexagon center
        coordinates. Uses the offset coordinate system for honeycomb layout.

        Args:
            input_shape: Shape of input image (height, width, channels)
        """
        # Output is scaled up by HEX_SCALE_FACTOR
        self.output_shape = (
            input_shape[0] * HEX_SCALE_FACTOR,
            input_shape[1] * HEX_SCALE_FACTOR,
            3
        )

        # Calculate grid dimensions with +2 for edge coverage
        cols = int(self.output_shape[1] / HEX_HORIZONTAL_SPACING) + 2
        # BUG FIX: Was using HEX_HORIZONTAL_SPACING instead of HEX_VERTICAL_SPACING
        # This caused too few rows for tall images
        rows = int(self.output_shape[0] / HEX_VERTICAL_SPACING) + 2

        # Generate hexagon centers using offset coordinate system
        # Odd columns are shifted down by half the vertical spacing
        self.hex_centers = [
            (
                int(HEX_HORIZONTAL_SPACING * col),
                int(HEX_VERTICAL_SPACING * row + (0.5 * HEX_VERTICAL_SPACING if col % 2 else 0))
            )
            for row in range(rows)
            for col in range(cols)
        ]

    def get_hex_bounds(self, center_x: int, center_y: int) -> tuple:
        """
        Get the bounding box for a hexagon, clipped to image bounds.

        Args:
            center_x: X coordinate of hexagon center
            center_y: Y coordinate of hexagon center

        Returns:
            Tuple of (x_start, y_start, x_end, y_end) clipped to output image bounds
        """
        x_start = max(center_x - self.hex_radius, 0)
        y_start = max(center_y - self.hex_radius, 0)
        x_end = min(center_x + self.hex_radius, self.output_shape[1])
        y_end = min(center_y + self.hex_radius, self.output_shape[0])
        return x_start, y_start, x_end, y_end

    def output_to_input_coords(self, center_x: int, center_y: int) -> tuple:
        """
        Convert output image coordinates to input image coordinates.

        Args:
            center_x: X coordinate in output space
            center_y: Y coordinate in output space

        Returns:
            Tuple of (input_x, input_y) coordinates
        """
        return (
            int(center_x / HEX_SCALE_FACTOR),
            int(center_y / HEX_SCALE_FACTOR)
        )

    def get_input_hex_radius(self) -> int:
        """Get the hexagon radius in input image coordinates."""
        return self.hex_radius // HEX_SCALE_FACTOR


class HexagonMask:
    """
    Utilities for creating and manipulating hexagonal masks.

    Masks are used to:
    - Sample average colors from hexagonal regions of the input image
    - Composite rendered hexagons onto the output image
    """

    @staticmethod
    def create(center_x: int, center_y: int, radius: int, shape: tuple) -> np.ndarray:
        """
        Create a hexagonal mask at the specified location.

        Uses matplotlib's RegularPolygon to generate accurate hexagon vertices,
        then fills with OpenCV. The hexagon is oriented with a flat top.

        Args:
            center_x: X coordinate of hexagon center
            center_y: Y coordinate of hexagon center
            radius: Radius of the hexagon (center to vertex)
            shape: Shape of the mask array (height, width)

        Returns:
            Binary mask as numpy array with 255 inside hexagon, 0 outside
        """
        mask = np.zeros(shape, dtype=np.uint8)

        # Create hexagon with flat-top orientation (pi/2 rotation)
        hexagon = RegularPolygon(
            (center_x, center_y),
            numVertices=HEX_NUM_VERTICES,
            radius=radius,
            orientation=HEX_ORIENTATION
        )

        # Get vertices and clip to image bounds
        coords = hexagon.get_verts()
        coords = np.clip(coords, [0, 0], [shape[1] - 1, shape[0] - 1]).astype(int)

        # Fill the hexagon
        mask = cv2.fillPoly(mask, [coords], 255)
        return mask

    @staticmethod
    def get_hexagon_vertices(center_x: int, center_y: int, radius: int) -> np.ndarray:
        """
        Get the vertices of a hexagon.

        Args:
            center_x: X coordinate of hexagon center
            center_y: Y coordinate of hexagon center
            radius: Radius of the hexagon

        Returns:
            Numpy array of vertex coordinates, shape (6, 2)
        """
        hexagon = RegularPolygon(
            (center_x, center_y),
            numVertices=HEX_NUM_VERTICES,
            radius=radius,
            orientation=HEX_ORIENTATION
        )
        return hexagon.get_verts().astype(int)


def average_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Calculate the average color of an image region defined by a mask.

    Args:
        image: Input image as numpy array (H, W, 3)
        mask: Binary mask defining the region (H, W)

    Returns:
        Average color as numpy array [R, G, B] rounded to integers
    """
    masked = cv2.bitwise_and(image, image, mask=mask)
    avg_color = cv2.mean(masked, mask=mask)[:3]
    return np.round(avg_color).astype(int)


def clip_point_to_hexagon(point: tuple, hex_coords: np.ndarray) -> tuple:
    """
    Clip a point to lie on or inside a hexagon boundary.

    If the point is inside the hexagon, returns it unchanged.
    If outside, returns the nearest point on the hexagon edge.

    Args:
        point: (x, y) coordinates of the point
        hex_coords: Array of hexagon vertex coordinates

    Returns:
        Clipped point coordinates as tuple
    """
    path = Path(hex_coords)
    if path.contains_point(point):
        return point

    # Find the nearest point on any edge
    min_dist = float('inf')
    nearest_point = point

    for i in range(len(hex_coords)):
        p1 = hex_coords[i]
        p2 = hex_coords[(i + 1) % len(hex_coords)]
        nearest = nearest_point_on_segment(point, p1, p2)
        dist = np.linalg.norm(np.array(nearest) - np.array(point))
        if dist < min_dist:
            min_dist = dist
            nearest_point = nearest

    return nearest_point


def nearest_point_on_segment(point: tuple, p1: tuple, p2: tuple) -> tuple:
    """
    Find the nearest point on a line segment to a given point.

    Uses vector projection to find the closest point on segment p1-p2.

    Args:
        point: The query point (x, y)
        p1: First endpoint of segment
        p2: Second endpoint of segment

    Returns:
        Nearest point on segment as tuple
    """
    px, py = point
    x1, y1 = p1
    x2, y2 = p2

    dx, dy = x2 - x1, y2 - y1

    # Handle degenerate case where segment is a point
    if dx == dy == 0:
        return p1

    # Project point onto line, then clamp to segment
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp to [0, 1]

    return (x1 + t * dx, y1 + t * dy)
