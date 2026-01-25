"""
Layer rendering for hexagon patterns.

This module handles the rendering of individual hexagon patterns with their
concentric layers. Each hexagon has 7 layers alternating between:

- Odd layers (7, 5, 3, 1): Solid background color (black or white based on brightness)
- Even layers (6, 4, 2): Patterned zones with two colors for color mixing

The layer system creates a distinctive visual style where:
- The outermost layer (7) provides contrast with the background
- Even layers use angular zones to mix two palette colors
- Inner odd layers provide separation between even layers
- Layer 1 (center) is the innermost solid region

The color mixing in even layers works by dividing the layer into 12 angular zones,
alternating between two colors. The ratio of zone sizes determines the visual
color blend, approximating any target color using only palette colors.
"""

import numpy as np
import cv2
import math
from matplotlib.patches import RegularPolygon

from .config import (
    HEX_ORIENTATION,
    HEX_NUM_VERTICES,
    NUM_LAYERS,
    NUM_ZONES,
    HEX_SCALE_FACTOR,
    LAYER_6_MAX_DIAMETER,
    LAYER_6_MIN_DIAMETER,
    INNER_LAYER_DIAMETER_RANGE,
    BRIGHTNESS_THRESHOLD,
    FLOAT_EPSILON,
)
from .geometry import HexagonMask, average_color, clip_point_to_hexagon
from .color import ColorPalette, get_background_color


class LayerRenderer:
    """
    Renders the multi-layer hexagon pattern.

    Each hexagon is rendered as a series of concentric layers from outside in.
    This class handles the complex geometry of the angular zones in even layers
    and coordinates color selection for visual color mixing.
    """

    def __init__(self, palette: ColorPalette, input_image: np.ndarray):
        """
        Initialize the layer renderer.

        Args:
            palette: The color palette to use for rendering
            input_image: The input image for sampling colors
        """
        self.palette = palette
        self.input_image = input_image

    def create_hex_pattern(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        avg_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Create the full multi-layer hexagon pattern.

        Renders all 7 layers from outside (layer 7) to center (layer 1).
        Odd layers are solid background, even layers have color-mixing zones.

        Args:
            center_x: X coordinate of hexagon center in output space
            center_y: Y coordinate of hexagon center in output space
            radius: Radius of the hexagon
            avg_rgb: Average color sampled from the input image

        Returns:
            Pattern as numpy array (2*radius, 2*radius, 3)
        """
        # Determine background color based on brightness
        # Dark areas get white background, light areas get black
        bw_value = 0 if np.mean(avg_rgb) >= BRIGHTNESS_THRESHOLD else 255
        bw_color = (bw_value, bw_value, bw_value)

        # Initialize pattern with background color
        pattern = np.full((2 * radius, 2 * radius, 3), bw_value, dtype=np.uint8)

        # Track layer areas and radii for color mixing calculations
        # Even layers need to know the area of the odd layer inside them
        layer_areas = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        layer_radii = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}

        # Track colors to avoid reusing in adjacent layers
        avoid_rgb = []

        # Render layers from outside in (7 down to 1)
        for i in range(NUM_LAYERS, 0, -1):
            if i % 2 == 1:
                # Odd layers: solid background color
                hex_radius = int(radius * (i / NUM_LAYERS))
                layer_radii[i] = hex_radius
                pattern, layer_area = self._fill_odd_layer(pattern, radius, hex_radius, bw_color)
                layer_areas[i] = layer_area
            else:
                # Even layers: patterned color mixing zones
                pattern, layer_area = self._fill_even_layer(
                    layer_index=i,
                    avg_rgb=avg_rgb,
                    radius=radius,
                    layer_radii=layer_radii,
                    avoid_rgb=avoid_rgb,
                    layer_areas=layer_areas,
                    pattern=pattern,
                    center_x=center_x,
                    center_y=center_y,
                    bw_color=bw_color
                )
                layer_areas[i] = layer_area

        return pattern

    def _fill_odd_layer(
        self,
        pattern: np.ndarray,
        radius: int,
        hex_radius: int,
        bw_color: tuple
    ) -> tuple:
        """
        Fill an odd layer with solid background color.

        Odd layers provide visual separation between the color-mixing even layers.
        They use the background color (black or white based on image brightness).

        Args:
            pattern: The pattern array to draw on
            radius: The full hexagon radius (for centering)
            hex_radius: The radius of this layer's hexagon
            bw_color: Background color tuple (R, G, B)

        Returns:
            Tuple of (updated pattern, layer area in pixels)
        """
        # Create hexagon centered in the pattern
        inner_hexagon = RegularPolygon(
            (radius, radius),
            numVertices=HEX_NUM_VERTICES,
            radius=hex_radius,
            orientation=HEX_ORIENTATION
        )
        inner_coords = inner_hexagon.get_verts().astype(int)

        # Fill with background color
        pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))

        # Calculate area for color mixing: A = (3 * sqrt(3) / 2) * r^2
        hex_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2

        return pattern, hex_area

    def _fill_even_layer(
        self,
        layer_index: int,
        avg_rgb: np.ndarray,
        radius: int,
        layer_radii: dict,
        avoid_rgb: list,
        layer_areas: dict,
        pattern: np.ndarray,
        center_x: int,
        center_y: int,
        bw_color: tuple
    ) -> tuple:
        """
        Fill an even layer with color-mixing angular zones.

        Even layers are the core of the color approximation algorithm. They divide
        the layer into 12 angular zones alternating between two palette colors.
        The relative sizes of the zones determine the visual color blend.

        The zone sizing is based on how close the primary color is to the target:
        - Perfect match: primary color gets 30-degree zones, secondary gets minimal
        - Poor match: zones sizes approach equality (30 degrees each)

        Args:
            layer_index: Which layer (6, 4, or 2)
            avg_rgb: Target color to approximate
            radius: Full hexagon radius
            layer_radii: Dict of radii for each layer
            avoid_rgb: Colors already used in outer layers
            layer_areas: Dict of areas for each layer
            pattern: Pattern array to draw on
            center_x: X coordinate in output space
            center_y: Y coordinate in output space
            bw_color: Background color

        Returns:
            Tuple of (updated pattern, layer area in pixels)
        """
        # Calculate layer diameter based on brightness.
        # Colors closer to mid-gray (128) get larger zones for more accurate mixing.
        # Extreme brightness values (0 or 255) get smaller zones.
        brightness = np.mean(avg_rgb)

        if layer_index == 6:
            # Outer even layer: varies from 192 to 256 based on brightness
            diameter = LAYER_6_MAX_DIAMETER - abs(brightness - 128) * (LAYER_6_MAX_DIAMETER - LAYER_6_MIN_DIAMETER) / 128
        else:
            # Inner even layers: based on outer odd layer radius with brightness adjustment
            diameter = (layer_radii[layer_index + 1] * 2) - abs(brightness - 128) * INNER_LAYER_DIAMETER_RANGE / 128

        hex_radius = int(diameter // 2)

        # Create the hexagon for this layer
        inner_hexagon = RegularPolygon(
            (radius, radius),
            numVertices=HEX_NUM_VERTICES,
            radius=hex_radius,
            orientation=HEX_ORIENTATION
        )
        inner_coords = inner_hexagon.get_verts().astype(int)

        # Sample average color at this layer's scale from input image
        scaled_center_x = center_x // HEX_SCALE_FACTOR
        scaled_center_y = center_y // HEX_SCALE_FACTOR
        scaled_radius = hex_radius // HEX_SCALE_FACTOR

        input_mask = HexagonMask.create(
            scaled_center_x, scaled_center_y, scaled_radius, self.input_image.shape[:2]
        )
        layer_avg_rgb = average_color(self.input_image, input_mask)

        # Select primary color (closest to target, avoiding already-used colors)
        color_1 = self.palette.closest_color(layer_avg_rgb, avoid_rgb)
        avoid_rgb.append(color_1)

        # Calculate distance from primary color to target
        dist_1 = np.linalg.norm(color_1 - layer_avg_rgb)

        # Find the palette color most similar to primary (for zone angle calculation)
        color_adj = min(
            self.palette.colors,
            key=lambda c: np.linalg.norm(c - color_1) if not np.array_equal(c, color_1) else float('inf')
        )
        dist_adj = np.linalg.norm(color_1 - color_adj)

        # Calculate zone angles based on color distance ratios.
        # percentage_off measures how "off" the primary color is from the target.
        # Higher values mean more equal zone sizes for better blending.
        percentage_off = min(dist_1, dist_adj) / max(dist_1, dist_adj) if dist_adj != 0 else 0
        even_angle = percentage_off * 60  # Primary color zone angle (0-60 degrees)
        odd_angle = 60 - even_angle  # Secondary color zone angle

        # Calculate areas for each zone type (used for secondary color selection)
        # Area of a circular sector: A = 0.5 * r^2 * theta
        even_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(even_angle))
        odd_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(odd_angle))
        layer_area = even_area + odd_area

        # Select secondary color that best approximates target when mixed with primary
        # Takes into account the background color area from the inner odd layer
        inner_layer_area = layer_areas[layer_index + 1] - layer_area
        color_2 = self.palette.select_mixing_color(
            layer_avg_rgb,
            color_1,
            even_area,
            odd_area,
            bw_color,
            inner_layer_area
        )
        avoid_rgb.append(color_2)

        # Convert to tuples for OpenCV
        color_1_rgb = tuple(map(int, color_1))
        color_2_rgb = tuple(map(int, color_2))

        # Draw the 12 angular zones
        # Start angle offset centers the pattern (30 degrees base - half of even_angle)
        angle_offset = 30 - (even_angle / 2)

        for zone in range(NUM_ZONES):
            # Alternate between even (primary) and odd (secondary) zones
            angle = even_angle if zone % 2 == 0 else odd_angle
            angle_color = color_1_rgb if zone % 2 == 0 else color_2_rgb

            start_angle = angle_offset
            end_angle = start_angle + angle
            angle_offset += angle

            # Calculate zone boundary points on the hexagon edge
            p1 = (
                radius + hex_radius * math.cos(math.radians(start_angle)),
                radius + hex_radius * math.sin(math.radians(start_angle))
            )
            p2 = (
                radius + hex_radius * math.cos(math.radians(end_angle)),
                radius + hex_radius * math.sin(math.radians(end_angle))
            )

            # Calculate midpoint and triangular apex for odd zones
            p12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # Distance from p1 to midpoint
            d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5 / 2

            # Height of equilateral triangle with base 2*d
            length_of_shorter_leg = d / math.sqrt(3)

            # Direction perpendicular to p1-p2
            dx = p12[0] - p1[0]
            dy = p12[1] - p1[1]

            length = (dx ** 2 + dy ** 2) ** 0.5
            if length > FLOAT_EPSILON:
                dx /= length
                dy /= length
                # p3 is the apex of the triangle, perpendicular to the edge
                p3 = (p12[0] + length_of_shorter_leg * dy, p12[1] - length_of_shorter_leg * dx)
            else:
                # Degenerate case: p1 and p2 are the same point
                p3 = p1

            # Clip all points to stay within the hexagon
            p1 = clip_point_to_hexagon(p1, inner_coords)
            p2 = clip_point_to_hexagon(p2, inner_coords)
            p3 = clip_point_to_hexagon(p3, inner_coords)

            # Draw the main triangular zone (center to edge)
            vertices1 = np.array([(radius, radius), p1, p2], dtype=np.int32)
            cv2.fillPoly(pattern, [vertices1], angle_color)

            # Odd zones (secondary color) get an additional triangular cap
            # This creates the characteristic "pinwheel" appearance
            if zone % 2 == 1:
                vertices2 = np.array([p1, p2, p3], dtype=np.int32)
                cv2.fillPoly(pattern, [vertices2], angle_color)

        return pattern, layer_area
