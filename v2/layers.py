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

Style augmentation features (all optional, backward compatible):
- Border styles: solid, double, glow effects around hexagon edges
- Fill styles: solid, radial gradient, linear gradient
- Noise texture: subtle grain for artistic effects

The module supports settings-based configuration through HexifySettings.
For backward compatibility, module-level constants are used when no
settings are provided.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from matplotlib.patches import RegularPolygon

from .color import ColorPalette, get_background_color
from .config import (
    BRIGHTNESS_THRESHOLD,
    ConfigResolver,
    FLOAT_EPSILON,
    HEX_NUM_VERTICES,
    HEX_ORIENTATION,
    HEX_SCALE_FACTOR,
    INNER_LAYER_DIAMETER_RANGE,
    LAYER_6_MAX_DIAMETER,
    LAYER_6_MIN_DIAMETER,
    NUM_LAYERS,
    NUM_ZONES,
)
from .geometry import HexagonMask, average_color, clip_point_to_hexagon
from .styles import (
    BorderStyle,
    FillStyle,
    add_hexagon_border,
    add_noise_texture,
    apply_radial_gradient,
    apply_linear_gradient,
)

if TYPE_CHECKING:
    from .settings import HexifySettings

# Module-level logger
logger = logging.getLogger(__name__)


class LayerRenderer:
    """
    Renders the multi-layer hexagon pattern.

    Each hexagon is rendered as a series of concentric layers from outside in.
    This class handles the complex geometry of the angular zones in even layers
    and coordinates color selection for visual color mixing.

    Style augmentation features (all optional, backward compatible):
    - border_width, border_color, border_style: Add borders around hexagons
    - fill_style: Control how hexagons are filled (solid, gradient)
    - noise_intensity: Add subtle noise texture for artistic effect
    """

    def __init__(
        self,
        palette: ColorPalette,
        input_image: np.ndarray,
        settings: Optional[HexifySettings] = None,
        border_width: int = 0,
        border_color: Tuple[int, int, int] = (0, 0, 0),
        border_style: BorderStyle = BorderStyle.NONE,
        fill_style: FillStyle = FillStyle.SOLID,
        noise_intensity: float = 0.0,
        gradient_inner_color: Optional[Tuple[int, int, int]] = None,
        gradient_outer_color: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Initialize the layer renderer.

        Args:
            palette: The color palette to use for rendering
            input_image: The input image for sampling colors
            settings: Optional HexifySettings for full configuration control
            border_width: Width of hexagon border in pixels (0 = no border)
            border_color: Border color as (R, G, B) tuple
            border_style: Style of border (NONE, SOLID, DOUBLE, GLOW)
            fill_style: Fill style (SOLID, RADIAL_GRADIENT, LINEAR_GRADIENT)
            noise_intensity: Noise texture intensity (0.0 = none, 0.05-0.15 typical)
            gradient_inner_color: Inner color for gradient fills (optional)
            gradient_outer_color: Outer color for gradient fills (optional)
        """
        self.palette = palette
        self.input_image = input_image
        self.settings = settings
        self._config = ConfigResolver(settings)

        # Style augmentation parameters (all optional, defaults preserve existing behavior)
        # Settings can override border settings if provided
        if settings is not None:
            self.border_width = settings.border_width
            self.border_color = settings.border_color
        else:
            self.border_width = border_width
            self.border_color = border_color

        self.border_style = border_style if self.border_width > 0 else BorderStyle.NONE
        self.fill_style = fill_style
        self.noise_intensity = noise_intensity
        self.gradient_inner_color = gradient_inner_color
        self.gradient_outer_color = gradient_outer_color

    def create_hex_pattern(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        avg_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Create the full multi-layer hexagon pattern.

        Renders all layers from outside to center.
        Odd layers are solid background, even layers have color-mixing zones.

        Args:
            center_x: X coordinate of hexagon center in output space
            center_y: Y coordinate of hexagon center in output space
            radius: Radius of the hexagon
            avg_rgb: Average color sampled from the input image

        Returns:
            Pattern as numpy array (2*radius, 2*radius, 3)
        """
        # Get configuration values
        num_layers = self._config.num_layers
        brightness_threshold = self._config.brightness_threshold

        # Determine background color based on brightness
        # Dark areas get white background, light areas get black
        bw_value = 0 if np.mean(avg_rgb) >= brightness_threshold else 255
        bw_color = (bw_value, bw_value, bw_value)

        # Initialize pattern with background color
        pattern = np.full((2 * radius, 2 * radius, 3), bw_value, dtype=np.uint8)

        # Track layer areas and radii for color mixing calculations
        # Even layers need to know the area of the odd layer inside them
        layer_areas = {i: 0 for i in range(num_layers, 0, -1)}
        layer_radii = {i: 0 for i in range(num_layers, 0, -1)}

        # Track colors to avoid reusing in adjacent layers
        avoid_rgb = []

        # Render layers from outside in
        for i in range(num_layers, 0, -1):
            if i % 2 == 1:
                # Odd layers: solid background color
                hex_radius = int(radius * (i / num_layers))
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

        # Apply style augmentations (only if enabled)
        pattern = self._apply_style_augmentations(pattern, radius)

        return pattern

    def _apply_style_augmentations(
        self,
        pattern: np.ndarray,
        radius: int
    ) -> np.ndarray:
        """
        Apply optional style augmentations to the pattern.

        This method applies any enabled style effects in the correct order:
        1. Gradient fill (if enabled)
        2. Noise texture (if enabled)
        3. Border (if enabled)

        Args:
            pattern: The base pattern to augment
            radius: Hexagon radius

        Returns:
            Pattern with style augmentations applied
        """
        # Get configuration values
        hex_orientation = self._config.hex_orientation
        hex_num_vertices = self._config.hex_num_vertices

        # Get hexagon coordinates for border and mask operations
        hexagon = RegularPolygon(
            (radius, radius),
            numVertices=hex_num_vertices,
            radius=radius,
            orientation=hex_orientation
        )
        hex_coords = hexagon.get_verts().astype(np.int32)

        # Create mask for selective effects
        mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
        cv2.fillPoly(mask, [hex_coords], 255)

        # Apply gradient fill if enabled
        if self.fill_style != FillStyle.SOLID:
            # Determine gradient colors (use average pattern colors if not specified)
            inner_color = self.gradient_inner_color
            outer_color = self.gradient_outer_color

            if inner_color is None or outer_color is None:
                # Default to subtle gradient based on pattern colors
                center_color = pattern[radius, radius].tolist()
                edge_color = pattern[0, radius].tolist() if radius > 0 else center_color

                if inner_color is None:
                    inner_color = tuple(center_color)
                if outer_color is None:
                    outer_color = tuple(edge_color)

            if self.fill_style == FillStyle.RADIAL_GRADIENT:
                pattern = apply_radial_gradient(
                    pattern,
                    center=(radius, radius),
                    radius=radius,
                    inner_color=inner_color,
                    outer_color=outer_color,
                    mask=mask
                )
            elif self.fill_style == FillStyle.LINEAR_GRADIENT:
                pattern = apply_linear_gradient(
                    pattern,
                    start_point=(0, 0),
                    end_point=(2 * radius, 2 * radius),
                    start_color=inner_color,
                    end_color=outer_color,
                    mask=mask
                )

        # Apply noise texture if enabled
        if self.noise_intensity > 0:
            pattern = add_noise_texture(pattern, self.noise_intensity, mask)

        # Apply border if enabled
        if self.border_width > 0 and self.border_style != BorderStyle.NONE:
            pattern = add_hexagon_border(
                pattern,
                hex_coords,
                border_width=self.border_width,
                border_color=self.border_color,
                border_style=self.border_style
            )

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
        # Get configuration values
        hex_num_vertices = self._config.hex_num_vertices
        hex_orientation = self._config.hex_orientation

        # Create hexagon centered in the pattern
        inner_hexagon = RegularPolygon(
            (radius, radius),
            numVertices=hex_num_vertices,
            radius=hex_radius,
            orientation=hex_orientation
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
        the layer into angular zones alternating between two palette colors.
        The relative sizes of the zones determine the visual color blend.

        The zone sizing is based on how close the primary color is to the target:
        - Perfect match: primary color gets larger zones, secondary gets minimal
        - Poor match: zones sizes approach equality

        Args:
            layer_index: Which layer (e.g., 6, 4, or 2 with default settings)
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
        # Get configuration values
        num_layers = self._config.num_layers
        hex_num_vertices = self._config.hex_num_vertices
        hex_orientation = self._config.hex_orientation
        hex_scale_factor = self._config.hex_scale_factor
        layer_6_max_diameter = self._config.layer_6_max_diameter
        layer_6_min_diameter = self._config.layer_6_min_diameter
        inner_layer_diameter_range = self._config.inner_layer_diameter_range
        num_zones = self._config.num_zones
        float_epsilon = self._config.float_epsilon

        # Calculate layer diameter based on brightness.
        # Colors closer to mid-gray (128) get larger zones for more accurate mixing.
        # Extreme brightness values (0 or 255) get smaller zones.
        brightness = np.mean(avg_rgb)

        # Determine which layer is the "outer even layer" based on num_layers
        outer_even_layer = num_layers - 1 if num_layers % 2 == 0 else num_layers - 1

        if layer_index == outer_even_layer or (num_layers == 7 and layer_index == 6):
            # Outer even layer: varies based on brightness
            diameter = layer_6_max_diameter - abs(brightness - 128) * (layer_6_max_diameter - layer_6_min_diameter) / 128
        else:
            # Inner even layers: based on outer odd layer radius with brightness adjustment
            diameter = (layer_radii[layer_index + 1] * 2) - abs(brightness - 128) * inner_layer_diameter_range / 128

        hex_radius = int(diameter // 2)

        # Create the hexagon for this layer
        inner_hexagon = RegularPolygon(
            (radius, radius),
            numVertices=hex_num_vertices,
            radius=hex_radius,
            orientation=hex_orientation
        )
        inner_coords = inner_hexagon.get_verts().astype(int)

        # Sample average color at this layer's scale from input image
        scaled_center_x = center_x // hex_scale_factor
        scaled_center_y = center_y // hex_scale_factor
        scaled_radius = hex_radius // hex_scale_factor

        input_mask = HexagonMask.create(
            scaled_center_x, scaled_center_y, scaled_radius, self.input_image.shape[:2],
            orientation=hex_orientation
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
        # NOTE: The triangular cap geometry in this algorithm was designed specifically
        # for 12 zones. Other zone counts will produce incorrect results.
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

        # Draw angular zones with vectorized angle calculations
        # Start angle offset centers the pattern (base angle - half of even_angle)
        # Base zone angle is 360/num_zones (30 degrees for 12 zones)
        base_zone_angle = 360 / num_zones  # Full zone angle
        initial_offset = base_zone_angle - (even_angle / 2)

        # Pre-compute all angles vectorized (alternating even_angle and odd_angle)
        zone_indices = np.arange(num_zones)
        zone_angles = np.where(zone_indices % 2 == 0, even_angle, odd_angle)

        # Cumulative sum gives end angles, shifted gives start angles
        cumulative_angles = np.cumsum(zone_angles)
        end_angles = initial_offset + cumulative_angles
        start_angles = np.concatenate([[initial_offset], end_angles[:-1]])

        # Convert to radians for vectorized trigonometry
        start_rads = np.radians(start_angles)
        end_rads = np.radians(end_angles)

        # Pre-compute all boundary points vectorized
        p1_x = radius + hex_radius * np.cos(start_rads)
        p1_y = radius + hex_radius * np.sin(start_rads)
        p2_x = radius + hex_radius * np.cos(end_rads)
        p2_y = radius + hex_radius * np.sin(end_rads)

        # Pre-compute midpoints vectorized
        p12_x = (p1_x + p2_x) / 2
        p12_y = (p1_y + p2_y) / 2

        # Pre-compute distances and triangle apex points vectorized
        d = np.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2) / 2
        length_of_shorter_leg = d / np.sqrt(3)

        # Direction perpendicular to p1-p2
        dx = p12_x - p1_x
        dy = p12_y - p1_y
        lengths = np.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero
        safe_lengths = np.where(lengths > float_epsilon, lengths, 1.0)
        dx_norm = dx / safe_lengths
        dy_norm = dy / safe_lengths

        # Calculate p3 (apex points) - use p1 for degenerate cases
        p3_x = np.where(lengths > float_epsilon,
                        p12_x + length_of_shorter_leg * dy_norm,
                        p1_x)
        p3_y = np.where(lengths > float_epsilon,
                        p12_y - length_of_shorter_leg * dx_norm,
                        p1_y)

        # Now iterate through zones (still need loop for fillPoly calls)
        for zone in range(num_zones):
            angle_color = color_1_rgb if zone % 2 == 0 else color_2_rgb

            # Get pre-computed points
            p1 = (p1_x[zone], p1_y[zone])
            p2 = (p2_x[zone], p2_y[zone])
            p3 = (p3_x[zone], p3_y[zone])

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
