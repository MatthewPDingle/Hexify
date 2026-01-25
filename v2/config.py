"""
Configuration constants for Hexify.

This module centralizes all magic numbers and configuration values used throughout
the hexagon pattern generation process. By extracting these values, we make the
algorithm easier to understand and tune.

The hexagon pattern algorithm works by:
1. Creating a grid of hexagons over the output image (16x scale of input)
2. For each hexagon, sampling the input image to get average color
3. Rendering 7 concentric layers alternating between solid color and patterned zones
4. Even layers use color mixing with 12 angular zones to approximate the target color

This module supports both:
1. Direct constant access (backward compatible): `from .config import HEX_WIDTH`
2. Settings-based configuration via HexifySettings class

When using settings-based configuration, the ConfigResolver class provides
computed properties that can be overridden by HexifySettings instances.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .settings import HexifySettings

# =============================================================================
# Image Scaling
# =============================================================================

# The output image is scaled up by this factor from the input.
# This allows for detailed hexagon patterns at high resolution.
HEX_SCALE_FACTOR = 16

# =============================================================================
# Hexagon Dimensions
# =============================================================================

# Width of each hexagon in output pixels.
# Height is calculated as width * sqrt(3)/2 to maintain regular hexagon proportions.
HEX_WIDTH = 256

# Calculated hexagon height maintaining regular hexagon proportions.
# For a regular flat-topped hexagon, height = width * sqrt(3)/2
HEX_HEIGHT = round(HEX_WIDTH * (math.sqrt(3) / 2))

# Radius is half the width, used for centering calculations
HEX_RADIUS = HEX_WIDTH // 2

# Horizontal spacing between hexagon centers.
# Hexagons overlap by 25% horizontally (3/4 of width between centers).
HEX_HORIZONTAL_SPACING = HEX_WIDTH * 0.75

# Vertical spacing between hexagon centers (full height)
HEX_VERTICAL_SPACING = HEX_HEIGHT

# =============================================================================
# Layer Configuration
# =============================================================================

# Total number of concentric layers in each hexagon.
# Layers alternate between odd (solid color) and even (patterned).
# Layer 7 is outermost, layer 1 is innermost.
NUM_LAYERS = 7

# Number of angular zones in even layers.
# 12 zones = 30 degrees each, alternating between two colors for color mixing.
NUM_ZONES = 12

# Base angle per zone before adjustment (360 / 12 = 30 degrees)
BASE_ZONE_ANGLE = 360 / NUM_ZONES  # 30 degrees

# =============================================================================
# Palette Generation
# =============================================================================

# Maximum pixels to sample for palette generation.
# Larger images are downscaled to this limit for faster K-means clustering.
MAX_PALETTE_SAMPLE_PIXELS = 1_000_000

# Random seed for K-means clustering to ensure reproducible results.
KMEANS_RANDOM_STATE = 42

# Number of K-means initializations for stable clustering.
KMEANS_N_INIT = 10

# =============================================================================
# Processing Defaults
# =============================================================================

# Default number of colors in the generated palette
DEFAULT_NUM_PALETTE_COLORS = 16

# Minimum required palette size (algorithm needs enough colors to work with)
MIN_PALETTE_COLORS = 5

# Default chunk size for parallel processing
DEFAULT_CHUNK_SIZE = 32

# =============================================================================
# Brightness Thresholds
# =============================================================================

# Threshold for determining black vs white background.
# Colors with average RGB below this use black background, above use white.
BRIGHTNESS_THRESHOLD = 128

# =============================================================================
# Layer Diameter Calculations
# =============================================================================

# These constants control how layer diameter varies with brightness.
# The goal is to make the pattern more compact for extreme brightness values
# (very dark or very light) to allow more of the background to show through.

# Maximum diameter for layer 6 (when brightness is at 128, mid-gray)
LAYER_6_MAX_DIAMETER = 256

# Minimum diameter for layer 6 (at extreme brightness 0 or 255)
LAYER_6_MIN_DIAMETER = 192

# Diameter reduction range for inner even layers based on brightness
INNER_LAYER_DIAMETER_RANGE = 64

# =============================================================================
# Geometric Constants
# =============================================================================

# Hexagon orientation in radians (pi/2 = flat top orientation)
HEX_ORIENTATION = math.pi / 2

# Number of vertices in a hexagon
HEX_NUM_VERTICES = 6

# Small threshold for floating point comparisons
FLOAT_EPSILON = 1e-6


# =============================================================================
# Configuration Resolver
# =============================================================================

class ConfigResolver:
    """
    Resolves configuration values with optional settings override.

    This class provides a unified interface for accessing configuration values.
    When initialized with settings, it uses those values; otherwise, it falls
    back to the module-level constants for backward compatibility.

    Usage:
        # Without settings (uses defaults)
        config = ConfigResolver()
        width = config.hex_width  # Returns HEX_WIDTH constant

        # With settings
        settings = HexifySettings(hex_width=128)
        config = ConfigResolver(settings)
        width = config.hex_width  # Returns 128
    """

    def __init__(self, settings: Optional[HexifySettings] = None):
        """
        Initialize the configuration resolver.

        Args:
            settings: Optional HexifySettings to use. If None, uses defaults.
        """
        self._settings = settings

    @property
    def settings(self) -> Optional[HexifySettings]:
        """Get the settings object, if any."""
        return self._settings

    # Image Scaling
    @property
    def hex_scale_factor(self) -> int:
        """Output image scale factor from input."""
        if self._settings is not None:
            return self._settings.hex_scale_factor
        return HEX_SCALE_FACTOR

    # Hexagon Dimensions
    @property
    def hex_width(self) -> int:
        """Width of each hexagon in output pixels."""
        if self._settings is not None:
            return self._settings.hex_width
        return HEX_WIDTH

    @property
    def hex_height(self) -> int:
        """Height of each hexagon maintaining regular proportions."""
        if self._settings is not None:
            return self._settings.hex_height
        return HEX_HEIGHT

    @property
    def hex_radius(self) -> int:
        """Hexagon radius (half of width)."""
        if self._settings is not None:
            return self._settings.hex_radius
        return HEX_RADIUS

    @property
    def hex_horizontal_spacing(self) -> float:
        """Horizontal spacing between hexagon centers."""
        if self._settings is not None:
            return self._settings.hex_horizontal_spacing
        return HEX_HORIZONTAL_SPACING

    @property
    def hex_vertical_spacing(self) -> int:
        """Vertical spacing between hexagon centers."""
        if self._settings is not None:
            return self._settings.hex_vertical_spacing
        return HEX_VERTICAL_SPACING

    # Layer Configuration
    @property
    def num_layers(self) -> int:
        """Total number of concentric layers in each hexagon."""
        if self._settings is not None:
            return self._settings.num_layers
        return NUM_LAYERS

    @property
    def num_zones(self) -> int:
        """Number of angular zones in even layers."""
        if self._settings is not None:
            return self._settings.num_zones
        return NUM_ZONES

    @property
    def base_zone_angle(self) -> float:
        """Base angle per zone in degrees."""
        if self._settings is not None:
            return self._settings.base_zone_angle
        return BASE_ZONE_ANGLE

    # Palette Generation
    @property
    def max_palette_sample_pixels(self) -> int:
        """Maximum pixels to sample for palette generation."""
        if self._settings is not None:
            return self._settings.max_palette_sample_pixels
        return MAX_PALETTE_SAMPLE_PIXELS

    @property
    def kmeans_random_state(self) -> int:
        """Random seed for K-means clustering."""
        if self._settings is not None:
            return self._settings.kmeans_random_state
        return KMEANS_RANDOM_STATE

    @property
    def kmeans_n_init(self) -> int:
        """Number of K-means initializations."""
        if self._settings is not None:
            return self._settings.kmeans_n_init
        return KMEANS_N_INIT

    # Processing Defaults
    @property
    def default_num_palette_colors(self) -> int:
        """Default number of colors in the palette."""
        if self._settings is not None:
            return self._settings.num_palette_colors
        return DEFAULT_NUM_PALETTE_COLORS

    @property
    def default_chunk_size(self) -> int:
        """Default chunk size for processing."""
        if self._settings is not None:
            return self._settings.chunk_size
        return DEFAULT_CHUNK_SIZE

    # Brightness Thresholds
    @property
    def brightness_threshold(self) -> int:
        """Threshold for black/white background selection."""
        if self._settings is not None:
            return self._settings.brightness_threshold
        return BRIGHTNESS_THRESHOLD

    # Layer Diameter Calculations
    @property
    def layer_6_max_diameter(self) -> int:
        """Maximum diameter for layer 6."""
        if self._settings is not None:
            return self._settings.layer_6_max_diameter
        return LAYER_6_MAX_DIAMETER

    @property
    def layer_6_min_diameter(self) -> int:
        """Minimum diameter for layer 6."""
        if self._settings is not None:
            return self._settings.layer_6_min_diameter
        return LAYER_6_MIN_DIAMETER

    @property
    def inner_layer_diameter_range(self) -> int:
        """Diameter reduction range for inner even layers."""
        if self._settings is not None:
            return self._settings.inner_layer_diameter_range
        return INNER_LAYER_DIAMETER_RANGE

    # Geometric Constants
    @property
    def hex_orientation(self) -> float:
        """Hexagon orientation in radians."""
        if self._settings is not None:
            return self._settings.hex_orientation_radians
        return HEX_ORIENTATION

    @property
    def hex_num_vertices(self) -> int:
        """Number of vertices in a hexagon."""
        return HEX_NUM_VERTICES

    @property
    def float_epsilon(self) -> float:
        """Small threshold for floating point comparisons."""
        return FLOAT_EPSILON
