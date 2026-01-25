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
"""

import math

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
