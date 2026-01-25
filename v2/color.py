"""
Color palette generation and selection for Hexify.

This module handles all color-related operations:
- Generating a color palette from an input image using K-means clustering
- Finding the closest palette color to a target color
- Selecting optimal color pairs for area-weighted color mixing

The color selection algorithm aims to approximate any target color using only
colors from a limited palette by strategically mixing two palette colors in
varying proportions across angular zones.

The module supports settings-based configuration through HexifySettings.
For backward compatibility, module-level constants are used when no
settings are provided.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .config import (
    BRIGHTNESS_THRESHOLD,
    ConfigResolver,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    MAX_PALETTE_SAMPLE_PIXELS,
)

if TYPE_CHECKING:
    from .settings import HexifySettings

# Lazy imports for heavy sklearn modules
_KMeans = None
_MiniBatchKMeans = None


def _get_kmeans():
    """Lazy load KMeans from sklearn."""
    global _KMeans
    if _KMeans is None:
        from sklearn.cluster import KMeans
        _KMeans = KMeans
    return _KMeans


def _get_minibatch_kmeans():
    """Lazy load MiniBatchKMeans from sklearn."""
    global _MiniBatchKMeans
    if _MiniBatchKMeans is None:
        from sklearn.cluster import MiniBatchKMeans
        _MiniBatchKMeans = MiniBatchKMeans
    return _MiniBatchKMeans

# Module-level logger
logger = logging.getLogger(__name__)


class ColorPalette:
    """
    Manages color palette generation and color selection from the palette.

    The palette is generated using K-means clustering on the input image pixels,
    which finds the most representative colors. The palette is then sorted by
    brightness for consistency across runs.

    Supports settings-based configuration through HexifySettings.
    For backward compatibility, parameters take precedence over settings.

    Attributes:
        num_colors: Number of colors to extract from the image
        fast_mode: If True, uses MiniBatchKMeans for faster (but potentially
                   less accurate) palette generation. Default is False.
    """

    def __init__(
        self,
        num_colors: int,
        fast_mode: bool = False,
        settings: Optional[HexifySettings] = None
    ):
        """
        Initialize the color palette.

        Args:
            num_colors: Number of colors to extract from the image
            fast_mode: If True, use MiniBatchKMeans for faster palette generation.
                       Default is False (use standard KMeans for better quality).
            settings: Optional HexifySettings for configuration.
                     When provided, uses settings for kmeans parameters.
        """
        self._config = ConfigResolver(settings)
        self.num_colors = num_colors
        self.fast_mode = fast_mode
        self.colors = None
        self.palette_hash = None

    def generate_from_image(self, image: np.ndarray) -> None:
        """
        Generate a color palette from an input image using K-means clustering.

        The algorithm:
        1. Downscales large images to speed up clustering
        2. Runs K-means to find cluster centers (representative colors)
        3. Sorts colors by brightness for reproducible ordering
        4. Rounds to integers and generates a hash for caching

        When fast_mode is enabled, uses MiniBatchKMeans which processes data
        in mini-batches for significantly faster training at the cost of
        slightly reduced clustering quality.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        # Get configuration values
        max_sample_pixels = self._config.max_palette_sample_pixels
        random_state = self._config.kmeans_random_state
        n_init = self._config.kmeans_n_init

        # Downscale large images to speed up K-means clustering.
        # Using INTER_AREA for high quality downsampling that preserves color accuracy.
        height, width = image.shape[:2]
        if height * width > max_sample_pixels:
            scale = np.sqrt(max_sample_pixels / (height * width))
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Reshape to list of pixels for clustering
        pixels = image.reshape(-1, 3)

        # Use fixed random state for reproducible palette generation
        if self.fast_mode:
            # MiniBatchKMeans is faster but may produce slightly different results
            MiniBatchKMeans = _get_minibatch_kmeans()
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_colors,
                random_state=random_state,
                n_init=n_init,
                batch_size=1024,  # Process 1024 samples at a time
                max_iter=100,
            )
        else:
            # Standard KMeans for best quality
            KMeans = _get_kmeans()
            kmeans = KMeans(
                n_clusters=self.num_colors,
                random_state=random_state,
                n_init=n_init
            )
        kmeans.fit(pixels)

        palette = kmeans.cluster_centers_

        # Sort by brightness (mean of RGB) for consistent ordering.
        # This ensures the same palette regardless of K-means initialization order.
        sorted_indices = np.argsort(np.mean(palette, axis=1))
        self.colors = palette[sorted_indices]

        # Round to integers for consistent hashing and to match typical color values
        self.colors = np.round(self.colors).astype(int)

        # Generate hash for cache key generation (identifies this specific palette)
        self.palette_hash = hashlib.sha256(self.colors.tobytes()).hexdigest()

    def closest_color(
        self, target_rgb: np.ndarray, avoid_colors: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Find the closest palette color to a target color.

        Uses Euclidean distance in RGB space to find the nearest match.
        Optionally avoids certain colors (useful when selecting multiple distinct colors).

        Args:
            target_rgb: Target color as numpy array [R, G, B]
            avoid_colors: List of colors to exclude from consideration

        Returns:
            The closest palette color as numpy array [R, G, B]

        Note:
            If all colors are in avoid_colors, returns the closest color anyway
            as a fallback to prevent None return values that would crash callers.
        """
        if avoid_colors is None:
            avoid_colors = []

        min_distance = float('inf')
        closest_color = None

        # Track fallback (closest ignoring avoid list) in case all colors are avoided.
        # BUG FIX: This prevents returning None when avoid_colors contains all palette colors,
        # which would cause crashes in color selection logic.
        fallback_color = None
        fallback_distance = float('inf')

        for color in self.colors:
            distance = np.linalg.norm(target_rgb - color)

            # Always track the absolute closest as fallback
            if distance < fallback_distance:
                fallback_distance = distance
                fallback_color = color

            # Skip avoided colors for primary selection
            if any(np.array_equal(color, avoided) for avoided in avoid_colors):
                continue

            if distance < min_distance:
                min_distance = distance
                closest_color = color

        # Return closest non-avoided color, or fallback if all were avoided
        return closest_color if closest_color is not None else fallback_color

    def select_mixing_color(
        self,
        target_rgb: np.ndarray,
        primary_color: np.ndarray,
        primary_area: float,
        secondary_area: float,
        background_color: tuple,
        background_area: float
    ) -> np.ndarray:
        """
        Select the best secondary color for area-weighted color mixing.

        This is the key color matching algorithm. Given a target color and constraints:
        - A primary color already chosen
        - Areas for primary, secondary, and background colors
        - The background color (black or white)

        Find the secondary palette color that, when mixed with the primary and background
        in their respective area proportions, best approximates the target color.

        The mixed color formula:
            mixed = (primary_area * primary + secondary_area * secondary + bg_area * bg) / total_area

        Args:
            target_rgb: The color we're trying to approximate
            primary_color: The first color already selected
            primary_area: Pixel area covered by primary color
            secondary_area: Pixel area covered by secondary color
            background_color: Background color tuple (R, G, B)
            background_area: Pixel area covered by background

        Returns:
            The optimal secondary color from the palette
        """
        min_distance = float('inf')
        best_secondary = None

        primary_color = np.array(primary_color)
        background_color = np.array(background_color)
        total_area = primary_area + secondary_area + background_area

        for color in self.colors:
            # Skip the primary color - we need a different secondary
            if np.array_equal(color, primary_color):
                continue

            color = np.array(color)

            # Calculate what the area-weighted mixed color would be
            mixed_rgb = (
                primary_area * primary_color +
                secondary_area * color +
                background_area * background_color
            ) / total_area

            # Find the color that produces the closest mix to target
            distance = np.linalg.norm(target_rgb - mixed_rgb)

            if distance < min_distance:
                min_distance = distance
                best_secondary = color

        return best_secondary


def get_background_color(
    avg_rgb: np.ndarray,
    threshold: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    Determine background color (black or white) based on average brightness.

    Lighter colors get black background for contrast.
    Darker colors get white background for contrast.

    Args:
        avg_rgb: Average color of the region
        threshold: Brightness threshold (default: BRIGHTNESS_THRESHOLD)

    Returns:
        Background color as tuple (R, G, B) - either (0, 0, 0) or (255, 255, 255)
    """
    if threshold is None:
        threshold = BRIGHTNESS_THRESHOLD

    brightness = np.mean(avg_rgb)
    if brightness < threshold:
        return (255, 255, 255)  # Dark colors get white background
    else:
        return (0, 0, 0)  # Light colors get black background


def get_background_value(
    avg_rgb: np.ndarray,
    threshold: Optional[int] = None
) -> int:
    """
    Get background value as single integer (0 or 255).

    Args:
        avg_rgb: Average color of the region
        threshold: Brightness threshold (default: BRIGHTNESS_THRESHOLD)

    Returns:
        0 for black background, 255 for white background
    """
    if threshold is None:
        threshold = BRIGHTNESS_THRESHOLD

    return 0 if np.mean(avg_rgb) >= threshold else 255
