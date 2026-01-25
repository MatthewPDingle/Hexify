"""
Main hexagon processor orchestrating the pattern generation.

This module provides the high-level HexagonProcessor class that coordinates
all the components:
- ColorPalette: Generates and manages the color palette
- HexagonGrid: Manages the hexagon layout
- LayerRenderer: Renders individual hexagon patterns

The processor handles parallelization, caching, and image I/O, delegating
the actual rendering work to the specialized components.

The processor supports a settings-based configuration system through the
HexifySettings class. For backward compatibility, all existing parameters
continue to work unchanged - if no settings are provided, defaults are used.

RGBA support:
- Set output_format="RGBA" for transparent background outside hexagons
- Hexagon interiors remain opaque, background becomes transparent
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

from .color import ColorPalette
from .config import DEFAULT_CHUNK_SIZE, ConfigResolver
from .exceptions import InvalidImageError
from .geometry import HexagonGrid, HexagonMask, average_color
from .layers import LayerRenderer
from .settings import HexifySettings, QuantizationMethod
from .styles import BorderStyle, FillStyle, apply_alpha_channel

# Lazy import for matplotlib (heavy module)
_plt = None


def _get_plt():
    """Lazy load matplotlib.pyplot."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt

# Module-level logger
logger = logging.getLogger(__name__)


class HexagonProcessor:
    """
    Main processor for generating hexagonal pattern images.

    Orchestrates the full pipeline:
    1. Generate color palette from input image
    2. Set up hexagon grid layout
    3. Process each hexagon in parallel:
       a. Sample average color from input
       b. Generate multi-layer pattern
       c. Composite onto output image
    4. Cache patterns by color for efficiency

    The processor can be configured in two ways:
    1. Traditional parameters (backward compatible):
       processor = HexagonProcessor(num_palette_colors=16)

    2. Settings-based configuration:
       settings = HexifySettings(hex_width=128, num_layers=5)
       processor = HexagonProcessor(settings=settings)

    When settings are provided, they take precedence over individual parameters.

    Attributes:
        settings: HexifySettings configuration (or None for defaults)
        num_palette_colors: Number of colors in the palette
        num_processes: Number of parallel workers
        hexagons_dir: Directory to save individual hexagon images
        chunk_size: Number of hexagons per processing chunk
        save_hexagons: Whether to save individual hexagon images
        fast_mode: Whether to use MiniBatchKMeans for faster palette generation
        output_format: Output image format - "RGB" (default) or "RGBA" for transparency
    """

    def __init__(
        self,
        num_palette_colors: int = 16,
        num_processes: Optional[int] = None,
        hexagons_dir: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        save_hexagons: bool = True,
        fast_mode: bool = False,
        settings: Optional[HexifySettings] = None,
        output_format: Literal["RGB", "RGBA"] = "RGB",
        style_params: Optional[dict] = None
    ):
        """
        Initialize the hexagon processor.

        Args:
            num_palette_colors: Number of colors to extract from image
            num_processes: Number of parallel workers (default: CPU count)
            hexagons_dir: Directory to save hexagon images
            chunk_size: Hexagons per processing chunk
            save_hexagons: Whether to save individual hexagons
            fast_mode: If True, use MiniBatchKMeans for faster palette generation
            settings: Optional HexifySettings for full configuration control.
                     When provided, overrides num_palette_colors, chunk_size,
                     and fast_mode parameters.
            output_format: Output format - "RGB" (default) or "RGBA" for transparency.
                          When RGBA, pixels outside hexagons are fully transparent.
            style_params: Optional dict of style parameters for LayerRenderer:
                         - border_width: Width of hexagon border in pixels (0 = none)
                         - border_color: Border color as (R, G, B) tuple
                         - border_style: BorderStyle enum value
                         - fill_style: FillStyle enum value
                         - noise_intensity: Noise texture intensity (0.0-1.0)
        """
        # Store settings for access by other components
        self.settings = settings
        self._config = ConfigResolver(settings)

        # Determine effective values based on settings or parameters
        if settings is not None:
            self.num_palette_colors = settings.num_palette_colors
            self.chunk_size = settings.chunk_size
            self.fast_mode = settings.quantization_method == QuantizationMethod.MINIBATCH_KMEANS
        else:
            self.num_palette_colors = num_palette_colors
            self.chunk_size = chunk_size
            self.fast_mode = fast_mode

        # These are always taken from parameters (not in settings)
        self.num_processes = num_processes or os.cpu_count()
        self.hexagons_dir = hexagons_dir
        self.save_hexagons = save_hexagons

        # Output format and style parameters
        self.output_format = output_format
        self.style_params = style_params or {}

        # Initialize components with settings
        self.palette = ColorPalette(
            self.num_palette_colors,
            fast_mode=self.fast_mode,
            settings=settings
        )
        self.grid = HexagonGrid(settings=settings)

        # Shared cache for parallel processing
        # Use Manager for shared objects (required for ProcessPoolExecutor)
        # Note: Manager().Value has some IPC overhead but is required for
        # sharing between processes spawned by ProcessPoolExecutor
        manager = Manager()
        self.hexagon_cache = manager.dict()
        self.cache_lock = manager.Lock()
        self.cache_hits = manager.Value('i', 0)
        self.cache_misses = manager.Value('i', 0)

        # Will be set during processing
        self.input_image = None
        self._palette_generated = False

    @property
    def hex_centers(self):
        """Get hexagon centers from grid (for backward compatibility)."""
        return self.grid.hex_centers

    @property
    def palette_hash(self):
        """Get palette hash (for backward compatibility)."""
        return self.palette.palette_hash

    def generate_palette(self, image: np.ndarray) -> None:
        """
        Generate the color palette from an image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        self.palette.generate_from_image(image)
        self._palette_generated = True

    def setup_hexagon_grid(self, input_shape: tuple) -> None:
        """
        Set up the hexagon grid based on input dimensions.

        Args:
            input_shape: Shape of input image (height, width, channels)
        """
        self.grid.setup(input_shape)

    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validate that the input image has the correct format.

        Args:
            image: Image to validate

        Raises:
            InvalidImageError: If the image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise InvalidImageError(
                f"Expected numpy array, got {type(image).__name__}",
                shape=None
            )

        if image.ndim != 3:
            raise InvalidImageError(
                f"Expected 3D array (H, W, C), got {image.ndim}D array",
                shape=image.shape
            )

        if image.shape[2] != 3:
            raise InvalidImageError(
                f"Expected 3 channels (RGB), got {image.shape[2]} channels",
                shape=image.shape
            )

        if image.shape[0] < 1 or image.shape[1] < 1:
            raise InvalidImageError(
                f"Image dimensions must be positive, got {image.shape[:2]}",
                shape=image.shape
            )

    def process_image(self, input_image: np.ndarray, pbar=None) -> np.ndarray:
        """
        Process an input image to generate hexagonal pattern output.

        This is the main entry point for image processing. It handles
        palette generation, grid setup, and parallel hexagon processing.

        Args:
            input_image: Input image as numpy array (H, W, 3) in RGB format
            pbar: Optional progress bar to update

        Returns:
            Output image as numpy array, HEX_SCALE_FACTOR times larger

        Raises:
            InvalidImageError: If the input image has invalid format
        """
        # Validate input
        self._validate_image(input_image)
        logger.info(f"Processing image with shape {input_image.shape}")

        self.input_image = input_image

        # Generate palette if not already done
        if not self._palette_generated:
            logger.debug("Generating color palette...")
            self.generate_palette(input_image)
            logger.debug(f"Palette generated with {self.num_palette_colors} colors")

        # Set up grid if not already done
        if self.grid.hex_centers is None:
            logger.debug("Setting up hexagon grid...")
            self.setup_hexagon_grid(input_image.shape)
            logger.debug(f"Grid created with {len(self.grid.hex_centers)} hexagons")

        return self._process_hexagons(pbar)

    def _process_hexagons(self, pbar=None) -> np.ndarray:
        """
        Process all hexagons in parallel.

        Divides hexagon centers into chunks and processes them using
        a process pool for parallelization.

        Args:
            pbar: Optional progress bar

        Returns:
            Complete output image (RGB or RGBA based on output_format)
        """
        # Create output image with appropriate number of channels
        if self.output_format == "RGBA":
            # RGBA: 4 channels, alpha starts at 0 (fully transparent)
            output_shape = (*self.grid.output_shape[:2], 4)
            output_image = np.zeros(output_shape, dtype=np.uint8)
        else:
            # RGB: 3 channels (default)
            output_image = np.zeros(self.grid.output_shape, dtype=np.uint8)

        # Track hexagon mask for RGBA alpha channel
        if self.output_format == "RGBA":
            combined_hex_mask = np.zeros(self.grid.output_shape[:2], dtype=np.uint8)

        # Divide into chunks for parallel processing
        hex_center_chunks = [
            self.grid.hex_centers[i:i + self.chunk_size]
            for i in range(0, len(self.grid.hex_centers), self.chunk_size)
        ]

        logger.info(f"Processing {len(self.grid.hex_centers)} hexagons in {len(hex_center_chunks)} chunks")
        logger.debug(f"Using {self.num_processes} worker processes")

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                executor.submit(self._process_hexagon_chunk, chunk)
                for chunk in hex_center_chunks
            ]

            for future in as_completed(futures):
                results = future.result()
                for result in results:
                    if result:
                        x_start, y_start, x_end, y_end, hex_pattern_masked, mask_slice = result
                        # Composite hexagon onto output (RGB channels only)
                        hex_slice = output_image[y_start:y_end, x_start:x_end]
                        hex_slice[mask_slice != 0, :3] = hex_pattern_masked[mask_slice != 0]

                        # For RGBA, also update alpha channel and track mask
                        if self.output_format == "RGBA":
                            hex_slice[mask_slice != 0, 3] = 255  # Fully opaque
                            combined_hex_mask[y_start:y_end, x_start:x_end][mask_slice != 0] = 255

                if pbar:
                    pbar.update(self.chunk_size)

        logger.info(f"Processing complete. Cache hit rate: {self.get_cache_hit_rate():.1%}")
        return output_image

    def _process_hexagon_chunk(
        self, centers: List[Tuple[int, int]]
    ) -> List[Optional[Tuple[int, int, int, int, np.ndarray, np.ndarray]]]:
        """
        Process a chunk of hexagons.

        For each hexagon:
        1. Sample average color from input image
        2. Check cache for existing pattern
        3. Generate new pattern if not cached
        4. Prepare pattern for compositing

        Args:
            centers: List of (center_x, center_y) tuples

        Returns:
            List of results for each hexagon (or None for out-of-bounds)
        """
        local_cache = {}
        results = []

        # Create layer renderer for this chunk with style parameters
        renderer = LayerRenderer(
            self.palette,
            self.input_image,
            settings=self.settings,
            **self.style_params
        )

        for center_x, center_y in centers:
            # Get clipped bounds
            x_start, y_start, x_end, y_end = self.grid.get_hex_bounds(center_x, center_y)

            if x_start < x_end and y_start < y_end:
                # Convert to input coordinates
                input_center_x, input_center_y = self.grid.output_to_input_coords(center_x, center_y)
                input_hex_radius = self.grid.get_input_hex_radius()

                # Sample average color from input
                input_mask = HexagonMask.create(
                    input_center_x, input_center_y, input_hex_radius, self.input_image.shape[:2]
                )
                avg_rgb = average_color(self.input_image, input_mask)
                avg_rgb_key = tuple(map(int, avg_rgb))

                # Check caches for existing pattern
                if avg_rgb_key in local_cache:
                    hex_pattern = local_cache[avg_rgb_key]
                    with self.cache_lock:
                        self.cache_hits.value += 1
                elif avg_rgb_key in self.hexagon_cache:
                    hex_pattern = self.hexagon_cache[avg_rgb_key]
                    local_cache[avg_rgb_key] = hex_pattern
                    with self.cache_lock:
                        self.cache_hits.value += 1
                else:
                    # Generate new pattern
                    hex_pattern = renderer.create_hex_pattern(
                        center_x, center_y, self.grid.hex_radius, avg_rgb
                    )

                    # Cache the pattern
                    local_cache[avg_rgb_key] = hex_pattern
                    with self.cache_lock:
                        self.hexagon_cache[avg_rgb_key] = hex_pattern
                        self.cache_misses.value += 1

                    # Optionally save to disk
                    if self.save_hexagons and self.hexagons_dir:
                        hex_filename = f"{self.palette.palette_hash[:6]}_hexagon_{avg_rgb_key[0]:03d}_{avg_rgb_key[1]:03d}_{avg_rgb_key[2]:03d}.png"
                        hex_path = os.path.join(self.hexagons_dir, hex_filename)
                        _get_plt().imsave(hex_path, hex_pattern)

                # Create mask for compositing
                full_mask = HexagonMask.create(
                    center_x, center_y, self.grid.hex_radius, self.grid.output_shape[:2]
                )
                mask = full_mask[y_start:y_end, x_start:x_end]

                # Calculate pattern crop coordinates
                pattern_y_start = y_start - (center_y - self.grid.hex_radius)
                pattern_x_start = x_start - (center_x - self.grid.hex_radius)
                pattern_y_end = pattern_y_start + (y_end - y_start)
                pattern_x_end = pattern_x_start + (x_end - x_start)

                # Crop and mask the pattern
                hex_pattern_cropped = hex_pattern[pattern_y_start:pattern_y_end, pattern_x_start:pattern_x_end]
                hex_pattern_masked = cv2.bitwise_and(hex_pattern_cropped, hex_pattern_cropped, mask=mask)

                results.append((x_start, y_start, x_end, y_end, hex_pattern_masked, mask))
            else:
                results.append(None)

        return results

    def get_cache_hit_rate(self) -> float:
        """
        Calculate the cache hit rate.

        Returns:
            Hit rate as float between 0 and 1
        """
        total = self.cache_hits.value + self.cache_misses.value
        if total == 0:
            return 0
        return self.cache_hits.value / total
