"""Hexagon pattern processor for image transformation.

This module provides the HexagonProcessor class for transforming images
into hexagonal pattern art using color palette extraction and geometric
pattern generation.
"""

from __future__ import annotations

from multiprocessing import Manager, Lock
from multiprocessing.managers import ValueProxy, DictProxy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeAlias, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
import math
import os
import hashlib
from sklearn.cluster import KMeans
from tqdm import tqdm

# Type Aliases for improved readability
RGB: TypeAlias = tuple[int, int, int]
Coordinate: TypeAlias = tuple[int, int]
ImageArray: TypeAlias = np.ndarray
HexagonResult: TypeAlias = tuple[int, int, int, int, np.ndarray, np.ndarray] | None


class HexagonProcessor:
    """Processes images into hexagonal pattern art.

    This class handles the transformation of input images into stylized
    hexagonal patterns by extracting color palettes and generating
    geometric hexagon patterns based on local image colors.

    Attributes:
        num_palette_colors: Number of colors in the extracted palette.
        num_processes: Number of parallel processes for processing.
        hexagons_dir: Directory to save individual hexagon patterns.
        chunk_size: Number of hexagons to process per chunk.
        save_hexagons: Whether to save individual hexagon images.
        palette: Extracted color palette from the image.
        palette_hash: SHA256 hash of the palette for caching.
        hex_centers: List of hexagon center coordinates.
    """

    # Validation constants
    MIN_IMAGE_SIZE: int = 16
    MIN_PALETTE_COLORS: int = 5
    MAX_PALETTE_COLORS: int = 256

    def __init__(
        self,
        num_palette_colors: int = 16,
        num_processes: int | None = None,
        hexagons_dir: str | None = None,
        chunk_size: int = 32,
        save_hexagons: bool = True,
    ) -> None:
        """Initialize the HexagonProcessor.

        Args:
            num_palette_colors: Number of colors for the palette (5-256).
            num_processes: Number of parallel processes. Defaults to CPU count.
            hexagons_dir: Directory path for saving hexagon images.
            chunk_size: Number of hexagons processed per chunk.
            save_hexagons: Whether to save individual hexagon images.

        Raises:
            ValueError: If num_palette_colors is outside valid range (5-256).
            ValueError: If num_processes is less than 1.
            ValueError: If chunk_size is less than 1.
        """
        # Validate palette colors
        if num_palette_colors < self.MIN_PALETTE_COLORS:
            raise ValueError(
                f"num_palette_colors must be at least {self.MIN_PALETTE_COLORS}, "
                f"got {num_palette_colors}"
            )
        if num_palette_colors > self.MAX_PALETTE_COLORS:
            raise ValueError(
                f"num_palette_colors must be at most {self.MAX_PALETTE_COLORS}, "
                f"got {num_palette_colors}"
            )

        # Validate process count
        effective_processes = num_processes or os.cpu_count() or 1
        if num_processes is not None and num_processes < 1:
            raise ValueError(f"num_processes must be at least 1, got {num_processes}")

        # Validate chunk size
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")

        self.num_palette_colors: int = num_palette_colors
        self.num_processes: int = effective_processes
        self.hexagons_dir: str | None = hexagons_dir
        self.hex_centers: list[Coordinate] | None = None
        self.chunk_size: int = chunk_size
        self.save_hexagons: bool = save_hexagons

        # Create a manager for shared objects
        manager = Manager()
        self.hexagon_cache: DictProxy[RGB, ImageArray] = manager.dict()
        self.cache_hits: ValueProxy[int] = manager.Value("i", 0)
        self.cache_misses: ValueProxy[int] = manager.Value("i", 0)
        self.cache_lock: Lock = manager.Lock()

        # Initialize palette and palette_hash
        self.palette: ImageArray | None = None
        self.palette_hash: str | None = None

        # Internal state
        self.input_image: ImageArray | None = None
        self.output_shape: tuple[int, int, int] | None = None
        self.hex_width: int = 0
        self.hex_height: int = 0
        self.hex_radius: int = 0

    def process_image(
        self, input_image: ImageArray, pbar: tqdm | None = None
    ) -> ImageArray:
        """Process an image through the hexagon filter.

        Transforms the input image into a hexagonal pattern at 16x resolution.
        Generates a color palette if not already set, sets up the hexagon grid,
        and processes all hexagons.

        Args:
            input_image: RGB image array with shape (height, width, 3).
            pbar: Optional progress bar for tracking processing.

        Returns:
            Processed image at 16x resolution as numpy array.

        Raises:
            ValueError: If image is smaller than 16x16 pixels.
            ValueError: If image does not have 3 color channels.
        """
        # Validate image dimensions
        if input_image.shape[0] < self.MIN_IMAGE_SIZE:
            raise ValueError(
                f"Image height must be at least {self.MIN_IMAGE_SIZE} pixels, "
                f"got {input_image.shape[0]}"
            )
        if input_image.shape[1] < self.MIN_IMAGE_SIZE:
            raise ValueError(
                f"Image width must be at least {self.MIN_IMAGE_SIZE} pixels, "
                f"got {input_image.shape[1]}"
            )
        if len(input_image.shape) != 3 or input_image.shape[2] != 3:
            raise ValueError(
                f"Image must have 3 color channels (RGB), got shape {input_image.shape}"
            )

        self.input_image = input_image
        if self.palette is None:
            self.generate_palette(input_image)
        if self.hex_centers is None:
            self.setup_hexagon_grid(input_image.shape)
        return self.process_hexagons(pbar)

    def generate_palette(self, image: ImageArray) -> None:
        """Generate a color palette from the input image using K-means clustering.

        Extracts the most representative colors from the image and sorts them
        by brightness for consistent results.

        Args:
            image: RGB image array with shape (height, width, 3).
        """
        # Resize the image if it's too large to speed up processing
        max_pixels: int = 1000000  # 1 million pixels
        height, width = image.shape[:2]
        if height * width > max_pixels:
            scale: float = np.sqrt(max_pixels / (height * width))
            new_height: int = int(height * scale)
            new_width: int = int(width * scale)
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        pixels: ImageArray = image.reshape(-1, 3)

        # Use a fixed random state for reproducibility
        kmeans = KMeans(n_clusters=self.num_palette_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        palette: ImageArray = kmeans.cluster_centers_

        # Sort the palette by brightness for consistency
        sorted_palette_indices: ImageArray = np.argsort(np.mean(palette, axis=1))
        self.palette = palette[sorted_palette_indices]

        # Round the palette values to integers for consistent hashing
        self.palette = np.round(self.palette).astype(int)

        # Generate hash from the sorted, rounded palette
        self.palette_hash = hashlib.sha256(self.palette.tobytes()).hexdigest()

    def setup_hexagon_grid(self, input_shape: tuple[int, ...]) -> None:
        """Set up the hexagon grid based on input image dimensions.

        Calculates hexagon centers for the output image which is 16x the
        input resolution.

        Args:
            input_shape: Shape tuple of the input image (height, width, channels).
        """
        self.output_shape = (input_shape[0] * 16, input_shape[1] * 16, 3)
        self.hex_width = 256
        self.hex_height = round(self.hex_width * (math.sqrt(3) / 2))
        self.hex_radius = self.hex_width // 2
        hex_horizontal_spacing: float = self.hex_width * 0.75
        hex_vertical_spacing: int = self.hex_height

        cols: int = int(self.output_shape[1] / hex_horizontal_spacing) + 2
        # BUG FIX: Was using hex_horizontal_spacing instead of hex_vertical_spacing for row calculation
        # This could result in too few rows for tall images
        rows: int = int(self.output_shape[0] / hex_vertical_spacing) + 2

        self.hex_centers = [
            (
                int(hex_horizontal_spacing * col),
                int(
                    hex_vertical_spacing * row
                    + (0.5 * hex_vertical_spacing if col % 2 else 0)
                ),
            )
            for row in range(rows)
            for col in range(cols)
        ]

    def process_hexagons(self, pbar: tqdm | None = None) -> ImageArray:
        """Process all hexagons in parallel and compose the output image.

        Uses multiprocessing to generate hexagon patterns for each grid
        position and combines them into the final output image.

        Args:
            pbar: Optional progress bar for tracking progress.

        Returns:
            The composed output image as a numpy array.
        """
        output_image: ImageArray = np.zeros(self.output_shape, dtype=np.uint8)

        # Group hex_centers into chunks
        hex_center_chunks: list[list[Coordinate]] = [
            self.hex_centers[i : i + self.chunk_size]
            for i in range(0, len(self.hex_centers), self.chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                executor.submit(self.process_hexagon_chunk, chunk)
                for chunk in hex_center_chunks
            ]

            for future in as_completed(futures):
                results: list[HexagonResult] = future.result()
                for result in results:
                    if result:
                        x_start, y_start, x_end, y_end, hex_pattern_masked, mask_slice = (
                            result
                        )
                        hex_slice = output_image[y_start:y_end, x_start:x_end]
                        hex_slice[mask_slice != 0] = hex_pattern_masked[mask_slice != 0]

                if pbar:
                    # BUG FIX: Update by actual chunk size, not self.chunk_size
                    # The final chunk may be smaller, causing progress bar overshoot
                    pbar.update(len(results))

        return output_image

    def process_hexagon_chunk(
        self, centers: list[Coordinate]
    ) -> list[HexagonResult]:
        """Process a chunk of hexagon centers.

        Generates hexagon patterns for a batch of center coordinates.
        Note: Caching was removed as inner layers depend on position (center_x, center_y),
        making cached patterns incorrect for different positions.

        Args:
            centers: List of (x, y) center coordinates for hexagons.

        Returns:
            List of hexagon results containing position and pattern data,
            or None for invalid positions.
        """
        results: list[HexagonResult] = []

        for center_x, center_y in centers:
            x_start: int = max(center_x - self.hex_radius, 0)
            y_start: int = max(center_y - self.hex_radius, 0)
            x_end: int = min(center_x + self.hex_radius, self.output_shape[1])
            y_end: int = min(center_y + self.hex_radius, self.output_shape[0])

            if x_start < x_end and y_start < y_end:
                input_center_x: int = int(center_x / 16)
                input_center_y: int = int(center_y / 16)
                input_hex_radius: int = self.hex_radius // 16

                input_mask: ImageArray = self.create_hex_mask(
                    input_center_x,
                    input_center_y,
                    input_hex_radius,
                    self.input_image.shape[:2],
                )
                avg_rgb: ImageArray = self.average_color(self.input_image, input_mask)

                hex_pattern: ImageArray = self.create_hex_pattern(
                    center_x, center_y, self.hex_radius, avg_rgb
                )

                if self.save_hexagons and self.hexagons_dir:
                    avg_rgb_key: RGB = tuple(map(int, avg_rgb))
                    hex_filename: str = (
                        f"{self.palette_hash[:6]}_hexagon_"
                        f"{avg_rgb_key[0]:03d}_{avg_rgb_key[1]:03d}_{avg_rgb_key[2]:03d}_"
                        f"{center_x}_{center_y}.png"
                    )
                    hex_path: str = os.path.join(self.hexagons_dir, hex_filename)
                    plt.imsave(hex_path, hex_pattern)

                full_mask: ImageArray = self.create_hex_mask(
                    center_x, center_y, self.hex_radius, self.output_shape[:2]
                )
                mask: ImageArray = full_mask[y_start:y_end, x_start:x_end]

                pattern_y_start: int = y_start - (center_y - self.hex_radius)
                pattern_x_start: int = x_start - (center_x - self.hex_radius)
                pattern_y_end: int = pattern_y_start + (y_end - y_start)
                pattern_x_end: int = pattern_x_start + (x_end - x_start)

                hex_pattern_cropped: ImageArray = hex_pattern[
                    pattern_y_start:pattern_y_end, pattern_x_start:pattern_x_end
                ]
                hex_pattern_masked: ImageArray = cv2.bitwise_and(
                    hex_pattern_cropped, hex_pattern_cropped, mask=mask
                )

                results.append(
                    (x_start, y_start, x_end, y_end, hex_pattern_masked, mask)
                )
            else:
                results.append(None)

        return results

    def get_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate for hexagon pattern caching.

        Returns:
            Cache hit rate as a float between 0.0 and 1.0.
        """
        total_accesses: int = self.cache_hits.value + self.cache_misses.value
        if total_accesses == 0:
            return 0.0
        return self.cache_hits.value / total_accesses

    @staticmethod
    def create_hex_mask(
        center_x: int, center_y: int, radius: int, shape: tuple[int, int]
    ) -> ImageArray:
        """Create a hexagonal mask at the specified position.

        Args:
            center_x: X coordinate of hexagon center.
            center_y: Y coordinate of hexagon center.
            radius: Radius of the hexagon.
            shape: Shape of the output mask (height, width).

        Returns:
            Binary mask array with hexagon filled with 255.
        """
        mask: ImageArray = np.zeros(shape, dtype=np.uint8)
        hexagon = RegularPolygon(
            (center_x, center_y), numVertices=6, radius=radius, orientation=np.pi / 2
        )
        coords: ImageArray = hexagon.get_verts()
        coords = np.clip(coords, [0, 0], [shape[1] - 1, shape[0] - 1]).astype(int)
        mask = cv2.fillPoly(mask, [coords], 255)
        return mask

    @staticmethod
    def average_color(image: ImageArray, mask: ImageArray) -> ImageArray:
        """Calculate the average color within a masked region.

        Args:
            image: RGB image array.
            mask: Binary mask array.

        Returns:
            Average RGB color as integer array.
        """
        masked: ImageArray = cv2.bitwise_and(image, image, mask=mask)
        avg_color: tuple[float, ...] = cv2.mean(masked, mask=mask)[:3]
        return np.round(avg_color).astype(int)

    def create_hex_pattern(
        self, center_x: int, center_y: int, radius: int, avg_rgb: ImageArray
    ) -> ImageArray:
        """Create a hexagon pattern with layered color zones.

        Generates a stylized hexagon pattern with alternating layers of
        colors selected from the palette based on the average color.

        Args:
            center_x: X coordinate of hexagon center in output space.
            center_y: Y coordinate of hexagon center in output space.
            radius: Radius of the hexagon pattern.
            avg_rgb: Average RGB color for the hexagon region.

        Returns:
            Hexagon pattern image as numpy array.
        """
        bw_rgb: int = 0 if np.mean(avg_rgb) < 128 else 255
        pattern: ImageArray = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
        if bw_rgb != 0:
            pattern = np.full((2 * radius, 2 * radius, 3), bw_rgb, dtype=np.uint8)

        layer_areas: dict[int, float] = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 1: 0}
        layer_radii: dict[int, int] = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 1: 0}

        bw_color: RGB = (0, 0, 0) if np.mean(avg_rgb) < 128 else (255, 255, 255)
        avoid_rgb: list[ImageArray] = []

        for i in range(7, 0, -1):
            if i % 2 == 1:
                hex_radius: int = int(radius * (i / 7))
                layer_radii[i] = hex_radius
                pattern, layer_area = self.fill_odd_layer(
                    pattern, radius, hex_radius, bw_color
                )
                layer_areas[i] = layer_area
            elif i % 2 == 0:
                pattern, layer_area = self.fill_even_layer(
                    i,
                    avg_rgb,
                    self.palette,
                    radius,
                    layer_radii,
                    avoid_rgb,
                    layer_areas,
                    pattern,
                    self.input_image,
                    center_x,
                    center_y,
                    bw_color,
                )
                layer_areas[i] = layer_area

        return pattern

    @staticmethod
    def fill_odd_layer(
        pattern: ImageArray, radius: int, hex_radius: int, bw_color: RGB
    ) -> tuple[ImageArray, float]:
        """Fill an odd-numbered layer with solid color.

        Args:
            pattern: The pattern image to draw on.
            radius: Base radius for positioning.
            hex_radius: Radius of this layer's hexagon.
            bw_color: Black or white color tuple.

        Returns:
            Tuple of (updated pattern, layer area).
        """
        inner_hexagon = RegularPolygon(
            (radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2
        )
        inner_coords: ImageArray = inner_hexagon.get_verts().astype(int)
        pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))
        hex_area: float = 3 * math.sqrt(3) * (hex_radius**2) / 2
        return pattern, hex_area

    def fill_even_layer(
        self,
        i: int,
        avg_rgb: ImageArray,
        palette_rgb: ImageArray,
        radius: int,
        layer_radii: dict[int, int],
        avoid_rgb: list[ImageArray],
        layer_areas: dict[int, float],
        pattern: ImageArray,
        input_image: ImageArray,
        center_x: int,
        center_y: int,
        bw_color: RGB,
    ) -> tuple[ImageArray, float]:
        """Fill an even-numbered layer with color zones.

        Creates alternating wedge patterns using colors from the palette.

        Args:
            i: Layer index (2, 4, or 6).
            avg_rgb: Average color for the region.
            palette_rgb: Color palette array.
            radius: Base radius for positioning.
            layer_radii: Dictionary of radii for each layer.
            avoid_rgb: List of colors to avoid using.
            layer_areas: Dictionary of areas for each layer.
            pattern: The pattern image to draw on.
            input_image: Original input image.
            center_x: X coordinate of hexagon center.
            center_y: Y coordinate of hexagon center.
            bw_color: Black or white background color.

        Returns:
            Tuple of (updated pattern, layer area).
        """
        brightness: float = np.mean(avg_rgb)
        if i == 6:
            diameter: float = 256 - abs(brightness - 128) * (256 - 192) / 128
        else:
            diameter = (layer_radii[i + 1] * 2) - abs(brightness - 128) * (64) / 128
        hex_radius: int = int(diameter // 2)
        inner_hexagon = RegularPolygon(
            (radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2
        )
        inner_coords: ImageArray = inner_hexagon.get_verts().astype(int)

        scaled_center_x: int = center_x // 16
        scaled_center_y: int = center_y // 16
        scaled_radius: int = hex_radius // 16

        input_mask: ImageArray = self.create_hex_mask(
            scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2]
        )
        avg_rgb = self.average_color(input_image, input_mask)

        color_1: ImageArray = self.closest_palette_color(avg_rgb, palette_rgb, avoid_rgb)
        avoid_rgb.append(color_1)
        dist_1: float = np.linalg.norm(color_1 - avg_rgb)

        color_adj: ImageArray = min(
            palette_rgb,
            key=lambda color: (
                np.linalg.norm(color - color_1)
                if not np.array_equal(color, color_1)
                else float("inf")
            ),
        )
        dist_adj: float = np.linalg.norm(color_1 - color_adj)

        percentage_off: float = (
            min(dist_1, dist_adj) / max(dist_1, dist_adj) if dist_adj != 0 else 0
        )
        even_angle: float = percentage_off * 60
        odd_angle: float = 60 - even_angle

        even_area: float = 6 * (0.5 * hex_radius * hex_radius * math.radians(even_angle))
        odd_area: float = 6 * (0.5 * hex_radius * hex_radius * math.radians(odd_angle))

        layer_area: float = even_area + odd_area

        color_2: ImageArray = self.select_color_2(
            avg_rgb,
            palette_rgb,
            color_1,
            even_area,
            odd_area,
            bw_color,
            layer_areas[i + 1] - layer_area,
        )
        avoid_rgb.append(color_2)

        color_1_rgb: RGB = tuple(map(int, color_1))
        color_2_rgb: RGB = tuple(map(int, color_2))

        angle_offset: float = 30 - (even_angle / 2)
        for zone in range(12):
            angle: float = even_angle if zone % 2 == 0 else odd_angle
            angle_color: RGB = color_1_rgb if zone % 2 == 0 else color_2_rgb

            start_angle: float = angle_offset
            end_angle: float = start_angle + angle
            angle_offset += angle

            p1: tuple[float, float] = (
                radius + hex_radius * math.cos(math.radians(start_angle)),
                radius + hex_radius * math.sin(math.radians(start_angle)),
            )
            p2: tuple[float, float] = (
                radius + hex_radius * math.cos(math.radians(end_angle)),
                radius + hex_radius * math.sin(math.radians(end_angle)),
            )

            p12: tuple[float, float] = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            d: float = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5 / 2

            length_of_shorter_leg: float = d / math.sqrt(3)

            dx: float = p12[0] - p1[0]
            dy: float = p12[1] - p1[1]

            length: float = (dx**2 + dy**2) ** 0.5
            if length > 1e-6:
                dx /= length
                dy /= length
                p3: tuple[float, float] = (
                    p12[0] + length_of_shorter_leg * dy,
                    p12[1] - length_of_shorter_leg * dx,
                )
            else:
                p3 = p1

            p1 = self.clip_point_to_hexagon(p1, inner_coords)
            p2 = self.clip_point_to_hexagon(p2, inner_coords)
            p3 = self.clip_point_to_hexagon(p3, inner_coords)

            vertices1: ImageArray = np.array(
                [(radius, radius), p1, p2], dtype=np.int32
            )
            cv2.fillPoly(pattern, [vertices1], angle_color)

            if zone % 2 == 1:
                vertices2: ImageArray = np.array([p1, p2, p3], dtype=np.int32)
                cv2.fillPoly(pattern, [vertices2], angle_color)

        return pattern, layer_area

    @staticmethod
    def closest_palette_color(
        avg_rgb: ImageArray,
        palette_rgb: ImageArray,
        avoid_rgb: list[ImageArray] | None = None,
    ) -> ImageArray:
        """Find the closest palette color to the target color.

        Args:
            avg_rgb: Target RGB color to match.
            palette_rgb: Array of palette colors.
            avoid_rgb: Optional list of colors to avoid.

        Returns:
            The closest palette color as numpy array.
        """
        min_distance: float = float("inf")
        closest_color: ImageArray | None = None
        # BUG FIX: Track the closest color ignoring avoid list as fallback
        # This prevents returning None when all colors are in avoid_rgb
        fallback_color: ImageArray | None = None
        fallback_distance: float = float("inf")

        for color in palette_rgb:
            distance: float = np.linalg.norm(avg_rgb - color)

            # Track fallback (closest color ignoring avoid list)
            if distance < fallback_distance:
                fallback_distance = distance
                fallback_color = color

            if avoid_rgb is not None and any(
                np.array_equal(color, avoid_color) for avoid_color in avoid_rgb
            ):
                continue

            if distance < min_distance:
                min_distance = distance
                closest_color = color

        # Return closest non-avoided color, or fallback to closest color if all avoided
        return closest_color if closest_color is not None else fallback_color

    @staticmethod
    def select_color_2(
        avg_rgb: ImageArray,
        palette_rgb: ImageArray,
        color_1: ImageArray,
        color_1_area: float,
        color_2_area: float,
        bw_color: RGB,
        bw_color_area: float,
    ) -> ImageArray:
        """Select a second color that best complements the first.

        Chooses a color from the palette that, when mixed with color_1 and
        the background, best approximates the target average color.

        Args:
            avg_rgb: Target average color.
            palette_rgb: Array of palette colors.
            color_1: First selected color.
            color_1_area: Area covered by color_1.
            color_2_area: Area to be covered by color_2.
            bw_color: Background black/white color.
            bw_color_area: Area covered by background.

        Returns:
            Best complementary color as numpy array.
        """
        min_distance: float = float("inf")
        best_color_2: ImageArray | None = None

        color_1 = np.array(color_1)
        bw_color_arr: ImageArray = np.array(bw_color)

        for color in palette_rgb:
            if np.array_equal(color, color_1):
                continue

            color = np.array(color)

            mixed_rgb: ImageArray = (
                color_1_area * color_1 + color_2_area * color + bw_color_area * bw_color_arr
            ) / (color_1_area + color_2_area + bw_color_area)

            distance: float = np.linalg.norm(avg_rgb - mixed_rgb)

            if distance < min_distance:
                min_distance = distance
                best_color_2 = color

        return best_color_2

    @staticmethod
    def clip_point_to_hexagon(
        point: tuple[float, float], hex_coords: ImageArray
    ) -> tuple[float, float]:
        """Clip a point to be inside or on the hexagon boundary.

        Args:
            point: Point coordinates (x, y).
            hex_coords: Array of hexagon vertex coordinates.

        Returns:
            Clipped point coordinates.
        """
        path = Path(hex_coords)
        if path.contains_point(point):
            return point

        min_dist: float = float("inf")
        nearest_point: tuple[float, float] = point
        for i in range(len(hex_coords)):
            p1: ImageArray = hex_coords[i]
            p2: ImageArray = hex_coords[(i + 1) % len(hex_coords)]
            nearest: tuple[float, float] = HexagonProcessor.nearest_point_on_segment(
                point, p1, p2
            )
            dist: float = np.linalg.norm(np.array(nearest) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest_point = nearest
        return nearest_point

    @staticmethod
    def nearest_point_on_segment(
        point: tuple[float, float],
        p1: ImageArray | tuple[float, float],
        p2: ImageArray | tuple[float, float],
    ) -> tuple[float, float]:
        """Find the nearest point on a line segment to a given point.

        Args:
            point: The point to find nearest segment point for.
            p1: First endpoint of the segment.
            p2: Second endpoint of the segment.

        Returns:
            Nearest point on the segment as (x, y) tuple.
        """
        px, py = point
        x1, y1 = p1
        x2, y2 = p2

        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            return (float(x1), float(y1))

        t: float = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        return (x1 + t * dx, y1 + t * dy)
