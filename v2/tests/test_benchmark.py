"""
Performance benchmarks for Hexify v2.

This module provides benchmarks for:
- Palette generation with KMeans vs MiniBatchKMeans
- Full image processing pipeline
- Cache hit rate analysis

Usage:
    pytest v2/tests/test_benchmark.py -v
    pytest v2/tests/test_benchmark.py -v --benchmark-only  # with pytest-benchmark

Note: Tests use synthetic images to ensure reproducibility across environments.
"""

import time
from typing import Tuple

import numpy as np
import pytest

# Import from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.color import ColorPalette
from v2.processor import HexagonProcessor


def create_test_image(size: Tuple[int, int] = (100, 100), seed: int = 42) -> np.ndarray:
    """
    Create a synthetic test image with various color regions.

    Args:
        size: Image size as (height, width)
        seed: Random seed for reproducibility

    Returns:
        RGB image as numpy array
    """
    np.random.seed(seed)
    height, width = size

    # Create a gradient background
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add horizontal gradient
    for x in range(width):
        image[:, x, 0] = int(255 * x / width)  # Red gradient

    # Add vertical gradient
    for y in range(height):
        image[y, :, 1] = int(255 * y / height)  # Green gradient

    # Add some random regions for color diversity
    for _ in range(5):
        x1 = np.random.randint(0, width - 20)
        y1 = np.random.randint(0, height - 20)
        x2 = x1 + np.random.randint(10, 20)
        y2 = y1 + np.random.randint(10, 20)
        color = np.random.randint(0, 256, 3)
        image[y1:y2, x1:x2] = color

    # Add blue channel based on position
    image[:, :, 2] = ((image[:, :, 0].astype(int) + image[:, :, 1].astype(int)) // 2).astype(np.uint8)

    return image


class TestPaletteGenerationBenchmark:
    """Benchmarks for palette generation comparing KMeans and MiniBatchKMeans."""

    @pytest.fixture
    def small_image(self):
        """Small test image (100x100)."""
        return create_test_image((100, 100))

    @pytest.fixture
    def medium_image(self):
        """Medium test image (500x500)."""
        return create_test_image((500, 500))

    @pytest.fixture
    def large_image(self):
        """Large test image (1000x1000)."""
        return create_test_image((1000, 1000))

    def test_kmeans_small_image(self, small_image):
        """Benchmark KMeans palette generation on small image."""
        palette = ColorPalette(num_colors=16, fast_mode=False)

        start = time.perf_counter()
        palette.generate_from_image(small_image)
        elapsed = time.perf_counter() - start

        assert palette.colors is not None
        assert len(palette.colors) == 16
        print(f"\nKMeans (100x100): {elapsed:.4f}s")

    def test_minibatch_kmeans_small_image(self, small_image):
        """Benchmark MiniBatchKMeans palette generation on small image."""
        palette = ColorPalette(num_colors=16, fast_mode=True)

        start = time.perf_counter()
        palette.generate_from_image(small_image)
        elapsed = time.perf_counter() - start

        assert palette.colors is not None
        assert len(palette.colors) == 16
        print(f"\nMiniBatchKMeans (100x100): {elapsed:.4f}s")

    def test_kmeans_medium_image(self, medium_image):
        """Benchmark KMeans palette generation on medium image."""
        palette = ColorPalette(num_colors=16, fast_mode=False)

        start = time.perf_counter()
        palette.generate_from_image(medium_image)
        elapsed = time.perf_counter() - start

        assert palette.colors is not None
        assert len(palette.colors) == 16
        print(f"\nKMeans (500x500): {elapsed:.4f}s")

    def test_minibatch_kmeans_medium_image(self, medium_image):
        """Benchmark MiniBatchKMeans palette generation on medium image."""
        palette = ColorPalette(num_colors=16, fast_mode=True)

        start = time.perf_counter()
        palette.generate_from_image(medium_image)
        elapsed = time.perf_counter() - start

        assert palette.colors is not None
        assert len(palette.colors) == 16
        print(f"\nMiniBatchKMeans (500x500): {elapsed:.4f}s")

    def test_kmeans_large_image(self, large_image):
        """Benchmark KMeans palette generation on large image."""
        palette = ColorPalette(num_colors=16, fast_mode=False)

        start = time.perf_counter()
        palette.generate_from_image(large_image)
        elapsed = time.perf_counter() - start

        assert palette.colors is not None
        assert len(palette.colors) == 16
        print(f"\nKMeans (1000x1000): {elapsed:.4f}s")

    def test_minibatch_kmeans_large_image(self, large_image):
        """Benchmark MiniBatchKMeans palette generation on large image."""
        palette = ColorPalette(num_colors=16, fast_mode=True)

        start = time.perf_counter()
        palette.generate_from_image(large_image)
        elapsed = time.perf_counter() - start

        assert palette.colors is not None
        assert len(palette.colors) == 16
        print(f"\nMiniBatchKMeans (1000x1000): {elapsed:.4f}s")

    def test_palette_quality_comparison(self, medium_image):
        """Compare palette quality between KMeans and MiniBatchKMeans."""
        palette_kmeans = ColorPalette(num_colors=16, fast_mode=False)
        palette_minibatch = ColorPalette(num_colors=16, fast_mode=True)

        palette_kmeans.generate_from_image(medium_image)
        palette_minibatch.generate_from_image(medium_image)

        # Both should produce valid palettes
        assert palette_kmeans.colors is not None
        assert palette_minibatch.colors is not None
        assert len(palette_kmeans.colors) == 16
        assert len(palette_minibatch.colors) == 16

        # Colors should be in valid range
        assert np.all(palette_kmeans.colors >= 0)
        assert np.all(palette_kmeans.colors <= 255)
        assert np.all(palette_minibatch.colors >= 0)
        assert np.all(palette_minibatch.colors <= 255)

        # Palettes may differ slightly due to different algorithms
        # but should have similar color distributions
        print(f"\nKMeans palette hash: {palette_kmeans.palette_hash[:16]}")
        print(f"MiniBatchKMeans palette hash: {palette_minibatch.palette_hash[:16]}")


class TestImageProcessingBenchmark:
    """Benchmarks for full image processing pipeline."""

    @pytest.fixture
    def small_image(self):
        """Small test image for processing."""
        return create_test_image((50, 50))

    def test_process_image_standard_mode(self, small_image):
        """Benchmark full image processing with standard KMeans."""
        processor = HexagonProcessor(
            num_palette_colors=8,
            num_processes=1,
            save_hexagons=False,
            fast_mode=False
        )

        start = time.perf_counter()
        output = processor.process_image(small_image)
        elapsed = time.perf_counter() - start

        assert output is not None
        assert output.shape[0] > small_image.shape[0]  # Should be scaled up
        assert output.shape[1] > small_image.shape[1]

        print(f"\nStandard mode (50x50): {elapsed:.4f}s")
        print(f"Cache hit rate: {processor.get_cache_hit_rate():.2%}")

    def test_process_image_fast_mode(self, small_image):
        """Benchmark full image processing with fast MiniBatchKMeans."""
        processor = HexagonProcessor(
            num_palette_colors=8,
            num_processes=1,
            save_hexagons=False,
            fast_mode=True
        )

        start = time.perf_counter()
        output = processor.process_image(small_image)
        elapsed = time.perf_counter() - start

        assert output is not None
        assert output.shape[0] > small_image.shape[0]
        assert output.shape[1] > small_image.shape[1]

        print(f"\nFast mode (50x50): {elapsed:.4f}s")
        print(f"Cache hit rate: {processor.get_cache_hit_rate():.2%}")

    def test_cache_effectiveness(self, small_image):
        """Test that cache is working effectively."""
        processor = HexagonProcessor(
            num_palette_colors=8,
            num_processes=1,
            save_hexagons=False,
            fast_mode=True
        )

        # Process the image
        processor.process_image(small_image)

        # Get cache statistics
        hits = processor.cache_hits.value
        misses = processor.cache_misses.value
        hit_rate = processor.get_cache_hit_rate()

        print(f"\nCache statistics:")
        print(f"  Hits: {hits}")
        print(f"  Misses: {misses}")
        print(f"  Hit rate: {hit_rate:.2%}")

        # For a typical image, we should have some cache hits
        # (identical colors in different regions reuse patterns)
        assert hits + misses > 0, "No cache operations recorded"


class TestSpeedComparison:
    """Direct speed comparisons between optimization modes."""

    def test_palette_speedup(self):
        """Measure speedup from using MiniBatchKMeans."""
        image = create_test_image((500, 500))

        # Standard KMeans
        palette_std = ColorPalette(num_colors=16, fast_mode=False)
        start = time.perf_counter()
        palette_std.generate_from_image(image)
        time_std = time.perf_counter() - start

        # MiniBatchKMeans
        palette_fast = ColorPalette(num_colors=16, fast_mode=True)
        start = time.perf_counter()
        palette_fast.generate_from_image(image)
        time_fast = time.perf_counter() - start

        speedup = time_std / time_fast if time_fast > 0 else float('inf')

        print(f"\nPalette generation speedup comparison:")
        print(f"  Standard KMeans: {time_std:.4f}s")
        print(f"  MiniBatch KMeans: {time_fast:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # MiniBatchKMeans should generally be faster (or at worst similar)
        # We don't assert specific speedup as it varies by system
        assert palette_std.colors is not None
        assert palette_fast.colors is not None


# Optional: pytest-benchmark compatible benchmarks
# These will run if pytest-benchmark is installed
try:
    import pytest_benchmark

    class TestBenchmarkPlugin:
        """Benchmarks using pytest-benchmark plugin."""

        def test_benchmark_kmeans(self, benchmark):
            """Benchmark KMeans with pytest-benchmark."""
            image = create_test_image((200, 200))
            palette = ColorPalette(num_colors=16, fast_mode=False)

            def run():
                palette.colors = None  # Reset
                palette.generate_from_image(image)

            benchmark(run)
            assert palette.colors is not None

        def test_benchmark_minibatch_kmeans(self, benchmark):
            """Benchmark MiniBatchKMeans with pytest-benchmark."""
            image = create_test_image((200, 200))
            palette = ColorPalette(num_colors=16, fast_mode=True)

            def run():
                palette.colors = None  # Reset
                palette.generate_from_image(image)

            benchmark(run)
            assert palette.colors is not None

except ImportError:
    # pytest-benchmark not installed, skip these tests
    pass


if __name__ == "__main__":
    # Run basic benchmarks when executed directly
    print("=" * 60)
    print("Hexify v2 Performance Benchmarks")
    print("=" * 60)

    # Create test images
    small = create_test_image((100, 100))
    medium = create_test_image((500, 500))
    large = create_test_image((1000, 1000))

    print("\n--- Palette Generation Benchmarks ---")

    for size_name, image in [("100x100", small), ("500x500", medium), ("1000x1000", large)]:
        print(f"\nImage size: {size_name}")

        # Standard KMeans
        palette = ColorPalette(num_colors=16, fast_mode=False)
        start = time.perf_counter()
        palette.generate_from_image(image)
        std_time = time.perf_counter() - start

        # MiniBatchKMeans
        palette = ColorPalette(num_colors=16, fast_mode=True)
        start = time.perf_counter()
        palette.generate_from_image(image)
        fast_time = time.perf_counter() - start

        speedup = std_time / fast_time if fast_time > 0 else 0
        print(f"  Standard: {std_time:.4f}s | Fast: {fast_time:.4f}s | Speedup: {speedup:.2f}x")

    print("\n--- Image Processing Benchmark ---")

    test_image = create_test_image((50, 50))

    # Standard mode
    processor = HexagonProcessor(
        num_palette_colors=8,
        num_processes=1,
        save_hexagons=False,
        fast_mode=False
    )
    start = time.perf_counter()
    output = processor.process_image(test_image)
    std_time = time.perf_counter() - start
    std_cache_rate = processor.get_cache_hit_rate()

    # Fast mode
    processor = HexagonProcessor(
        num_palette_colors=8,
        num_processes=1,
        save_hexagons=False,
        fast_mode=True
    )
    start = time.perf_counter()
    output = processor.process_image(test_image)
    fast_time = time.perf_counter() - start
    fast_cache_rate = processor.get_cache_hit_rate()

    print(f"Processing 50x50 image:")
    print(f"  Standard: {std_time:.4f}s (cache: {std_cache_rate:.2%})")
    print(f"  Fast: {fast_time:.4f}s (cache: {fast_cache_rate:.2%})")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
