"""
Preset configurations for common Hexify use cases.

This module provides pre-configured HexifySettings for common scenarios,
making it easy to quickly select an appropriate configuration.

Usage:
    from v2.presets import get_preset, PRESETS

    settings = get_preset("fast")
    processor = HexagonProcessor(settings=settings)

    # Or access presets directly
    settings = PRESETS["detailed"]
"""

from typing import Dict

from .settings import (
    HexifySettings,
    ColorSpace,
    HexOrientation,
    QuantizationMethod,
)


# =============================================================================
# Preset Definitions
# =============================================================================

PRESETS: Dict[str, HexifySettings] = {
    # Default settings - balanced quality and performance
    "default": HexifySettings(),

    # Fast processing - uses MiniBatch K-means and larger chunks
    "fast": HexifySettings(
        quantization_method=QuantizationMethod.MINIBATCH_KMEANS,
        chunk_size=200,
        kmeans_n_init=5,  # Fewer initializations for speed
    ),

    # Detailed output - smaller hexagons for more detail
    "detailed": HexifySettings(
        hex_width=128,  # Smaller hexagons = more detail
        num_layers=7,
        num_zones=12,
        chunk_size=64,  # Smaller chunks for detailed work
    ),

    # Minimal/artistic - larger hexagons for stylized look
    "minimal": HexifySettings(
        hex_width=512,  # Larger hexagons
        num_layers=5,
        num_zones=6,
        num_palette_colors=8,  # Fewer colors for more stylized look
    ),

    # High quality - best color matching (slower)
    "high_quality": HexifySettings(
        hex_width=256,
        num_layers=7,
        num_zones=12,
        color_space=ColorSpace.RGB,  # Can be changed to LAB when implemented
        kmeans_n_init=20,  # More initializations for better palette
        num_palette_colors=24,  # More colors for better matching
        max_palette_sample_pixels=2_000_000,  # Sample more pixels
    ),

    # Large format - for very large output images
    "large_format": HexifySettings(
        hex_width=512,
        hex_scale_factor=32,  # Even larger scale
        num_layers=7,
        num_zones=12,
        chunk_size=32,  # Smaller chunks due to larger hexagons
    ),

    # Thumbnail - for small preview images
    "thumbnail": HexifySettings(
        hex_width=64,
        hex_scale_factor=8,
        num_layers=5,
        num_zones=6,
        chunk_size=200,  # Larger chunks for small hexagons
        quantization_method=QuantizationMethod.MINIBATCH_KMEANS,
    ),

    # Pointy top orientation variant
    "pointy": HexifySettings(
        orientation=HexOrientation.POINTY_TOP,
    ),

    # Video processing optimized
    "video": HexifySettings(
        quantization_method=QuantizationMethod.MINIBATCH_KMEANS,
        chunk_size=150,
        kmeans_n_init=5,
        max_palette_sample_pixels=500_000,  # Faster palette generation
    ),
}


# =============================================================================
# Preset Access Functions
# =============================================================================

def get_preset(name: str) -> HexifySettings:
    """
    Get a preset configuration by name.

    Args:
        name: Name of the preset (case-insensitive)

    Returns:
        HexifySettings instance for the requested preset

    Raises:
        KeyError: If the preset name is not found

    Example:
        >>> settings = get_preset("fast")
        >>> settings.quantization_method
        <QuantizationMethod.MINIBATCH_KMEANS: 'minibatch_kmeans'>
    """
    name_lower = name.lower()
    if name_lower not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    return PRESETS[name_lower].copy()


def list_presets() -> Dict[str, str]:
    """
    Get a dictionary of preset names and their descriptions.

    Returns:
        Dictionary mapping preset names to brief descriptions
    """
    descriptions = {
        "default": "Balanced quality and performance",
        "fast": "Optimized for speed with MiniBatch K-means",
        "detailed": "Smaller hexagons for more detail",
        "minimal": "Large hexagons for stylized look",
        "high_quality": "Best color matching (slower)",
        "large_format": "For very large output images",
        "thumbnail": "Quick previews with small output",
        "pointy": "Pointy-top hexagon orientation",
        "video": "Optimized for video processing",
    }
    return descriptions


def create_custom_preset(
    base: str = "default",
    **overrides
) -> HexifySettings:
    """
    Create a custom preset based on an existing one.

    Args:
        base: Name of the base preset to start from
        **overrides: Settings to override from the base

    Returns:
        New HexifySettings with overrides applied

    Example:
        >>> settings = create_custom_preset(
        ...     base="fast",
        ...     hex_width=128,
        ...     num_palette_colors=32
        ... )
    """
    base_settings = get_preset(base)
    return base_settings.copy(**overrides)
