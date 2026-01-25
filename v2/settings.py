"""
Configuration settings for Hexify.

This module provides a configurable settings system that allows users to
customize hexagon generation parameters. Settings can be loaded from YAML
files or created programmatically.

The settings system maintains backward compatibility - if no settings are
provided, the default values from config.py are used.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import math


class ColorSpace(Enum):
    """Color space for color matching algorithms."""
    RGB = "rgb"      # Standard RGB color space (current default)
    LAB = "lab"      # CIE LAB - better perceptual matching
    HSV = "hsv"      # Hue-Saturation-Value space


class HexOrientation(Enum):
    """Orientation of hexagons in the grid."""
    FLAT_TOP = "flat_top"      # Current default (pi/2 rotation)
    POINTY_TOP = "pointy_top"  # No rotation (0 radians)


class QuantizationMethod(Enum):
    """Method for color palette quantization."""
    KMEANS = "kmeans"                    # Standard K-means (current default)
    MINIBATCH_KMEANS = "minibatch_kmeans"  # Faster, slightly less accurate


@dataclass
class HexifySettings:
    """
    Configuration settings for Hexify hexagon pattern generation.

    This dataclass provides a clean interface for configuring all aspects
    of hexagon generation, from geometry to color processing.

    Attributes:
        hex_width: Width of each hexagon in output pixels (default: 256)
        hex_scale_factor: Output image scale factor from input (default: 16)
        num_layers: Number of concentric layers in each hexagon (default: 7)
        num_zones: Number of angular zones in even layers (default: 12)
        orientation: Hexagon orientation - flat top or pointy top
        color_space: Color space for color matching
        quantization_method: Method for palette generation
        border_width: Width of hexagon borders (0 = no border)
        border_color: RGB color tuple for borders
        chunk_size: Number of hexagons per processing chunk
        num_palette_colors: Number of colors in the palette (default: 16)
        kmeans_random_state: Random seed for K-means reproducibility
        kmeans_n_init: Number of K-means initializations
        max_palette_sample_pixels: Max pixels to sample for palette generation
        brightness_threshold: Threshold for black/white background selection

    Example:
        >>> settings = HexifySettings(hex_width=128, num_layers=5)
        >>> processor = HexagonProcessor(settings=settings)
    """

    # Hexagon geometry
    hex_width: int = 256
    hex_scale_factor: int = 16
    num_layers: int = 7
    num_zones: int = 12
    orientation: HexOrientation = HexOrientation.FLAT_TOP

    # Color settings
    color_space: ColorSpace = ColorSpace.RGB
    quantization_method: QuantizationMethod = QuantizationMethod.KMEANS
    num_palette_colors: int = 16

    # K-means settings
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10
    max_palette_sample_pixels: int = 1_000_000

    # Style settings (for future use)
    border_width: int = 0
    border_color: Tuple[int, int, int] = (0, 0, 0)

    # Processing settings
    chunk_size: int = 100

    # Brightness threshold for background selection
    brightness_threshold: int = 128

    # Layer diameter settings
    layer_6_max_diameter: int = 256
    layer_6_min_diameter: int = 192
    inner_layer_diameter_range: int = 64

    def __post_init__(self):
        """Validate settings after initialization."""
        if self.hex_width < 16:
            raise ValueError("hex_width must be at least 16")
        if self.hex_scale_factor < 1:
            raise ValueError("hex_scale_factor must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.num_zones < 2:
            raise ValueError("num_zones must be at least 2")
        if self.num_palette_colors < 2:
            raise ValueError("num_palette_colors must be at least 2")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    @property
    def hex_height(self) -> int:
        """Calculate hexagon height for regular hexagon proportions."""
        return round(self.hex_width * (math.sqrt(3) / 2))

    @property
    def hex_radius(self) -> int:
        """Calculate hexagon radius (half of width)."""
        return self.hex_width // 2

    @property
    def hex_horizontal_spacing(self) -> float:
        """Calculate horizontal spacing between hexagon centers."""
        return self.hex_width * 0.75

    @property
    def hex_vertical_spacing(self) -> int:
        """Calculate vertical spacing between hexagon centers."""
        return self.hex_height

    @property
    def hex_orientation_radians(self) -> float:
        """Get hexagon orientation in radians."""
        if self.orientation == HexOrientation.FLAT_TOP:
            return math.pi / 2
        else:  # POINTY_TOP
            return 0.0

    @property
    def base_zone_angle(self) -> float:
        """Calculate base angle per zone in degrees."""
        return 360 / self.num_zones

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to a dictionary.

        Enum values are converted to their string values for serialization.

        Returns:
            Dictionary representation of settings
        """
        return {
            'hex_width': self.hex_width,
            'hex_scale_factor': self.hex_scale_factor,
            'num_layers': self.num_layers,
            'num_zones': self.num_zones,
            'orientation': self.orientation.value,
            'color_space': self.color_space.value,
            'quantization_method': self.quantization_method.value,
            'num_palette_colors': self.num_palette_colors,
            'kmeans_random_state': self.kmeans_random_state,
            'kmeans_n_init': self.kmeans_n_init,
            'max_palette_sample_pixels': self.max_palette_sample_pixels,
            'border_width': self.border_width,
            'border_color': list(self.border_color),
            'chunk_size': self.chunk_size,
            'brightness_threshold': self.brightness_threshold,
            'layer_6_max_diameter': self.layer_6_max_diameter,
            'layer_6_min_diameter': self.layer_6_min_diameter,
            'inner_layer_diameter_range': self.inner_layer_diameter_range,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HexifySettings':
        """
        Create settings from a dictionary.

        Enum string values are converted back to enum instances.

        Args:
            data: Dictionary with settings values

        Returns:
            HexifySettings instance
        """
        # Make a copy to avoid modifying the input
        data = data.copy()

        # Convert enum string values to enum instances
        if 'orientation' in data and isinstance(data['orientation'], str):
            data['orientation'] = HexOrientation(data['orientation'])
        if 'color_space' in data and isinstance(data['color_space'], str):
            data['color_space'] = ColorSpace(data['color_space'])
        if 'quantization_method' in data and isinstance(data['quantization_method'], str):
            data['quantization_method'] = QuantizationMethod(data['quantization_method'])

        # Convert border_color list to tuple if needed
        if 'border_color' in data and isinstance(data['border_color'], list):
            data['border_color'] = tuple(data['border_color'])

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> 'HexifySettings':
        """
        Load settings from a YAML file.

        Requires PyYAML to be installed. Falls back to JSON if YAML is
        not available.

        Args:
            path: Path to YAML configuration file

        Returns:
            HexifySettings instance

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If the file does not exist
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config file support. "
                "Install it with: pip install pyyaml"
            )

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """
        Save settings to a YAML file.

        Requires PyYAML to be installed.

        Args:
            path: Path to save the YAML configuration file

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config file support. "
                "Install it with: pip install pyyaml"
            )

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_json(cls, path: str) -> 'HexifySettings':
        """
        Load settings from a JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            HexifySettings instance

        Raises:
            FileNotFoundError: If the file does not exist
        """
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_json(self, path: str, indent: int = 2) -> None:
        """
        Save settings to a JSON file.

        Args:
            path: Path to save the JSON configuration file
            indent: Indentation level for pretty printing
        """
        import json

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    def copy(self, **overrides) -> 'HexifySettings':
        """
        Create a copy of settings with optional overrides.

        Args:
            **overrides: Setting values to override in the copy

        Returns:
            New HexifySettings instance with overrides applied
        """
        data = self.to_dict()

        # Convert enums back for from_dict processing
        for key, value in overrides.items():
            if isinstance(value, Enum):
                data[key] = value.value
            else:
                data[key] = value

        return self.from_dict(data)
