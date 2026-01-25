"""
Hexify v2 - Clean Architecture Refactor

This package provides a cleanly architected version of the Hexify hexagonal
pattern generator. The code has been restructured for clarity while maintaining
identical output to the original implementation.

Modules:
    config: All configuration constants and magic numbers
    color: Color palette generation and selection
    geometry: Hexagon grid and mask operations
    layers: Multi-layer hexagon pattern rendering
    processor: Main orchestration class
    exceptions: Custom exception classes
    settings: Configurable settings dataclass
    styles: Style augmentation utilities (borders, gradients, noise, transparency)
    cli: Command-line interface

Usage:
    # Python API:
    from v2 import HexagonProcessor

    processor = HexagonProcessor(num_palette_colors=16)
    output = processor.process_image(input_image)

    # With style effects:
    processor = HexagonProcessor(
        num_palette_colors=16,
        style_params={"border_width": 2, "border_style": BorderStyle.SOLID}
    )

    # With RGBA output (transparent background):
    processor = HexagonProcessor(output_format="RGBA")
    output = processor.process_image(input_image)  # 4-channel output

    # Command line:
    python -m v2 input.png -o output.png
    python -m v2 input.png --preset fast

Exceptions:
    HexifyError: Base exception for all Hexify errors
    InvalidImageError: Raised when input image format is invalid
    PaletteError: Raised when palette operations fail
"""

from .color import ColorPalette
from .exceptions import (
    GridError,
    GridNotSetupError,
    HexifyError,
    InsufficientColorsError,
    InvalidImageError,
    PaletteError,
    PaletteNotGeneratedError,
)
from .geometry import HexagonGrid, HexagonMask
from .layers import LayerRenderer
from .processor import HexagonProcessor
from .settings import HexifySettings, ColorSpace, HexOrientation, QuantizationMethod
from .presets import PRESETS, get_preset, list_presets, create_custom_preset
from .config import ConfigResolver
from .styles import (
    BorderStyle,
    FillStyle,
    STYLE_EFFECTS,
    add_hexagon_border,
    add_noise_texture,
    apply_alpha_channel,
    apply_radial_gradient,
    apply_linear_gradient,
    apply_vignette,
    apply_style_preset,
    get_style_preset,
)

__all__ = [
    # Main classes
    'HexagonProcessor',
    'ColorPalette',
    'HexagonGrid',
    'HexagonMask',
    'LayerRenderer',
    # Settings and Configuration
    'HexifySettings',
    'ColorSpace',
    'HexOrientation',
    'QuantizationMethod',
    'ConfigResolver',
    # Presets
    'PRESETS',
    'get_preset',
    'list_presets',
    'create_custom_preset',
    # Style augmentation
    'BorderStyle',
    'FillStyle',
    'STYLE_EFFECTS',
    'add_hexagon_border',
    'add_noise_texture',
    'apply_alpha_channel',
    'apply_radial_gradient',
    'apply_linear_gradient',
    'apply_vignette',
    'apply_style_preset',
    'get_style_preset',
    # Exceptions
    'HexifyError',
    'InvalidImageError',
    'PaletteError',
    'PaletteNotGeneratedError',
    'InsufficientColorsError',
    'GridError',
    'GridNotSetupError',
]

__version__ = '2.2.0'
