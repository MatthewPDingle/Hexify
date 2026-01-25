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

Usage:
    from v2 import HexagonProcessor

    processor = HexagonProcessor(num_palette_colors=16)
    output = processor.process_image(input_image)

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

__all__ = [
    # Main classes
    'HexagonProcessor',
    'ColorPalette',
    'HexagonGrid',
    'HexagonMask',
    'LayerRenderer',
    # Exceptions
    'HexifyError',
    'InvalidImageError',
    'PaletteError',
    'PaletteNotGeneratedError',
    'InsufficientColorsError',
    'GridError',
    'GridNotSetupError',
]

__version__ = '2.1.0'
