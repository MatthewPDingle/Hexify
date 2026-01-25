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

Usage:
    from v2 import HexagonProcessor

    processor = HexagonProcessor(num_palette_colors=16)
    output = processor.process_image(input_image)
"""

from .processor import HexagonProcessor
from .color import ColorPalette
from .geometry import HexagonGrid, HexagonMask
from .layers import LayerRenderer

__all__ = [
    'HexagonProcessor',
    'ColorPalette',
    'HexagonGrid',
    'HexagonMask',
    'LayerRenderer',
]

__version__ = '2.0.0'
