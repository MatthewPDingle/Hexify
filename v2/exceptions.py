"""
Custom exceptions for Hexify v2.

This module defines domain-specific exceptions that provide clear error messages
and allow callers to handle specific error conditions appropriately.
"""


class HexifyError(Exception):
    """Base exception for all Hexify errors."""
    pass


class InvalidImageError(HexifyError):
    """Raised when an input image has invalid format or dimensions."""

    def __init__(self, message: str, shape: tuple = None):
        """
        Initialize the exception.

        Args:
            message: Description of the validation failure
            shape: The actual shape of the invalid image (optional)
        """
        self.shape = shape
        super().__init__(message)


class PaletteError(HexifyError):
    """Raised when palette generation or color selection fails."""
    pass


class PaletteNotGeneratedError(PaletteError):
    """Raised when attempting to use a palette before generation."""

    def __init__(self):
        super().__init__("Palette must be generated before use. Call generate_from_image() first.")


class InsufficientColorsError(PaletteError):
    """Raised when the palette has fewer colors than required."""

    def __init__(self, required: int, available: int):
        """
        Initialize the exception.

        Args:
            required: Number of colors required
            available: Number of colors available
        """
        self.required = required
        self.available = available
        super().__init__(f"Palette has {available} colors but {required} are required.")


class GridError(HexifyError):
    """Raised when hexagon grid setup or operations fail."""
    pass


class GridNotSetupError(GridError):
    """Raised when attempting to use grid before setup."""

    def __init__(self):
        super().__init__("Grid must be set up before use. Call setup() first.")
