"""
Style augmentation for hexagon patterns.

This module provides style rendering utilities for enhancing hexagon patterns
with borders, gradients, noise textures, and transparency effects.

All style features are opt-in to maintain backward compatibility with the
existing rendering pipeline.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from enum import Enum


class BorderStyle(Enum):
    """Border styles for hexagon edges."""
    NONE = "none"
    SOLID = "solid"
    DOUBLE = "double"
    GLOW = "glow"


class FillStyle(Enum):
    """Fill styles for hexagon patterns."""
    SOLID = "solid"           # Current default
    RADIAL_GRADIENT = "radial_gradient"
    LINEAR_GRADIENT = "linear_gradient"


def add_hexagon_border(
    pattern: np.ndarray,
    hex_coords: np.ndarray,
    border_width: int = 2,
    border_color: Tuple[int, int, int] = (0, 0, 0),
    border_style: BorderStyle = BorderStyle.SOLID
) -> np.ndarray:
    """
    Add a border around the hexagon edge.

    Draws a border along the hexagon perimeter using the specified style.
    The border is drawn on top of the existing pattern.

    Args:
        pattern: The pattern array to draw on (H, W, 3 or 4)
        hex_coords: Array of hexagon vertex coordinates, shape (N, 2)
        border_width: Width of the border in pixels
        border_color: Border color as (R, G, B) tuple
        border_style: Style of the border (SOLID, DOUBLE, or GLOW)

    Returns:
        Pattern with border applied
    """
    if border_style == BorderStyle.NONE or border_width <= 0:
        return pattern

    result = pattern.copy()
    coords = hex_coords.astype(np.int32)

    if border_style == BorderStyle.SOLID:
        # Simple solid border
        cv2.polylines(result, [coords], isClosed=True, color=border_color, thickness=border_width)

    elif border_style == BorderStyle.DOUBLE:
        # Double line border - outer and inner lines
        outer_width = border_width
        inner_width = max(1, border_width // 2)
        gap = max(1, border_width // 2)

        # Draw outer line
        cv2.polylines(result, [coords], isClosed=True, color=border_color, thickness=outer_width)

        # Calculate inner hexagon coords (scaled down)
        center = np.mean(coords, axis=0)
        scale_factor = 1 - (outer_width + gap) / np.linalg.norm(coords[0] - center)
        inner_coords = (center + (coords - center) * scale_factor).astype(np.int32)
        cv2.polylines(result, [inner_coords], isClosed=True, color=border_color, thickness=inner_width)

    elif border_style == BorderStyle.GLOW:
        # Glow effect - multiple fading layers
        num_layers = min(border_width, 5)
        for i in range(num_layers, 0, -1):
            # Calculate alpha for this layer (fades outward)
            alpha = i / num_layers
            layer_width = border_width - (num_layers - i)

            if layer_width > 0:
                # Blend color toward white for glow effect
                glow_color = tuple(
                    int(c * alpha + 255 * (1 - alpha) * 0.5)
                    for c in border_color
                )
                cv2.polylines(result, [coords], isClosed=True, color=glow_color, thickness=layer_width)

        # Draw the core border
        cv2.polylines(result, [coords], isClosed=True, color=border_color, thickness=1)

    return result


def apply_radial_gradient(
    pattern: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    inner_color: Tuple[int, int, int],
    outer_color: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply a radial gradient fill.

    Creates a smooth color transition from the center outward.
    If a mask is provided, the gradient is only applied within the masked region.

    Args:
        pattern: The pattern array to modify (H, W, 3 or 4)
        center: Center point of the gradient (x, y)
        radius: Radius of the gradient in pixels
        inner_color: Color at the center as (R, G, B)
        outer_color: Color at the edge as (R, G, B)
        mask: Optional binary mask (H, W) where gradient is applied

    Returns:
        Pattern with radial gradient applied
    """
    result = pattern.copy()
    h, w = pattern.shape[:2]

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Calculate distance from center for each pixel
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Normalize distance to [0, 1] range
    normalized_dist = np.clip(dist / max(radius, 1), 0, 1)

    # Interpolate colors
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        gradient[:, :, i] = (
            inner_color[i] * (1 - normalized_dist) +
            outer_color[i] * normalized_dist
        )

    gradient = gradient.astype(np.uint8)

    # Apply with mask if provided
    if mask is not None:
        mask_bool = mask > 0
        # Handle both RGB and RGBA patterns
        for i in range(3):
            result[:, :, i] = np.where(mask_bool, gradient[:, :, i], result[:, :, i])
    else:
        result[:, :, :3] = gradient

    return result


def apply_linear_gradient(
    pattern: np.ndarray,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    start_color: Tuple[int, int, int],
    end_color: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply a linear gradient fill.

    Creates a smooth color transition along a line from start to end point.
    If a mask is provided, the gradient is only applied within the masked region.

    Args:
        pattern: The pattern array to modify (H, W, 3 or 4)
        start_point: Start point of the gradient (x, y)
        end_point: End point of the gradient (x, y)
        start_color: Color at the start as (R, G, B)
        end_color: Color at the end as (R, G, B)
        mask: Optional binary mask (H, W) where gradient is applied

    Returns:
        Pattern with linear gradient applied
    """
    result = pattern.copy()
    h, w = pattern.shape[:2]

    # Calculate gradient direction vector
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    length = np.sqrt(dx**2 + dy**2)

    if length < 1:
        return result

    # Normalize direction
    dx /= length
    dy /= length

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Project each pixel onto the gradient line
    # Distance along the gradient direction from start point
    proj_dist = (x - start_point[0]) * dx + (y - start_point[1]) * dy

    # Normalize to [0, 1] range
    normalized_dist = np.clip(proj_dist / length, 0, 1)

    # Interpolate colors
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        gradient[:, :, i] = (
            start_color[i] * (1 - normalized_dist) +
            end_color[i] * normalized_dist
        )

    gradient = gradient.astype(np.uint8)

    # Apply with mask if provided
    if mask is not None:
        mask_bool = mask > 0
        for i in range(3):
            result[:, :, i] = np.where(mask_bool, gradient[:, :, i], result[:, :, i])
    else:
        result[:, :, :3] = gradient

    return result


def add_noise_texture(
    pattern: np.ndarray,
    intensity: float = 0.1,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Add subtle noise/grain for artistic effect.

    Applies random noise to the pattern to create a textured appearance.
    The noise is additive and respects the color channel ranges [0, 255].

    Args:
        pattern: The pattern array to modify (H, W, 3 or 4)
        intensity: Noise intensity from 0.0 (none) to 1.0 (maximum).
                   Values around 0.05-0.15 work well for subtle texture.
        mask: Optional binary mask (H, W) where noise is applied

    Returns:
        Pattern with noise texture applied
    """
    if intensity <= 0:
        return pattern

    result = pattern.copy()
    h, w = pattern.shape[:2]

    # Clamp intensity to reasonable range
    intensity = min(max(intensity, 0), 1.0)

    # Generate random noise
    # Scale is based on intensity - max deviation of ~25 at intensity=0.1
    noise_scale = intensity * 255 * 0.1
    noise = np.random.randn(h, w, 3) * noise_scale

    # Apply noise to RGB channels
    if mask is not None:
        mask_3d = np.expand_dims(mask > 0, axis=2)
        noisy = result[:, :, :3].astype(np.float32) + noise * mask_3d
    else:
        noisy = result[:, :, :3].astype(np.float32) + noise

    # Clip to valid range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    result[:, :, :3] = noisy

    return result


def apply_alpha_channel(
    pattern: np.ndarray,
    mask: np.ndarray,
    background_alpha: int = 0
) -> np.ndarray:
    """
    Convert RGB pattern to RGBA with transparency.

    Creates an RGBA image where the alpha channel is determined by the mask.
    Pixels inside the mask are fully opaque, pixels outside have the specified
    background alpha value.

    Args:
        pattern: The pattern array (H, W, 3) in RGB format
        mask: Binary mask (H, W) defining the opaque region
        background_alpha: Alpha value for pixels outside the mask (0-255).
                         0 = fully transparent, 255 = fully opaque.

    Returns:
        Pattern as RGBA image (H, W, 4)
    """
    h, w = pattern.shape[:2]

    # Create RGBA output
    if pattern.shape[2] == 4:
        result = pattern.copy()
    else:
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = pattern[:, :, :3]

    # Create alpha channel from mask
    alpha = np.where(mask > 0, 255, background_alpha).astype(np.uint8)
    result[:, :, 3] = alpha

    return result


def apply_vignette(
    pattern: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    strength: float = 0.3,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply a vignette effect (darkening towards edges).

    Creates a subtle darkening effect that increases with distance from center.

    Args:
        pattern: The pattern array to modify (H, W, 3 or 4)
        center: Center point of the vignette (x, y)
        radius: Radius where darkening begins
        strength: Vignette strength from 0.0 (none) to 1.0 (maximum darkness)
        mask: Optional binary mask (H, W) where vignette is applied

    Returns:
        Pattern with vignette effect applied
    """
    if strength <= 0:
        return pattern

    result = pattern.copy()
    h, w = pattern.shape[:2]

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Calculate distance from center
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Calculate vignette factor (1 at center, decreasing toward edges)
    vignette = 1 - np.clip((dist / max(radius, 1)) * strength, 0, strength)

    # Apply vignette to RGB channels
    if mask is not None:
        mask_bool = mask > 0
        for i in range(3):
            darkened = (result[:, :, i] * vignette).astype(np.uint8)
            result[:, :, i] = np.where(mask_bool, darkened, result[:, :, i])
    else:
        for i in range(3):
            result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)

    return result


# =============================================================================
# Style Effect Presets
# =============================================================================

STYLE_EFFECTS: Dict[str, Dict[str, Any]] = {
    "classic": {},  # No effects, current behavior
    "outlined": {
        "border_width": 2,
        "border_color": (0, 0, 0),
        "border_style": BorderStyle.SOLID
    },
    "glowing": {
        "border_width": 3,
        "border_style": BorderStyle.GLOW,
        "border_color": (255, 255, 255)
    },
    "textured": {
        "noise_intensity": 0.05
    },
    "neon": {
        "border_width": 2,
        "border_color": (255, 255, 255),
        "border_style": BorderStyle.GLOW,
        "noise_intensity": 0.02
    },
    "vintage": {
        "noise_intensity": 0.08,
        "vignette_strength": 0.2
    },
    "double_border": {
        "border_width": 4,
        "border_color": (0, 0, 0),
        "border_style": BorderStyle.DOUBLE
    },
}


def get_style_preset(name: str) -> Dict[str, Any]:
    """
    Get a style preset by name.

    Args:
        name: Name of the style preset

    Returns:
        Dictionary of style parameters

    Raises:
        ValueError: If the preset name is not found
    """
    if name not in STYLE_EFFECTS:
        available = ", ".join(STYLE_EFFECTS.keys())
        raise ValueError(f"Unknown style preset '{name}'. Available: {available}")
    return STYLE_EFFECTS[name].copy()


def apply_style_preset(
    pattern: np.ndarray,
    hex_coords: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    preset_name: str,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply a named style preset to a hexagon pattern.

    Convenience function that applies all effects from a preset.

    Args:
        pattern: The pattern array to modify
        hex_coords: Hexagon vertex coordinates
        center: Center point of the hexagon
        radius: Radius of the hexagon
        preset_name: Name of the style preset to apply
        mask: Optional binary mask for effects

    Returns:
        Pattern with style preset applied
    """
    style = get_style_preset(preset_name)

    if not style:
        return pattern

    result = pattern.copy()

    # Apply noise if specified
    if "noise_intensity" in style:
        result = add_noise_texture(result, style["noise_intensity"], mask)

    # Apply vignette if specified
    if "vignette_strength" in style:
        result = apply_vignette(result, center, radius, style["vignette_strength"], mask)

    # Apply border if specified
    if "border_width" in style:
        result = add_hexagon_border(
            result,
            hex_coords,
            border_width=style.get("border_width", 2),
            border_color=style.get("border_color", (0, 0, 0)),
            border_style=style.get("border_style", BorderStyle.SOLID)
        )

    return result
